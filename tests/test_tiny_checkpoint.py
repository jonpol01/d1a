"""A tiny random Gemma 4 checkpoint in the served layout, built here without downloading anything: d1a.training.train writes it (one
LoRA step on a 6-layer random base with sliding and KV-shared layers), d1a.backends.checkpoint loads it and d1a.serving.serve answers with
it. Every scoring path is checked against an independent reference: each question as its own plain causal row through the
transformers model, with transformers' own causal and sliding-window masks, and the readout positions found by token id.
The checks hold for any weights, so every PR runs them without the real checkpoints; each mutation test breaks one piece
of the path and requires the checks to catch it (#33)."""
import json
import math
import sys

import pytest
import torch

S, F = "sliding_attention", "full_attention"
WINDOW = 4   # far below the states and branches below, so the sliding mask matters


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(checkpoint dir, tokenizer, model) for a random Gemma 4 trained one LoRA step by d1a.training.train."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    from d1a.training import train
    from d1a.backends.checkpoint import LoadOptions, load
    from d1a.backends.torch import GEMMA_SPECIAL
    root = tmp_path_factory.mktemp("tiny-gemma4")
    words = "it is charged twice which team billing shipping refund angry the customer how bad no yes".split()
    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", *GEMMA_SPECIAL, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    config = Gemma4TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
                              head_dim=16, global_head_dim=32, num_key_value_heads=1, num_global_key_value_heads=1, num_kv_shared_layers=2,
                              hidden_size_per_layer_input=8, vocab_size_per_layer_input=len(vocab), sliding_window=WINDOW,
                              layer_types=[S, S, F, S, S, F], pad_token_id=0, bos_token_id=2, eos_token_id=3)
    torch.manual_seed(0)
    Gemma4ForCausalLM(config).save_pretrained(root / "base")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>").save_pretrained(root / "base")
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(8)]
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = ["d1a.training.train", "--base", str(root / "base"), "--data", str(root / "data.jsonl"), "--device", "cpu", "--batch", "2",
            "--lr", "1e-2", "--lora", "4", "--max_steps", "1", "--out", str(root / "checkpoint")]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", argv); train.main()
    tok, model = load(str(root / "checkpoint"), "cpu", LoadOptions(dtype=torch.float32, temperature=1.0))
    return root / "checkpoint", tok, model.eval()


def request(state_words=12):
    """A System One request with all three question types; the state is several sliding windows long."""
    from d1a.core.api import SystemOneRequest
    return SystemOneRequest(model="d1a-latest", state=" ".join(["the customer is charged twice it"] * (state_words // 6 + 1)), questions={
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": "refund it", "refund": None}},
        "angry": {"type": "noul", "instructions": "is the customer angry"},
        "how": {"type": "score", "instructions": "how bad is it", "criteria": ["no", "it is bad", "twice bad"]}})


def reference_logits(model, tok, rec):
    """Raw pointer logits per question (temperature 1), computed without d1a.backends.torch's encoder, masks or forms: each question
    is the plain causal row <bos><state>state <q>instr (<opt>option</opt>)... <decide> run through the transformers model
    with its own masks, read at the </opt> and <decide> tokens found by id, and scored by the head's formula."""
    from d1a.backends.torch import layout, user_tokens
    leading, (s_id, q_id, o_id, c_id, d_id), _ = layout(tok)
    state = leading + [s_id] + user_tokens(tok, rec["state"])
    out = []
    with torch.no_grad():
        for q in rec["questions"]:
            row = state + [q_id] + user_tokens(tok, q["instr"])
            for o in q["options"]: row += [o_id] + user_tokens(tok, o) + [c_id]
            row += [d_id]
            h = model.lm(input_ids=torch.tensor([row]), position_ids=torch.arange(len(row))[None]).last_hidden_state[0].float()
            ends = [i for i, t in enumerate(row) if t == c_id]
            head = model.head
            out.append((head.k(h[ends]) @ head.q(h[len(row) - 1])) / math.sqrt(head.q.out_features))
    return out


def record_of(req):
    from d1a.core.api import to_record
    return to_record(req)[0]


def gap(paths, reference):
    """The largest probability difference between any scoring path and the reference (inf if a path has another shape)."""
    worst = 0.0
    for got in paths:
        if len(got) != len(reference): return math.inf
        for p, z in zip(got, reference):
            p = torch.as_tensor(p).flatten()
            if p.shape != z.shape: return math.inf
            worst = max(worst, (p - torch.softmax(z, -1)).abs().max().item())
    return worst


def scoring_paths(model, tok, rec):
    """Probabilities from every way d1a scores one request: the packed pass, the rows form, a state-prefix miss and hit,
    and the serving batch."""
    with torch.no_grad():
        enc = model.encode(tok, rec)
        packed = model.probs(enc)
        rows = [torch.softmax(z, -1) for z in model.forward_rows_batch([enc])[0]]
        miss, prefix = model.probs_and_prefix(enc)
        hit = model.probs_with_prefix(enc, prefix)
        batch = model.probs_batch([enc, enc], [None, prefix], [False, False])[0]
    return [packed, rows, miss, hit, *batch]


def served(ck_dir, tok, model, req):
    from d1a.backends.checkpoint import Checkpoint
    from d1a.serving.serve import Server
    s = Server(Checkpoint(str(ck_dir)), tok, model, "cpu")
    try: return s.answer(req)["answers"]
    finally: s.close()


def served_gap(ck_dir, tok, model, req):
    """The largest difference between the served answers (rounded to 4 decimals) and the reference at the model's
    temperature, each answer read from its documented field independently of d1a.core.api: a choice's probabilities by
    criteria key, a noul's P(true) as its "yes" option, a score's probabilities by level index ("0" = the first level)."""
    reference = dict(zip(req.questions, reference_logits(model, tok, record_of(req))))
    answers, worst = served(ck_dir, tok, model, req), 0.0
    for qid, q in req.questions.items():
        expected, a = torch.softmax(reference[qid] / model.head.temperature, -1), answers[qid]
        if a["type"] != q.type: return math.inf
        if q.type == "noul": got = [1 - a["noul"], a["noul"]]
        else:
            keys = list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria))]
            if sorted(a["probabilities"]) != sorted(keys): return math.inf
            got = [a["probabilities"][k] for k in keys]
            if q.type == "choice" and a["choice"] != keys[int(expected.argmax())]: return math.inf
        worst = max(worst, (torch.tensor(got) - expected).abs().max().item())
    return worst


def test_every_scoring_path_matches_the_reference(tiny):
    _, tok, model = tiny
    assert model.backbone.name == "gemma4" and not model.hybrid and model.sliding_window == WINDOW
    for words in (3, 12, 30):                       # inside one window, a few windows, many windows
        rec = record_of(request(words))
        assert gap(scoring_paths(model, tok, rec), reference_logits(model, tok, rec)) < 1e-5


def test_question_order_does_not_change_any_answer(tiny):
    """Questions never see each other (block-causal mask), so reordering them only reorders the answers."""
    _, tok, model = tiny
    rec = record_of(request())
    flipped = {**rec, "questions": rec["questions"][::-1]}
    with torch.no_grad():
        a, b = model.probs(model.encode(tok, rec)), model.probs(model.encode(tok, flipped))
    for x, y in zip(a, b[::-1]):
        assert torch.allclose(x, y, atol=1e-6)


def test_served_answers_are_keyed_and_tempered(tiny):
    """d1a.serving.serve end to end: each key's probability is its own option's, at the checkpoint's temperature."""
    from d1a.backends.checkpoint import LoadOptions, load
    ck_dir, tok, model = tiny
    assert served_gap(ck_dir, tok, model, request()) < 1e-4   # 4-decimal answers
    _, hot = load(str(ck_dir), "cpu", LoadOptions(dtype=torch.float32, temperature=2.5))
    assert hot.eval().head.temperature == 2.5
    assert served_gap(ck_dir, tok, hot, request()) < 1e-4


# --- mutations: each breaks one piece of the scoring path, and the checks above must catch it -------------------------

def shift(field, by):
    def mutate(mp):
        import d1a.backends.torch as M
        real = M.encode
        def encode(*a, **k):
            enc = real(*a, **k)
            if field == "opt_idx": enc["opt_idx"] = [[i + by for i in oi] for oi in enc["opt_idx"]]
            else: enc["decide_idx"] = [i + by for i in enc["decide_idx"]]
            return enc
        mp.setattr(M, "encode", encode)
    return mutate


def leaky_mask(mp):
    """Questions attend to each other's tokens."""
    import d1a.backends.torch as M
    real = M.branch_mask_batch
    mp.setattr(M, "branch_mask_batch", lambda segs, *a, **k: real([[0] * len(seg) for seg in segs], *a, **k))


def no_sliding_window(mp):
    """The sliding layers get the full mask."""
    import d1a.backends.torch as M
    real = M.branch_masks
    mp.setattr(M, "branch_masks", lambda encs, device, dtype, window, length=None: real(encs, device, dtype, None, length=length))


def positions_continue(mp):
    """Branch positions run on across questions instead of restarting after the state."""
    import d1a.backends.torch as M
    real = M.encode
    mp.setattr(M, "encode", lambda *a, **k: {**(e := real(*a, **k)), "pos": list(range(len(e["ids"])))})


def no_bos(mp):
    """The leading <bos> Gemma's attention relies on is dropped."""
    import d1a.backends.torch as M
    real = M.layout
    mp.setattr(M, "layout", lambda tok: ([], *real(tok)[1:]))


@pytest.mark.parametrize("mutate", [shift("opt_idx", -1), shift("decide_idx", -1), leaky_mask, no_sliding_window, positions_continue],
                         ids=["option-readout-off-by-one", "decide-readout-off-by-one", "questions-see-each-other", "sliding-window-ignored", "positions-do-not-restart"])
def test_scoring_mutations_are_caught(tiny, mutate, monkeypatch):
    _, tok, model = tiny
    rec = record_of(request(30))
    mutate(monkeypatch)
    try: paths = scoring_paths(model, tok, rec)
    except ValueError as e:   # caught by d1a.backends.torch.rows_of's layout check instead
        assert "branch layout mismatch" in str(e); return
    assert gap(paths, reference_logits(model, tok, rec)) > 1e-3


def test_dropping_the_leading_bos_is_caught(tiny, monkeypatch):
    """The reference reads the layout too, so this mutation is checked against the unmutated reference."""
    _, tok, model = tiny
    rec = record_of(request(30))
    reference = reference_logits(model, tok, rec)
    no_bos(monkeypatch)
    assert gap(scoring_paths(model, tok, rec), reference) > 1e-3


def ignored_temperature(mp):
    import d1a.backends.torch as M
    mp.setattr(M.PointerHead, "forward", lambda self, hd, ho: (self.k(ho) @ self.q(hd)) * self.scale)
    mp.setattr(M.PointerHead, "many", lambda self, hd, ho, owner: (self.k(ho) * self.q(hd)[owner]).sum(-1) * self.scale)


def answers_misread(qtype, how):
    """d1a.serving.serve maps one question type's probabilities to the wrong options."""
    def mutate(mp):
        import d1a.serving.serve as Srv
        real = Srv.to_answers
        mp.setattr(Srv, "to_answers", lambda probs, meta: real([how(p) if m["type"] == qtype else p for p, m in zip(probs, meta)], meta))
    return mutate


@pytest.mark.parametrize("mutate", [ignored_temperature, answers_misread("noul", lambda p: p[::-1]), answers_misread("score", lambda p: p[::-1]),
                                    answers_misread("choice", lambda p: p[1:] + p[:1])],
                         ids=["temperature-ignored", "noul-reads-no", "score-levels-reversed", "choice-keys-shifted"])
def test_serving_mutations_are_caught(tiny, mutate, monkeypatch):
    from d1a.backends.checkpoint import LoadOptions, load
    ck_dir, tok, _ = tiny
    _, hot = load(str(ck_dir), "cpu", LoadOptions(dtype=torch.float32, temperature=2.5))
    mutate(monkeypatch)
    assert served_gap(ck_dir, tok, hot.eval(), request()) > 1e-2
