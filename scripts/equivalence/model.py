"""d1a.model against the module at --ref, bit for bit, on the committed tiny Gemma 4 (tests/golden/tiny-gemma4/base) and a
tiny random Qwen3.5 (hybrid): the module-level functions (encode, masks, rows_of, contexts) on generated records, then the
old and new DecisionModel built from the same seed (their weights checked equal first) through every scoring path
(forward_batch, the row form, probs, the prefix miss and hit, probs_batch, the shared prefix) and training-mode
gradients.

    uv run python scripts/equivalence/model.py                   # against the commit before the rewrite (default origin/main)
"""
import random
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Checker, arguments, module_at  # noqa: E402

import d1a.model as new  # noqa: E402

WORDS = "the customer is charged twice which team billing shipping refund angry how bad no yes order late lost box arrived wrong size".split()


def qwen_base(root):
    """A 2-layer random Qwen3.5 (one Gated DeltaNet layer) with a word-level tokenizer, saved like a Hub snapshot."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from d1a.model import SPECIAL
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *WORDS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"], pad_token_id=1)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(root / "qwen")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(root / "qwen")
    return root / "qwen"


def records(rng, n):
    out = []
    for _ in range(n):
        qs = [{"instr": " ".join(rng.choices(WORDS, k=rng.randint(1, 5))), "options": [" ".join(rng.choices(WORDS, k=rng.randint(1, 3))) for _ in range(rng.randint(1, 5))],
               "label": 0} for _ in range(rng.randint(1, 5))]
        out.append({"state": " ".join(rng.choices(WORDS, k=rng.choice([1, 6, 30, 90]))) + rng.choice(["", " <|box_end|>", " <bos>"]), "questions": qs})
    return out


def main():
    a = arguments(__doc__.split("\n")[0], "origin/main", 60)
    old, rng, check = module_at(a.ref, "d1a/model.py"), random.Random(a.seed), Checker()
    for name in ("SPECIAL", "GEMMA_SPECIAL", "MAX_STATE", "MAX_BRANCH", "MAX_PACKED", "SERVE_MAX_STATE", "SERVE_MAX_BRANCH", "SERVE_MAX_PACKED",
                 "ROW_PASS_TOKENS", "MAX_TRAIN_STATE", "SERVE_MAX_STATE_8K", "SERVE_MAX_BRANCH_8K", "MAX_TRAIN_STATE_8K", "SCORING_INTERFACE", "EAGER_STATES"):
        check.equal(name, getattr(old, name), getattr(new, name))
    for s in (384, 1000, 65536, 100, 70000):
        check.same("training_context", lambda: old.training_context(s), lambda: new.training_context(s))
    check.same("rows_per_pass", lambda: old.rows_per_pass([[0] * 30] * 5, 270), lambda: new.rows_per_pass([[0] * 30] * 5, 270))
    qwen = qwen_base(Path(tempfile.mkdtemp()))
    for base in (str(ROOT / "tests/golden/tiny-gemma4/base"), str(qwen)):
        tok = new.load_tokenizer(base)
        check.same("layout", lambda: old.layout(tok)[:2], lambda: new.layout(tok)[:2])
        recs = records(rng, a.iterations)
        for rec in recs:
            for kw in ({}, {"max_state": 8}, {"max_state": 8, "strict": True}, {"max_branch": 20}):
                check.same("encode", lambda: old.encode(tok, rec, **kw), lambda: new.encode(tok, rec, **kw))
            check.same("fits", lambda: old.fits(rec, tok), lambda: new.fits(rec, tok))
        encs = [new.encode(tok, r, max_state=new.SERVE_MAX_STATE, max_branch=new.SERVE_MAX_BRANCH) for r in recs]
        for e in encs[:20]:
            check.same("rows_of", lambda: old.rows_of(e), lambda: new.rows_of(e))
            check.same("branch_mask", lambda: old.branch_mask(e["seg"], "cpu"), lambda: new.branch_mask(e["seg"], "cpu"))
            for window in (None, 4):
                check.same("branch_masks", lambda: old.branch_masks([e, encs[0]], "cpu", torch.float32, window), lambda: new.branch_masks([e, encs[0]], "cpu", torch.float32, window))
        models = []
        for m in (old, new):
            torch.manual_seed(0)
            models.append(m.DecisionModel(base, tok, "cpu", lora=4))
        mo, mn = models
        check.equal("weights", {k: v for k, v in mo.state_dict().items()}, {k: v for k, v in mn.state_dict().items()})
        mo.eval(); mn.eval()
        for model in models:
            model.head.temperature = 1.7
        with torch.no_grad():
            check.same("forward_batch", lambda: mo.forward_batch(encs[:6]), lambda: mn.forward_batch(encs[:6]))
            check.same("forward_rows_batch", lambda: mo.forward_rows_batch(encs[:6]), lambda: mn.forward_rows_batch(encs[:6]))
            if mo.hybrid:
                check.same("shared_prefix", lambda: mo.forward_batch(encs[:6], True), lambda: mn.forward_batch(encs[:6], True))
            for e in encs[:12]:
                check.same("probs", lambda: mo.probs(e), lambda: mn.probs(e))
                po, pn = mo.probs_and_prefix(e), mn.probs_and_prefix(e)
                check.equal("probs_and_prefix", po[0], pn[0])
                check.same("probs_with_prefix", lambda: mo.probs_with_prefix(e, po[1]), lambda: mn.probs_with_prefix(e, pn[1]))
                check.same("hidden", lambda: mo.hidden(e), lambda: mn.hidden(e))
            check.same("probs_batch", lambda: mo.probs_batch(encs[:4], [None] * 4, [False, True, False, True])[0],
                       lambda: mn.probs_batch(encs[:4], [None] * 4, [False, True, False, True])[0])
        for model in models:
            model.train()
            model.zero_grad()
            torch.manual_seed(1)   # LoRA dropout is on in training: the same masks for both models
            sum(z.sum() for zs in model.forward_batch(encs[:4]) for z in zs).backward()
        check.equal("gradients", [p.grad for p in mo.trainable_parameters()], [p.grad for p in mn.trainable_parameters()])
        print(f"same on {Path(base).name}", flush=True)
    print(f"d1a.model: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
