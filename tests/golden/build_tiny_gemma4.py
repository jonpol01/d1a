"""Build the conformance fixture in tests/golden/tiny-gemma4/ (committed; rerun only to change it on purpose):

    uv run python tests/golden/build_tiny_gemma4.py

A random 6-layer Gemma 4 base (sliding and KV-shared layers, a 4-token window) with a word-level tokenizer, a checkpoint
d1a.training.train writes from it (one LoRA step), and golden.json: a fixed set of requests, their token ids and the fp32 CPU
probabilities of every question, in scripts/golden_vectors.py's d1a-golden format. tests/test_conformance.py requires
the same ids exactly and the same probabilities to 1e-5, so a change to the model code that moves any answer fails CI.
The weights are committed rather than rebuilt in the test, so a new torch release's initialisation cannot move them.
"""
import json
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tiny-gemma4"
RELATIVE = OUT.relative_to(ROOT)
WORDS = ("the customer is charged twice which team billing shipping refund angry how bad no yes order late lost box "
         "arrived wrong size damaged it was again please help today").split()
S, F = "sliding_attention", "full_attention"


def build_base(base):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    from d1a.backends.torch import GEMMA_SPECIAL
    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<bos>", "<eos>", *GEMMA_SPECIAL, *WORDS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    config = Gemma4TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
                              head_dim=16, global_head_dim=32, num_key_value_heads=1, num_global_key_value_heads=1, num_kv_shared_layers=2,
                              hidden_size_per_layer_input=8, vocab_size_per_layer_input=len(vocab), sliding_window=4,
                              layer_types=[S, S, F, S, S, F], pad_token_id=0, bos_token_id=2, eos_token_id=3)
    torch.manual_seed(0)
    Gemma4ForCausalLM(config).save_pretrained(base)
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>").save_pretrained(base)


def sentence(rng, lo, hi):
    return " ".join(rng.choices(WORDS, k=rng.randint(lo, hi)))


def requests(rng, n):
    """System One requests of every shape: all three question types, states inside and far beyond the window, 1-6
    questions, option descriptions or none."""
    out = []
    for i in range(n):
        questions = {}
        for j in range(rng.randint(1, 6)):
            kind = rng.choice(["choice", "noul", "score"])
            q = {"type": kind, "instructions": sentence(rng, 1, 6)}
            if kind == "choice": q["criteria"] = {w: rng.choice([None, sentence(rng, 1, 4)]) for w in rng.sample(WORDS, rng.randint(1, 7))}
            if kind == "score": q["criteria"] = [sentence(rng, 1, 3) for _ in range(rng.randint(1, 5))]
            if kind == "noul" and rng.random() < 0.5: q["criteria"] = {"true": sentence(rng, 1, 3), "false": sentence(rng, 1, 3)}
            questions[f"q{j}"] = q
        out.append({"state": sentence(rng, 1, rng.choice([3, 12, 60])), "questions": questions})
    return out


def training_rows(rng):
    rows = []
    for req in requests(rng, 12):
        for q in req["questions"].values():
            q["label"] = rng.choice(list(q["criteria"])) if q["type"] == "choice" else rng.random() < 0.5 if q["type"] == "noul" else rng.randrange(len(q["criteria"]))
        rows.append(req)
    return rows


def main():
    from d1a.training import train
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.backends.checkpoint import LoadOptions, load, read_meta, write_meta
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT / "base").mkdir(parents=True)
    build_base(OUT / "base")
    rng = random.Random(20261005)
    data = OUT / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in training_rows(rng)), encoding="utf-8")
    sys.argv = ["d1a.training.train", "--base", str(RELATIVE / "base"), "--data", str(data.relative_to(ROOT)), "--device", "cpu", "--batch", "2",
                "--lr", "1e-2", "--lora", "4", "--max_steps", "1", "--out", str(RELATIVE / "checkpoint")]
    train.main()
    data.unlink()
    meta = read_meta(OUT / "checkpoint"); meta.base = str(RELATIVE / "base"); meta.extra.pop("args", None); write_meta(OUT / "checkpoint", meta)
    adapter = json.loads((OUT / "checkpoint" / "adapter_config.json").read_text(encoding="utf-8"))
    adapter["base_model_name_or_path"] = str(RELATIVE / "base")
    (OUT / "checkpoint" / "adapter_config.json").write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")
    for extra in ("training_metrics.json", "training_config.json", "README.md"):
        (OUT / "checkpoint" / extra).unlink(missing_ok=True)
    tok, model = load(str(RELATIVE / "checkpoint"), "cpu", LoadOptions(dtype=torch.float32, backend="torch"))
    model.eval()
    records = []
    with torch.no_grad():
        for i, req in enumerate(requests(rng, 40)):
            rec, meta_q = to_record(SystemOneRequest.model_validate(req))
            enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            probs = model.probs(enc)
            records.append({"id": f"tiny:{i}", "kind": "tiny", "request": req,
                            "input": {"state": rec["state"], "questions": [{"instr": q["instr"], "options": q["options"]} for q in rec["questions"]]},
                            "ids": enc["ids"], "questions": [{"qid": m["id"], "type": m["type"], "keys": m["keys"], "label": None, "probs": [float(x) for x in p.tolist()]}
                                                             for m, p in zip(meta_q, probs)]})
    golden = {"format": "d1a-golden", "version": 1, "run": str(RELATIVE / "checkpoint"), "base": str(RELATIVE / "base"), "backend": "torch",
              "dtype": "float32", "device": "cpu", "d1a_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None, "torch": torch.__version__, "platform": platform.platform(), "records": records}
    (OUT / "golden.json").write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(records)} records, {sum(len(r['questions']) for r in records)} questions")


if __name__ == "__main__":
    main()
