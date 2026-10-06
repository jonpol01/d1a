"""d1a.training.train against the module at --ref, on the CPU, bit for bit: main() trains tiny models (the committed Gemma 4 in
tests/golden/tiny-gemma4 and a 2-layer random Qwen3.5 built here) under a matrix of options, once per version, and the
adapter tensors, the head, training_config.json, training_metrics.json and the log must be identical (timings aside).
A run stopped and resumed is compared too.

    uv run python scripts/equivalence/train.py                   # against c04827e3, the last commit before the rewrite
"""
import contextlib
import io
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Checker, arguments, module_at  # noqa: E402

import d1a.training.train as new  # noqa: E402

GEMMA = ROOT / "tests/golden/tiny-gemma4/base"
TIMINGS = ("wall_seconds", "step_seconds", "optimizer_seconds", "resume_seconds", "resume_write_seconds", "backbone_save_seconds", "peak_rss_bytes", "peak_device_bytes")
WORDS = "the customer is charged twice which team billing shipping refund angry how bad no yes order late lost box arrived wrong size".split()


def data_file(root, rng, n=16):
    rows = []
    for i in range(n):
        criteria = {w: rng.choice([None, " ".join(rng.choices(WORDS, k=3))]) for w in rng.sample(["billing", "shipping", "refund", "order", "box"], rng.randint(3, 5))}
        rows.append({"state": " ".join(rng.choices(WORDS, k=rng.randint(3, 40))), "questions": {
            "team": {"type": "choice", "instructions": "which team", "criteria": criteria, "label": rng.choice(list(criteria))},
            "angry": {"type": "noul", "instructions": "is the customer angry", "label": rng.random() < 0.5},
            "how": {"type": "score", "instructions": "how bad", "criteria": ["no", "bad", "twice bad"], "label": rng.randrange(3)}}})
    path = root / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def qwen_base(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from d1a.backends.torch import SPECIAL
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *WORDS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"], pad_token_id=1)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(root / "qwen")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(root / "qwen")
    return root / "qwen"


def run(module, argv):
    """main() with argv -> (outcome, log with timings and paths masked)."""
    out = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["d1a.training.train", *argv]
    try:
        with contextlib.redirect_stdout(out):
            module.main()
        outcome = "ok"
    except SystemExit as e:
        outcome = f"exit {e.code}"
    except Exception as e:
        outcome = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = old_argv
    log = re.sub(r"\d+\.\d+s/rec", "Xs/rec", out.getvalue())
    return outcome, log


def results(out_dir):
    """What a run left, timings masked and its own path replaced."""
    out = {}
    out_dir = Path(out_dir)
    if (out_dir / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file
        out["adapter"] = {k: v.tolist() for k, v in sorted(load_file(str(out_dir / "adapter_model.safetensors")).items())}
        from d1a.backends.checkpoint import read_meta
        meta = read_meta(out_dir).to_dict()
        out["head"] = {k: v.tolist() for k, v in meta.pop("head").items()}
        out["meta"] = json.loads(json.dumps(meta, default=str).replace(str(out_dir), "OUT"))
    for name in ("training_config.json", "training_metrics.json"):
        if (out_dir / name).exists():
            value = json.loads((out_dir / name).read_text(encoding="utf-8").replace(str(out_dir), "OUT"))
            for key in TIMINGS:
                value.pop(key, None)
            out[name] = value
    out["files"] = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
    return out


def main():
    a = arguments(__doc__.split("\n")[0], "c04827e3", 1)
    import os
    os.chdir(ROOT)   # the fixture checkpoint names its base by a repo-relative path
    old, check = module_at(a.ref, "d1a/training/train.py"), Checker()
    work = Path(tempfile.mkdtemp())
    data = data_file(work, random.Random(a.seed))
    qwen = qwen_base(work)
    common = ["--data", str(data), "--device", "cpu", "--lora", "4", "--seed", str(a.seed)]
    gemma = ["--base", str(GEMMA), *common]
    hybrid = ["--base", str(qwen), *common]
    matrix = {
        "plain": gemma + ["--batch", "2", "--accum", "2", "--epochs", "2"],
        "length_sort": gemma + ["--batch", "2", "--accum", "2", "--length_sort", "1"],
        "none_pairs": gemma + ["--batch", "2", "--accum", "2", "--p_none_pair", "0.5"],
        "none_pair_max_state": gemma + ["--batch", "2", "--accum", "2", "--p_none_pair", "0.6", "--none_pair_max_state", "20", "--length_sort", "1"],
        "row_budget": gemma + ["--batch", "3", "--accum", "2", "--row_budget", "70"],
        "head_lr_no_decay": gemma + ["--head_lr", "1e-3", "--weight_decay", "0", "--accum", "3"],
        "max_steps": gemma + ["--accum", "2", "--max_steps", "3", "--epochs", "3"],
        "checkpointing": gemma + ["--accum", "2", "--checkpointing", "1"],
        "init_from": ["--base", "tests/golden/tiny-gemma4/base", *common, "--accum", "2", "--init_from", "tests/golden/tiny-gemma4/checkpoint"],
        "mix": gemma + ["--accum", "2", "--train_sources", "custom", "--synthetic_repeat", "2"],
        "hybrid_rows": hybrid + ["--batch", "2", "--accum", "2"],
        "hybrid_shared_prefix_ceiling": hybrid + ["--batch", "2", "--accum", "2", "--shared_prefix", "1", "--length_sort", "1", "--pass_tokens_max", "4000",
                                                 "--p_none_pair", "0.5", "--none_pair_max_state", "30"],
        "refused_pass_tokens_on_gemma": gemma + ["--length_sort", "1", "--pass_tokens_max", "4000"],
        "refused_args": gemma + ["--accum", "0"],
    }
    for name, argv in matrix.items():
        got = []
        for module, side in ((old, "old"), (new, "new")):
            out = work / f"{name}-{side}"
            outcome, log = run(module, argv + ["--out", str(out)])
            got.append((outcome, log.replace(str(out), "OUT"), results(out)))
        check.equal(name, got[0], got[1])
        print(f"same: {name} ({got[0][0]})", flush=True)
    # a run stopped after step 2 and resumed must equal itself across versions, and the straight run
    got = []
    for module, side in ((old, "old"), (new, "new")):
        out = work / f"resume-{side}"
        argv = gemma + ["--accum", "2", "--epochs", "2", "--save_every_steps", "1", "--out", str(out)]
        first = run(module, argv + ["--stop_after", "2"])
        second = run(module, argv + ["--resume", "1"])
        got.append((first[0], first[1].replace(str(out), "OUT"), second[0], second[1].replace(str(out), "OUT"), results(out)))
    check.equal("stop and resume", got[0], got[1])
    print(f"same: stop and resume ({got[0][2]})", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"d1a.training.train: identical to {a.ref} on {check.count} runs")


if __name__ == "__main__":
    main()
