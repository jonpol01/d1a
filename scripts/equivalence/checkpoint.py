"""d1a.backends.checkpoint against the module at --ref: the path and metadata helpers, head.pt's bytes, MLX export configs and
their refusals, LoadOptions.from_env, the backend decision for every device and option, every load-time refusal,
warm_start, and loading tests/golden/tiny-gemma4 under a grid of options (merge, lora_scale, temperature, dtype) with
the answers compared bit for bit.

    uv run python scripts/equivalence/checkpoint.py              # against 97706065, the last commit before the rewrite
"""
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Checker, arguments, module_at  # noqa: E402

import d1a.backends.checkpoint as new  # noqa: E402

FIXTURE = "tests/golden/tiny-gemma4"


def main():
    a = arguments(__doc__.split("\n")[0], "97706065", 200)
    os.chdir(ROOT)
    old, rng, check = module_at(a.ref, "d1a/backends/checkpoint.py"), random.Random(a.seed), Checker()
    for name in ("HUB_ID", "EXPORT_CONFIG", "EXPORT_HEAD", "EXPORT_FORMAT", "EXPORT_VERSION"):
        check.equal(name, getattr(old, name).pattern if name == "HUB_ID" else getattr(old, name), getattr(new, name).pattern if name == "HUB_ID" else getattr(new, name))
    check.equal("Meta.KNOWN", old.Meta.KNOWN, new.Meta.KNOWN)
    check.equal("LoadOptions fields", [(f.name, f.default) for f in old.LoadOptions.__dataclass_fields__.values()],
                [(f.name, f.default) for f in new.LoadOptions.__dataclass_fields__.values()])
    check.equal("COMPAT_FIELDS", old.Checkpoint.COMPAT_FIELDS, new.Checkpoint.COMPAT_FIELDS)

    # paths and metadata
    for run in ["JohnP1/d1a-e2b", "JohnP1/d1a-e2b@v0.4", "a/b@c@d", "x", "", FIXTURE, FIXTURE + "/checkpoint", "owner/na me", "o.w-n/n_a.m-e@t.a-g"]:
        check.same("is_hub_id " + run, lambda: old.is_hub_id(run), lambda: new.is_hub_id(run))
    check.same("resolve_run", lambda: old.resolve_run(FIXTURE), lambda: new.resolve_run(FIXTURE))
    for _ in range(a.iterations):
        d = {k: rng.choice([1, "x", None, [1], {"a": 1}]) for k in rng.sample(list(old.Meta.KNOWN) + ["args", "suite_sha256", "extra"], rng.randint(1, 8))}
        d["base"] = "b"
        check.same("Meta round trip", lambda: old.Meta.from_dict(d).to_dict(), lambda: new.Meta.from_dict(d).to_dict())
    tmp = Path(tempfile.mkdtemp())
    for m, sub in ((old, "o"), (new, "n")):
        (tmp / sub).mkdir(); m.write_meta(tmp / sub, m.Meta(base="b", head={"q.weight": torch.ones(2)}, lora=4, extra={"args": {"x": 1}}))
    check.equal("head.pt bytes", (tmp / "o/head.pt").read_bytes() == (tmp / "n/head.pt").read_bytes(), True)
    check.same("read_meta", lambda: old.read_meta(tmp / "o").to_dict(), lambda: new.read_meta(tmp / "o").to_dict())
    for names in ([], ["model.safetensors"], ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors", "other.safetensors", "model.bin"]):
        d = Path(tempfile.mkdtemp())
        for n in names: (d / n).write_bytes(b"x")
        check.same("weight_shards", lambda: [p.name for p in old.weight_shards(d)], lambda: [p.name for p in new.weight_shards(d)])

    # MLX export configs
    from safetensors.torch import save_file
    for cfg in ({"format": "d1a-mlx", "format_version": 1}, {"format": "d1a-mlx", "format_version": 2}, {"format": "x", "format_version": 1},
                {"format": "d1a-mlx", "format_version": "1"}, {"format": "d1a-mlx"}, None):
        d = Path(tempfile.mkdtemp())
        if cfg is not None: (d / "d1a_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        check.same("read_export", lambda: old.read_export(d), lambda: new.read_export(d))
    d = Path(tempfile.mkdtemp())
    save_file({"q.weight": torch.arange(4.0)}, str(d / "head.safetensors"))
    cfg = {"base": "b", "base_revision": None, "lora": 4, "head_dim": 8, "temperature": 1.5, "head": "head.safetensors", "source": "src/x"}
    check.same("export_meta", lambda: old.export_meta(d, cfg).to_dict(), lambda: new.export_meta(d, cfg).to_dict())

    # load options
    keys = {"D1A_DTYPE": ["bf16", "fp16", "fp32", "", "fp8"], "D1A_MERGE": ["0", "1", ""], "D1A_ATTN": ["sdpa", "eager", ""], "D1A_LORA_SCALE": ["0.5", "1", "x"],
            "D1A_TEMPERATURE": ["", "2.0", "x"], "D1A_BACKEND": ["", "torch", "mlx", "auto", "metal"], "D1A_CUDA_GRAPHS": ["0", "1", "", "2"],
            "D1A_FUSED": ["0", "1", ""], "D1A_PLE_FLASH": ["0", "1", ""]}
    for _ in range(a.iterations * 5):
        env = {k: rng.choice(v) for k, v in keys.items() if rng.random() < 0.6}
        check.same("LoadOptions.from_env", lambda: repr(old.LoadOptions.from_env(env)), lambda: repr(new.LoadOptions.from_env(env)))
    check.same("mlx_available", old.mlx_available, new.mlx_available)
    check.same("fused_available", old.fused_available, new.fused_available)

    # the fixture checkpoint
    run = FIXTURE + "/checkpoint"
    ck_old, ck_new = old.Checkpoint(run), new.Checkpoint(run)
    for method in ("adapter_config", "weights_sha256", "release_date", "hybrid_base", "mlx_base"):
        check.same(method, getattr(ck_old, method), getattr(ck_new, method))
    check.same("text_config", lambda: ck_old.text_config().to_dict(), lambda: ck_new.text_config().to_dict())
    for device in ("cpu", "mps", "cuda"):
        for backend in (None, "torch", "mlx", "auto", "metal"):
            for dtype in (None, torch.float32, torch.bfloat16):
                o, n = old.LoadOptions(backend=backend, dtype=dtype), new.LoadOptions(backend=backend, dtype=dtype)
                check.same(f"backend {device} {backend} {dtype}", lambda: ck_old.backend(device, o), lambda: ck_new.backend(device, n))
    from d1a.core.api import SystemOneRequest, to_record
    golden = json.loads(Path(FIXTURE, "golden.json").read_text(encoding="utf-8"))["records"][:12]
    for opts in ({}, {"merge": False}, {"lora_scale": 0.5}, {"temperature": 2.0}, {"dtype": torch.bfloat16}, {"lora_scale": 0.0, "merge": False}):
        def answers(m):
            tok, model = m.Checkpoint(run).load("cpu", m.LoadOptions(backend="torch", **opts))
            with torch.no_grad():
                out = [[p.tolist() for p in model.probs(model.encode(tok, to_record(SystemOneRequest.model_validate(r["request"]))[0]))] for r in golden]
            return out, model.head.temperature, model.dtype, getattr(model, "lora_scale", None)
        check.same(f"load {opts}", lambda: answers(old), lambda: answers(new))

    # refusals
    def refused(meta_change=None, export=None):
        d = Path(tempfile.mkdtemp()) / "ck"; shutil.copytree(run, d)
        if meta_change:
            meta = old.read_meta(d); [setattr(meta, k, v) for k, v in meta_change.items()]; old.write_meta(d, meta)
        if export:
            save_file({k: v for k, v in old.read_meta(d).head.items()}, str(d / "head.safetensors"))
            config = {"format": "d1a-mlx", "format_version": 1, "base": FIXTURE + "/base", "base_revision": None, "lora": 4, "head_dim": 256,
                      "temperature": 1.0, "head": "head.safetensors", "source": "src/run"}
            (d / "d1a_config.json").write_text(json.dumps(config), encoding="utf-8")
        return d
    for change in ({"option_isolation": True}, {"weights": "full"}):
        d = refused(change)
        check.same(f"refuse {change}", lambda: old.Checkpoint(d), lambda: new.Checkpoint(d))
    d = refused(export=True)
    for backend in ("torch", None, "mlx", "auto"):
        check.same(f"export backend {backend}", lambda: old.Checkpoint(d).backend("cpu", old.LoadOptions(backend=backend)), lambda: new.Checkpoint(d).backend("cpu", new.LoadOptions(backend=backend)))
    for opts in ({"merge": False}, {"lora_scale": 0.5}):
        check.same(f"_load_mlx {opts}", lambda: old.Checkpoint(d)._load_mlx(None, old.LoadOptions(**opts)), lambda: new.Checkpoint(d)._load_mlx(None, new.LoadOptions(**opts)))
    check.same("warm_start from an export", lambda: old.Checkpoint(d).warm_start(None, None), lambda: new.Checkpoint(d).warm_start(None, None))

    # warm start
    from d1a.backends.torch import DecisionModel, load_tokenizer
    tok = load_tokenizer(FIXTURE + "/base")
    for change in ({}, {"lora": 8}, {"head_dim": 64}, {"base_revision": "abc"}, {"base": "other"}):
        def warm(m):
            torch.manual_seed(0)
            model = DecisionModel(FIXTURE + "/base", tok, "cpu", lora=8 if change.get("lora") == 8 else 4, head_dim=change.get("head_dim", 256))
            ours = m.read_meta(run); [setattr(ours, k, v) for k, v in change.items()]
            provenance = m.Checkpoint(run).warm_start(model, ours)
            return provenance, {k: v.tolist() for k, v in model.head.state_dict().items()}
        check.same(f"warm_start {change}", lambda: warm(old), lambda: warm(new))
    print(f"d1a.backends.checkpoint: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
