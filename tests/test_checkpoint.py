"""d1a.backends.checkpoint on the committed tiny checkpoint (tests/golden/tiny-gemma4) and on head.pt files written here: the
metadata schema, the D1A_* options, which backend a load uses, what loading applies (merge, LoRA scale, temperature,
dtype), the refusals, and delta-training warm starts."""
import dataclasses
import importlib.machinery
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch

from d1a.backends.checkpoint import Checkpoint, LoadOptions, Meta, fused_available, is_hub_id, read_meta, write_meta

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/golden/tiny-gemma4"


@pytest.fixture
def at_root(monkeypatch):
    monkeypatch.chdir(ROOT)   # the fixture checkpoint names its base by a repo-relative path


def copy_of_fixture(tmp_path):
    shutil.copytree(ROOT / FIXTURE, tmp_path / "tiny")
    return tmp_path / "tiny"


# --- metadata -----------------------------------------------------------------------------------------------------------

def test_head_pt_has_one_schema(tmp_path):
    """Old files read with the same defaults everywhere, and unknown keys survive a read-modify-write."""
    meta = Meta.from_dict({"head": {"w": torch.zeros(1)}, "base": "Qwen/Qwen2.5-0.5B", "lora": 16, "args": {"lr": 1}, "suite_sha256": "abc"})
    assert (meta.head_dim, meta.option_isolation, meta.temperature, meta.holdout, meta.weights_dtype, meta.weights) == (256, False, 1.0, [], "fp32", "lora")
    assert meta.extra == {"args": {"lr": 1}, "suite_sha256": "abc"}
    meta.temperature = 2.3
    meta.extra["temperature_fit"] = {"n": 10}
    write_meta(tmp_path, meta)
    back = read_meta(tmp_path)
    assert (back.temperature, back.lora, back.extra) == (2.3, 16, {"args": {"lr": 1}, "suite_sha256": "abc", "temperature_fit": {"n": 10}})
    assert Meta.from_dict({"base": "b", "temperature": 3.0, "extra": {"temperature": 9.0}}).to_dict()["temperature"] == 3.0   # a known field wins


def test_head_pt_never_runs_code(tmp_path):
    """head.pt may come from any Hub repo: one carrying a pickled callable is refused, and the callable never runs."""
    import os
    torch.save({"base": "b", "head": {}, "hook": os.getcwd}, tmp_path / "head.pt")
    with pytest.raises(Exception, match="(?i)weights.only|unsupported|global"):
        read_meta(tmp_path)


def test_unsupported_checkpoints_are_refused(tmp_path):
    """Kev's runs D1A no longer loads, as they arrive: a head.pt saying so. D1A itself never writes a full-weight run."""
    meta = Meta(base="b", head={})
    for change, message in (({"option_isolation": True}, "option_isolation"), ({"weights": "full"}, "full-weight")):
        run = tmp_path / message
        run.mkdir()
        torch.save(dataclasses.replace(meta, **change).to_dict(), run / "head.pt")
        with pytest.raises(ValueError, match=message):
            Checkpoint(run)
    with pytest.raises(ValueError, match="LoRA runs only"):
        write_meta(tmp_path / "full-weight", dataclasses.replace(meta, weights="full"))


# --- the d1a-torch format (#64) -----------------------------------------------------------------------------------------

def converted_fixture(tmp_path):
    """The committed tiny checkpoint (head.pt only, as saved before D1A 0.4) rewritten by write_meta: both formats."""
    run = copy_of_fixture(tmp_path) / "checkpoint"
    write_meta(run, read_meta(run))
    return run


def same(a, b):
    return all(torch.equal(a.head[k], b.head[k]) for k in a.head) and a.head.keys() == b.head.keys() and \
        dataclasses.replace(a, head=None) == dataclasses.replace(b, head=None)


def test_every_reader_path(at_root, tmp_path, monkeypatch):
    import json
    legacy = read_meta(FIXTURE + "/checkpoint")                                    # head.pt alone
    both = converted_fixture(tmp_path)
    cfg = json.loads((both / "d1a_config.json").read_text(encoding="utf-8"))
    assert (cfg["format"], cfg["format_version"], cfg["weights"], cfg["head"]) == ("d1a-torch", 1, "lora", "head.safetensors")
    assert same(read_meta(both), legacy)                                           # both, agreeing
    (both / "head.pt").unlink()
    def no_pickle(*a, **k): raise AssertionError("torch.load on the d1a-torch path")
    monkeypatch.setattr(torch, "load", no_pickle)
    assert same(read_meta(both), legacy)                                           # the new files alone: nothing unpickled
    monkeypatch.undo()
    torch.save(dataclasses.replace(legacy, temperature=legacy.temperature + 1).to_dict(), both / "head.pt")
    with pytest.raises(ValueError, match=f"temperature differs between d1a_config.json and head.pt \\({legacy.temperature!r} and {legacy.temperature + 1!r}\\)"):
        read_meta(both)                                                            # both, disagreeing: never a silent pick
    cfg["format_version"] = 2
    (both / "d1a_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="d1a-torch versions 1-1"):
        read_meta(both)


def test_the_new_format_answers_as_the_old(at_root, tmp_path):
    """The golden requests' probabilities through the converted run, bit for bit those of its head.pt."""
    run = converted_fixture(tmp_path)
    (run / "head.pt").unlink()
    tok, old = Checkpoint(FIXTURE + "/checkpoint").load("cpu", LoadOptions(backend="torch"))
    _, new = Checkpoint(run).load("cpu", LoadOptions(backend="torch"))
    assert torch.equal(probs(tok, old), probs(tok, new))


def test_metadata_must_be_json(tmp_path):
    from pathlib import PurePosixPath
    meta = Meta(base="b", head={"w": torch.zeros(1)}, extra={"args": {"lr": 1e-4, "data": PurePosixPath("runs/x.jsonl")}})
    with pytest.raises(ValueError, match=r"extra\.args\.data is a PurePosixPath"):
        write_meta(tmp_path, meta)
    assert not any(tmp_path.iterdir())                                             # checked before anything is written


def test_hub_ids():
    assert is_hub_id("JohnP1/d1a-e2b") and is_hub_id("JohnP1/d1a-e2b@v0.4")
    assert not is_hub_id("x") and not is_hub_id(FIXTURE) and not is_hub_id("a/b@c@d")


# --- options ------------------------------------------------------------------------------------------------------------

def test_options_from_the_environment():
    """LoadOptions.from_env is the only reader of the D1A_* load variables; an explicit value equal to a default is kept,
    so d1a.serving.serve can tell "asked for fp32" or "turned it off" from "said nothing"."""
    assert LoadOptions.from_env({}) == LoadOptions()
    assert LoadOptions.from_env({"D1A_DTYPE": "bf16", "D1A_MERGE": "0", "D1A_ATTN": "sdpa", "D1A_TEMPERATURE": "1.0", "D1A_LORA_SCALE": "0.5"}) == \
        LoadOptions(dtype=torch.bfloat16, merge=False, attn="sdpa", lora_scale=0.5, temperature=1.0)
    assert LoadOptions.from_env({"D1A_DTYPE": "fp32"}).dtype is torch.float32
    assert [LoadOptions.from_env({"D1A_BACKEND": b} if b else {}).backend for b in ("", "torch", "mlx", "auto")] == [None, "torch", "mlx", "auto"]
    for name in ("D1A_CUDA_GRAPHS", "D1A_FUSED", "D1A_PLE_FLASH"):
        field = {"D1A_CUDA_GRAPHS": "cuda_graphs", "D1A_FUSED": "fused", "D1A_PLE_FLASH": "ple_flash"}[name]
        assert [getattr(LoadOptions.from_env(e), field) for e in ({}, {name: "0"}, {name: "1"})] == [None, False, True]
    with pytest.raises(ValueError, match="D1A_BACKEND"):
        LoadOptions.from_env({"D1A_BACKEND": "metal"})


def test_fused_kernels_need_the_pinned_flash_linear_attention(monkeypatch):
    """d1a.serving.serve's CUDA default: on only with flash-linear-attention importable at fused_qwen35.FLA_VERSION (it is not in the
    serve extra), so a plain install serves unfused instead of failing to import it."""
    monkeypatch.setitem(sys.modules, "fla", None)
    assert not fused_available()
    fla = types.ModuleType("fla"); fla.__spec__ = importlib.machinery.ModuleSpec("fla", None); fla.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "d1a.backends.fused_qwen35", types.SimpleNamespace(FLA_VERSION="9.9.9"))
    assert fused_available()
    fla.__version__ = "9.9.8"
    assert not fused_available()


def test_which_backend_a_load_uses(at_root, monkeypatch):
    import d1a.backends.checkpoint as C
    ck = Checkpoint(FIXTURE + "/checkpoint")
    assert ck.mlx_base() and not ck.hybrid_base()                                      # Gemma 4
    for available in (True, False):
        monkeypatch.setattr(C, "mlx_available", lambda: available)
        assert ck.backend("mps", LoadOptions(backend="auto")) == ("mlx" if available else "torch")
        assert ck.backend("mps", LoadOptions(backend="auto", dtype=torch.float32)) == "torch"   # fp32 asks for the exact path
        assert ck.backend("cuda", LoadOptions(backend="auto")) == ck.backend("cpu", LoadOptions(backend="auto")) == "torch"
    assert ck.backend("mps", LoadOptions()) == "torch" and ck.backend("cpu", LoadOptions(backend="mlx")) == "mlx"
    with pytest.raises(ValueError, match="unknown backend"):
        ck.backend("cpu", LoadOptions(backend="metal"))


# --- loading ------------------------------------------------------------------------------------------------------------

def probs(tok, model):
    from d1a.core.api import SystemOneRequest, to_record
    request = SystemOneRequest.model_validate({"state": "the customer is charged twice", "questions": {
        "team": {"type": "choice", "criteria": {"billing": None, "shipping": None, "refund": None}}, "angry": {"type": "noul"}}})
    with torch.no_grad():
        return torch.cat(model.probs(model.encode(tok, to_record(request)[0])))


def test_what_loading_applies(at_root):
    run = FIXTURE + "/checkpoint"
    tok, merged = Checkpoint(run).load("cpu", LoadOptions(backend="torch"))
    _, unmerged = Checkpoint(run).load("cpu", LoadOptions(backend="torch", merge=False))
    assert not merged.training and merged.head.temperature == read_meta(run).temperature
    assert (probs(tok, merged) - probs(tok, unmerged)).abs().max() < 1e-6                  # merging is exact in fp32
    _, base_only = Checkpoint(run).load("cpu", LoadOptions(backend="torch", merge=False, lora_scale=0.0))
    assert base_only.lora_scale == 0.0 and (probs(tok, base_only) - probs(tok, merged)).abs().max() > 1e-6   # scale 0: the base alone
    _, hot = Checkpoint(run).load("cpu", LoadOptions(backend="torch", temperature=2.0))
    assert hot.head.temperature == 2.0
    _, half = Checkpoint(run).load("cpu", LoadOptions(backend="torch", dtype=torch.bfloat16))
    assert half.dtype == "bfloat16" and (probs(tok, half) - probs(tok, merged)).abs().max() < 0.05


def test_mlx_refusals(at_root, tmp_path):
    import d1a.backends.checkpoint as C
    from safetensors.torch import save_file
    if not C.mlx_available():
        pytest.skip("mlx-lm is not installed")
    export = copy_of_fixture(tmp_path) / "checkpoint"
    save_file(read_meta(export).head, str(export / "head.safetensors"))
    config = ('{"format": "d1a-mlx", "format_version": 1, "base": "b", "base_revision": null, "lora": 4, "head_dim": 256, '
              '"temperature": 1.0, "head": "head.safetensors", "source": "src/run"}')
    (export / "d1a_config.json").write_text(config, encoding="utf-8")
    ck = Checkpoint(export)
    with pytest.raises(ValueError, match="always merges"):
        ck._load_mlx(None, LoadOptions(merge=False))
    with pytest.raises(ValueError, match="lora_scale needs its source checkpoint src/run"):
        ck._load_mlx(None, LoadOptions(lora_scale=0.5))


# --- warm starts --------------------------------------------------------------------------------------------------------

def test_warm_start_compares_the_architecture_first(at_root):
    from d1a.backends.torch import DecisionModel, load_tokenizer
    run = FIXTURE + "/checkpoint"
    tok, ours = load_tokenizer(FIXTURE + "/base"), read_meta(run)
    model = DecisionModel(FIXTURE + "/base", tok, "cpu", lora=4)
    provenance = Checkpoint(run).warm_start(model, ours)
    assert provenance["init_from"] == run and provenance["tensors"] > 0 and len(provenance["head_sha256"]) == 64
    assert all(torch.equal(model.head.state_dict()[k], v) for k, v in ours.head.items())
    assert Checkpoint(run).warm_start(DecisionModel(FIXTURE + "/base", tok, "cpu", lora=4), dataclasses.replace(ours, base_revision="abc"))   # None matches any
    for change in ({"lora": 8}, {"head_dim": 64}, {"base": "other"}):
        with pytest.raises(ValueError, match=f"{next(iter(change))} is"):
            Checkpoint(run).warm_start(model, dataclasses.replace(ours, **change))
    with pytest.raises(ValueError, match="adapter tensors this model does not have"):
        Checkpoint(run).warm_start(DecisionModel(FIXTURE + "/base", tok, "cpu", lora=4, lora_targets="qv"), ours)
