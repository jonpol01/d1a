# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: Gemma 4 tests (from jonpol01/kev); package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); the tests of the removed Modal app and autoresearch left; the d1a.core.api tests rewritten as tests/test_system_one.py; the d1a.training.data tests rewritten as tests/test_data.py; the d1a.backends.checkpoint tests rewritten as tests/test_checkpoint.py; the encoding, mask and pointer-head tests rewritten as tests/test_encoding.py; the serving and pass-sizing tests rewritten as tests/test_serving.py; the training tests (epoch planning, --row_budget, non-finite gradients, gradient norms, resume points) rewritten as tests/test_train.py, the shared-prefix tests as tests/test_shared_prefix.py and the long-row LocalPredictor tests as tests/test_predictors.py, on a tiny hybrid Qwen3.5 now built in tests/conftest.py; the soft-target and date-fact test and two unused stand-ins left, their checks made by tests/test_data.py, tests/test_train.py and tests/test_system_one.py.
"""Fast tests of D1A's additions that have no file of their own, with no real weights: the server's startup checks,
latency, /metrics and idle unload, media decoding, the presets and MCP tools, the backbone families, Gemma 4's shared
layers and MLX per-layer embeddings, the training context, calibration's refusals, a non-finite loss, --extra_suites,
the library against the server, release notes and D1A's suite partitions.

    uv run --extra serve python -m pytest tests/test_unit.py -q
"""
from pathlib import Path

import pytest
import torch
from d1a.backends.torch import DecisionModel


def test_serve_self_check():
    """The startup self-check passes sound probabilities and refuses the silent failures: non-finite or unnormalised."""
    from d1a.serving.serve import self_check
    self_check(lambda rec: ([[0.7, 0.3], [0.2, 0.8]], {}))
    for bad in ([[float("nan"), 0.5], [0.2, 0.8]], [[0.7, 0.7], [0.2, 0.8]]):
        with pytest.raises(SystemExit, match="probabilities"): self_check(lambda rec, bad=bad: (bad, {}))


def test_serve_device_probe():
    """A device that cannot run a kernel is reported unusable instead of crashing the server process."""
    from d1a.serving.serve import usable
    assert usable("cpu") and not usable("no-such-device")


def test_nonfinite_loss_skips_its_batch(train_hybrid, tmp_path, monkeypatch):
    """A non-finite loss skips its micro-batch and the run finishes (counted in training_metrics.json); MAX_NONFINITE in a
    row end it. (A NaN gradient from a finite loss: tests/test_train.py::test_a_nan_gradient_skips_its_optimizer_step.)"""
    from d1a.training import train
    from d1a.eval.suite import read_json
    real, calls = train.batch_loss, []
    def flaky(bad):
        def batch_loss(*args, **kw):
            calls.append(1)
            if len(calls) in bad: raise train.NonFinite("non-finite training loss")
            return real(*args, **kw)
        return batch_loss
    monkeypatch.setattr(train, "batch_loss", flaky({2}))
    train_hybrid(tmp_path / "one", "--lora", "4")
    assert read_json(tmp_path / "one/training_metrics.json")["nonfinite_skipped"] == {"microbatches": 1}
    calls.clear(); monkeypatch.setattr(train, "batch_loss", flaky(set(range(2, 2 + train.MAX_NONFINITE))))
    with pytest.raises(train.NonFinite): train_hybrid(tmp_path / "many", "--lora", "4")


def test_library_decide_matches_the_server(train_hybrid, tmp_path):
    """d1a.D1A answers a question set exactly as d1a.serving.serve's Server does for the same checkpoint and request."""
    import d1a
    from d1a.core.api import SystemOneRequest
    from d1a.serving.serve import Server
    train_hybrid(tmp_path / "ck", "--lora", "4", "--max_steps", "2")
    m = d1a.D1A.load(str(tmp_path / "ck"), device="cpu")
    questions = {"team": {"type": "choice", "instr": "which team", "criteria": {"billing": "billing", "shipping": "shipping"}},
                 "angry": {"type": "noul", "instr": "is the customer angry"}}
    got = m.decide("the customer is charged twice", questions)
    server = Server(m.checkpoint, m.tok, m.model, "cpu")
    try: want = server.answer(SystemOneRequest(model="d1a-latest", state="the customer is charged twice", questions=questions))["answers"]
    finally: server.close()
    assert got == want and set(got) == {"team", "angry"} and got["team"]["choice"] in ("billing", "shipping")


def test_serve_latency_summary():
    """/v1/models reports the recent batches' model time by nearest rank; nothing before the first batch."""
    from d1a.serving.serve import latency_summary
    assert latency_summary([]) is None
    assert latency_summary([float(i) for i in range(1, 101)]) == {"recent": 100, "p50_ms": 50.0, "p95_ms": 95.0, "max_ms": 100.0}
    assert latency_summary([7.0]) == {"recent": 1, "p50_ms": 7.0, "p95_ms": 7.0, "max_ms": 7.0}


def test_presets_are_valid_requests_and_advice_fails_safe():
    """Every preset is a valid System One request, and advise() turns unsure answers into the safe action: route up,
    ask a person, do not close a card."""
    from d1a.core.api import SystemOneRequest
    from d1a.agents.presets import PRESETS, advise, fail_up
    for kind, qs in PRESETS.items(): SystemOneRequest(model="d1a-latest", state="x", questions=qs)
    assert fail_up({"small": 0.65, "medium": 0.3, "large": 0.05}) == "medium"          # unsure about small: one tier up
    assert fail_up({"small": 0.2, "medium": 0.2, "large": 0.6}) == "large"
    gate = lambda a, k, d: {"decision": {"choice": max({"allow": a, "ask": k, "deny": d}, key=lambda x: {"allow": a, "ask": k, "deny": d}[x]),
                                         "probabilities": {"allow": a, "ask": k, "deny": d}}}
    assert advise("gate", gate(0.7, 0.1, 0.2))["decision"] == "ask"                     # likely fine, not sure: ask
    assert advise("gate", gate(0.4, 0.05, 0.55))["decision"] == "deny"
    judge = {"next": {"choice": "complete"}, "done": {"noul": 0.6}, "human": {"noul": 0.1}, "blocked": {"noul": 0.1}}
    assert advise("judge", judge)["next"] == "consult"                                   # not sure it is done: do not close
    intake = {"consult": {"noul": 0.3}, "has_target": {"noul": 0.07}, "clear_done": {"noul": 0.9}, "worker": {"choice": "developer"},
              "tier": {"probabilities": {"small": 0.1, "medium": 0.8, "large": 0.1}}}
    assert advise("intake", intake)["consult"] is True                                   # no target named: ask first


def test_mcp_server_tools(monkeypatch):
    """d1a.agents.mcp_server exposes the presets as MCP tools; each sends its preset's questions and returns advice."""
    pytest.importorskip("mcp")
    import asyncio
    from d1a.agents import mcp_server
    from d1a.agents.presets import PRESETS
    sent = []
    def fake_ask(state, questions, use_case=None):
        sent.append((state, questions))
        return {"answers": {"decision": {"choice": "allow", "probabilities": {"allow": 0.95, "ask": 0.03, "deny": 0.02}}}, "latency_ms": 1.0}
    monkeypatch.setattr(mcp_server, "ask", fake_ask)
    monkeypatch.setattr(mcp_server, "temperatures", lambda: {"temperature": 1.0, "use_case_temperatures": {}})
    names = {t.name for t in asyncio.run(mcp_server.server().list_tools())}
    assert names == {"d1a_intake", "d1a_judge", "d1a_tier", "d1a_gate", "d1a_route", "d1a_decide"}
    out = mcp_server.d1a_gate("developer", "fix the CI", "gh run view 1 --log-failed")
    assert out["advice"] == {"decision": "allow"} and sent[0][1] is PRESETS["gate"] and "command: gh run view" in sent[0][0]


def test_extra_suites_join_the_run(train_hybrid, tmp_path, capsys):
    """--extra_suites adds a frozen suite's training partition (checked against that suite's own manifest) to the run."""
    from d1a.eval.suite import load_split
    n = len(load_split("evals/devtools-v1", "train"))
    train_hybrid(tmp_path / "out", "--lora", "4", "--max_steps", "1", "--extra_suites", "evals/devtools-v1")
    out = capsys.readouterr().out
    assert f"extra suite devtools-v1: {n} training records" in out and "training requests" in out


def test_backbone_families(monkeypatch):
    """d1a.backends.backbone picks the family from the config: Qwen3.5 (DeltaNet) runs rows with its cache and extra LoRA names,
    Gemma 4 (sliding layers) the packed form with a second mask, a plain model neither. MLX runs Gemma 4 and Qwen3.5
    only, and Qwen3.5's CUDA kernels load through Qwen35 alone (fused only on merged weights)."""
    import sys
    from types import SimpleNamespace
    from transformers import DynamicCache
    from d1a.backends.backbone import Attention, Gemma4, Qwen35, for_config
    qwen = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    gemma = SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"], sliding_window=512, model_type="gemma4_text")
    sliding = SimpleNamespace(layer_types=["sliding_attention", "full_attention"], sliding_window=1024, model_type="gemma3_text")
    plain = SimpleNamespace(layer_types=None)
    bq, bg, bs, bp = for_config(qwen), for_config(gemma), for_config(sliding), for_config(plain)
    assert (type(bq), type(bg), type(bs), type(bp)) == (Qwen35, Gemma4, Gemma4, Attention)
    assert (bq.hybrid, bg.hybrid, bp.hybrid) == (True, False, False)
    assert (bq.sliding_window, bg.sliding_window, bp.sliding_window) == (None, 512, None)
    assert (bq.mlx, bg.mlx, bs.mlx, bp.mlx) == (True, True, False, False)
    assert "in_proj_qkv" in bq.lora_extra and bg.lora_extra == () and (bq.prefix_min_tokens, bg.prefix_min_tokens) == (0, 384)
    assert isinstance(bg.new_cache(), DynamicCache)
    assert Attention.unwrap(SimpleNamespace(language_model="text")) == "text"
    fused = []
    monkeypatch.setitem(sys.modules, "d1a.backends.fused_qwen35", SimpleNamespace(fuse=fused.append))
    monkeypatch.setitem(sys.modules, "d1a.backends.cuda_graphs", SimpleNamespace(CudaGraphs=lambda lm, pad: ("graphs", lm, pad)))
    on = SimpleNamespace(fused=True, cuda_graphs=True)
    for bb, merged in ((bg, True), (bq, False), (bq, True)):
        m = SimpleNamespace(lm="lm", pad_id=0, graphs=None)
        bb.serve_cuda(m, on, merged)
        assert m.graphs == (None if bb is bg else ("graphs", "lm", 0))
    assert fused == ["lm"]


def test_backbone_mlx_cache_copy():
    """The MLX copy rule follows the family: Qwen3.5's recurrent DeltaNet states are copied by mlx-lm's merge, Gemma 4's
    plain and rotating KV caches by replicate (d1a.backends.mlx)."""
    from types import SimpleNamespace
    from d1a.backends.backbone import Attention, Gemma4, Qwen35
    assert Qwen35.mlx_cache_copy == "merge" and Attention.mlx_cache_copy == Gemma4.mlx_cache_copy == "replicate"
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
    from d1a.backends.backbone import for_mlx
    assert type(for_mlx([KVCache(), RotatingKVCache(max_size=8)])) is Attention
    assert type(for_mlx([KVCache(), ArraysCache(size=2)])) is Qwen35


def test_media_audio_is_mono_16k_and_bounded():
    # the browser and phones record 44.1/48 kHz stereo; Gemma 4's feature extractor reads 16 kHz mono only
    sf = pytest.importorskip("soundfile")
    import io, numpy as np
    from d1a.serving.media import MAX_AUDIO_S, SAMPLE_RATE, decode_audio
    def wav(seconds, sr, channels):
        buf = io.BytesIO(); sf.write(buf, np.full((int(seconds * sr), channels), 0.25, dtype="float32"), sr, format="WAV"); return buf.getvalue()
    out = decode_audio(wav(1.5, 48_000, 2))
    assert out.ndim == 1 and len(out) == int(1.5 * SAMPLE_RATE) and abs(float(out.mean()) - 0.25) < 1e-3
    with pytest.raises(ValueError):
        decode_audio(wav(MAX_AUDIO_S + 1, 8_000, 1))


def test_max_state_lifts_row_and_packed_limits_together():
    from d1a.backends.torch import MAX_BRANCH, MAX_PACKED, MAX_STATE, MAX_TRAIN_STATE, SERVE_MAX_BRANCH, SERVE_MAX_STATE, training_context
    from d1a.eval.suite import CONTEXT
    assert training_context() == {k: v for k, v in CONTEXT.items() if k != "truncate"} == {"max_state": MAX_STATE, "max_branch": MAX_BRANCH, "max_packed": MAX_PACKED}
    long = 12 * MAX_STATE
    lifted = training_context(long)
    assert lifted["max_branch"] - MAX_BRANCH == lifted["max_packed"] - MAX_PACKED == long - MAX_STATE
    assert MAX_TRAIN_STATE == SERVE_MAX_STATE == 65536                                   # 64k states train and serve
    assert training_context(MAX_TRAIN_STATE)["max_branch"] <= SERVE_MAX_BRANCH          # the served row limit still fits a training branch
    with pytest.raises(ValueError):
        training_context(MAX_TRAIN_STATE + 1)


def test_calibration_refuses_rows_from_the_checkpoints_own_training_unless_told(tmp_path, monkeypatch):
    # a temperature fitted on held-out items of the checkpoint's own training corpus is in distribution and ships
    # overconfident; the fit must refuse such rows with the reasons, and an explicit override must be recorded in head.pt
    import json
    from d1a.training import calibrate
    from d1a.backends.checkpoint import Meta, read_meta, write_meta
    from d1a.eval.suite import digest
    suite = "evals/v7/decision-v7"
    run = tmp_path / "ckpt"; run.mkdir()
    write_meta(run, Meta(base="b", extra={"suite_sha256": digest(f"{suite}/manifest.json"), "args": {"suite": suite}}))
    reads = tmp_path / "cal"; reads.mkdir()
    rows = [{"id": f"r{i}", "source": "agnews", "task": "agnews", "type": "choice", "keys": ["a", "b"], "variant": "clean", "group": i,
             "question": "q", "label": i % 2, "p": [0.73, 0.27], "logits": [1.0, 0.0], "inference_temperature": 1.0} for i in range(10)]
    (reads / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (reads / "report.json").write_text(json.dumps({"suite_sha256": digest(f"{suite}/manifest.json"), "split": "calibration"}), encoding="utf-8")
    monkeypatch.setattr(calibrate, "fit_temperature", lambda rows, **kw: 1.5)
    monkeypatch.setattr(calibrate, "cross_validated_temperature", lambda rows, **kw: {
        "temperatures": [1.5], "raw": {"ece": 0.1}, "out_of_fold": {"ece": 0.05}, "separated": True,
        "ece_ci95": {"raw": [0.0, 0.2], "out_of_fold": [0.0, 0.1], "delta": [-0.1, 0.0]}})
    with pytest.raises(SystemExit) as refused:
        calibrate.main(["--run", str(run), "--rows", str(reads / "rows.json")])
    why = str(refused.value)
    assert "is training data of the checkpoint" in why and "training source(s)" in why and "calibration partition" in why
    assert read_meta(run).temperature == 1.0                      # nothing written
    assert calibrate.main(["--run", str(run), "--rows", str(reads / "rows.json"), "--allow-in-distribution"]) == 1.5
    meta = read_meta(run)
    assert meta.temperature == 1.5 and meta.extra["temperature_fit"]["in_distribution"]["problems"]
    with pytest.raises(SystemExit):                              # a manual value needs a reason
        calibrate.main(["--run", str(run), "--temperature", "2"])
    calibrate.main(["--run", str(run), "--temperature", "2", "--reason", "copied from a pool fit"])
    assert read_meta(run).extra["temperature_fit"] == {"method": "manual", "reason": "copied from a pool fit"}


def test_video_frames_are_sampled_evenly_and_never_repeated():
    # a clip becomes at most MAX_VIDEO_FRAMES frames spread over its whole length (a long clip must not be read from its
    # start only), a short one keeps each frame once, and bytes PyAV cannot read are a 422, not a crash
    av = pytest.importorskip("av")
    import io
    import numpy as np
    from d1a.serving.media import MAX_VIDEO_FRAMES, decode_video

    def clip(n):
        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as out:
            s = out.add_stream("mpeg4", rate=8); s.width = s.height = 32; s.pix_fmt = "yuv420p"
            for i in range(n):
                for packet in s.encode(av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 5, np.uint8), format="rgb24")): out.mux(packet)
            for packet in s.encode(): out.mux(packet)
        return buf.getvalue()

    frames, meta = decode_video(clip(40))
    idx = list(meta.frames_indices)
    assert len(frames) == MAX_VIDEO_FRAMES and idx[0] == 0 and idx[-1] == 39 and idx == sorted(set(idx))
    frames, meta = decode_video(clip(5))
    assert len(frames) == 5 and list(meta.frames_indices) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        decode_video(b"not a video")


def test_media_model_loads_on_demand_and_never_unloads_while_in_use():
    # an idle media server must give its ~10 GB back, but a request in flight keeps the model it is using
    from d1a.serving.media import OnDemand
    loads = []
    od = OnDemand(lambda: loads.append(1) or object(), idle_s=60, device="cpu")
    assert od.model is None and not loads
    with od.use() as m:
        assert m is od.model and len(loads) == 1
        assert not od.reap(now=od.last + 10_000)       # busy: kept however long ago the last request ended
    assert not od.reap(now=od.last + 30)               # idle, but not long enough
    assert od.reap(now=od.last + 61) and od.model is None   # not +60: (last + 60) - last can round to 59.999...
    with od.use(): pass
    assert len(loads) == 2                             # the next request loads it again


def test_idle_server_unloads_and_models_never_loads_it(monkeypatch):
    # --idle-unload: the playground polls /v1/models, so that must answer without loading (else the model never goes idle);
    # an unloaded Server is closed and actually freed (its atexit hook held it, and the model with it)
    import gc, torch, weakref
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    from d1a.serving.media import OnDemand

    class Model:
        prefix_min_tokens, backend, dtype = 0, "torch", "float32"
        head = SimpleNamespace(temperature=1.5)
        def encode(self, tok, rec, **kw): return rec
        def probs_batch(self, encs, cached, keep): return [[torch.tensor([0.5, 0.5])] for _ in encs], [None for _ in encs]

    loaded = []
    def load():
        s = serve.Server(SimpleNamespace(release_date=lambda: "2026-01-01"), None, Model(), "cpu")
        loaded.append(weakref.ref(s)); return s
    od = OnDemand(load, idle_s=60, device="cpu")
    monkeypatch.setattr(serve.app.state, "models", od, raising=False)
    monkeypatch.setattr(serve.app.state, "card", {"run": "r"}, raising=False)
    with TestClient(serve.app) as client:
        assert client.get("/v1/models").json()["models"][0]["loaded"] is False and not loaded
        with serve.server() as s: assert s.probs({"ids": [1, 9], "seg": [0, 1]})[0] == [[0.5, 0.5]]
        assert len(loaded) == 1 and client.get("/v1/models").json()["models"][0]["batches"]["count"] == 1
    del s
    assert od.reap(now=od.last + 61)
    gc.collect()
    assert loaded[0]() is None   # closed and freed


def test_media_span_joins_the_state_and_leaves_every_branch_as_it_was():
    # with_media: the photo's tokens go after <state>; each question's branch must be the same tokens with the same readout
    # offsets, its positions shifted by the span, and the prefix cache must not key the request by its token ids (two
    # photos of one size have the same placeholder ids)
    import numpy as np
    from d1a.serving.media import with_media
    from d1a.backends.torch import encode, layout, load_tokenizer, rows_of
    from d1a.serving.serve import PrefixCache
    tok = load_tokenizer("Qwen/Qwen2.5-0.5B")
    rec = {"state": "left at the door", "questions": [{"instr": "Damaged?", "options": ["yes", "no"], "label": 0}]}
    enc = encode(tok, rec)
    n_head = len(layout(tok)[0]) + 1
    m = with_media(enc, n_head, [7, 8, 8, 9], [1, 2], np.zeros((2, 4), np.float32))
    S0, _, r0 = rows_of(enc); S1, P1, r1 = rows_of(m)
    assert S1 == S0[:n_head] + [7, 8, 8, 9] + S0[n_head:] and P1 == list(range(len(S1)))
    assert [(r["ids"], r["decide"], r["opts"]) for r in r1] == [(r["ids"], r["decide"], r["opts"]) for r in r0]
    assert [r["pos"] for r in r1] == [[p + 4 for p in r["pos"]] for r in r0]
    assert m["media"][0] == [n_head + 1, n_head + 2]
    assert PrefixCache(size=4, min_tokens=0).plan([m])[0] == [None]


@pytest.mark.parametrize("bits", [4, None])
def test_ple_on_flash_matches_the_in_memory_table(tmp_path, bits):
    # the per-layer embeddings are read from the weight file per request (#71); every looked-up value must equal the
    # in-memory (Quantized)Embedding's, repeated and out-of-order ids included
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    import numpy as np
    from d1a.backends.mlx import PLE, FlashEmbedding
    mx.set_default_device(mx.cpu)
    emb = nn.Embedding(512, 128)
    emb.weight = emb.weight.astype(mx.bfloat16)
    if bits: emb = nn.QuantizedEmbedding.from_embedding(emb, group_size=64, bits=bits)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {f"language_model.model.{PLE}.{k}": v for k, v in emb.parameters().items()})
    ids = mx.array(np.array([[5, 511, 5, 0], [300, 7, 7, 64]], dtype=np.int32))
    got = FlashEmbedding([tmp_path / "model.safetensors"], emb)(ids)
    assert got.shape == (2, 4, 128) and got.dtype == mx.bfloat16
    assert bool(mx.array_equal(got, emb(ids)))


def test_the_package_version_has_release_notes():
    # releases publish CHANGELOG.md's section for the pyproject.toml version, so a version bump without notes fails here,
    # not at release time; the section extractor must drop the file's trailing link references
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_notes", Path(__file__).parent.parent / "scripts" / "release_notes.py")
    rn = importlib.util.module_from_spec(spec); spec.loader.exec_module(rn)
    version = rn.package_version()
    assert rn.check(f"v{version}") == []
    assert rn.check("v99.0.0") and rn.check("version-2")
    body = rn.section("1.2.0", "## [Unreleased]\n\n## [1.2.0] - 2026-01-02\n\n- a\n\n## 1.0.0 - x\n\n- b\n\n[1.2.0]: https://x\n")
    assert body == "- a"
    # GitHub renders every line break of a release text, so wrapped lines are joined; blocks stay as they are
    assert rn.unwrap("para\nwraps\n\n- item\n  continues\n- next\n\n| a |\n|---|\n\n### H\ntext") == \
        "para wraps\n\n- item continues\n- next\n\n| a |\n|---|\n\n### H\ntext"


def test_d1a_suite_partitions_are_pinned_verified_and_eval_ones_never_train(tmp_path, monkeypatch):
    """d1a.eval.suites: `<suite>:<partition>` resolves to the dataset file at the manifest's revision only when its sha256 and
    record count match; an eval partition is refused for training (no held-out leakage); other arguments pass through."""
    import json
    import huggingface_hub
    from d1a.eval import suites
    data = tmp_path / "hub"; data.mkdir()
    (data / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8"); (data / "dev.jsonl").write_text('{"a": 3}\n', encoding="utf-8")
    fetched = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path, repo_type, revision: (fetched.append((repo, path, revision)), str(data / path))[1])
    suite = tmp_path / "evals/d1a/toy"; suite.mkdir(parents=True)
    parts = {"train": {"path": "train.jsonl", "role": "train", "sha256": suites.sha256(data / "train.jsonl"), "records": 2},
             "dev": {"path": "dev.jsonl", "role": "eval", "sha256": suites.sha256(data / "dev.jsonl"), "records": 1}}
    (suite / "manifest.json").write_text(json.dumps({"format": "d1a-suite", "version": 1, "dataset": "me/toy", "revision": "abc123", "partitions": parts}), encoding="utf-8")
    assert suites.resolve(f"{suite}:train", purpose="train") == str(data / "train.jsonl") and fetched == [("me/toy", "train.jsonl", "abc123")]
    assert suites.resolve(f"{suite}:dev") == str(data / "dev.jsonl")
    with pytest.raises(ValueError, match="eval partition"): suites.resolve(f"{suite}:dev", purpose="train")
    with pytest.raises(ValueError, match="no partition"): suites.resolve(f"{suite}:test")
    (data / "train.jsonl").write_text('{"a": 1}\n{"a": 9}\n', encoding="utf-8")   # the file changed under the same revision
    with pytest.raises(ValueError, match="does not match"): suites.resolve(f"{suite}:train", purpose="train")
    assert suites.resolve("mine.jsonl") == "mine.jsonl" and suites.resolve(str(tmp_path / "x:y")) == str(tmp_path / "x:y")


def test_gemma4_shared_layers_run_only_the_read_positions(tmp_path):
    """Gemma 4's KV-shared layers read keys and values from earlier layers, so the packed forward runs them over the
    positions the head reads alone (d1a.backends.backbone.Gemma4.picked_hidden): the same logits and, with a LoRA, the same
    gradients as the whole sequence, over states past the sliding window."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    from d1a.training.data import materialize
    from d1a.backends.torch import GEMMA_SPECIAL, load_tokenizer
    words = "it is charged twice which team billing shipping refund angry the customer how calm annoyed".split()
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *GEMMA_SPECIAL, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    s, f = "sliding_attention", "full_attention"
    torch.manual_seed(0)
    Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=2,
                                       head_dim=32, global_head_dim=64, num_key_value_heads=1, num_kv_shared_layers=2, hidden_size_per_layer_input=16,
                                       vocab_size_per_layer_input=len(vocab), sliding_window=8, layer_types=[s, s, f, s, s, f], pad_token_id=1)).save_pretrained(tmp_path)
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=GEMMA_SPECIAL).save_pretrained(tmp_path)
    tok = load_tokenizer(str(tmp_path))
    m = DecisionModel(str(tmp_path), tok, "cpu", lora=4); m.train()
    for mod in m.modules():   # dropout off, so both passes see the same function
        if hasattr(mod, "lora_dropout"):
            for k in mod.lora_dropout: mod.lora_dropout[k] = torch.nn.Identity()
    questions = {"team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": "billing", "src": "x"},
                 "angry": {"type": "noul", "instructions": "is the customer angry", "label": True, "src": "x"},
                 "level": {"type": "score", "instructions": "how angry", "criteria": ["calm", "annoyed", "angry"], "label": 1, "src": "x"}}
    encs = [m.encode(tok, materialize({"state": "the customer is charged twice" + " it" * n, "questions": questions})) for n in (0, 5, 17)]
    params = [p for p in m.parameters() if p.requires_grad]
    seen = []   # sequence length the last (shared) layer runs over
    m.lm.get_base_model().layers[-1].register_forward_hook(lambda mod, args, out: seen.append(args[0].shape[1]))
    def run():
        out = m.forward_batch(encs)
        return [z.detach() for r in out for z in r], torch.autograd.grad(sum(z.logsumexp(-1) - z[0] for r in out for z in r), params)
    picked, gp = run()
    read = max(len({i for d, oi in zip(e["decide_idx"], e["opt_idx"]) for i in (d, *oi)}) for e in encs)
    assert seen == [read] and read < max(len(e["ids"]) for e in encs) // 4
    m.backbone.picked_hidden = lambda *a: None   # the whole sequence through every layer
    whole, gw = run()
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(picked, whole))
    assert sum((a - b).abs().sum() for a, b in zip(gp, gw)) / sum(b.abs().sum() for b in gw) < 1e-5


def test_metrics_reports_the_server_without_loading_it():
    """GET /metrics: Prometheus text; an unloaded model gives d1a_loaded 0 and no per-model lines; a loaded one adds requests,
    batches, queue, batch latency quantiles and the prefix cache."""
    from collections import deque
    from queue import Queue
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    serve.app.state.models = SimpleNamespace(model=None, idle_s=None)
    with TestClient(serve.app) as client:
        text = client.get("/metrics").text
    assert "d1a_loaded 0" in text and "# TYPE d1a_loaded gauge" in text and "d1a_requests_total" not in text
    q = Queue(); q.put(1); q.put(2)
    loaded = SimpleNamespace(batched_requests=7, batches=3, queue=q, batch_ms=deque([10.0, 20.0, 30.0]), device="cpu",
                             prefix_cache=SimpleNamespace(hits=4, misses=1, entries={"a": 1}), model=SimpleNamespace(backend="torch"))
    serve.app.state.models = SimpleNamespace(model=loaded, idle_s=None)
    with TestClient(serve.app) as client:
        text = client.get("/metrics").text
    for line in ("d1a_loaded 1", "d1a_requests_total 7", "d1a_batches_total 3", "d1a_queue_depth 2", 'd1a_batch_latency_ms{quantile="0.5"} 20.0',
                 'd1a_batch_latency_ms{quantile="1"} 30.0', "d1a_prefix_cache_hits_total 4", "d1a_prefix_cache_misses_total 1", "d1a_prefix_cache_states 1",
                 "d1a_device_memory_bytes 0", "# TYPE d1a_requests_total counter"):
        assert line in text, line


def test_metrics_needs_the_api_key_when_one_is_set(monkeypatch):
    """With D1A_API_KEY set, /metrics is behind the same bearer check as /v1 (request counts, queue and memory are not public)."""
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    monkeypatch.setattr(serve, "API_KEY", "secret")
    serve.app.state.models = SimpleNamespace(model=None, idle_s=None)
    with TestClient(serve.app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"authorization": "Bearer wrong"}).status_code == 401
        ok = client.get("/metrics", headers={"authorization": "Bearer secret"})
    assert ok.status_code == 200 and "d1a_loaded 0" in ok.text
