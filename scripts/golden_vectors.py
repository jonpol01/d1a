"""D1A golden vectors: per-question probabilities of a checkpoint on a fixed record set, and a comparison of another
backend (the MLX exports) against them.

    # the reference: the fp32 torch path on CPU (a Hugging Face CPU job; ~20 GB of RAM for Gemma 4 E2B)
    python scripts/golden_vectors.py score --run JohnP1/d1a-e2b@v0.1-1epoch --device cpu --out golden.json \
        --upload JohnP1/d1a-golden
    # a candidate on this Mac against it
    python scripts/golden_vectors.py compare --golden golden.json --run exports/d1a-e2b-mlx-4bit --out report.json

The record set: every 6th record of each variant of the decision-v7 development split (~200), the five playground presets
(scripts/golden_presets.json, copied from playground/src/lib/d1a.ts) and three long records whose states (~1k tokens,
development states joined) pass Gemma 4's 512-token sliding window. Records are encoded as d1a.serve encodes
them (serving context), and each record stores its input, its token ids and the probabilities, so the file is a
self-contained test vector: an implementation must reproduce the ids exactly and the probabilities to its tolerance.
"""
import argparse, json, os, platform, subprocess, time
from pathlib import Path

import numpy as np

GOLDEN_FORMAT, GOLDEN_VERSION = "d1a-golden", 1
SUITE, STRIDE = "evals/v7/decision-v7", 6
PRESETS = Path(__file__).with_name("golden_presets.json")
LONG_RECORDS, LONG_GROUP, LONG_STATE_CHARS = 3, 40, 4000


def record_set():
    """[(id, kind, internal record, per-question meta)] in a fixed order."""
    from d1a.api import SystemOneRequest, to_record
    from d1a.data import materialize
    from d1a.suite import load_split
    dev = load_split(SUITE, "development")
    out = []
    variants = sorted({r["_meta"]["variant"] for r in dev})
    for r in [r for v in variants for r in [r for r in dev if r["_meta"]["variant"] == v][::STRIDE]]:
        rec = materialize(r)
        out.append((r["_meta"]["id"], f"{r['_meta']['variant']}", rec, [{"qid": q["qid"], "type": q["qtype"], "keys": q["keys"]} for q in rec["questions"]]))
    for p in json.loads(PRESETS.read_text(encoding="utf-8")):
        rec, meta = to_record(SystemOneRequest.model_validate({"model": "d1a-latest", "state": p["state"], "questions": p["questions"]}))
        for q in rec["questions"]: q["label"] = None
        out.append((f"preset:{p['name']}", "preset", rec, [{"qid": qid, "type": m["type"], "keys": m["keys"]} for qid, m in zip(p["questions"], meta)]))
    clean = [materialize(r) for r in dev if r["_meta"]["variant"] == "clean"]
    short = [r for r in clean if all(len(q["options"]) <= 10 for q in r["questions"])]   # questions with few options keep the rows short
    for i in range(LONG_RECORDS):
        group, chars = [], 0
        for r in short[i * LONG_GROUP:]:
            if chars >= LONG_STATE_CHARS: break
            group.append(r); chars += len(r["state"]) + 2
        qs = [dict(q) for r in group[:3] for q in r["questions"]]
        out.append((f"long:{i}", "long", {"state": "\n\n".join(r["state"] for r in group), "questions": qs},
                    [{"qid": q["qid"], "type": q["qtype"], "keys": q["keys"]} for q in qs]))
    return out


def encode(model, tok, rec):
    from d1a.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    return model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)


def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception: return None


def score(a):
    import torch
    from d1a.checkpoint import Checkpoint, LoadOptions
    torch.set_num_threads(os.cpu_count() or 1)
    ck = Checkpoint(a.run)
    tok, m = ck.load(a.device, LoadOptions(backend="torch"))   # fp32, adapter merged: the path every reported number uses
    t0, recs = time.time(), []
    for i, (rid, kind, rec, meta) in enumerate(record_set()):
        enc = encode(m, tok, rec)
        ps = m.probs(enc)
        recs.append({"id": rid, "kind": kind, "input": {"state": rec["state"], "questions": [{"instr": q["instr"], "options": q["options"]} for q in rec["questions"]]},
                     "ids": enc["ids"], "questions": [{**qm, "label": q["label"], "probs": [float(x) for x in p.tolist()]} for qm, q, p in zip(meta, rec["questions"], ps)]})
        if i % 10 == 0: print(f"{i} records, {len(enc['ids'])} tokens, {time.time() - t0:.0f} s", flush=True)
    out = {"format": GOLDEN_FORMAT, "version": GOLDEN_VERSION, "run": a.run, "resolved": ck.path, "base": ck.meta.base,
           "base_revision": ck.meta.base_revision, "temperature": m.head.temperature, "backend": m.backend, "dtype": m.dtype,
           "device": a.device, "d1a_commit": git_commit(), "torch": torch.__version__, "platform": platform.platform(),
           "record_set": {"suite": SUITE, "split": "development", "stride": STRIDE, "presets": PRESETS.name, "long": LONG_RECORDS},
           "seconds": round(time.time() - t0, 1), "records": recs}
    Path(a.out).write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(recs)} records, {sum(len(r['questions']) for r in recs)} questions in {out['seconds']} s")
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.upload, repo_type="dataset", private=True, exist_ok=True)
        name = a.run.replace("/", "--").replace("@", "--")
        api.upload_file(path_or_fileobj=a.out, path_in_repo=f"{name}/{m.backend}-{m.dtype}-{a.device}.json", repo_id=a.upload, repo_type="dataset",
                        commit_message=f"Golden vectors: {a.run}, {m.backend} {m.dtype} on {a.device}")
        print(f"uploaded to {a.upload}")


def summary(pairs, dev):
    """pairs: [(candidate probs, reference probs, label or None)]; dev: which pairs are labelled development questions."""
    d = np.array([float(np.abs(np.asarray(p) - np.asarray(r)).max()) for p, r, _ in pairs])
    flips = [(int(np.argmax(p)), int(np.argmax(r)), float(np.sort(r)[-1] - np.sort(r)[-2])) for p, r, _ in pairs]
    out = {"questions": len(pairs), "max_dp": float(d.max()), "mean_dp": float(d.mean()), "p95_dp": float(np.percentile(d, 95)),
           "argmax_flips": sum(a != b for a, b, _ in flips), "flip_margins": sorted(round(m, 4) for a, b, m in flips if a != b)}
    from d1a.metrics import ece
    lab = [(p, r, y) for (p, r, y), keep in zip(pairs, dev) if keep and y is not None]
    for name, k in (("candidate", 0), ("reference", 1)):
        conf = [float(np.max(x[k])) for x in lab]; correct = [int(np.argmax(x[k])) == x[2] for x in lab]
        out[name] = {"labelled": len(lab), "accuracy": float(np.mean(correct)), "ece": ece(conf, correct)}
    return out


def compare(a):
    from d1a.checkpoint import Checkpoint, LoadOptions
    golden = json.loads(Path(a.golden).read_text(encoding="utf-8"))
    if golden.get("format") != GOLDEN_FORMAT: raise ValueError(f"{a.golden} is not a {GOLDEN_FORMAT} file")
    ck = Checkpoint(a.run)
    tok, m = ck.load(a.device, LoadOptions(backend=a.backend))
    pairs, dev, by_kind, id_mismatch = [], [], {}, []
    for r in golden["records"]:
        rec = {"state": r["input"]["state"], "questions": [{**q, "label": None} for q in r["input"]["questions"]]}
        enc = encode(m, tok, rec)
        if enc["ids"] != r["ids"]: id_mismatch.append(r["id"])
        for p, q in zip(m.probs(enc), r["questions"]):
            pair = (p.float().numpy(), q["probs"], q["label"])
            pairs.append(pair); dev.append(r["kind"] not in ("preset", "long")); by_kind.setdefault(r["kind"] if r["kind"] in ("preset", "long") else "development", []).append(pair)
    report = {"golden": a.golden, "reference": {k: golden[k] for k in ("run", "backend", "dtype", "device", "d1a_commit")},
              "candidate": {"run": a.run, "backend": m.backend, "dtype": m.dtype}, "token_id_mismatches": id_mismatch,
              "all": summary(pairs, dev), "by_kind": {k: summary(v, [False] * len(v)) for k, v in by_kind.items()}}
    for k in report["by_kind"].values(): k.pop("candidate"); k.pop("reference")
    text = json.dumps(report, indent=2)
    if a.out: Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--run", required=True); s.add_argument("--device", default="cpu")
    s.add_argument("--out", required=True); s.add_argument("--upload", default="", help="private HF dataset repo to upload the file to")
    c = sub.add_parser("compare"); c.add_argument("--golden", required=True); c.add_argument("--run", required=True)
    c.add_argument("--device", default="mps"); c.add_argument("--backend", default="mlx"); c.add_argument("--out", default="")
    a = ap.parse_args()
    (score if a.cmd == "score" else compare)(a)


if __name__ == "__main__":
    main()
