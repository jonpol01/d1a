"""d1a.benchmark against the module at --ref: prediction_rows, labels, summarize, paired_flip and evaluate_records on generated
records with stand-in predictors (failures, skipped long records, concurrency, logits, kernels), every written file
compared byte for byte; then main() end to end on evals/smoke-v1 (each mode, argument errors, --help).

    uv run python scripts/equivalence/benchmark.py               # against 34113f45, the last commit before the rewrite
"""
import contextlib, io, json, math, random, shutil, sys, tempfile, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import arguments, module_at  # noqa: E402
warnings.simplefilter("ignore")
import d1a.benchmark as new
from d1a.model import ContextOverflow
from d1a.api import question_keys
args = arguments(__doc__.split("\n")[0], "34113f45", 400)
old = module_at(args.ref, "d1a/benchmark.py")
rng = random.Random(args.seed)
def run(f):
    try: return ("ok", f())
    except SystemExit as e: return ("exit", e.code)
    except Exception as e: return ("err", type(e).__name__, str(e))
def same(tag, a, b):
    ja, jb = json.dumps(a, default=repr), json.dumps(b, default=repr)
    if ja != jb: raise AssertionError((tag, ja[:800], jb[:800]))
def record(i):
    qs = {}
    for j in range(rng.randint(1, 3)):
        t = rng.choice(["choice", "noul", "score"])
        if t == "choice": crit = {f"o{k}": None for k in range(rng.randint(1, 4))}; label = rng.choice(list(crit))
        elif t == "noul": crit = None; label = rng.choice([True, False])
        else: crit = ["lo", "mid", "hi"][: rng.randint(1, 3)]; label = rng.randrange(len(crit))
        q = {"type": t, "instructions": "q", "label": label, "src": rng.choice(["t1", "t2", "unknowable_x", "unknowable_control"])}
        if crit is not None: q["criteria"] = crit
        qs[f"q{j}"] = q
    variant = rng.choice(["clean"] * 4 + ["permuted", "none_present"])
    meta = {"id": f"r{i}" + ("" if variant == "clean" else f"/{variant}"), "group_id": f"r{i}", "source": rng.choice(["s1", "s2", "unknowable", "unknowable_control"]), "variant": variant}
    if variant != "clean" and rng.random() < 0.5: meta["parent_id"] = f"r{rng.randint(0, i)}"
    if rng.random() < 0.3: meta.update(pair_id=f"p{i // 2}", sibling="a" if i % 2 == 0 else "b")
    if rng.random() < 0.2: meta["control_id"] = f"r{rng.randint(0, 9)}"
    return {"state": "s", "questions": qs, "_meta": meta}
def predict(rec, mode):
    out = {"probabilities": {}, "latency_ms": round(rng.random() * 10, 3)}
    if mode == "logits": out.update(logits={}, inference_temperature=rng.choice([1.0, 2.0]))
    for qid, q in rec["questions"].items():
        keys = question_keys(q["type"], q.get("criteria")); z = [rng.gauss(0, 2) for _ in keys]
        e = [math.exp(x - max(z)) for x in z]; s = sum(e); p = [round(x / s, 4) for x in e]
        if rng.random() < 0.02: p[0] = float("nan")
        if rng.random() < 0.02: p = [x * 0.9 for x in p]
        out["probabilities"][qid] = dict(zip(keys, p))
        if mode == "logits": out["logits"][qid] = dict(zip(keys, z))
    if rng.random() < 0.02: out["probabilities"].pop(next(iter(out["probabilities"])))
    if rng.random() < 0.05: out["kernels"] = "sdpa-flash"
    return out
class Fake:
    def __init__(self, preds, fail, concurrency):
        self.preds, self.fail, self.temperature = preds, fail, 1.5
        if concurrency > 1: self.concurrency = concurrency
    def __call__(self, rec):
        i = rec["_meta"]["id"]
        if i in self.fail: raise self.fail[i]
        return self.preds[i]
n = 0
for it in range(args.iterations):
    recs = [record(i) for i in range(rng.randint(1, 25))]
    seen = set(); recs = [r for r in recs if not (r["_meta"]["id"] in seen or seen.add(r["_meta"]["id"]))]
    mode = rng.choice(["plain", "logits"]); preds = {r["_meta"]["id"]: predict(r, mode) for r in recs}
    for r in recs:
        same("rows", run(lambda: old.prediction_rows(r, preds[r["_meta"]["id"]])), run(lambda: new.prediction_rows(r, preds[r["_meta"]["id"]]))); n += 1
        for q in r["questions"].values(): same("labels", run(lambda: old.labels(q)), run(lambda: new.labels(q))); n += 1
    allrows = []
    for r in recs:
        res = run(lambda: old.prediction_rows(r, preds[r["_meta"]["id"]]))
        if res[0] == "ok": allrows += res[1]
    T = rng.choice([1.0, 1.7]); held = tuple(rng.sample(["s1", "s2", "unknowable"], rng.randint(0, 2)))
    same("summarize", run(lambda: old.summarize(allrows, T, held)), run(lambda: new.summarize(allrows, T, held))); n += 1
    same("paired_flip", run(lambda: old.paired_flip(allrows)), run(lambda: new.paired_flip(allrows))); n += 1
    fail = {}
    for r in recs:
        if rng.random() < 0.08: fail[r["_meta"]["id"]] = ContextOverflow("state exceeds 384 tokens")
        elif rng.random() < 0.02: fail[r["_meta"]["id"]] = RuntimeError("boom")
    conc = rng.choice([1, 1, 3]); skip = rng.random() < 0.5
    outs = []
    for m in (old, new):
        d = Path(tempfile.mkdtemp()) / "out"
        res = run(lambda: m.evaluate_records(recs, Fake(preds, fail, conc), d, temperature=T, heldout_sources=held, skip_overlong=skip))
        files = {f.name: f.read_bytes() for f in sorted(d.iterdir())} if d.exists() else {}
        outs.append((res, files)); shutil.rmtree(d.parent)
    same("evaluate_records", [outs[0][0], sorted(outs[0][1])], [outs[1][0], sorted(outs[1][1])])
    assert outs[0][1] == outs[1][1], ("files differ", [k for k in outs[0][1] if outs[0][1].get(k) != outs[1][1].get(k)])
    n += 1


from d1a.suite import load_split
class Local:
    temperature = 1.3
    def __init__(self, run, device, opts, context): self.context = context
    def __call__(self, rec):
        if len(json.dumps(rec["state"])) > 2500 and self.context["max_state"] < 1000: raise ContextOverflow("state exceeds 384 tokens")
        out = {"probabilities": {}, "logits": {}, "inference_temperature": 1.3, "latency_ms": float(len(json.dumps(rec["state"])) % 17)}
        for qid, q in rec["questions"].items():
            keys = question_keys(q["type"], q.get("criteria")); r = random.Random(json.dumps(rec["state"]) + qid)
            z = [r.gauss(0, 2) for _ in keys]; e = [math.exp(x - max(z)) for x in z]; s = sum(e)
            out["probabilities"][qid] = {k: x / s for k, x in zip(keys, e)}; out["logits"][qid] = dict(zip(keys, z))
        return out
data = Path(tempfile.mkdtemp()) / "data.jsonl"
recs = load_split("evals/smoke-v1", "development")
data.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "_meta"}) + "\n" for r in recs[:30]), encoding="utf-8")
cases = [["--suite", "evals/smoke-v1"], ["--suite", "evals/smoke-v1", "--split", "calibration"], ["--data", str(data)], ["--data", str(data), "--context", "serving"],
         ["--suite", "evals/smoke-v1", "--rotations", "3"], ["--suite", "evals/smoke-v1", "--date_facts"], ["--data", str(data), "--suite", "evals/smoke-v1"], ["--rotations", "0", "--suite", "x"], []]
for case in cases:
    got = []
    for m in (old, new):
        m.LocalPredictor = Local
        out = Path(tempfile.mkdtemp()) / "o"; argv = ["benchmark", "--out", str(out), "--device", "cpu"] + (["--run", "x"] if case and "--rotations" != case[0] else []) + case
        sys.argv = argv; buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            try: m.main(); code = 0
            except SystemExit as e: code = e.code
            except Exception as e: code = f'{type(e).__name__}: {e}'
        files = {f.name: f.read_bytes() for f in sorted(out.iterdir())} if out.exists() else {}
        got.append((code, buf.getvalue().replace(str(out), "OUT"), err.getvalue(), {k: v.replace(str(out).encode(), b"OUT") for k, v in files.items()}))
        shutil.rmtree(out.parent)
    assert got[0] == got[1], (case, [x[:3] for x in got])
    n += 1
for m in (old, new):
    sys.argv = ["benchmark", "--help"]; buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: m.main()
        except SystemExit: pass
    m.help = buf.getvalue()
assert old.help == new.help; n += 1
print(f"d1a.benchmark: identical to {args.ref} on {n} checks, the command line included")
