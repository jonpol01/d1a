"""d1a.suite against the module at --ref: its constants, the path and hash helpers, the bytes its writers produce,
validate_training, and load_split on every frozen partition in git, on tampered copies, through a stand-in Hub mirror and
against a private one.

    uv run python scripts/equivalence/suite.py                   # against 34113f45, the last commit before the rewrite
"""
import contextlib, hashlib, io, random, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import arguments, module_at  # noqa: E402
import d1a.suite as new
import huggingface_hub
args = arguments(__doc__.split("\n")[0], "34113f45", 3000)
old = module_at(args.ref, "d1a/suite.py")
rng = random.Random(args.seed); n = 0
def run(f):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): r = f()
        return ("ok", repr(r), buf.getvalue())
    except Exception as e: return ("err", type(e).__name__, str(e), buf.getvalue())
def same(tag, fo, fn):
    global n; n += 1; a, b = run(fo), run(fn)
    assert a == b, (tag, str(a)[:500], str(b)[:500])
for name in ("SPLITS", "CONTEXT", "SERVING_CONTEXT", "SERVING_CONTEXT_8K", "ADMISSION_BRANCH_HEADROOM", "SUITES_DATASET", "PRIVATE_DATASET",
             "SUITES_REVISION", "GIT_LIMIT", "ADMISSION_TOKENIZER", "SYNTHETIC_SOURCES", "REMOVED_SUITES", "ENCODING"):
    assert getattr(old, name) == getattr(new, name), name
assert issubclass(new.RemovedSuite, ValueError)
paths = ["evals/external/scienthoon-v1", "/root/kev/evals/external/scienthoon-v1", "evals/v7/decision-v7", "x/y", "", ".", "evals", "a/evals/b/evals/c", Path("evals/external/scienthoon-v1/")]
for p in paths:
    for f in ("suite_key", "removed_suite", "refuse_removed"): same(f, lambda: getattr(old, f)(p), lambda: getattr(new, f)(p))
WORDS = ["Ünï", " ", "\t", "\n", " ", "a", "B", "ß", "İ", "日本", "\x85"]
for _ in range(args.iterations):
    t = "".join(rng.choices(WORDS, k=rng.randint(0, 10)))
    for f in ("normalise_text", "text_digest", "record_digest"): same(f, lambda: getattr(old, f)(t), lambda: getattr(new, f)(t))
    v = {"s": t, "n": rng.random(), "l": [t, None, True]}
    same("record_digest", lambda: old.record_digest(v), lambda: new.record_digest(v))
    d = Path(tempfile.mkdtemp())
    for m, sub in ((old, "o"), (new, "n")):
        (d / sub).mkdir(); m.write_json(d / sub / "a.json", v); m.write_json(d / sub / "b.json", v, atomic=True); m.write_jsonl(d / sub / "c.jsonl", [v, {"t": t}])
    for fname in ("a.json", "b.json", "c.jsonl"):
        assert (d / "o" / fname).read_bytes() == (d / "n" / fname).read_bytes(); n += 1
        same("digest", lambda: old.digest(d / "o" / fname), lambda: new.digest(d / "n" / fname))
    same("read_jsonl", lambda: old.read_jsonl(d / "o" / "c.jsonl"), lambda: new.read_jsonl(d / "o" / "c.jsonl"))
    same("read_json", lambda: old.read_json(d / "o" / "a.json"), lambda: new.read_json(d / "o" / "a.json"))
    bad = {"x": float("nan")}
    same("write_json nan", lambda: old.write_json(d / "o" / "z.json", bad), lambda: new.write_json(d / "n" / "z.json", bad))
big = Path(tempfile.mkdtemp()) / "big.bin"; big.write_bytes(bytes(rng.getrandbits(8) for _ in range(3 * (1 << 20) + 17)))
same("digest big", lambda: old.digest(big), lambda: new.digest(big))
# validate_training
shapes = ["atom", "held_and_or", "final_negation", "nested_and", "rand:3", "weird"]
for _ in range(args.iterations):
    recs = []
    for _ in range(rng.randint(0, 4)):
        meta = {"source": rng.choice(["compositional", "agnews", "mmlu", "x", "contrastive"])}
        if rng.random() < 0.9: meta["family"] = rng.choice(shapes)
        if rng.random() < 0.3: meta["structure"] = rng.choice(list(old.HELD_OUT_KEYS) + ["and(0,1)"])
        recs.append({"_meta": meta})
    man = {k: rng.sample(["compositional", "agnews", "x", "contrastive", "mmlu"], rng.randint(0, 3)) for k in ("trainable_sources", "eval_only_sources", "holdout_sources") if rng.random() < 0.6}
    same("validate_training", lambda: old.validate_training(recs, man), lambda: new.validate_training(recs, man))
# every frozen suite partition in git
for manifest in sorted(Path("evals").rglob("manifest.json")):
    sdir = manifest.parent
    same("read_manifest", lambda: old.read_manifest(sdir), lambda: new.read_manifest(sdir))
    for split in ("train", "calibration", "development", "test", "bogus"):
        if split != "bogus" and not (sdir / f"{split}.jsonl").exists(): continue
        for allow in (False, True):
            same(f"load_split {sdir}:{split}", lambda: old.load_split(sdir, split, allow), lambda: new.load_split(sdir, split, allow))
# tampered copies, missing partitions (fetch through a stub), private mirror refusal
root = Path(tempfile.mkdtemp()) / "evals" / "x" / "s"; root.mkdir(parents=True)
payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
def fake(repo, path, repo_type, revision):
    out = Path(tempfile.mkdtemp()) / "p.jsonl"; out.write_bytes(payload if "bad" not in repo else b'{"x": 1}\n'); return str(out)
huggingface_hub.hf_hub_download = fake
for files, mirror in (({"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}, None), ({"train.jsonl": {"sha256": "0" * 64, "records": 1}}, None),
                      ({"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 2}}, None), ({}, None),
                      ({"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}, {"dataset": "bad/x", "revision": "abc"})):
    for m in (old, new):
        (root / "train.jsonl").unlink(missing_ok=True)
        old.write_json(root / "manifest.json", {"files": files, **({"mirror": mirror} if mirror else {})})
    a = []
    for m in (old, new):
        (root / "train.jsonl").unlink(missing_ok=True); a.append(run(lambda: m.load_split(root, "train")))
    assert a[0] == a[1], a; n += 1
outside = Path(tempfile.mkdtemp()) / "s"; outside.mkdir(); old.write_json(outside / "manifest.json", {"files": {}})
same("outside evals", lambda: old.load_split(outside, "train"), lambda: new.load_split(outside, "train"))
from huggingface_hub.errors import RepositoryNotFoundError
import httpx
def denied(*a, **k): raise RepositoryNotFoundError("404", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
huggingface_hub.hf_hub_download = denied
old.write_json(root / "manifest.json", {"files": {"test.jsonl": {"sha256": "0", "records": 1}}, "mirror": {"dataset": "p/x", "revision": "r"}})
same("private", lambda: old.load_split(root, "test", True), lambda: new.load_split(root, "test", True))
lock = Path(tempfile.mkdtemp()) / "a" / "b.lock"
with new.file_lock(lock): pass
assert lock.exists(); n += 1
print(f"d1a.suite: identical to {args.ref} on {n} checks, every frozen partition in git included")
