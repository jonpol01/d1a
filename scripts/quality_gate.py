"""The quality gate for model, loading and serving changes (AGENTS.md, Quality bar): real weights, base against head.

Two d1a.serve instances, one per version, answer the same requests: every demo example (regenerated from a
d1a-playground checkout, or read from a file) and the labeler replay states (private PR text: kept under runs/, never
committed). Each request goes to both, alternating which goes first. A third run, base against base, measures the
noise floor of that setup. --suites also scores frozen suites with d1a.benchmark from each version.

FAIL on any changed choice, any probability moving more than --tol, any request one side answers and the other does
not, head's latency above the floor (its paired ratio's 95% interval entirely above the floor's), or a suite whose
accuracy drops or whose calibration (ECE, NLL) gets worse. Exit 0 PASS, 1 FAIL, 2 when the gate cannot run.

    python scripts/quality_gate.py --base origin/main                        # head: this working tree
    python scripts/quality_gate.py --base 0c601b2a --head origin/main --playground ../d1a-playground \\
        --suites v7/decision-v7,pr-labels:development --suite-run JohnP1/d1a-e2b@v0.2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDIA_GROUPS = ("demo:photo", "demo:voice", "demo:video")   # left out of the overall latency: a few slow requests each
LEARNING_ENV = ("D1A_FEEDBACK_LOG", "D1A_OUTCOME_CALIBRATOR", "D1A_OUTCOME_MEMORY")


# --- the rules (tests/test_quality_gate.py) -----------------------------------------------------------------------------

def pick(answer):
    """(the side the answer takes, its probabilities) for any answer type: a choice's option, a score's most likely level,
    a yes/no answer's side of 0.5."""
    if answer["type"] == "noul":
        return ("true" if answer["noul"] >= 0.5 else "false"), {"true": answer["noul"]}
    probs = answer["probabilities"]
    return (answer["choice"] if answer["type"] == "choice" else max(probs, key=probs.get)), probs


def compare(records, tol):
    """records: [{"group", "name", "base": answers | None, "head": answers | None}] -> per-group counts and the failures."""
    groups, failures = defaultdict(lambda: {"requests": 0, "questions": 0, "flips": 0, "max_dp": 0.0}), []
    for r in records:
        g, name = groups[r["group"]], f"{r['group']} | {r['name']}"
        g["requests"] += 1
        if r["base"] is None or r["head"] is None:
            failures.append(f"{name}: answered by {'neither side' if r['base'] is r['head'] else 'one side only'}"); continue
        if set(r["base"]) != set(r["head"]):
            failures.append(f"{name}: different questions answered"); continue
        for q, a in r["base"].items():
            (side_b, pb), (side_h, ph) = pick(a), pick(r["head"][q])
            if set(pb) != set(ph):
                failures.append(f"{name} / {q}: different options"); continue
            g["questions"] += 1
            dp = max(abs(pb[k] - ph[k]) for k in pb)
            g["max_dp"] = max(g["max_dp"], dp)
            if side_b != side_h:
                g["flips"] += 1; failures.append(f"{name} / {q}: {side_b} -> {side_h}")
            elif dp > tol:
                failures.append(f"{name} / {q}: a probability moved {dp:.2e} (> {tol:g})")
    return dict(groups), failures


def paired_ratio(pairs, seed=0, draws=2000):
    """Median of head/base per request, with a bootstrap 95% interval; pairs: [(base_ms, head_ms)]."""
    if not pairs:
        return None
    r = np.array([h / b for b, h in pairs])
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(r, len(r))) for _ in range(draws)]
    return {"ratio": float(np.median(r)), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "n": len(r)}


def latency_failure(head, floor):
    """Head is slower only when its whole interval sits above the floor's: the floor holds the setup's own bias (the
    second server runs about 1% slower) and its noise."""
    if head is None or floor is None:
        return None
    if head["lo"] > floor["hi"]:
        return f"latency: head/base {head['ratio']:.4f} [{head['lo']:.4f}-{head['hi']:.4f}] is above the floor {floor['ratio']:.4f} [{floor['lo']:.4f}-{floor['hi']:.4f}]"
    return None


def suite_failures(name, base, head):
    """base, head: {"acc", "ece", "nll"} of one suite (report.json's calibrated_clean)."""
    out = []
    for key, worse in (("acc", lambda b, h: h < b), ("ece", lambda b, h: h > b), ("nll", lambda b, h: h > b)):
        b, h = base.get(key), head.get(key)
        if b is not None and h is not None and worse(b, h) and abs(h - b) > 1e-12:
            out.append(f"suite {name}: {key} {b:.4f} -> {h:.4f}")
    return out


# --- requests -----------------------------------------------------------------------------------------------------------

DEMO_DUMP = """
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createJiti } from "jiti";
const ROOT = process.cwd();
globalThis.React = (await import("react")).default;
const jiti = createJiti(join(ROOT, "scripts/demo_smoke.mjs"), { jsx: true, alias: { "@": join(ROOT, "src") } });
const { smokeRequests } = await jiti.import(join(ROOT, "src/components/uses/smoke.ts"));
process.stdout.write(JSON.stringify(smokeRequests().map((r) => r.kind === "text"
  ? { demo: r.demo, name: r.name, path: "/v1/systemone", body: r.body }
  : { demo: r.demo, name: r.name, path: "/v1/systemone/media", body: { model: "d1a-latest", questions: r.questions,
      media: { type: r.mediaType, data: readFileSync(join(ROOT, "public/samples", r.file)).toString("base64") } } })));
"""


def demo_requests(playground, path):
    """The playground's demo examples as its smoke test sends them (the same module), or a file of them."""
    if playground:
        out = subprocess.run(["node", "--input-type=module", "-e", DEMO_DUMP], cwd=playground, capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(f"could not list the demos in {playground} (npm ci there first?):\n{out.stderr[-2000:]}")
        reqs = json.loads(out.stdout)
    elif path and Path(path).exists():
        reqs = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        return []
    return [(f"demo:{r['demo']}", r["name"], r["path"], r["body"]) for r in reqs]


def replay_requests(kit):
    """One request per distinct labeler state, with the labeler's questions (kit: labeler-calls.jsonl, questions.json)."""
    kit = Path(kit)
    if not (kit / "labeler-calls.jsonl").exists():
        return []
    questions, seen, out = json.loads((kit / "questions.json").read_text(encoding="utf-8")), set(), []
    for line in (kit / "labeler-calls.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("event") == "d1a_call" and r["state"] not in seen:
            seen.add(r["state"])
            out.append(("labeler-replay", f"state {len(seen)}", "/v1/systemone", {"state": r["state"], "model": "d1a-latest", "questions": questions}))
    return out


# --- versions and servers -----------------------------------------------------------------------------------------------

@contextmanager
def checkout(ref):
    """A directory holding `ref`: itself when it is one, else a temporary detached git worktree."""
    if Path(ref).is_dir():
        yield Path(ref).resolve(); return
    path = Path(tempfile.mkdtemp(prefix="quality-gate-")) / "src"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(path), ref], cwd=ROOT, check=True)
    try:
        yield path
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=ROOT, check=False)
        shutil.rmtree(path.parent, ignore_errors=True)


def dependencies(path):
    project = tomllib.loads((Path(path) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return project.get("dependencies"), project.get("optional-dependencies")


@contextmanager
def server(path, port, run, log):
    """d1a.serve from `path` (its own d1a first on sys.path). Load options from this shell (D1A_BACKEND, D1A_DTYPE, ...)
    apply to both sides alike; the learning settings are dropped, so no answer is logged or recalibrated."""
    env = {k: v for k, v in os.environ.items() if k not in LEARNING_ENV}
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, "-m", "d1a.serve", "--run", run, "--port", str(port)], cwd=path, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise SystemExit(f"d1a.serve from {path} exited ({proc.returncode}); see {log}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read(); break
            except (urllib.error.URLError, OSError):
                time.sleep(5)
        else:
            raise SystemExit(f"d1a.serve from {path} did not answer on :{port}; see {log}")
        yield port
    finally:
        proc.terminate()
        try: proc.wait(30)
        except subprocess.TimeoutExpired: proc.kill()


def post(port, path, body):
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            answers = json.loads(r.read())["answers"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        answers = None
    return answers, (time.perf_counter() - t) * 1000


def interleave(reqs, base_port, head_port):
    """Every request to both, base first on even items and head first on odd ones."""
    post(base_port, reqs[0][2], reqs[0][3]); post(head_port, reqs[0][2], reqs[0][3])   # the first request loads lazily
    records = []
    for i, (group, name, path, body) in enumerate(reqs):
        got = {}
        for side, port in ((("base", base_port), ("head", head_port)) if i % 2 == 0 else (("head", head_port), ("base", base_port))):
            got[side] = post(port, path, body)
        records.append({"group": group, "name": name, "base": got["base"][0], "head": got["head"][0], "base_ms": got["base"][1], "head_ms": got["head"][1]})
    return records


def run_pair(base_dir, head_dir, reqs, run, ports, logs):
    with server(base_dir, ports[0], run, f"{logs}.base.log"), server(head_dir, ports[1], run, f"{logs}.head.log"):
        return interleave(reqs, ports[0], ports[1])


def latency(records):
    """Paired ratios per group and over every text request."""
    ok = [r for r in records if r["base"] is not None and r["head"] is not None]
    groups = defaultdict(list)
    for r in ok:
        groups[r["group"]].append((r["base_ms"], r["head_ms"]))
    out = {g: paired_ratio(p) for g, p in groups.items()}
    out["all text"] = paired_ratio([(r["base_ms"], r["head_ms"]) for r in ok if r["group"] not in MEDIA_GROUPS])
    return out


# --- suites -------------------------------------------------------------------------------------------------------------

def benchmark(path, run, suite, out):
    if (Path(out) / "report.json").exists():
        return
    arg = ["--data", str(ROOT / "evals/d1a" / suite)] if ":" in suite else ["--suite", str(ROOT / "evals" / suite)]
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "log.txt", "w", encoding="utf-8") as log:
        code = subprocess.run([sys.executable, "-m", "d1a.benchmark", "--run", run, *arg, "--out", str(out)], cwd=path, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"d1a.benchmark from {path} failed on {suite}; see {out}/log.txt")


def suite_records(base_out, head_out):
    """The two runs' predictions as compare() records, and their accuracy and calibration."""
    def preds(d):
        return {(p.get("request_sha256"), p["id"]): p["prediction"]["probabilities"]
                for p in map(json.loads, (Path(d) / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    def answers(probs):
        return None if probs is None else {q: {"type": "choice", "choice": max(p, key=p.get), "probabilities": p} for q, p in probs.items()}
    b, h = preds(base_out), preds(head_out)
    records = [{"group": "suite", "name": str(k[1]), "base": answers(b.get(k)), "head": answers(h.get(k))} for k in sorted(set(b) | set(h), key=str)]
    def metrics(d):
        rep = json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))
        c = rep.get("calibrated_clean") or rep.get("clean") or {}
        return {k: c.get(k) for k in ("acc", "ece", "nll")}
    return records, metrics(base_out), metrics(head_out)


# --- the gate -----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="git ref or directory of the version to compare against")
    ap.add_argument("--head", default=str(ROOT), help="git ref or directory of the new version (default: this working tree)")
    ap.add_argument("--run", default="JohnP1/d1a-e4b-mlx-q8@v0.4", help="the weights both servers load (default: what the Mac mini serves)")
    ap.add_argument("--playground", help="a d1a-playground checkout (npm ci done): its demo examples, as its smoke test builds them")
    ap.add_argument("--demo-requests", default=str(ROOT / "runs/labeler-replay/demo-requests.json"), help="the demo requests as a file, without --playground")
    ap.add_argument("--replay", default=str(ROOT / "runs/labeler-replay"), help="the labeler replay kit (labeler-calls.jsonl, questions.json); private, never committed")
    ap.add_argument("--suites", default="", help="comma-separated suites to score too: evals/ directories, or <d1a suite>:<partition>")
    ap.add_argument("--suite-run", help="weights for --suites (default: --run)")
    ap.add_argument("--tol", type=float, default=1e-6, help="largest probability move allowed (default 1e-6)")
    ap.add_argument("--ports", default="8101,8102")
    ap.add_argument("--out", default=str(ROOT / "runs/quality-gate"))
    a = ap.parse_args()
    out, ports = Path(a.out), [int(p) for p in a.ports.split(",")]
    out.mkdir(parents=True, exist_ok=True)
    demos, replay = demo_requests(a.playground, a.demo_requests), replay_requests(a.replay)
    print(f"requests: {len(demos)} demo ({a.playground or a.demo_requests}), {len(replay)} labeler replay ({a.replay})"
          + ("" if demos and replay else "; a missing source is not checked"), flush=True)
    reqs = demos + replay
    if not reqs:
        raise SystemExit("no requests: give --playground or --demo-requests, and/or --replay")
    report, failures = {"base": a.base, "head": a.head, "run": a.run, "requests": len(reqs)}, []
    with ExitStack() as stack:
        base_dir, head_dir = stack.enter_context(checkout(a.base)), stack.enter_context(checkout(a.head))
        if dependencies(base_dir) != dependencies(head_dir):
            raise SystemExit("base and head declare different dependencies: run each in its own environment instead")
        floor = latency(run_pair(base_dir, base_dir, reqs, a.run, ports, out / "floor"))
        records = run_pair(base_dir, head_dir, reqs, a.run, ports, out / "gate")
        groups, failures = compare(records, a.tol)
        lat = latency(records)
        failures += [f for f in [latency_failure(lat["all text"], floor["all text"])] if f]
        report |= {"groups": groups, "latency": lat, "floor": floor}
        for suite in filter(None, a.suites.split(",")):
            d = out / "suites" / suite.replace("/", "_").replace(":", "_")
            for side, path in (("base", base_dir), ("head", head_dir)):
                benchmark(path, a.suite_run or a.run, suite, d / side)
            recs, mb, mh = suite_records(d / "base", d / "head")
            g, f = compare(recs, a.tol)
            failures += [f"suite {suite}: {x}" for x in f] + suite_failures(suite, mb, mh)
            report.setdefault("suites", {})[suite] = {**g.get("suite", {}), "base": mb, "head": mh}
    report["failures"] = failures
    report["verdict"] = "FAIL" if failures else "PASS"
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    fmt = lambda r: "–" if r is None else f"{r['ratio']:.4f} [{r['lo']:.4f}–{r['hi']:.4f}]"
    print("| group | requests | questions | choice flips | max \\|dp\\| | head/base latency | floor base/base |")
    print("|---|---|---|---|---|---|---|")
    for g, s in groups.items():
        print(f"| {g} | {s['requests']} | {s['questions']} | {s['flips']} | {s['max_dp']:.1e} | {fmt(lat.get(g))} | {fmt(floor.get(g))} |")
    print(f"| all text | | | | | {fmt(lat['all text'])} | {fmt(floor['all text'])} |")
    for suite, s in report.get("suites", {}).items():
        print(f"| suite {suite} | {s.get('requests', 0)} | {s.get('questions', 0)} | {s.get('flips', 0)} | {s.get('max_dp', 0.0):.1e} | acc {s['base']['acc']} → {s['head']['acc']} | ece {s['base']['ece']} → {s['head']['ece']} |")
    for f in failures[:50]:
        print(f"FAIL {f}")
    print(f"quality gate: {report['verdict']} ({len(failures)} failures; {out / 'report.json'})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
