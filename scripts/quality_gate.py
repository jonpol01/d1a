"""The quality gate for model, loading and serving changes (AGENTS.md, Quality bar): real weights, base against head.

Two d1a.serving.serve instances, one per version, answer the same requests: every demo example (regenerated from a
d1a-playground checkout, or read from a file) and the labeler replay states (private PR text: kept under runs/, never
committed). Each request goes to both, alternating which goes first. A third run, base against base, measures the
noise floor of that setup. --suites also scores frozen suites with d1a.eval.benchmark from each version.

FAIL on any changed choice, any probability moving more than --tol, any request one side answers and the other does
not, head's latency above the floor (its paired ratio's 95% interval entirely above the floor's), or a suite whose
accuracy drops or whose calibration (ECE, NLL) gets worse. Exit 0 PASS, 1 FAIL, 2 when the gate cannot run.

    python scripts/quality_gate.py --base origin/main                        # head: this working tree
    python scripts/quality_gate.py --base 0c601b2a --head origin/main --playground ../d1a-playground \\
        --suites v7/decision-v7,pr-labels:development --suite-run JohnP1/d1a-e2b@v0.2
    python scripts/quality_gate.py --base origin/main --head <pr branch> --post-status     # sets the PR's required check
    python scripts/quality_gate.py --base origin/main --head origin/main --run JohnP1/d1a-e4b-mlx-q8@v0.4 \
        --head-run <new checkpoint> --card-suites --playground ../d1a-playground    # a new checkpoint against the served one
    python scripts/quality_gate.py --base <old pin> --head <new pin> --playground . --post-status jonpol01/d1a-playground@<pr head>

--post-status sets the required `quality-gate` commit status (CI leaves it pending on a pull request that needs the gate,
.github/workflows/quality-gate.yml) to the verdict, on the exact commit tested: the head checkout's commit, which must
have no uncommitted changes; or, for a playground pull request that moves the D1A pin, that pull request's commit, whose
mini.sh must pin the head commit.
"""
import argparse
import json
import os
import re
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
CARD_SUITES = ("v7/decision-v7", "v4/transfer-v4", "hard-v1", "devtools-v1", "documents-v1")   # Kev-4B's card suites we can score (#58, #167)
SERVER_EXTRAS = ("serve", "mlx", "media")   # what the gate's two servers import
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


def compare(records, tol, changes=None):
    """records: [{"group", "name", "base": answers | None, "head": answers | None}] -> per-group counts and the failures.
    With a `changes` list (a new checkpoint, whose answers are meant to move), choice flips and probability moves go there
    instead of the failures; a request one side does not answer, or answers with other questions or options, still fails."""
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
            out = failures if changes is None else changes
            if side_b != side_h:
                g["flips"] += 1; out.append(f"{name} / {q}: {side_b} -> {side_h}")
            elif dp > tol:
                out.append(f"{name} / {q}: a probability moved {dp:.2e} (> {tol:g})")
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


def suite_runs(run, suite_run=None, head_run=None):
    """(base weights, head weights) for --suites: a new checkpoint (head_run) is scored against the one it replaces (run);
    otherwise both code versions score the same weights (suite_run, default run)."""
    return (run, head_run) if head_run else (suite_run or run,) * 2


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


def requirements(path, extras=SERVER_EXTRAS):
    """A version's declared requirements: its dependencies and those of the extras the gate's servers use."""
    project = tomllib.loads((Path(path) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional = project.get("optional-dependencies", {})
    return list(project.get("dependencies", [])) + [r for e in extras for r in optional.get(e, [])]


def unmet(reqs):
    """The requirements this environment does not satisfy (both versions run in it, so each must be met here)."""
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement
    out = []
    for text in reqs:
        r = Requirement(text)
        if r.marker is not None and not r.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = version(r.name)
        except PackageNotFoundError:
            out.append(f"{text} (not installed)"); continue
        if r.specifier and not r.specifier.contains(installed, prereleases=True):
            out.append(f"{text} (installed {installed})")
    return out


def eval_partitions(root=ROOT):
    """Every D1A suite's evaluation partitions as <suite>:<partition>, leaving out the composites whose parts are scored
    anyway: a partition named "all", or named <a>-<b> after two others in its manifest."""
    out = []
    for manifest in sorted((Path(root) / "evals/d1a").glob("*/manifest.json")):
        parts = json.loads(manifest.read_text(encoding="utf-8"))["partitions"]
        for name, p in parts.items():
            composite = name == "all" or any(name == f"{a}-{b}" for a in parts for b in parts if a != b)
            if p["role"] == "eval" and not composite:
                out.append(f"{manifest.parent.name}:{name}")
    return out


def suite_list(suites, card, every=False):
    """--suites, plus the five card suites with --card-suites or --all-suites, plus every D1A evaluation partition with
    --all-suites (each once, in that order)."""
    return list(dict.fromkeys([*filter(None, suites.split(",")), *(CARD_SUITES if card or every else ()),
                               *(eval_partitions() if every else ())]))


def module(path, name):
    """A module's path in the checkout at `path`: d1a's layout since #60 (d1a.serving.serve, d1a.eval.benchmark), or the
    flat one before it (d1a.serve, d1a.benchmark), so a version from either side can be gated, e.g. the Mac mini's pin."""
    new = {"serve": "d1a.serving.serve", "benchmark": "d1a.eval.benchmark"}[name]
    return new if (Path(path) / (new.replace(".", "/") + ".py")).exists() else f"d1a.{name}"


@contextmanager
def server(path, port, run, log):
    """d1a.serving.serve from `path` (its own d1a first on sys.path). Load options from this shell (D1A_BACKEND, D1A_DTYPE, ...)
    apply to both sides alike; the learning settings are dropped, so no answer is logged or recalibrated."""
    env = {k: v for k, v in os.environ.items() if k not in LEARNING_ENV}
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, "-m", module(path, "serve"), "--run", run, "--port", str(port)], cwd=path, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise SystemExit(f"d1a.serving.serve from {path} exited ({proc.returncode}); see {log}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read(); break
            except (urllib.error.URLError, OSError):
                time.sleep(5)
        else:
            raise SystemExit(f"d1a.serving.serve from {path} did not answer on :{port}; see {log}")
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


def run_pair(base_dir, head_dir, reqs, run, ports, logs, head_run=None):
    """Both servers answer every request, interleaved; the head server loads `head_run` when given (a new checkpoint)."""
    with server(base_dir, ports[0], run, f"{logs}.base.log"), server(head_dir, ports[1], head_run or run, f"{logs}.head.log"):
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
    # a D1A partition at the serving context, as the server would read it: at the training context 770 of pr-labels:test's
    # 953 PRs were skipped as too long, and the gate scored only the short ones
    arg = ["--data", str(ROOT / "evals/d1a" / suite), "--context", "serving"] if ":" in suite else ["--suite", str(ROOT / "evals" / suite)]
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)   # d1a.eval.benchmark refuses an existing --out (an unfinished run left one)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_name(out.name + ".log")
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, "-m", module(path, "benchmark"), "--run", run, *arg, "--out", str(out)], cwd=path, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"d1a.eval.benchmark from {path} failed on {suite}; see {log}")


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

# --- the required status (--post-status) ---------------------------------------------------------------------------------

STATUS_CONTEXT = "quality-gate"


def git(directory, *args):
    return subprocess.run(["git", *args], cwd=directory, capture_output=True, text=True, check=True).stdout.strip()


def commit_of(directory):
    """(the commit a checkout is at, whether its tracked files are exactly that commit)."""
    return git(directory, "rev-parse", "HEAD"), git(directory, "status", "--porcelain", "--untracked-files=no") == ""


def repo_of(directory):
    """owner/name of a checkout's origin (https or ssh remote)."""
    url = git(directory, "remote", "get-url", "origin").removesuffix(".git")
    return "/".join(url.replace(":", "/").split("/")[-2:])


def status_target(value, head_commit, head_dirty, default_repo):
    """(repo, commit) --post-status sets the status on: `value` ("owner/repo@commit"), or the head's own commit, which must
    be clean (a result for files that are not that commit describes no commit)."""
    if value:
        repo, sep, commit = value.partition("@")
        if not sep or "/" not in repo or len(commit) < 7:
            raise SystemExit(f"--post-status: give owner/repo@commit, not {value!r}")
        return repo, commit
    if head_dirty:
        raise SystemExit("--post-status: the head checkout has uncommitted changes; commit them, or gate a ref")
    return default_repo, head_commit


def pinned(text):
    """The D1A commit a d1a-playground mini.sh pins (D1A_SHA="...")."""
    m = re.search(r'^D1A_SHA="([0-9a-f]{40})"', text, re.M)
    return m[1] if m else None


def status_description(report, base_commit):
    """At most 140 characters (GitHub's limit): the verdict, the requests, flips, the largest move and the latency."""
    if report["verdict"] == "FAIL":
        text = f"FAIL: {len(report['failures'])} failures, e.g. {report['failures'][0]}"
    else:
        groups, lat, floor = report["groups"].values(), report["latency"]["all text"], report["floor"]["all text"]
        text = (f"PASS vs {base_commit[:8]}: {report['requests']} requests, {sum(g['flips'] for g in groups)} flips, "
                f"max dp {max(g['max_dp'] for g in groups):.0e}, latency {lat['ratio']:.3f} (floor {floor['ratio']:.3f})"
                + (f", {len(report['suites'])} suites" if report.get("suites") else ""))
    return text if len(text) <= 140 else text[:139] + "…"


def post_status(repo, commit, state, description):
    subprocess.run(["gh", "api", f"repos/{repo}/statuses/{commit}", "-f", f"state={state}", "-f", f"context={STATUS_CONTEXT}",
                    "-f", f"description={description}"], check=True, capture_output=True)
    print(f"status {STATUS_CONTEXT}={state} on {repo}@{commit[:8]}: {description}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="git ref or directory of the version to compare against")
    ap.add_argument("--head", default=str(ROOT), help="git ref or directory of the new version (default: this working tree)")
    ap.add_argument("--run", default="JohnP1/d1a-e4b-mlx-q8@v0.4", help="the weights both servers load (default: what the Mac mini serves)")
    ap.add_argument("--playground", help="a d1a-playground checkout (npm ci done): its demo examples, as its smoke test builds them")
    ap.add_argument("--demo-requests", default=str(ROOT / "runs/labeler-replay/demo-requests.json"), help="the demo requests as a file, without --playground")
    ap.add_argument("--replay", default=str(ROOT / "runs/labeler-replay"), help="the labeler replay kit (labeler-calls.jsonl, questions.json); private, never committed")
    ap.add_argument("--suites", default="", help="comma-separated suites to score too: evals/ directories, or <d1a suite>:<partition>")
    ap.add_argument("--card-suites", action="store_true", help=f"also score the five card suites ({', '.join(CARD_SUITES)}); required for a fine-tune (AGENTS.md)")
    ap.add_argument("--all-suites", action="store_true", help="score every suite: the card suites and every D1A evaluation partition, "
                    "with the demos and the labeler replay; the rule for any model or checkpoint change (AGENTS.md)")
    ap.add_argument("--suite-run", help="weights for --suites (default: --run)")
    ap.add_argument("--head-run", help="a new checkpoint: the head server and head suites load these weights, base keeps --run. Changed "
                    "answers are then reported, not failed; suites (no lower accuracy, no worse calibration) and latency still gate it")
    ap.add_argument("--tol", type=float, default=1e-6, help="largest probability move allowed (default 1e-6)")
    ap.add_argument("--ports", default="8101,8102")
    ap.add_argument("--out", default=str(ROOT / "runs/quality-gate"))
    ap.add_argument("--post-status", nargs="?", const="", metavar="OWNER/REPO@COMMIT",
                    help="set the required quality-gate status to the verdict: on the head's commit (no value), or on that commit")
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
    changes = None
    with ExitStack() as stack:
        base_dir, head_dir = stack.enter_context(checkout(a.base)), stack.enter_context(checkout(a.head))
        missing = {side: unmet(requirements(d)) for side, d in (("base", base_dir), ("head", head_dir))}
        if any(missing.values()):
            raise SystemExit("this environment does not meet: " + "; ".join(f"{side}: {', '.join(m)}" for side, m in missing.items() if m))
        rb, rh = set(requirements(base_dir)), set(requirements(head_dir))
        report["dependencies"] = {"base_only": sorted(rb - rh), "head_only": sorted(rh - rb)}   # met here either way
        base_commit, _ = commit_of(base_dir)
        head_commit, head_clean = commit_of(head_dir)
        report |= {"base_commit": base_commit, "head_commit": head_commit}
        target = None
        if a.post_status is not None:   # checked before the run: a status for the wrong commit must never be possible
            target = status_target(a.post_status, head_commit, not head_clean, repo_of(head_dir))
            if target[0] != repo_of(head_dir):   # a playground pull request: its mini.sh must pin exactly the head
                mini = subprocess.run(["gh", "api", f"repos/{target[0]}/contents/mini.sh?ref={target[1]}", "-H", "Accept: application/vnd.github.raw"],
                                      capture_output=True, text=True, check=True).stdout
                if pinned(mini) != head_commit:
                    raise SystemExit(f"--post-status: {target[0]}@{target[1][:8]} pins {pinned(mini)}, not the head {head_commit}")
        floor = latency(run_pair(base_dir, base_dir, reqs, a.run, ports, out / "floor"))
        records = run_pair(base_dir, head_dir, reqs, a.run, ports, out / "gate", head_run=a.head_run)
        changes = [] if a.head_run else None
        groups, failures = compare(records, a.tol, changes)
        lat = latency(records)
        failures += [f for f in [latency_failure(lat["all text"], floor["all text"])] if f]
        report |= {"groups": groups, "latency": lat, "floor": floor}
        for suite in suite_list(a.suites, a.card_suites, a.all_suites):
            d = out / "suites" / suite.replace("/", "_").replace(":", "_")
            for side, path, run in zip(("base", "head"), (base_dir, head_dir), suite_runs(a.run, a.suite_run, a.head_run)):
                benchmark(path, run, suite, d / side)
            recs, mb, mh = suite_records(d / "base", d / "head")
            g, f = compare(recs, a.tol, changes)
            failures += [f"suite {suite}: {x}" for x in f] + suite_failures(suite, mb, mh)
            report.setdefault("suites", {})[suite] = {**g.get("suite", {}), "base": mb, "head": mh}
    report["failures"] = failures
    if a.head_run:
        report |= {"head_run": a.head_run, "changes": changes}
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
    if a.head_run:   # a new checkpoint: its answers are meant to move; these are for review, not failures
        print(f"changes against {a.run} (expected for a new checkpoint, not failures): {len(changes)}")
        for c in changes[:30]:
            print(f"CHANGE {c}")
    for f in failures[:50]:
        print(f"FAIL {f}")
    print(f"quality gate: {report['verdict']} ({len(failures)} failures; {out / 'report.json'})")
    if target:
        post_status(*target, "success" if report["verdict"] == "PASS" else "failure", status_description(report, report["base_commit"]))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
