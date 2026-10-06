"""scripts/quality_gate.py's rules, on synthetic answers: what passes, what fails, and how latency is judged against the
floor. No server and no weights; the real run is local (AGENTS.md, Quality bar)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("quality_gate", ROOT / "scripts/quality_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def choice(p):
    return {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}


def record(base, head, group="demo:routing", name="en: Quick fact"):
    return {"group": group, "name": name, "base": base, "head": head}


def test_answers_must_match():
    same = {"route": choice({"small": 0.7, "large": 0.3}), "urgent": {"type": "noul", "noul": 0.2}}
    groups, failures = gate.compare([record(same, json.loads(json.dumps(same)))], tol=1e-6)
    assert failures == [] and groups["demo:routing"] == {"requests": 1, "questions": 2, "flips": 0, "max_dp": 0.0}

    def fails(head, tol=1e-6):
        return gate.compare([record(same, {**same, **head})], tol)[1]
    assert fails({"route": choice({"small": 0.4, "large": 0.6})}) == ["demo:routing | en: Quick fact / route: small -> large"]
    assert fails({"urgent": {"type": "noul", "noul": 0.51}}) == ["demo:routing | en: Quick fact / urgent: false -> true"]
    assert fails({"urgent": {"type": "noul", "noul": 0.2 + 1e-7}}) == []                       # within --tol
    assert "moved" in fails({"urgent": {"type": "noul", "noul": 0.21}})[0]                    # same side, beyond --tol
    assert "different options" in fails({"route": choice({"small": 0.7, "medium": 0.3})})[0]
    assert gate.compare([record(same, None)], 1e-6)[1] == ["demo:routing | en: Quick fact: answered by one side only"]
    score = {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
    assert gate.pick(score) == ("1", score["probabilities"])


def test_latency_is_judged_against_the_floor():
    flat = gate.paired_ratio([(100.0, 100.0)] * 50)
    assert (flat["ratio"], flat["lo"], flat["hi"], flat["n"]) == (1.0, 1.0, 1.0, 50)
    floor = {"ratio": 1.008, "lo": 1.001, "hi": 1.012}
    assert gate.latency_failure({"ratio": 1.0, "lo": 0.998, "hi": 1.003}, floor) is None      # inside the floor
    assert gate.latency_failure({"ratio": 1.02, "lo": 1.010, "hi": 1.03}, floor) is None      # overlaps it
    assert "above the floor" in gate.latency_failure({"ratio": 1.05, "lo": 1.03, "hi": 1.07}, floor)
    slower = gate.paired_ratio([(100.0, 110.0 + i % 3) for i in range(60)])
    assert slower["lo"] > 1.09 and gate.latency_failure(slower, floor)
    assert gate.paired_ratio([(100.0, 103.0), (100.0, 97.0)] * 30, seed=1) == gate.paired_ratio([(100.0, 103.0), (100.0, 97.0)] * 30, seed=1)


def test_a_suite_must_not_get_worse():
    base = {"acc": 0.80, "ece": 0.05, "nll": 0.50}
    assert gate.suite_failures("v7/decision-v7", base, dict(base)) == []
    assert gate.suite_failures("v7/decision-v7", base, {"acc": 0.81, "ece": 0.04, "nll": 0.49}) == []
    assert gate.suite_failures("v7/decision-v7", base, {"acc": 0.79, "ece": 0.06, "nll": 0.51}) == [
        "suite v7/decision-v7: acc 0.8000 -> 0.7900", "suite v7/decision-v7: ece 0.0500 -> 0.0600", "suite v7/decision-v7: nll 0.5000 -> 0.5100"]


def test_a_suite_run_starts_from_an_empty_out(tmp_path, monkeypatch):
    """d1a.eval.benchmark refuses an existing --out: an unfinished run's folder is cleared, and the log lives beside it."""
    out = tmp_path / "suites" / "v7_decision-v7" / "base"
    out.mkdir(parents=True); (out / "rows.json").write_text("[]", encoding="utf-8")   # an unfinished earlier run

    def fake_run(cmd, cwd, stdout, stderr):
        target = Path(cmd[cmd.index("--out") + 1])
        assert not target.exists()
        target.mkdir(); (target / "report.json").write_text("{}", encoding="utf-8")
        return type("Done", (), {"returncode": 0})()
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    gate.benchmark(tmp_path, "JohnP1/d1a-e2b@v0.2", "v7/decision-v7", out)
    assert sorted(p.name for p in out.iterdir()) == ["report.json"] and out.with_name("base.log").exists()
    gate.benchmark(tmp_path, "JohnP1/d1a-e2b@v0.2", "v7/decision-v7", out)   # a finished run is kept, not rerun


def test_requests_from_files(tmp_path):
    (tmp_path / "demo.json").write_text(json.dumps([{"demo": "routing", "name": "en: Quick fact", "path": "/v1/systemone", "body": {"state": "s"}}]), encoding="utf-8")
    assert gate.demo_requests(None, tmp_path / "demo.json") == [("demo:routing", "en: Quick fact", "/v1/systemone", {"state": "s"})]
    assert gate.demo_requests(None, tmp_path / "missing.json") == []
    (tmp_path / "questions.json").write_text(json.dumps({"type": {"type": "choice", "criteria": {"a": None, "b": None}}}), encoding="utf-8")
    calls = [{"event": "d1a_call", "state": "pr 1"}, {"event": "other", "state": "x"}, {"event": "d1a_call", "state": "pr 1"}, {"event": "d1a_call", "state": "pr 2"}]
    (tmp_path / "labeler-calls.jsonl").write_text("\n".join(map(json.dumps, calls)), encoding="utf-8")
    reqs = gate.replay_requests(tmp_path)
    assert [(g, n, b["state"]) for g, n, _, b in reqs] == [("labeler-replay", "state 1", "pr 1"), ("labeler-replay", "state 2", "pr 2")]
    assert gate.replay_requests(tmp_path / "nowhere") == []


def test_both_versions_must_be_runnable_here(tmp_path):
    """The two servers share this environment: every requirement of either version must be met here, declared alike or not."""
    from importlib.metadata import version
    numpy = version("numpy")
    assert gate.unmet([f"numpy=={numpy}", "numpy>=1", "pytest", "nothing-like-this; python_version < '3'"]) == []
    assert gate.unmet(["numpy>=999"]) == [f"numpy>=999 (installed {numpy})"]
    assert gate.unmet(["no-such-package-here>=1"]) == ["no-such-package-here>=1 (not installed)"]
    toml = ('[project]\nname = "x"\ndependencies = ["numpy>=1"]\n'
            '[project.optional-dependencies]\nserve = ["pytest"]\ntrain = ["unused-extra"]\n')
    (tmp_path / "pyproject.toml").write_text(toml, encoding="utf-8")
    assert gate.requirements(tmp_path) == ["numpy>=1", "pytest"]                    # the server extras only


def test_card_suites():
    assert gate.CARD_SUITES == ("v7/decision-v7", "v4/transfer-v4", "hard-v1", "devtools-v1", "documents-v1")
    assert all((ROOT / "evals" / s / "manifest.json").exists() for s in gate.CARD_SUITES)
    assert gate.suite_list("", False) == [] and gate.suite_list("hard-v1,pr-labels:development", True) == \
        ["hard-v1", "pr-labels:development", "v7/decision-v7", "v4/transfer-v4", "devtools-v1", "documents-v1"]


def test_the_status_goes_on_the_exact_commit_tested():
    """--post-status: the head's own commit only when its checkout is clean; another repository's commit when named."""
    import pytest
    head = "a" * 40
    assert gate.status_target("", head, False, "jonpol01/d1a") == ("jonpol01/d1a", head)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        gate.status_target("", head, True, "jonpol01/d1a")
    assert gate.status_target("jonpol01/d1a-playground@" + "b" * 40, head, True, "jonpol01/d1a") == ("jonpol01/d1a-playground", "b" * 40)
    for bad in ("jonpol01/d1a-playground", "repo@" + "b" * 40, "o/r@abc"):
        with pytest.raises(SystemExit, match="owner/repo@commit"):
            gate.status_target(bad, head, False, "jonpol01/d1a")
    assert gate.pinned('X=1\nD1A_SHA="' + "c" * 40 + '"   # the pin\n') == "c" * 40 and gate.pinned("D1A_SHA=") is None


def test_the_status_text_fits_and_says_why():
    groups = {"demo:routing": {"requests": 145, "questions": 300, "flips": 0, "max_dp": 0.0}, "labeler-replay": {"requests": 92, "questions": 276, "flips": 0, "max_dp": 3e-7}}
    report = {"verdict": "PASS", "requests": 237, "groups": groups, "latency": {"all text": {"ratio": 0.9991}}, "floor": {"all text": {"ratio": 1.0029}}}
    assert gate.status_description(report, "d" * 40) == "PASS vs dddddddd: 237 requests, 0 flips, max dp 3e-07, latency 0.999 (floor 1.003)"
    failed = gate.status_description({"verdict": "FAIL", "failures": ["demo:routing | en: Quick fact / route: small -> large" * 5]}, "d" * 40)
    assert failed.startswith("FAIL: 1 failures, e.g. demo:routing") and len(failed) == 140


def test_a_new_checkpoint_reports_its_changes_and_still_fails_on_structure():
    """--head-run: the head loads other weights, so changed answers are listed for review, not failed; a request one side
    does not answer, or answers with other options, still fails, and without --head-run nothing changes."""
    same = {"route": choice({"small": 0.7, "large": 0.3})}
    moved = {"route": choice({"small": 0.4, "large": 0.6})}
    records = [record(same, moved), record(same, {"route": choice({"small": 0.69, "large": 0.31})}, name="b"), record(same, None, name="c")]
    changes = []
    groups, failures = gate.compare(records, 1e-6, changes)
    assert changes == ["demo:routing | en: Quick fact / route: small -> large", "demo:routing | b / route: a probability moved 1.00e-02 (> 1e-06)"]
    assert failures == ["demo:routing | c: answered by one side only"] and groups["demo:routing"]["flips"] == 1
    assert gate.compare(records, 1e-6)[1][:2] == changes                        # without --head-run they are failures, as before
    assert gate.suite_runs("old", None, "new") == ("old", "new")                 # a new checkpoint scores its own suites
    assert gate.suite_runs("old", "s", None) == ("s", "s") and gate.suite_runs("old") == ("old", "old")


def test_the_head_server_loads_the_new_checkpoint(monkeypatch):
    from contextlib import contextmanager
    started = []

    @contextmanager
    def fake_server(path, port, run, log):
        started.append((path, run)); yield

    monkeypatch.setattr(gate, "server", fake_server)
    monkeypatch.setattr(gate, "interleave", lambda reqs, p0, p1: [])
    gate.run_pair("base", "head", [], "old", [1, 2], "x", head_run="new")
    gate.run_pair("base", "head", [], "old", [1, 2], "y")
    assert started == [("base", "old"), ("head", "new"), ("base", "old"), ("head", "old")]


def test_each_checkout_runs_its_own_module_paths(tmp_path):
    """A version from before the package layout (#60), such as the Mac mini's pin, serves as d1a.serve; a newer one as d1a.serving.serve."""
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "d1a").mkdir(parents=True); (old / "d1a/serve.py").write_text("", encoding="utf-8"); (old / "d1a/benchmark.py").write_text("", encoding="utf-8")
    for f in ("d1a/serving/serve.py", "d1a/eval/benchmark.py", "d1a/serve.py"):   # the new layout keeps one-release shims at the old paths
        (new / f).parent.mkdir(parents=True, exist_ok=True); (new / f).write_text("", encoding="utf-8")
    assert (gate.module(old, "serve"), gate.module(old, "benchmark")) == ("d1a.serve", "d1a.benchmark")
    assert (gate.module(new, "serve"), gate.module(new, "benchmark")) == ("d1a.serving.serve", "d1a.eval.benchmark")
    assert gate.module(ROOT, "serve") == "d1a.serving.serve"
