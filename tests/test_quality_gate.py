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
