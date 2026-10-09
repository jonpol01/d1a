"""recipes/swe-verifier/zero_shot.py's --per-issue draw: the released dev and test files (eval/dev-*.jsonl, eval/test-*.jsonl)
were drawn with rng.sample per issue (as at 4903f797), and --mixed-only with shuffle, one success and one failure first
(07791f1e). Both are pinned here on synthetic rows against verbatim copies of those algorithms, so the files stay
reproducible from the code. No model, no download."""
import importlib.util
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("swe_verifier_zero_shot", ROOT / "recipes/swe-verifier/zero_shot.py")
zero_shot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zero_shot)


def draw_4903f797(rows, per_issue, rng):
    """zero_shot.py's --per-issue draw at 4903f797, verbatim."""
    by_issue = {}
    for r in rows: by_issue.setdefault(r["instance_id"], []).append(r)
    return [r for iid in sorted(by_issue) for r in rng.sample(by_issue[iid], min(per_issue, len(by_issue[iid])))]


def draw_mixed_07791f1e(rows, per_issue, rng):
    """zero_shot.py's --per-issue --mixed-only draw from 07791f1e, verbatim."""
    by_issue = {}
    for r in rows: by_issue.setdefault(r["instance_id"], []).append(r)
    by_issue = {k: v for k, v in by_issue.items() if 0 < sum(r["target"] for r in v) < len(v)}
    out = []
    for iid in sorted(by_issue):
        runs = by_issue[iid]; rng.shuffle(runs)
        keep = {id(next(x for x in runs if x["target"])), id(next(x for x in runs if not x["target"]))}
        runs.sort(key=lambda r: id(r) not in keep)
        out += runs[:per_issue]
    return out


def synthetic_rows(seed):
    """Runs of 12 issues in shuffled order: 1 to 15 runs each, some all-failed, some all-resolved, most mixed."""
    rng = random.Random(seed); rows = []
    for i in range(12):
        n = rng.randint(1, 15); p = (0.0, 1.0, 0.3, 0.6)[i % 4]
        rows += [{"instance_id": f"org__repo{i % 3}-{i}", "run": f"{i}.{j}", "target": rng.random() < p} for j in range(n)]
    rng.shuffle(rows)
    return rows


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("per_issue", [1, 4, 8])
def test_per_issue_draw_matches_4903f797(seed, per_issue):
    rows = synthetic_rows(seed)
    got = zero_shot.per_issue_draw(rows, per_issue, False, random.Random(seed))
    assert [r["run"] for r in got] == [r["run"] for r in draw_4903f797(rows, per_issue, random.Random(seed))]


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("per_issue", [2, 8])
def test_mixed_only_draw_is_unchanged(seed, per_issue):
    rows = synthetic_rows(seed)
    got = zero_shot.per_issue_draw(rows, per_issue, True, random.Random(seed))
    assert [r["run"] for r in got] == [r["run"] for r in draw_mixed_07791f1e(rows, per_issue, random.Random(seed))]
    for iid in {r["instance_id"] for r in got}:   # every kept issue keeps both outcomes
        assert {r["target"] for r in got if r["instance_id"] == iid} == {True, False}
