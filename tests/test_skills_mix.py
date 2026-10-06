"""recipes/skills/mix.py (#175): the skills stage's mix takes every hard-v1 and devtools-v1 training record and a fixed
replay of the rest, the same bytes for the same seed, and its smoke file holds the longest states (the memory worst case).
Synthetic records stand in for the suites."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("skills_mix", ROOT / "recipes/skills/mix.py")
mix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mix)

SIZES = {"evals/hard-v1": 60, "evals/devtools-v1": 53, "evals/documents-v1": 52, "evals/d1a/ja-jglue:train": 60, "evals/d1a/pr-labels:train": 75}


def load(source):
    return [{"state": source + " " + "x" * i, "src": source, "i": i} for i in range(SIZES[source])]


def test_all_skills_records_and_a_capped_replay():
    plan = (("evals/hard-v1", 0), ("evals/devtools-v1", 0), ("evals/documents-v1", 10), ("evals/d1a/ja-jglue:train", 6),
            ("evals/d1a/pr-labels:train", 100))
    out, counts, pools = mix.mix(plan, seed=0, load=load)
    assert counts == {"evals/hard-v1": 60, "evals/devtools-v1": 53, "evals/documents-v1": 10, "evals/d1a/ja-jglue:train": 6,
                      "evals/d1a/pr-labels:train": 75}   # 0 = all; a cap above the pool takes the pool
    assert pools == SIZES and len(out) == sum(counts.values())
    assert sorted(r["i"] for r in out if r["src"] == "evals/hard-v1") == list(range(60))   # every skills record, once
    assert out == mix.mix(plan, seed=0, load=load)[0] and out != mix.mix(plan, seed=1, load=load)[0]


def test_the_shipped_plan_is_kev_sized():
    """Every hard-v1 and devtools-v1 record plus 4,000 replayed (1,000 of them decision-v7 through the trainer): 15,320 = 1,915 steps of 8."""
    plan = dict(mix.PLAN)
    assert plan["evals/hard-v1"] == 0 and plan["evals/devtools-v1"] == 0
    assert 6000 + 5320 + sum(n for s, n in mix.PLAN if n) + 1000 == 1915 * 8


def test_the_smoke_records_are_the_longest_states():
    out, _, _ = mix.mix(seed=0, load=load)
    longest = sorted(out, key=mix.state_size)[-5:]
    assert min(mix.state_size(r) for r in longest) >= max(mix.state_size(r) for r in out if r not in longest)
    assert mix.state_size({"state": {"a": "bb"}}) == len('{"a": "bb"}')
