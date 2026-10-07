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


def test_train_sources_are_the_skills_suites_and_every_d1a_train_partition():
    """The one list of what a fine-tune must keep (#167): every frozen suite with a train partition (decision-v7's is
    replayed by the trainer) and every D1A partition whose role is train, read from the manifests."""
    import json
    from d1a.eval.suites import SKILLS, train_sources
    assert train_sources() == ["evals/hard-v1", "evals/devtools-v1", "evals/documents-v1", "evals/d1a/ja-jglue:train",
                               "evals/d1a/pr-labels:train", "evals/d1a/pr-labels:train-ja", "evals/d1a/pr-labels:train-blast",
                               "evals/d1a/routing:factory-train", "evals/d1a/routing:generic-train"]
    frozen = {f"evals/{m.parent.relative_to(ROOT / 'evals')}" for m in (ROOT / "evals").glob("**/manifest.json")
              if m.relative_to(ROOT / "evals").parts[0] != "d1a"   # the checkout itself may sit in a folder named d1a (CI does)
              and "train" in (json.loads(m.read_text(encoding="utf-8")).get("partitions") or [])}
    assert frozen == set(SKILLS)


def test_a_new_stage_replays_every_training_source_and_recorded_is_v05s_plan(monkeypatch, tmp_path):
    """v0.5's stage (PLAN alone) replayed no routing and none of the blast or Japanese PRs, and the labeler's severity
    slipped (#187). A new stage appends every source PLAN leaves out, so PLAN's own draws are unchanged."""
    import pytest
    from d1a.eval.suites import train_sources
    plan = mix.with_replay(mix.PLAN, 300)
    assert plan[:len(mix.PLAN)] == mix.PLAN and [s for s, _ in plan] != [s for s, _ in mix.PLAN]
    assert sorted(s for s, _ in plan) == sorted(train_sources())
    assert dict(plan[len(mix.PLAN):]) == {"evals/d1a/pr-labels:train-ja": 300, "evals/d1a/pr-labels:train-blast": 300,
                                          "evals/d1a/routing:factory-train": 300, "evals/d1a/routing:generic-train": 300}
    monkeypatch.setattr("sys.argv", ["mix.py", "--out", str(tmp_path / "m.jsonl"), "--replay-rest", "0"])
    with pytest.raises(SystemExit):   # 0 would read as "all" in a plan; a stage that replays none of a source forgets it
        mix.main()
