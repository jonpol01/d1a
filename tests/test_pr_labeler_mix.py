"""recipes/pr-labeler/mix.py replays the skills suites (#167): a fine-tune that leaves hard-v1, devtools-v1 and
documents-v1 out loses them. Synthetic suite lines stand in for the private datasets."""
import importlib.util
import inspect
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pr_mix", ROOT / "recipes/pr-labeler/mix.py")
mix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mix)


def fake(monkeypatch):
    def lines(ref):
        n = {"train": 40, "train-ja": 6, "factory-train": 5, "generic-train": 5}.get(ref.rpartition(":")[2], 10)
        return [f'{{"ref": "{ref}", "i": {i}, "questions": {{}}}}\n' for i in range(n)]
    monkeypatch.setattr(mix, "lines", lines)
    monkeypatch.setattr(mix, "skill_lines", lambda suite: [f'{{"skill": "{suite}", "i": {i}}}\n' for i in range(30)])


def before_skills(en=0, en_skip=False, extra=("train-ja",), replay_ja=500, replay_routing=300, seed=0):
    """The mix as it was before skills replay (the v0.3 and v0.4 mixes), to pin replay_skills=0 to it."""
    rng = random.Random(seed)
    english = mix.lines(f"{mix.PR}:train")
    if en:
        keep = [x for x in english if mix.rare(x)]; rest = [x for x in english if not mix.rare(x)]; rng.shuffle(rest)
        english = keep + (rest[max(0, en - len(keep)):] if en_skip else rest[:max(0, en - len(keep))])
    more = [x for name in extra for x in mix.lines(f"{mix.PR}:{name}")]
    ja = mix.lines(f"{mix.JA}:train"); rng.shuffle(ja)
    routing = mix.lines(f"{mix.ROUTING}:factory-train") + mix.lines(f"{mix.ROUTING}:generic-train"); rng.shuffle(routing)
    out = english + more + ja[:replay_ja] + routing[:replay_routing]; rng.shuffle(out)
    return out


def test_no_skills_replay_is_the_old_mix_byte_for_byte(monkeypatch):
    fake(monkeypatch)
    for kw in ({}, {"en": 12, "seed": 3}, {"en": 12, "en_skip": True, "replay_ja": 2, "replay_routing": 4}):
        out, counts = mix.mix(**kw, replay_skills=0)
        assert out == before_skills(**kw) and counts["skills"] == {}


def test_each_skills_suite_is_replayed(monkeypatch):
    fake(monkeypatch)
    out, counts = mix.mix(replay_ja=2, replay_routing=2, replay_skills=7, seed=1)
    assert counts["skills"] == {s: 7 for s in mix.SKILLS}
    assert {s: sum(f'"skill": "{s}"' in x for x in out) for s in mix.SKILLS} == {s: 7 for s in mix.SKILLS}
    assert out == mix.mix(replay_ja=2, replay_routing=2, replay_skills=7, seed=1)[0]          # the same arguments, the same bytes
    assert mix.mix(replay_ja=2, replay_routing=2, replay_skills=100, seed=1)[1]["skills"] == {s: 30 for s in mix.SKILLS}   # all there is
    assert mix.SKILLS == ("evals/hard-v1", "evals/devtools-v1", "evals/documents-v1")
    assert inspect.signature(mix.mix).parameters["replay_skills"].default > 0                 # replaying them is the default


def test_a_new_mix_must_cover_every_training_source(monkeypatch, tmp_path):
    """A fine-tune forgets what it neither trains on nor replays (#167, #187): the CLI refuses a new mix that leaves a
    training source out (here the blast PRs, or routing); --replay-skills 0 still rebuilds the v0.3/v0.4 mixes."""
    import pytest
    assert mix.uncovered(["train-ja", "train-blast"], 500, 300, 500) == []
    assert mix.uncovered(["train-ja"], 500, 300, 500) == ["evals/d1a/pr-labels:train-blast"]
    assert mix.uncovered(["train-ja", "train-blast"], 500, 0, 500) == ["evals/d1a/routing:factory-train", "evals/d1a/routing:generic-train"]
    monkeypatch.setattr("sys.argv", ["mix.py", "--out", str(tmp_path / "m.jsonl"), "--extra", "train-ja"])
    with pytest.raises(SystemExit):
        mix.main()
    fake(monkeypatch)   # the defaults cover everything, and the mix is written
    monkeypatch.setattr(mix, "digest", lambda path: "sha")
    monkeypatch.setattr("sys.argv", ["mix.py", "--out", str(tmp_path / "m.jsonl")])
    mix.main()
    assert (tmp_path / "m.jsonl").read_text(encoding="utf-8").count("train-blast") == 10


def test_the_hf_job_defaults_pass_the_coverage_check():
    """train_job.sh forwards its own EXTRA and replay settings to mix.py, so the CLI's new default never applies there:
    the job's defaults themselves must cover every training source, or every default launch fails (hermes-prbot, #194)."""
    import re
    job = (ROOT / "recipes/pr-labeler/train_job.sh").read_text(encoding="utf-8")
    default = lambda name: re.search(rf'{name}=\$\{{{name}:-"?([^}}"]*)"?\}}', job)[1]
    assert mix.uncovered(default("EXTRA").split(), int(default("REPLAY_JA")), int(default("REPLAY_ROUTING")), 500) == []
