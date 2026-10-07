"""The skills stage's training mix (#175): every hard-v1 and devtools-v1 training record (the skills D1A-E4B never got a
full stage of, #167), and replay of documents-v1, Japanese JGLUE and PR labels, so the stage does not forget them.
decision-v7 replay is not here: d1a.training.train mixes it in itself (--suite evals/v7/decision-v7 --replay N).

    uv run python recipes/skills/mix.py --out /data/train.jsonl                  # a new stage: PLAN + 300 of every other source
    uv run python recipes/skills/mix.py --out /data/train.jsonl --smoke /data/smoke.jsonl   # and the 40 longest states
    uv run python recipes/skills/mix.py --out /data/train.jsonl --recorded       # v0.5's mix, byte for byte (14,320 records)

A new stage replays every training source PLAN leaves out (d1a.eval.suites.train_sources: routing, the other PR
partitions), so it keeps them: v0.5's stage, PLAN alone, had no routing and the labeler's severity slipped (#187).

Writes the mix and, beside it, `<out>.json`: the counts, each source's size, the frozen suites' manifest sha256 and the
mix's own sha256. --smoke also writes the records with the longest states, for a smoke run that checks memory first.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from d1a.eval.suite import digest, load_split
from d1a.eval.suites import resolve, train_sources

# (source, records; 0 = all): frozen suites (evals/<name>, their train partition) or D1A suite partitions (evals/d1a/...)
PLAN = (("evals/hard-v1", 0), ("evals/devtools-v1", 0), ("evals/documents-v1", 1000),
        ("evals/d1a/ja-jglue:train", 650), ("evals/d1a/pr-labels:train", 1350))


def with_replay(plan, n):
    """The plan, then n records of every training source it leaves out (appended, so the plan's own draws are unchanged)."""
    return tuple(plan) + tuple((s, n) for s in train_sources() if s not in dict(plan))


def records(source):
    """A source's training records: a frozen suite's train partition (checked against its manifest), or a D1A suite partition."""
    if ":" in source:
        with open(resolve(source, purpose="train"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return load_split(ROOT_DIR / source, "train")


def state_size(record):
    state = record["state"]
    return len(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))


def mix(plan=PLAN, seed=0, load=records):
    """-> (records, counts, pool sizes). One RNG, drawn in the plan's order, so the same plan and seed give the same mix."""
    rng, out, counts, pools = random.Random(seed), [], {}, {}
    for source, n in plan:
        pool = list(load(source))
        rng.shuffle(pool)
        take = pool if n == 0 else pool[:n]
        out += take; counts[source] = len(take); pools[source] = len(pool)
    rng.shuffle(out)
    return out, counts, pools


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", help="also write the records with the longest states here")
    ap.add_argument("--smoke-records", type=int, default=40)
    ap.add_argument("--replay-rest", type=int, default=300, help="records replayed from each training source PLAN leaves out")
    ap.add_argument("--recorded", action="store_true", help="PLAN alone: the mix v0.5 was trained on (not for new runs)")
    a = ap.parse_args()
    if not a.recorded and a.replay_rest < 1:
        ap.error("--replay-rest must be at least 1: a stage that replays none of a source forgets it (#167); --recorded rebuilds v0.5's mix")
    plan = PLAN if a.recorded else with_replay(PLAN, a.replay_rest)
    out, counts, pools = mix(plan, seed=a.seed)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    Path(a.out).write_text(text, encoding="utf-8")
    manifests = {s: digest(ROOT_DIR / s / "manifest.json") for s, _ in plan if ":" not in s}
    meta = {"plan": plan, "counts": counts, "pool": pools, "total": len(out), "seed": a.seed, "manifests": manifests,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    Path(a.out + ".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    if a.smoke:
        longest = sorted(out, key=state_size)[-a.smoke_records:]
        Path(a.smoke).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in longest), encoding="utf-8")
    print(json.dumps({"counts": counts, "total": len(out)}), flush=True)


if __name__ == "__main__":
    main()
