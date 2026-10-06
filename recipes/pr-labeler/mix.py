"""The PR labeler's training mix, from pinned D1A suites (d1a.eval.suites): English PRs (every rare-label PR plus a sample of
the rest), extra PR partitions, and replay of Japanese JGLUE, routing and the skills suites (hard-v1, devtools-v1,
documents-v1: a fine-tune that leaves them out loses them, #167; recipes/README.md). Writes the mix and, beside it, `<out>.json`:
every input's suite reference and sha256, the parameters, and the mix's own sha256, so a checkpoint's training data is
named exactly.

    uv run python recipes/pr-labeler/mix.py --out /data/train.jsonl --en 3500 --en-skip 1 --extra train-ja train-blast
    ... --replay-skills 0      # the mixes of v0.3 and v0.4, byte for byte (no skills replay; not for new runs)

decision-v7 replay is not here: d1a.training.train mixes it in itself (--suite evals/v7/decision-v7 --replay N).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from d1a.eval.suite import digest, load_split  # noqa: E402
from d1a.eval.suites import resolve  # noqa: E402

PR, JA, ROUTING = "evals/d1a/pr-labels", "evals/d1a/ja-jglue", "evals/d1a/routing"
RARE_SEVERITIES = ("P0", "P1", "P4")
SKILLS = ("evals/hard-v1", "evals/devtools-v1", "evals/documents-v1")   # Kev's frozen skills suites: their train partitions


def rare(line):
    """A PR whose labels are rare in train: it carries a blast question, or a P0, P1 or P4 severity."""
    q = json.loads(line)["questions"]
    return "blast" in q or q.get("sev", {}).get("label") in RARE_SEVERITIES


def lines(ref):
    with open(resolve(ref, purpose="train"), encoding="utf-8") as f:
        return [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]


def skill_lines(suite):
    """A frozen skills suite's train partition as JSONL lines (checked against its manifest's sha256, fetched if needed)."""
    return [json.dumps(r, ensure_ascii=False) + "\n" for r in load_split(Path(ROOT_DIR / suite), "train")]


def mix(en=0, en_skip=False, extra=("train-ja",), replay_ja=500, replay_routing=300, seed=0, replay_skills=500):
    """-> (mixed lines, counts). One RNG, drawn in a fixed order (the English sample, then JGLUE, routing, each skills suite
    when replay_skills > 0, the mix), so the same arguments give the same bytes, and replay_skills=0 gives the mixes made
    before it existed."""
    rng = random.Random(seed)
    english = lines(f"{PR}:train")
    if en:
        keep = [x for x in english if rare(x)]; rest = [x for x in english if not rare(x)]; rng.shuffle(rest)
        # en_skip continues a previous round with the same en and seed: the PRs it left out, plus its rare-label PRs again
        # (every P0/P1/P4 PR is rare-label, so without them those labels fade)
        english = keep + (rest[max(0, en - len(keep)):] if en_skip else rest[:max(0, en - len(keep))])
    more = [x for name in extra for x in lines(f"{PR}:{name}")]
    ja = lines(f"{JA}:train"); rng.shuffle(ja)
    routing = lines(f"{ROUTING}:factory-train") + lines(f"{ROUTING}:generic-train"); rng.shuffle(routing)
    skills = {}
    if replay_skills:
        for suite in SKILLS:
            pool = skill_lines(suite); rng.shuffle(pool); skills[suite] = pool[:replay_skills]
    out = english + more + ja[:replay_ja] + routing[:replay_routing] + [x for s in SKILLS for x in skills.get(s, [])]; rng.shuffle(out)
    return out, {"english": len(english), "extra": len(more), "ja": min(replay_ja, len(ja)), "routing": min(replay_routing, len(routing)),
                 "skills": {s: len(v) for s, v in skills.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--en", type=int, default=0, help="English PRs: every rare-label PR plus a sample of the rest up to this many; 0 = all")
    ap.add_argument("--en-skip", type=int, choices=[0, 1], default=0, help="1 = the PRs a previous round with the same --en and seed left out, plus its rare-label PRs")
    ap.add_argument("--extra", nargs="*", default=["train-ja"], help=f"more training partitions of {PR}")
    ap.add_argument("--replay-ja", type=int, default=500); ap.add_argument("--replay-routing", type=int, default=300)
    ap.add_argument("--replay-skills", type=int, default=500, help=f"train records replayed from each of {', '.join(SKILLS)} (0: none, the "
                    "v0.3/v0.4 mixes; a fine-tune without them loses them, #167)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, counts = mix(a.en, bool(a.en_skip), a.extra, a.replay_ja, a.replay_routing, a.seed, a.replay_skills)
    Path(a.out).write_text("".join(out), encoding="utf-8")
    inputs = [f"{PR}:train", *(f"{PR}:{n}" for n in a.extra), f"{JA}:train", f"{ROUTING}:factory-train", f"{ROUTING}:generic-train"]
    suites = {s: json.loads((Path(s) / "manifest.json").read_text(encoding="utf-8")) for s in (PR, JA, ROUTING)}
    record = {"inputs": {ref: suites[ref.rpartition(":")[0]]["partitions"][ref.rpartition(":")[2]]["sha256"] for ref in inputs}
              | {f"{s}:train (manifest sha256)": digest(ROOT_DIR / s / "manifest.json") for s in counts["skills"]},
              "params": {k: v for k, v in vars(a).items() if k != "out"}, "counts": counts, "records": len(out),
              "sha256": hashlib.sha256("".join(out).encode("utf-8")).hexdigest()}
    Path(f"{a.out}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"train records {len(out)} | English PRs {counts['english']} | extra {counts['extra']} | skills {counts['skills']} | sha256 {record['sha256'][:12]}")


if __name__ == "__main__":
    main()
