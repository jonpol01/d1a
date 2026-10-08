"""Checks a mix recipes/mix/build_mix.py built, independently of the build (CPU only, no model):

- the mix's sha256 is its sidecar's, and d1a.training.data.load_records reads every line;
- every line is, byte for byte, json.dumps(record, ensure_ascii=False) of the source row its index line names, with that
  row's state sha256;
- decision-v7's records pass the trainer's validate_training against the decision-v7 manifest;
- no normalised state of an eval partition or eval-only kit is in the mix, no PR id of a pr-labels eval partition or the
  hand-labelled kit, no state twice; a word 5-shingle screen of the whole mix against each family's eval partitions (any
  eval-only kit counting as a PR one) finds no Jaccard >= 0.8;
- the sidecar's records by source match the index, and d1a.training.train's source check accepts the sidecar.
Also reports label shares, and with --tokenizer the exact token lengths. Writes <mix>.verify.json; exits 1 on a failure.

    HF_HUB_OFFLINE=1 uv run python recipes/mix/verify_mix.py /data/train.jsonl --replay runs/labeler-replay --human <file>
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mix as bm  # noqa: E402
from d1a.eval.suite import read_manifest, validate_training  # noqa: E402

# the shingle families, as the builder's but with every eval-only kit against the PR sources
FAMILIES = {**bm.FAMILIES, "pr-labels": ("pr-labels:", "EVAL-ONLY")}


def verify(text, index, sidecar, train, evals, root=bm.ROOT_DIR):
    """-> report with "failures" (empty when the mix passes). text: the mix file's contents; index: its index lines;
    sidecar: its <mix>.json; train {plan name: records}; evals {name: records}."""
    from d1a.training.train import required_sources, sidecar_counts, source_names
    lines = text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")
    raw = [json.loads(x) for x in lines]
    out, fail = {"lines": len(lines), "index_lines": len(index)}, []
    out["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if out["sha256"] != sidecar.get("sha256"): fail.append("the mix's sha256 is not its sidecar's")
    if len(lines) != len(index): fail.append("the index has a different number of lines than the mix")
    equal = sum(1 for x, ix in zip(lines, index) if ix["row"] < len(train.get(ix["source"], ())) and
                bm.line(train[ix["source"]][ix["row"]]) == x + "\n" and bm.state_hash(train[ix["source"]][ix["row"]]) == ix["state_sha256"])
    out["equal_to_source"] = equal
    if equal != len(lines): fail.append(f"{len(lines) - equal} line(s) differ from the source row their index names")
    dv = [r for r, x in zip(raw, index) if x["source"] == bm.short(bm.DECISION_V7)]
    if dv:
        try:
            validate_training(dv, read_manifest(Path(root) / bm.DECISION_V7)); out["decision_v7_validate_training"] = f"ok ({len(dv)})"
        except Exception as e:
            out["decision_v7_validate_training"] = f"{type(e).__name__}: {str(e)[:200]}"; fail.append("decision-v7 records fail validate_training")
    mh = Counter(bm.state_hash(r) for r in raw)
    ids = {str(r.get("id")).split(":")[0] for r in raw if r.get("id")}
    ov = {}
    for k, rs in evals.items():
        ns = sum(bm.state_hash(r) in mh for r in rs)
        ni = sum(str(r.get("id")).split(":")[0] in ids for r in rs if r.get("id") and k.startswith(bm.PR_ID_EVALS))
        if ns or ni: ov[k] = {"states": ns, "pr_ids": ni}
    out["eval_partitions_read"] = len(evals); out["eval_overlap"] = ov
    if "eval_partitions" in sidecar and sorted(sidecar["eval_partitions"]) != sorted(evals):
        fail.append(f"the build screened other eval partitions than this check read: {sorted(set(sidecar['eval_partitions']) ^ set(evals))}")
    out["repeated_states"] = sum(v - 1 for v in mh.values() if v > 1)
    if ov: fail.append(f"eval states or PR ids in the mix: {ov}")
    if out["repeated_states"]: fail.append(f"{out['repeated_states']} repeated state(s)")
    scr = {}
    for fam, pre in FAMILIES.items():
        E = [(k, bm.word_shingles(bm.state_text(r))) for k, rs in evals.items() if k.startswith(pre) for r in rs]
        shingles = [s for _, s in E]
        ix = defaultdict(list)
        for j, (_, s) in enumerate(E):
            for x in s: ix[x].append(j)
        hits, mx = Counter(), 0
        for r, x in zip(raw, index):
            if x["source"].split(":")[0] != fam: continue
            s = bm.word_shingles(bm.state_text(r)); c = Counter()
            for y in s:
                for j in ix.get(y, ()): c[j] += 1
            best, j = bm.nearest(s, c, shingles)
            bk = E[j][0] if j is not None else None
            mx = max(mx, best)
            for th in (0.5, bm.NEAR_SHINGLE):
                if best >= th: hits[f"{th}|{bk}"] += 1
        scr[fam] = {"max_jaccard": round(mx, 3), **dict(sorted(hits.items()))}
        if mx >= bm.NEAR_SHINGLE: fail.append(f"{fam}: a record is a near-duplicate of an eval state (Jaccard {mx:.3f})")
    out["shingle_screen_final_mix"] = scr
    by_source = Counter(x["source"] for x in index)
    out["by_source"] = dict(by_source)
    try:
        counts = sidecar_counts(sidecar)
    except ValueError as e:
        counts = {}; fail.append(f"d1a.training.train cannot read the sidecar: {e}")
    if {k: v for k, v in counts.items() if v} != dict(by_source): fail.append("the sidecar's records by source are not the index's")
    names = source_names(required_sources())
    covered = {names[n] for n, v in counts.items() if v and n in names}
    if sidecar.get("dv7_replay_by_trainer", 0) > 0: covered.add(names[bm.short(bm.DECISION_V7)])
    allowed = {names[s] for s in (sidecar.get("plan") or {}).get("allow_missing_sources") or [] if s in names}
    out["sources_missing"] = [s for s in required_sources() if s not in covered]
    if refused := [s for s in out["sources_missing"] if s not in allowed]:
        fail.append(f"the mix leaves out {', '.join(refused)} without naming it in the plan's allow_missing_sources")
    pr = [r for r, x in zip(raw, index) if x["source"].startswith("pr-labels")]
    for q in ("sev", "blast", "type"):
        c = Counter(r["questions"][q]["label"] for r in pr if q in r["questions"]); t = sum(c.values())
        if t: out[f"pr_{q}"] = {k: f"{v} ({100 * v / t:.1f}%)" for k, v in sorted(c.items())}
    out["failures"] = fail
    return out


def token_lengths(mix, name, revision, context=5120):
    from d1a.core.encoding import encode, load_tokenizer, training_context
    from d1a.training.data import load_records, materialize
    tok, ctx = load_tokenizer(name, revision=revision), training_context(context)
    lens = [len(encode(tok, materialize(r), strict=True, max_state=ctx["max_state"], max_branch=ctx["max_branch"])["ids"]) for r in load_records(mix)]
    return {"sum": sum(lens), "mean": round(sum(lens) / max(1, len(lens))), "max": max(lens, default=0),
            "over_max_packed": sum(n > ctx["max_packed"] for n in lens)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mix")
    ap.add_argument("--root", default=str(bm.ROOT_DIR), help="the checkout whose evals/ holds the pinned suites")
    ap.add_argument("--replay", help="the labeler replay kit, as the build had it"); ap.add_argument("--human", help="the owner's hand-labelled PRs")
    ap.add_argument("--tokenizer", help="also count exact token lengths with this tokenizer (e.g. google/gemma-4-E4B)")
    ap.add_argument("--revision"); ap.add_argument("--context", type=int, default=5120)
    a = ap.parse_args()
    from d1a.training.data import load_records
    text = Path(a.mix).read_text(encoding="utf-8")
    index = [json.loads(x) for x in Path(a.mix + ".index.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    sidecar = json.loads(Path(a.mix + ".json").read_text(encoding="utf-8"))
    out = verify(text, index, sidecar, bm.load_train(a.root), bm.load_evals(a.root, a.replay, a.human), a.root)
    out = {"mix": a.mix, "loads_with_load_records": len(load_records(a.mix)), **out}
    if out["loads_with_load_records"] != out["lines"]: out["failures"].append("load_records reads a different number of records")
    if a.tokenizer:
        out["tokens"] = token_lengths(a.mix, a.tokenizer, a.revision, a.context)
    Path(a.mix + ".verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(1 if out["failures"] else 0)


if __name__ == "__main__":
    main()
