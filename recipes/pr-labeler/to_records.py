"""hermes_prs.jsonl -> D1A records for the PR labeler: the labeler's own state text and its three questions, worded exactly
as production asks them (labeler_spec.py), with the maintainers' labels as targets.

type  <- the single type/* label;  sev <- P0..P4;  blast <- sweeper:blast-* (renamed to the labeler's review:blast-*).
A question is included only when the PR carries its label. Severity follows the labeler's own rules where hermes-agent's
convention differs: docs-only PRs and dependency bumps without an advisory are P4 there (hermes-agent marks them P3). Split by creation time so evaluation PRs are newer than every
training PR of its group: within each group (a record's rarest label: blast, then P0/P1/P4, then type), the newest 10% test, the next 10% development, the rest train (the
fetch is balanced per type and reaches further back for rare types, so one global cut would make the test set mostly bugs).
Usage: to_records.py prs.jsonl OUT_DIR"""
import importlib.util, json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("L", HERE / "labeler_spec.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
SEVS = set(L.SEVS)


DOCS = (".md", ".mdx", ".rst", ".txt")


def labeler_rule_sev(r):
    """The labeler's P4 rules: docs-only, and dependency bumps without a CVE/GHSA. None when no rule applies."""
    files = [f["filename"] for f in r["files"]]
    if files and all(f.lower().endswith(DOCS) or f.startswith(("website/", "docs/")) for f in files): return "P4"
    title, body = r["pr"]["title"] or "", r["pr"]["body"] or ""
    bump = (r["pr"]["user"]["login"] or "").startswith("dependabot") or re.match(r"(?i)^(build\(deps|bump |chore\(deps)", title)
    if bump and not re.search(r"(?i)CVE-|GHSA-|security", title + body): return "P4"
    return None


def record(r):
    labels = r["labels"]; qs = {}
    types = [l for l in labels if l in L.TYPES]
    if len(types) == 1: qs["type"] = {**L.D1A_QUESTIONS["type"], "label": types[0]}
    blast = [l.replace("sweeper:blast-", "review:blast-") for l in labels if l.startswith("sweeper:blast-")]
    if len(blast) == 1 and blast[0] in L.BLASTS: qs["blast"] = {**L.D1A_QUESTIONS["blast"], "label": blast[0]}
    sev = [l for l in labels if l in SEVS]
    if len(sev) == 1: qs["sev"] = {**L.D1A_QUESTIONS["sev"], "label": labeler_rule_sev(r) or sev[0]}
    if not qs: return None
    return {"state": L._pr_state(r["pr"], r["files"]), "questions": qs, "_created": r["created"], "_number": r["number"]}


def rarest(x):
    """The split group: a record's rarest label (blast, then P0/P1/P4, then its type), so every rare label reaches dev and test."""
    q = x["questions"]
    if "blast" in q: return q["blast"]["label"]
    if q.get("sev", {}).get("label") in ("P0", "P1", "P4"): return q["sev"]["label"]
    return q.get("type", {}).get("label", "-")


def main(src, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    recs = [x for x in (record(json.loads(l)) for l in open(src)) if x]
    groups = {}
    for x in recs: groups.setdefault(rarest(x), []).append(x)
    train, dev, test = [], [], []
    for xs in groups.values():
        xs.sort(key=lambda x: x["_created"]); n = len(xs)
        train += xs[:int(n * 0.8)]; dev += xs[int(n * 0.8):int(n * 0.9)]; test += xs[int(n * 0.9):]
    for xs in (train, dev, test): xs.sort(key=lambda x: x["_created"])
    for name, xs in (("train", train), ("development", dev), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for x in xs: f.write(json.dumps({"state": x["state"], "questions": x["questions"], "id": f"hermes-agent#{x['_number']}"}, ensure_ascii=False) + "\n")
        c = Counter(f"{q}={v['label']}" for x in xs for q, v in x["questions"].items())
        print(f"{name}: {len(xs)} PRs ({xs[0]['_created'][:10]}..{xs[-1]['_created'][:10]})", dict(sorted(c.items())))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
