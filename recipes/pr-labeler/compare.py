"""Before/after table from two scoring folders: accuracy per question and accuracy at p >= 0.7, for each named set.
Usage: compare.py SCORES_DIR test test_ja real17   (reads before-<set>.json and after-<set>.json)"""
import json, sys
from pathlib import Path
F = Path(sys.argv[1])
def load(name):
    p = F / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None
for d in sys.argv[2:]:
    b, a = load(f"before-{d}"), load(f"after-{d}")
    if not (a and b): continue
    print(f"== {d}")
    for q in ("type", "blast", "sev", "ALL"):
        def st(x):
            rows = [r for r in x["rows"] if q == "ALL" or r["q"] == q]
            if not rows: return None
            hi = [r for r in rows if r["p"] >= 0.7]
            return len(rows), sum(r["label"] == r["pred"] for r in rows) / len(rows), len(hi), (sum(r["label"] == r["pred"] for r in hi) / len(hi) if hi else float("nan"))
        sb, sa = st(b), st(a)
        if not sa: continue
        print(f"  {q:6s} n={sa[0]:4d}  before {sb[1]:6.1%}  after {sa[1]:6.1%}  ({sa[1]-sb[1]:+.1%})   p>=0.7: before {sb[3]:.0%} on {sb[2]}, after {sa[3]:.0%} on {sa[2]}")
