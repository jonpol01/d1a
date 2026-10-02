"""Before/after tables from train_job.sh's eval folder: per question (type, blast, sev) and overall accuracy, and accuracy
and coverage at p >= 0.7, on the English test, Japanese test and real PRs; plus the forgetting checks against the starting
model. Usage: report_job.py EVAL_DIR   (download it with `hf download REPO --include 'PREFIX/eval/*'`)"""
import json, sys
from pathlib import Path

E = Path(sys.argv[1])
BASE = {"after-decision-v7": 0.842, "after-ja": 0.826, "after-routing-factory": 0.917}   # D1A-E4B v0.3 (the default INIT); update when INIT changes


def rows(name):
    p = E / name / "rows.json"
    return json.loads(p.read_text()) if p.exists() else None


def stats(rs, q=None):
    rs = [r for r in rs if q is None or r["question"] == q]
    if not rs: return None
    ok = [max(range(len(r["p"])), key=r["p"].__getitem__) == r["label"] for r in rs]
    hi = [o for r, o in zip(rs, ok) if max(r["p"]) >= 0.7]
    return {"n": len(rs), "acc": sum(ok) / len(rs), "hi_n": len(hi), "hi_acc": sum(hi) / len(hi) if hi else None}


out = {}
print(f"{'set':10s} {'question':8s} {'before':>8s} {'after':>8s}   at p>=0.7 before -> after (labels kept)")
for d in ("test", "test_ja", "real17"):
    b, a = rows(f"before-pr-{d}"), rows(f"after-pr-{d}")
    if not (a and b): print(d, "missing"); continue
    for q in ("type", "blast", "sev", None):
        sb, sa = stats(b, q), stats(a, q)
        if not sa: continue
        name = q or "ALL"
        out[f"{d}/{name}"] = {"before": sb, "after": sa}
        hb = f"{sb['hi_acc']:.0%} ({sb['hi_n']})" if sb["hi_acc"] is not None else "-"
        ha = f"{sa['hi_acc']:.0%} ({sa['hi_n']})" if sa["hi_acc"] is not None else "-"
        print(f"{d:10s} {name:8s} {sb['acc']:8.1%} {sa['acc']:8.1%}   {hb} -> {ha}   n={sa['n']}")
print()
for name, base in BASE.items():
    rep = E / name / "report.json"
    if rep.exists():
        acc = json.loads(rep.read_text())["clean"]["acc"]; out[name] = {"before": base, "after": acc}
        print(f"forgetting check {name:24s} start {base:.3f} -> {acc:.3f} ({acc - base:+.3f})")
json.dump(out, open(E / "summary.json", "w"), indent=1)
