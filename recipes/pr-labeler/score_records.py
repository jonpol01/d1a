"""Score a D1A checkpoint on PR-labeler records (dev or test): accuracy per question, calibration, and accuracy above
confidence thresholds. Usage: score_records.py RUN records.jsonl OUT.json [limit]"""
import json, sys, time
from collections import Counter, defaultdict
from d1a import D1A

run, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
recs = [json.loads(l) for l in open(path)][:limit]
m = D1A.load(run)
rows, t0 = [], time.time()
for i, r in enumerate(recs):
    qs = {k: {kk: vv for kk, vv in q.items() if kk != "label"} for k, q in r["questions"].items()}
    ans = m.decide(r["state"], qs)
    for k, q in r["questions"].items():
        a = ans[k]; p = a["probabilities"]
        rows.append({"id": r["id"], "q": k, "label": q["label"], "pred": a["choice"], "p": max(p.values()), "p_label": p.get(q["label"], 0.0)})
    if i % 100 == 99: print(i + 1, "records", round(time.time() - t0), "s", flush=True)
summary = {}
for q in sorted({x["q"] for x in rows}):
    xs = [x for x in rows if x["q"] == q]
    acc = sum(x["label"] == x["pred"] for x in xs) / len(xs)
    by = defaultdict(lambda: [0, 0])
    for x in xs: by[x["label"]][0] += x["label"] == x["pred"]; by[x["label"]][1] += 1
    hi = [x for x in xs if x["p"] >= 0.7]
    summary[q] = {"n": len(xs), "acc": round(acc, 3), "per_label_recall": {k: f"{a}/{n}" for k, (a, n) in sorted(by.items())},
                  "at_p>=0.7": {"kept": len(hi), "acc": round(sum(x["label"] == x["pred"] for x in hi) / len(hi), 3) if hi else None},
                  "predicted": dict(Counter(x["pred"] for x in xs))}
json.dump({"run": run, "data": path, "summary": summary, "rows": rows}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
