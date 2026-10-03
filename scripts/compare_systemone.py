"""Score any System One server on D1A record files, through its HTTP API: D1A (d1a.serve), Clef through Ollama, Jev, or
anything else that answers POST /v1/systemone. Every server gets the same requests (the record's state and questions,
labels removed) from the same client, so their numbers compare directly.

    python scripts/compare_systemone.py --endpoint http://127.0.0.1:11434 --model clef-flash --out runs/compare/clef-flash \\
        --data test=records/test.jsonl --data jglue=jglue/development.jsonl

Per record file and question: accuracy (argmax against the label), accuracy and coverage at p >= 0.7, expected
calibration error (10 bins on the top probability) and median / p95 request latency. Rows go to OUT/rows.jsonl as they
come back, so an interrupted run continues where it stopped; OUT/summary.json holds the table.
"""
import argparse, json, statistics, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path


def ask(endpoint, model, rec, timeout):
    body = {"model": model, "state": rec["state"],
            "questions": {k: {kk: vv for kk, vv in q.items() if kk in ("type", "instructions", "criteria")} for k, q in rec["questions"].items()}}   # no label, no record extras
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["answers"], (time.perf_counter() - t) * 1000


def read(q, a):
    """-> (predicted label, its probability, the label's probability) in the record's label space."""
    if q["type"] == "noul":
        p = a["noul"]
        pred = p >= 0.5
        label = q["label"] if isinstance(q["label"], bool) else str(q["label"]).lower() in ("true", "yes", "1")
        return pred, max(p, 1 - p), p if label else 1 - p
    probs = a["probabilities"]
    if q["type"] == "score":
        probs = {int(k): v for k, v in probs.items()}
    pred = max(probs, key=probs.get)
    return pred, probs[pred], probs.get(q["label"], 0.0)


def summarize(rows):
    def stats(rs):
        ok = [r["pred"] == r["label"] for r in rs]
        hi = [o for r, o in zip(rs, ok) if r["p"] >= 0.7]
        bins = defaultdict(list)
        for r, o in zip(rs, ok): bins[min(int(r["p"] * 10), 9)].append((r["p"], o))
        ece = sum(len(b) / len(rs) * abs(statistics.mean(p for p, _ in b) - statistics.mean(o for _, o in b)) for b in bins.values())
        return {"n": len(rs), "acc": round(sum(ok) / len(rs), 4), "acc_p70": round(sum(hi) / len(hi), 4) if hi else None,
                "coverage_p70": round(len(hi) / len(rs), 4), "ece": round(ece, 4)}
    out = {}
    for f in sorted({r["file"] for r in rows}):
        rs = [r for r in rows if r["file"] == f]
        ms = sorted({r["id"]: r["ms"] for r in rs}.values())
        out[f] = {"all": stats(rs), **{q: stats([r for r in rs if r["q"] == q]) for q in sorted({r["q"] for r in rs})},
                  "latency_ms": {"p50": round(statistics.median(ms), 1), "p95": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="server base URL, e.g. http://127.0.0.1:8009")
    ap.add_argument("--model", default="d1a-latest")
    ap.add_argument("--data", action="append", required=True, help="NAME=records.jsonl (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N records of each file (0 = all)")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    done = {(r["file"], r["id"]) for r in rows if "error" not in r}   # failed requests are retried
    with open(path, "a", encoding="utf-8") as f:
        for spec in a.data:
            name, _, file = spec.partition("=")
            recs = [json.loads(l) for l in open(file, encoding="utf-8") if l.strip()]
            recs = recs[:a.limit] if a.limit else recs
            t0, n = time.time(), 0
            for i, rec in enumerate(recs):
                rid = rec.get("id", i)
                if (name, rid) in done: continue
                try:
                    answers, ms = ask(a.endpoint, a.model, rec, a.timeout)
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                    detail = e.read().decode()[:200] if isinstance(e, urllib.error.HTTPError) else str(e)[:200]
                    f.write(json.dumps({"file": name, "id": rid, "error": detail}) + "\n"); f.flush(); continue
                for qid, q in rec["questions"].items():
                    pred, p, pl = read(q, answers[qid])
                    row = {"file": name, "id": rid, "q": qid, "label": q["label"], "pred": pred, "p": p, "p_label": pl, "ms": round(ms, 1)}
                    rows.append(row); f.write(json.dumps(row) + "\n")
                f.flush(); n += 1
                if n % 50 == 0: print(f"{name}: {i + 1}/{len(recs)} ({time.time() - t0:.0f} s)", flush=True)
            print(f"{name}: done", flush=True)
    rows = [r for r in rows if "error" not in r]
    errors = sum(1 for l in open(path, encoding="utf-8") if '"error"' in l)
    summary = {"endpoint": a.endpoint, "model": a.model, "errors": errors, "sets": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
