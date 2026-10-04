"""Learning from outcomes. D1A logs each decision; the world later reports what actually happened (a patch passed its
tests, a maintainer corrected a label, a routed task failed); the outcomes improve the model under a promotion gate.

    from d1a.feedback import FeedbackLog
    log = FeedbackLog("runs/feedback/verifier.jsonl")
    did = log.decision(state, questions, answers, run="JohnP1/d1a-e4b-mlx-q8@v0.4", meta={"repo": "psf/requests"})
    ...                                        # later, when the outcome is known
    log.outcome(did, {"resolved": True})

    python -m d1a.feedback status runs/feedback/verifier.jsonl
    python -m d1a.feedback records runs/feedback/verifier.jsonl --out feedback.jsonl     # for d1a.train --data
    python -m d1a.feedback calibrate runs/feedback/verifier.jsonl --out calibrator.json

The loop has three steps, cheapest first:
1. recalibrate: an OutcomeCalibrator refits each yes/no question's probability on the outcomes (Platt scaling on the
   logit). Unlike the checkpoint's single temperature it also corrects a shifted base rate, which is what a new kind of
   input usually brings (a verifier that says yes to 77% of patches when 20% pass).
2. retrain: records() turns resolved decisions into labelled requests for a small LoRA update (d1a.train --data).
3. promote: gate() compares a candidate with the current model on held-out outcomes and promotes it only when its log
   loss is better with a bootstrap interval clear of zero and no frozen suite regressed beyond the tolerance.
The log is append-only JSONL (one "decision" or "outcome" event per line), so it can be written by several processes and
replayed. Only yes/no ("noul") questions are calibrated and gated for now; choice questions are logged and exported.
"""
import argparse
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _p_true(answer):
    """A yes/no answer's probability of "true" (System One answer shape), or None for other question types."""
    return float(answer["noul"]) if isinstance(answer, dict) and "noul" in answer else None


class FeedbackLog:
    def __init__(self, path):
        self.path = Path(path)

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")

    def decision(self, state, questions, answers, run, meta=None):
        """Record a decision as it was made: the request, the answers and the model `run` (repo@tag). -> its id."""
        did = uuid.uuid4().hex
        self._append({"kind": "decision", "id": did, "ts": time.time(), "run": run, "state": state, "questions": questions,
                      "answers": answers, "meta": meta or {}})
        return did

    def outcome(self, did, labels, meta=None):
        """Record what actually happened for decision `did`: {question id: label} (bool for yes/no, option key for choice)."""
        self._append({"kind": "outcome", "id": did, "ts": time.time(), "labels": labels, "meta": meta or {}})

    def events(self):
        if not self.path.exists(): return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def resolved(self):
        """Decisions with an outcome, oldest first; the latest outcome for a decision wins."""
        decisions, outcomes = {}, {}
        for e in self.events():
            (decisions if e["kind"] == "decision" else outcomes)[e["id"]] = e
        return [{**d, "labels": outcomes[i]["labels"], "outcome_ts": outcomes[i]["ts"]} for i, d in decisions.items() if i in outcomes]

    def pending(self):
        done = {e["id"] for e in self.events() if e["kind"] == "outcome"}
        return [e for e in self.events() if e["kind"] == "decision" and e["id"] not in done]


def records(resolved, src="feedback"):
    """Resolved decisions as labelled requests (d1a.data's format), keeping only the questions that got an outcome."""
    out = []
    for d in resolved:
        qs = {}
        for qid, label in d["labels"].items():
            q = d["questions"].get(qid)
            if q is None: continue
            qs[qid] = {"type": q["type"], "instructions": q.get("instr") or q.get("instructions"), "label": label, "src": src,
                       **({"criteria": q["criteria"]} if "criteria" in q else {})}
        if qs: out.append({"state": d["state"], "questions": qs, "meta": {**d.get("meta", {}), "feedback_id": d["id"], "run": d["run"]}})
    return out


def pairs(resolved, qid):
    """(p_true, outcome) arrays for one yes/no question across resolved decisions."""
    p, y = [], []
    for d in resolved:
        pt = _p_true(d["answers"].get(qid))
        if pt is not None and qid in d["labels"]: p.append(pt); y.append(bool(d["labels"][qid]))
    return np.array(p), np.array(y)


class OutcomeCalibrator:
    """Per yes/no question: P' = sigmoid(a * logit(P) + b), fitted to outcomes by maximum likelihood (Platt scaling)."""

    def __init__(self, params=None):
        self.params = dict(params or {})   # {question id: [a, b, n]}

    def fit(self, resolved, min_outcomes=20, l2=1e-3):
        for qid in {q for d in resolved for q in d["labels"]}:
            p, y = pairs(resolved, qid)
            if len(y) < min_outcomes or y.all() or not y.any(): continue
            x = _logit(p); a, b = 1.0, 0.0
            loss = lambda a, b: float(log_loss(y, 1 / (1 + np.exp(-np.clip(a * x + b, -40, 40)))).mean() + l2 / 2 * (a - 1) ** 2)
            for _ in range(100):   # Newton steps with backtracking: the logits of an overconfident model are nearly collinear with the intercept
                z = 1 / (1 + np.exp(-np.clip(a * x + b, -40, 40))); g = z - y; w = z * (1 - z) + 1e-9
                grad = np.array([(g * x).mean() + l2 * (a - 1), g.mean()])
                hess = np.array([[(w * x * x).mean() + l2, (w * x).mean()], [(w * x).mean(), w.mean()]])
                step, t, now = np.linalg.solve(hess, grad), 1.0, loss(a, b)
                while loss(a - t * step[0], b - t * step[1]) > now and t > 1e-6: t /= 2
                a, b = a - t * step[0], b - t * step[1]
                if abs(t * step).max() < 1e-9: break
            self.params[qid] = [float(a), float(b), int(len(y))]
        return self

    def p(self, qid, p_true):
        if qid not in self.params: return p_true
        a, b, _ = self.params[qid]
        return float(1 / (1 + math.exp(-max(-40.0, min(40.0, a * float(_logit(p_true)) + b)))))

    def apply(self, answers):
        """Answers with their yes/no probabilities recalibrated; other answers unchanged."""
        return {qid: ({**ans, "noul": round(self.p(qid, ans["noul"]), 4)} if _p_true(ans) is not None else ans) for qid, ans in answers.items()}

    def save(self, path): Path(path).write_text(json.dumps({"kind": "d1a-outcome-calibrator", "params": self.params}, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path): return cls(json.loads(Path(path).read_text(encoding="utf-8"))["params"])


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gate(y, p_candidate, p_incumbent, groups=None, suite_deltas=None, tolerance=0.01, n=2000, seed=0):
    """Promote the candidate only if its mean log loss on held-out outcomes is lower with a 95% bootstrap interval (over
    `groups`, e.g. repositories, else over items) entirely below zero, and no frozen suite's accuracy dropped by more
    than `tolerance` (suite_deltas: {suite: candidate minus incumbent accuracy}). -> {"promote": bool, ...}"""
    d = log_loss(y, p_candidate) - log_loss(y, p_incumbent)
    groups = np.arange(len(d)) if groups is None else np.asarray(groups)
    keys = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in keys}; rng = np.random.default_rng(seed)
    boots = np.sort([d[np.concatenate([idx[g] for g in rng.choice(keys, len(keys))])].mean() for _ in range(n)])
    lo, hi = float(boots[int(0.025 * n)]), float(boots[int(0.975 * n) - 1])
    regressions = {s: v for s, v in (suite_deltas or {}).items() if v < -tolerance}
    reasons = []
    if hi >= 0: reasons.append(f"log loss not clearly lower (delta {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])")
    if regressions: reasons.append("frozen suites regressed: " + ", ".join(f"{s} {v:+.3f}" for s, v in regressions.items()))
    return {"promote": not reasons, "delta_log_loss": float(d.mean()), "ci": [lo, hi], "regressions": regressions, "reasons": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("log")
    r = sub.add_parser("records"); r.add_argument("log"); r.add_argument("--out", required=True); r.add_argument("--src", default="feedback")
    c = sub.add_parser("calibrate"); c.add_argument("log"); c.add_argument("--out", required=True); c.add_argument("--min-outcomes", type=int, default=20)
    a = ap.parse_args(argv)
    log = FeedbackLog(a.log); res = log.resolved()
    if a.cmd == "status":
        print(f"{len(res)} resolved, {len(log.pending())} pending decisions")
        for qid in sorted({q for d in res for q in d["labels"]}):
            p, y = pairs(res, qid)
            if len(y): print(f"  {qid}: {len(y)} outcomes, {y.mean():.1%} true, mean P {p.mean():.3f}, log loss {log_loss(y, p).mean():.4f}")
    elif a.cmd == "records":
        recs = records(res, a.src); Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        print(f"wrote {len(recs)} labelled requests to {a.out}")
    else:
        cal = OutcomeCalibrator().fit(res, a.min_outcomes); cal.save(a.out)
        print(f"wrote {a.out}: " + ", ".join(f"{q} a={v[0]:.3f} b={v[1]:+.3f} (n={v[2]})" for q, v in cal.params.items()))


if __name__ == "__main__":
    main()
