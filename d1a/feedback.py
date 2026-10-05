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
   input usually brings (a verifier that says yes to 77% of patches when 20% pass). A choice question gets its own
   temperature, which changes how sure the answer is but never which option it picks.
2. retrain: records() turns resolved decisions into labelled requests for a small LoRA update (d1a.train --data).
3. promote: gate() compares a candidate with the current model on held-out outcomes and promotes it only when its log
   loss is better with a bootstrap interval clear of zero and no frozen suite regressed beyond the tolerance.
   `python -m d1a.feedback promote <log> --calibrator <file>` runs this for the calibrator on a live log: it fits on
   part of the outcomes (split by meta["group"], e.g. a pull request), gates each question on the rest, and rewrites
   the file d1a.serve reads only for the questions that pass.
The log is append-only JSONL (one "decision" or "outcome" event per line), so it can be written by several processes and
replayed. An outcome may name its source in meta["src"] (e.g. "human" for a person's correction, "reviewer" for another
model's judgment); each question keeps the label from the most trusted source (PREFER), then the latest.
"""
import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np

EPS = 1e-6
PREFER = ("human",)   # outcome sources whose label wins over any other source's, whatever the order they arrived in


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _p_true(answer):
    """A yes/no answer's probability of "true" (System One answer shape), or None for other question types."""
    return float(answer["noul"]) if isinstance(answer, dict) and "noul" in answer else None


def _choice_probs(answer):
    """A choice answer's {option: probability} (System One answer shape), or None for other question types."""
    probs = answer.get("probabilities") if isinstance(answer, dict) else None
    return {k: float(v) for k, v in probs.items()} if isinstance(probs, dict) and probs else None


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

    def resolved(self, src=None):
        """Decisions with an outcome, oldest first. Each question's label comes from the outcomes of the most trusted source
        (PREFER), the latest of them; `src` keeps only outcomes from that source. label_src says where each label came from."""
        decisions, outcomes = {}, {}
        for e in self.events():
            if e["kind"] == "decision": decisions[e["id"]] = e
            elif src is None or e.get("meta", {}).get("src") == src: outcomes.setdefault(e["id"], []).append(e)
        out = []
        for i, d in decisions.items():
            if i not in outcomes: continue
            labels, label_src = {}, {}
            for o in sorted(outcomes[i], key=lambda o: (o.get("meta", {}).get("src") in PREFER, o["ts"])):   # trusted and latest last
                for qid, label in o["labels"].items(): labels[qid], label_src[qid] = label, o.get("meta", {}).get("src")
            group = next((o["meta"]["group"] for o in reversed(outcomes[i]) if o.get("meta", {}).get("group")), None)
            out.append({**d, "labels": labels, "label_src": label_src, "group": group, "outcome_ts": max(o["ts"] for o in outcomes[i])})
        return out

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
            qs[qid] = {"type": q["type"], "instructions": q.get("instr") or q.get("instructions"), "label": label,
                       "src": d.get("label_src", {}).get(qid) or src,
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


def choice_pairs(resolved, qid):
    """([{option: probability}], [observed option]) for one choice question; outcomes naming an option the decision did not
    offer are left out."""
    probs, labels = [], []
    for d in resolved:
        pr = _choice_probs(d["answers"].get(qid))
        if pr is not None and d["labels"].get(qid) in pr: probs.append(pr); labels.append(d["labels"][qid])
    return probs, labels


def _rescale(probs, s):
    """{option: p ** s}, renormalised: temperature 1/s, the same order of options."""
    lp = {k: s * math.log(max(v, EPS)) for k, v in probs.items()}; top = max(lp.values())
    z = sum(math.exp(v - top) for v in lp.values())
    return {k: math.exp(v - top) / z for k, v in lp.items()}


class OutcomeCalibrator:
    """Per yes/no question: P' = sigmoid(a * logit(P) + b), fitted to outcomes by maximum likelihood (Platt scaling). Per
    choice question: P'(option) proportional to P(option) ** s, the inverse temperature s fitted the same way."""

    def __init__(self, params=None):
        self.params = dict(params or {})   # {yes/no question id: [a, b, n], choice question id: {"s": s, "n": n}}

    def fit(self, resolved, min_outcomes=20, l2=1e-3):
        for qid in {q for d in resolved for q in d["labels"]}:
            probs, labels = choice_pairs(resolved, qid)
            if probs:
                if len(labels) >= min_outcomes and len(set(labels)) > 1: self.params[qid] = {"s": self._fit_s(probs, labels, l2), "n": len(labels)}
                continue
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

    @staticmethod
    def _fit_s(probs, labels, l2):
        """Inverse temperature by Newton's method on the mean log loss, which is convex in s (its second derivative is the
        variance of log P under the rescaled answer); l2 pulls towards s = 1, the model as it is."""
        lps = [np.log(np.clip(np.array(list(pr.values())), EPS, 1)) for pr in probs]
        ys = [list(pr).index(label) for pr, label in zip(probs, labels)]
        def stats(s):
            loss = grad = hess = 0.0
            for lp, y in zip(lps, ys):
                z = s * lp; q = np.exp(z - z.max()); q /= q.sum(); m = float(q @ lp)
                loss += float(np.log(np.exp(z - z.max()).sum()) + z.max() - z[y]); grad += m - float(lp[y]); hess += float(q @ (lp - m) ** 2)
            n = len(ys)
            return loss / n + l2 / 2 * (s - 1) ** 2, grad / n + l2 * (s - 1), hess / n + l2
        s = 1.0
        for _ in range(100):
            loss, grad, hess = stats(s); step, t = grad / hess, 1.0
            while t > 1e-6 and not 0.05 <= s - t * step <= 20: t /= 2   # keep s in [0.05, 20]
            while t > 1e-6 and stats(s - t * step)[0] > loss: t /= 2
            s -= t * step
            if abs(t * step) < 1e-9: break
        return float(s)

    def p(self, qid, p_true):
        if not isinstance(self.params.get(qid), list): return p_true
        a, b, _ = self.params[qid]
        return float(1 / (1 + math.exp(-max(-40.0, min(40.0, a * float(_logit(p_true)) + b)))))

    def probs(self, qid, probs):
        """A choice answer's {option: probability} recalibrated (the same order of options)."""
        return _rescale(probs, self.params[qid]["s"]) if isinstance(self.params.get(qid), dict) else probs

    def apply(self, answers):
        """Answers with their yes/no and choice probabilities recalibrated; the chosen option never changes."""
        out = {}
        for qid, ans in answers.items():
            if _p_true(ans) is not None: ans = {**ans, "noul": round(self.p(qid, ans["noul"]), 4)}
            elif _choice_probs(ans) is not None and isinstance(self.params.get(qid), dict):
                ans = {**ans, "probabilities": {k: round(v, 4) for k, v in self.probs(qid, _choice_probs(ans)).items()}}
            out[qid] = ans
        return out

    def save(self, path): Path(path).write_text(json.dumps({"kind": "d1a-outcome-calibrator", "params": self.params}, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path): return cls(json.loads(Path(path).read_text(encoding="utf-8"))["params"])


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def choice_log_loss(probs, labels):
    """-log P(observed option) per item: the multi-class log loss of choice answers ({option: probability} each)."""
    return -np.log(np.clip([pr.get(label, 0.0) for pr, label in zip(probs, labels)], EPS, 1))


def gate(y, p_candidate, p_incumbent, groups=None, suite_deltas=None, tolerance=0.01, n=2000, seed=0):
    """Promote the candidate only if its mean log loss on held-out outcomes is lower with a 95% bootstrap interval (over
    `groups`, e.g. repositories, else over items) entirely below zero, and no frozen suite's accuracy dropped by more
    than `tolerance` (suite_deltas: {suite: candidate minus incumbent accuracy}). -> {"promote": bool, ...}"""
    return _gate(log_loss(y, p_candidate) - log_loss(y, p_incumbent), groups, suite_deltas, tolerance, n, seed)


def gate_choice(labels, probs_candidate, probs_incumbent, groups=None, suite_deltas=None, tolerance=0.01, n=2000, seed=0):
    """gate() for a choice question: the same rule on the multi-class log loss ({option: probability} per item)."""
    d = choice_log_loss(probs_candidate, labels) - choice_log_loss(probs_incumbent, labels)
    return _gate(d, groups, suite_deltas, tolerance, n, seed)


def held_out(group, share=0.3):
    """Whether a group (e.g. one pull request: all its decisions together) is on the held-out side; fixed by its name."""
    return int(hashlib.sha256(str(group).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def _group(d): return d.get("group") or d["id"]


def split(resolved, share=0.3):
    """(fit, held out): each decision goes with its group (outcome meta["group"], else itself), so no group is on both sides."""
    return [d for d in resolved if not held_out(_group(d), share)], [d for d in resolved if held_out(_group(d), share)]


def promote(resolved, current=None, share=0.3, min_outcomes=20, **gate_kw):
    """The promotion gate run on a live log. Each decision's group (its outcome's meta["group"], else the decision itself)
    goes to the fit or the held-out side by held_out(); a calibrator fitted on the fit side replaces the current one for a
    question only if it passes gate()/gate_choice() on the held-out side against the current calibrator (or the raw answers).
    -> (the calibrator to serve, {question: report}); questions that fail keep the current parameters."""
    current = current or OutcomeCalibrator()
    fit, test = split(resolved, share)
    cand = OutcomeCalibrator().fit(fit, min_outcomes)
    out, report = OutcomeCalibrator(current.params), {}
    for qid, params in cand.params.items():
        if isinstance(params, dict):
            rows = [d for d in test if _choice_probs(d["answers"].get(qid)) is not None and d["labels"].get(qid) in _choice_probs(d["answers"][qid])]
            raw = [_choice_probs(d["answers"][qid]) for d in rows]; labels = [d["labels"][qid] for d in rows]
            g = gate_choice(labels, [cand.probs(qid, p) for p in raw], [current.probs(qid, p) for p in raw], [_group(d) for d in rows], **gate_kw) if rows else None
        else:
            rows = [d for d in test if _p_true(d["answers"].get(qid)) is not None and qid in d["labels"]]
            y = [bool(d["labels"][qid]) for d in rows]; p = [_p_true(d["answers"][qid]) for d in rows]
            g = gate(y, [cand.p(qid, x) for x in p], [current.p(qid, x) for x in p], [_group(d) for d in rows], **gate_kw) if rows else None
        ok = bool(g and g["promote"])
        if ok: out.params[qid] = params
        report[qid] = {"promote": ok, "fit": len(fit), "held_out": len(rows), **({k: g[k] for k in ("delta_log_loss", "ci", "reasons")} if g else {"reasons": ["no held-out outcomes"]})}
    return out, report


def _gate(d, groups, suite_deltas, tolerance, n, seed):
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
    r = sub.add_parser("records"); r.add_argument("log"); r.add_argument("--out", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("log"); c.add_argument("--out", required=True); c.add_argument("--min-outcomes", type=int, default=20)
    pr = sub.add_parser("promote", help="fit on part of the log, gate on the rest, and update the served calibrator only where it passes")
    pr.add_argument("log"); pr.add_argument("--calibrator", required=True, help="the file d1a.serve's D1A_OUTCOME_CALIBRATOR reads")
    pr.add_argument("--min-outcomes", type=int, default=20); pr.add_argument("--held-out", type=float, default=0.3)
    for x in (s, r, c, pr): x.add_argument("--src", help="only outcomes from this source (meta src, e.g. human or reviewer)")
    a = ap.parse_args(argv)
    log = FeedbackLog(a.log); res = log.resolved(a.src)
    if a.cmd == "status":
        print(f"{len(res)} resolved, {len(log.pending())} pending decisions")
        for qid in sorted({q for d in res for q in d["labels"]}):
            p, y = pairs(res, qid); probs, labels = choice_pairs(res, qid)
            if len(y): print(f"  {qid}: {len(y)} outcomes, {y.mean():.1%} true, mean P {p.mean():.3f}, log loss {log_loss(y, p).mean():.4f}")
            if labels:
                top = np.mean([max(pr, key=pr.get) == label for pr, label in zip(probs, labels)])
                print(f"  {qid}: {len(labels)} outcomes, top choice right {top:.1%}, log loss {choice_log_loss(probs, labels).mean():.4f}")
    elif a.cmd == "records":
        recs = records(res, a.src or "feedback"); Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        print(f"wrote {len(recs)} labelled requests to {a.out}")
    elif a.cmd == "promote":
        path = Path(a.calibrator); current = OutcomeCalibrator.load(path) if path.exists() else None
        cal, report = promote(res, current, a.held_out, a.min_outcomes)
        if any(r["promote"] for r in report.values()):   # written whole and renamed into place: the server never reads half a file
            tmp = path.with_name(path.name + ".tmp"); cal.save(tmp); os.replace(tmp, path)
        print(json.dumps({"promoted": sorted(q for q, r in report.items() if r["promote"]), "report": report}))
    else:
        cal = OutcomeCalibrator().fit(res, a.min_outcomes); cal.save(a.out)
        print(f"wrote {a.out}: " + ", ".join(f"{q} s={v['s']:.3f} (n={v['n']})" if isinstance(v, dict) else f"{q} a={v[0]:.3f} b={v[1]:+.3f} (n={v[2]})"
                                             for q, v in cal.params.items()))


if __name__ == "__main__":
    main()
