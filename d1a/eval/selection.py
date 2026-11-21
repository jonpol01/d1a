"""Selection: which of a day's candidate items belong in its report. D1A answers one yes/no question per candidate, with
the fixed question id "include" and the day's whole candidate list as the state, and is scored against a teacher's or a
person's choice (the gold) and against the rule the use case would otherwise run on. D1A selects; it writes no prose.

    python -m d1a.eval.selection precheck days.jsonl --server http://127.0.0.1:8009 [--save-scores scores.jsonl] [--out r.json]
    python -m d1a.eval.selection report   days.jsonl --scores scores.jsonl [--out report.json]

A row of days.jsonl is one day (one group), the rows in time order:

    {"group": "2026-11-20", "use_case": "<name>", "candidates": [{"id": "c1", "facts": "<one line>", "type": "<category>", "z": 3.1}],
     "gold": ["c1"], "gold_src": "teacher", "generator_misses": 0, "gold_alt": ["c1"]}

gold_src is "teacher" or "human"; generator_misses counts the teacher's facts that matched no candidate (the recall of
whatever produced the candidates); gold_alt, optional, is a second teacher draft of the same day. A scores file has one
{"group", "scores": {candidate id: p(include)}} per day.

precheck asks whether D1A's zero-shot p(include) ranks a day's candidates better than the rule, per day (AUC of gold
against not gold), on the teacher's days:
- the rule: a logistic model logit = a*|z| + b_type with a small ridge penalty, fitted leave-one-day-out, so each day is
  scored by a model that never saw it; and, secondary, |z| alone;
- the paired per-day difference, D1A minus the rule, with a bootstrap that resamples whole days (10,000, seed 0);
- it passes when at least 14 days count and the interval's lower bound is above 0 against both. A day whose candidates are
  all gold, or none, has no AUC and is counted out with its reason. Days a person chose are reported beside, not judged.
If it fails, the use case runs on the rule and D1A does not select.

report adds selection at three thresholds, fitted on the first 70% of days in time order and reported on the rest:
p >= 0.5; the threshold a served calibrator would apply (Platt scaling through d1a.learning.feedback.OutcomeCalibrator,
then p' >= 0.5); and the threshold that maximises micro F1 on the fit days, which is a ceiling, not a setting. The rule is
thresholded the same ways. It also gives the teacher's agreement with its own second draft (a ceiling for any selector)
and the generator's misses.
"""
import argparse
import ipaddress
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

QUESTION = "include"
SOURCES = ("teacher", "human")
MIN_DAYS, SAMPLES, SEED, FIT_SHARE, L2 = 14, 10_000, 0, 0.7, 1.0
ID = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")   # a candidate id, as the question and the state's lines name it
CONTROL = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]")   # a newline in a candidate's text could forge another line of the state


# --- days and requests --------------------------------------------------------------------------------------------------

def check_days(days):
    """The rows' problems (empty when every day can be scored)."""
    problems, groups = [], set()
    for i, d in enumerate(days):
        where = f"row {i + 1} ({d.get('group')!r})"
        if not isinstance(d.get("group"), str) or not d["group"].strip(): problems.append(f"{where}: group, a non-empty string")
        elif d["group"] in groups: problems.append(f"{where}: the group appears twice")
        groups.add(d.get("group"))
        if d.get("gold_src") not in SOURCES:   # a teacher's labels are never "judge", the blind judge's source
            problems.append(f"{where}: gold_src {d.get('gold_src')!r}, one of {list(SOURCES)}")
        cands = d.get("candidates")
        if not isinstance(cands, list) or not cands: problems.append(f"{where}: candidates, a non-empty list"); continue
        ids = [c.get("id") for c in cands if isinstance(c, dict)]
        if len(ids) != len(cands) or len(set(ids)) != len(ids) or not all(isinstance(x, str) and ID.match(x) for x in ids):
            problems.append(f"{where}: every candidate needs a distinct id of letters, digits, '_', '.' or '-' (at most 32)")
        if any(isinstance(v, str) and CONTROL.search(v) for v in [d.get("group")] + [c.get(k) for c in cands if isinstance(c, dict) for k in ("type", "facts")]):
            problems.append(f"{where}: a group, type or facts with a newline or another control character")
        if not all(isinstance(c.get("z"), (int, float)) and not isinstance(c.get("z"), bool) and math.isfinite(c["z"]) for c in cands):
            problems.append(f"{where}: every candidate needs a finite z")
        if not all(isinstance(c.get("type"), str) and isinstance(c.get("facts"), str) for c in cands):
            problems.append(f"{where}: every candidate needs a type and facts")
        for key in ("gold", "gold_alt"):
            if key in d and not (isinstance(d[key], list) and set(d[key]) <= set(ids)): problems.append(f"{where}: {key} names a candidate the day does not have")
        if "gold" not in d: problems.append(f"{where}: gold, the list of chosen candidate ids")
        if not (isinstance(d.get("generator_misses", 0), int) and d.get("generator_misses", 0) >= 0): problems.append(f"{where}: generator_misses, an integer >= 0")
    return problems


def load_days(path):
    days = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = check_days(days)
    if problems: raise ValueError("; ".join(problems[:10]) + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
    return days


def state(day):
    """The day's state document: every candidate, one line each."""
    return "\n".join([f"Candidate items for {day['group']}:"] + [f"{c['id']} ({c['type']}, z {c['z']:+.1f}): {c['facts']}" for c in day["candidates"]])


def request(day, candidate):
    """One System One request: the day's state and one yes/no question about one candidate. The question id is the same for
    every candidate, so a calibrator fitted on outcomes learns one question, not one per position in the list."""
    q = {"type": "noul", "instructions": f"Does item {candidate['id']} belong in this day's report?"}
    return {"state": state(day), "questions": {QUESTION: q}, **({"use_case": day["use_case"]} if day.get("use_case") else {})}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None   # a redirect is an error: urllib would carry the key along


def _loopback(host):
    try: return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def score_server(days, server, api_key=None, timeout=120, retries=3):
    """{group: {candidate id: p(include)}} from a System One server, one request per candidate (the server keeps the day's
    state prefix cached between them). A key goes only over https or to this machine, and never follows a redirect."""
    url = urllib.parse.urlsplit(server)
    if url.scheme not in ("http", "https"): raise ValueError(f"{server}: an http or https URL")
    if api_key and url.scheme != "https" and not _loopback(url.hostname or ""):
        raise ValueError(f"{server}: a key is sent only over https or to a loopback address")
    opener = urllib.request.build_opener(_NoRedirect)
    out = {}
    for d in days:
        out[d["group"]] = {}
        for c in d["candidates"]:
            body = json.dumps(request(d, c)).encode()
            headers = {"content-type": "application/json", **({"authorization": f"Bearer {api_key}"} if api_key else {})}
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(f"{server.rstrip('/')}/v1/systemone", data=body, headers=headers, method="POST")
                    with opener.open(req, timeout=timeout) as r: answer = json.loads(r.read())
                    out[d["group"]][c["id"]] = float(answer["answers"][QUESTION]["noul"]); break
                except Exception:
                    if attempt == retries - 1: raise
                    time.sleep(2 ** attempt)
    return out


def load_scores(path):
    return {r["group"]: {k: float(v) for k, v in r["scores"].items()} for r in
            (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())}


# --- the rule -----------------------------------------------------------------------------------------------------------

def _features(cands, types):
    return np.array([[1.0, abs(c["z"])] + [float(c["type"] == t) for t in types] for c in cands])


def _logistic(X, y, l2):
    """Ridge-penalised logistic regression (no penalty on the intercept) by Newton's method with backtracking."""
    w = np.zeros(X.shape[1]); pen = np.full(X.shape[1], l2); pen[0] = 1e-6
    loss = lambda w: float(np.logaddexp(0, X @ w).sum() - y @ (X @ w) + 0.5 * (pen * w * w).sum())
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -40, 40)))
        g = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step, t, now = np.linalg.solve(H, g), 1.0, loss(w)
        while loss(w - t * step) > now and t > 1e-6: t /= 2
        w = w - t * step
        if np.abs(t * step).max() < 1e-9: break
    return w


def rule_scores(days, l2=L2):
    """{group: {candidate id: p}} from the rule (a*|z| + b_type), each day scored by a fit on every other day."""
    types = sorted({c["type"] for d in days for c in d["candidates"]})
    X = [_features(d["candidates"], types) for d in days]
    y = [np.array([float(c["id"] in set(d["gold"])) for c in d["candidates"]]) for d in days]
    out = {}
    for i, d in enumerate(days):
        rest = [j for j in range(len(days)) if j != i]
        w = _logistic(np.concatenate([X[j] for j in rest]), np.concatenate([y[j] for j in rest]), l2)
        out[d["group"]] = dict(zip((c["id"] for c in d["candidates"]), (1 / (1 + np.exp(-np.clip(X[i] @ w, -40, 40)))).tolist()))
    return out


def z_scores(days):
    return {d["group"]: {c["id"]: abs(c["z"]) for c in d["candidates"]} for d in days}


# --- ranking ------------------------------------------------------------------------------------------------------------

def auc(scores, positive):
    """The probability that a gold candidate outscores a non-gold one, ties counting half; None without both kinds."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    if y.all() or not y.any(): return None
    diff = s[y][:, None] - s[~y][None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def day_interval(values, samples=SAMPLES, seed=SEED):
    """The mean and a 95% percentile interval over `samples` resamples of whole days."""
    v = np.asarray(values, float)
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(samples, len(v)))].mean(axis=1)
    return float(v.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ranking(days, scores, rules, samples=SAMPLES, seed=SEED):
    """Per-day AUC for D1A and for each rule, and D1A minus each rule, paired by day, with its day-resampled interval."""
    rows, excluded = [], []
    for d in days:
        gold, ids = set(d["gold"]), [c["id"] for c in d["candidates"]]
        missing = [i for i in ids if i not in scores.get(d["group"], {})]
        if missing: raise ValueError(f"{d['group']}: no D1A score for {missing[:5]}")
        pos = [i in gold for i in ids]
        if all(pos) or not any(pos):
            excluded.append({"group": d["group"], "why": "every candidate is gold" if all(pos) else "no candidate is gold"}); continue
        rows.append({"group": d["group"], "d1a": auc([scores[d["group"]][i] for i in ids], pos),
                     **{name: auc([r[d["group"]][i] for i in ids], pos) for name, r in rules.items()}})
    out = {"n_days": len(rows), "excluded": excluded, "per_day": rows, "vs": {}}
    for name in rules:
        if not rows: out["vs"][name] = {"mean": None, "ci": None}; continue
        mean, ci = day_interval([r["d1a"] - r[name] for r in rows], samples, seed)
        out["vs"][name] = {"mean": round(mean, 5), "ci": [round(x, 5) for x in ci], "d1a_auc": round(float(np.mean([r["d1a"] for r in rows])), 4),
                           "rule_auc": round(float(np.mean([r[name] for r in rows])), 4)}
    return out


def precheck(days, scores, samples=SAMPLES, seed=SEED, min_days=MIN_DAYS):
    """The pre-check report: ranking on the teacher's days with a pass or fail, and the people's days beside it."""
    rules = {"rule": rule_scores(days), "z": z_scores(days)}
    teacher = ranking([d for d in days if d["gold_src"] == "teacher"], scores, rules, samples, seed)
    reasons = [] if teacher["n_days"] >= min_days else [f"{teacher['n_days']} teacher days count, fewer than {min_days}"]
    for name, v in teacher["vs"].items():
        if v["ci"] is not None and v["ci"][0] <= 0: reasons.append(f"against {name}: the interval's lower bound {v['ci'][0]:+.4f} is not above 0")
    human = [d for d in days if d["gold_src"] == "human"]
    return {"question": QUESTION, "teacher": teacher, "human": ranking(human, scores, rules, samples, seed) if human else None,
            "verdict": "pass" if not reasons else "fail", "reasons": reasons, "samples": samples, "seed": seed, "min_days": min_days}


# --- selection ----------------------------------------------------------------------------------------------------------

def _pairs(days, scores):
    p = np.array([scores[d["group"]][c["id"]] for d in days for c in d["candidates"]], float)
    y = np.array([c["id"] in set(d["gold"]) for d in days for c in d["candidates"]], bool)
    return p, y


def prf(chosen, gold):
    tp = int((chosen & gold).sum()); fp = int((chosen & ~gold).sum()); fn = int((~chosen & gold).sum())
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "chosen": tp + fp, "gold": tp + fn}


def best_threshold(p, y):
    """The threshold among the scores that maximises micro F1 (choosing p >= it)."""
    best = (-1.0, 1.0)
    for t in sorted(set(p.tolist())):
        f1 = prf(p >= t, y)["f1"] or 0.0
        if f1 > best[0]: best = (f1, t)
    return best[1]


def calibrated_threshold(p, y):
    """The raw p at which a calibrator fitted on (p, y) as a served calibrator is fitted (d1a.learning.feedback's Platt
    scaling) gives p' = 0.5, as a function p -> chosen; None when it cannot be fitted (too few outcomes, one class only)."""
    from d1a.learning.feedback import OutcomeCalibrator
    rows = [{"answers": {QUESTION: {"type": "noul", "noul": float(x)}}, "labels": {QUESTION: bool(t)}} for x, t in zip(p, y)]
    cal = OutcomeCalibrator()._fit(rows)
    if QUESTION not in cal.params: return None
    return lambda q: np.array([cal.p(QUESTION, float(x)) >= 0.5 for x in q], bool)


def selection(days, scores, fit_share=FIT_SHARE):
    """Precision, recall and F1 on the last (1 - fit_share) of days, at p >= 0.5, at the calibrated threshold and at the
    fit days' best-F1 threshold (a ceiling), for D1A and for the rule."""
    cut = int(round(fit_share * len(days)))
    fit, test = days[:cut], days[cut:]
    if not fit or not test: return {"fit_days": len(fit), "test_days": len(test), "skipped": "too few days to split"}
    out = {"fit_days": len(fit), "test_days": len(test), "fit_groups": [fit[0]["group"], fit[-1]["group"]],
           "test_groups": [test[0]["group"], test[-1]["group"]]}
    for name, s in (("d1a", scores), ("rule", rule_scores(days))):
        pf, yf = _pairs(fit, s); pt, yt = _pairs(test, s)
        cal = calibrated_threshold(pf, yf)
        out[name] = {"p>=0.5": prf(pt >= 0.5, yt), "calibrated": prf(cal(pt), yt) if cal else None,
                     "fitted_f1_ceiling": {**prf(pt >= (t := best_threshold(pf, yf)), yt), "threshold": round(t, 5)}}
    return out


def agreement(days):
    """The teacher's micro F1 against its own second draft, over the days that have one: a ceiling for any selector."""
    both = [d for d in days if "gold_alt" in d]
    if not both: return None
    a = np.array([c["id"] in set(d["gold"]) for d in both for c in d["candidates"]], bool)
    b = np.array([c["id"] in set(d["gold_alt"]) for d in both for c in d["candidates"]], bool)
    return {"days": len(both), **prf(b, a)}


def report(days, scores, samples=SAMPLES, seed=SEED, min_days=MIN_DAYS):
    return {**precheck(days, scores, samples, seed, min_days), "selection": selection(days, scores), "teacher_self_agreement": agreement(days),
            "generator_misses": sum(d.get("generator_misses", 0) for d in days), "days": len(days)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["precheck", "report"])
    ap.add_argument("days", help="one JSON row per day, in time order")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server", help="a System One server to score every candidate on")
    src.add_argument("--scores", help="saved scores ({group, scores: {id: p}} per line) instead of a server")
    ap.add_argument("--api-key-file", help="a file holding the server's bearer key")
    ap.add_argument("--save-scores", help="write the server's scores here, so a report can be redone without it")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-days", type=int, default=MIN_DAYS)
    ap.add_argument("--out", help="the report's JSON file (default: stdout)")
    a = ap.parse_args(argv)
    days = load_days(a.days)
    if a.server:
        key = Path(a.api_key_file).read_text(encoding="utf-8").strip() if a.api_key_file else None
        scores = score_server(days, a.server, key)
        if a.save_scores:
            Path(a.save_scores).write_text("".join(json.dumps({"group": g, "scores": s}) + "\n" for g, s in scores.items()), encoding="utf-8")
    else:
        scores = load_scores(a.scores)
    out = (precheck if a.mode == "precheck" else report)(days, scores, a.samples, a.seed, a.min_days)
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out: Path(a.out).write_text(text + "\n", encoding="utf-8")
    else: print(text)
    print(f"{a.mode}: {out['verdict']} ({out['teacher']['n_days']} teacher days" + (f"; {'; '.join(out['reasons'])}" if out["reasons"] else "") + ")", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
