"""Learned decision flags (#237): a named group of a choice question's options and a threshold, raised when the answer's top
option is in the group or the group's summed probability reaches the threshold. A flag is served beside the answer and
never changes it: severity's P0/P1, which the top answer often misses even when the model gives them weight.

    python -m d1a.learning.flags fit rows.json --name sev_severe --question sev --options P0,P1 --budget 0.15 \\
        --run JohnP1/d1a-e4b-mlx-q8@v0.6 --kit hermes-agent-2026-10-08 --created-max 2026-10-08T17:40:54Z
    python -m d1a.learning.flags gate rows.json --flag '<the JSON fit printed>'

rows.json is d1a.eval.benchmark's per-question rows (question, keys, label, p, group). A flag reads the model's own
probabilities (space "raw": the checkpoint's temperature, before d1a.learning.feedback's outcome calibrator) and holds
for the one served run it was fitted for. It is fitted and gated only on labelled kits offline, never on the live log:
live severity outcomes are human corrections only, too few and biased (#237). The objective is recall at a flag-rate
budget: the threshold is the (1 - budget) quantile of the group's probability on the fit kit, since the flag rate is
what reviewers feel and needs no labels, and it holds when the base rate drifts.

Two optional settings (#296):
- mode "shadow": the flag is computed and kept in the decision log's meta, but not served, so the response is the one it
  would be without the flag. A blind check then collects labels the flag never influenced.
- track {"window", "every", "clamp", "alert_rate"}: t follows the flag rate without labels. Every `every` new logged
  decisions, t is refit as the (1 - budget) quantile of the group's probability over the last `window` decisions of the
  flag's run (scores only, one per distinct state, no replays), clamped to `clamp`. Below `window` decisions, the
  offline t stays. d1a.learning.loop's tick runs it, and reports when the last 250 decisions' flag rate exceeds
  `alert_rate` or t sits at a clamp. A threshold fitted on older scores fires more often when the scores drift, and a
  fixed t did (#237: rate 0.194 against a 0.15 budget).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np

from d1a.eval.suite import read_json

FIELDS = ("question", "options", "t", "space", "run", "budget", "fitted_on")
ALERT_WINDOW = 250   # decisions the tick's flag-rate alert looks back over


def group_p(probabilities, options):
    """The summed probability of the flag's options in one answer's {option: probability}."""
    return float(sum(probabilities.get(o, 0.0) for o in options))


def raised(probabilities, top, flag):
    """Whether the flag is on for one answer: its top option is in the group, or the group's probability reaches t."""
    return top in flag["options"] or group_p(probabilities, flag["options"]) >= flag["t"]


def compute(raw_answers, flags, run):
    """{question: {name: {"on", "p", "t", "space"}}} for the flags in `flags` that apply: fitted for `run`, in the raw
    space, on a choice question these answers hold. Read on raw_answers, the answers before the outcome calibrator."""
    out = {}
    for name, f in (flags or {}).items():
        raw = raw_answers.get(f["question"])
        if f["run"] != run or f["space"] != "raw" or raw is None or raw.get("type") != "choice":
            continue
        p = group_p(raw["probabilities"], f["options"])
        out.setdefault(f["question"], {})[name] = {"on": raised(raw["probabilities"], raw["choice"], f), "p": round(p, 4),
                                                   "t": f["t"], "space": "raw", **({"shadow": True} if f.get("mode") == "shadow" else {})}
    return out


def attach(answers, computed):
    """The answers with each question's served flags added as answers[q]["flags"]; nothing else in an answer changes.
    A shadow flag is left out: it is only logged."""
    for q, fl in computed.items():
        shown = {name: v for name, v in fl.items() if not v.get("shadow")}
        if q in answers and shown: answers[q].setdefault("flags", {}).update(shown)
    return answers


def annotate(answers, raw_answers, flags, run):
    """attach(answers, compute(raw_answers, flags, run)): with no flag that applies, `answers` is returned as it came."""
    return attach(answers, compute(raw_answers, flags, run))


class Configured:
    """The flags in force for a server: d1a.learning.settings' "flags" (the file D1A_LEARNING names), re-read when the
    file changes, so a fitted flag is served or withdrawn without a restart. No file, or no flags: none."""
    SECTION = "flags"

    def __init__(self, path=None):
        from d1a.learning import settings
        self._settings, self.path = settings, (path or settings.path_from_env())
        self._mtime, self._flags = None, {}

    def current(self):
        if self.path is None: return {}
        try: mtime = self.path.stat().st_mtime
        except OSError: return {}
        if mtime != self._mtime:
            self._flags, self._mtime = self._settings.load(self.path)[0].get(self.SECTION) or {}, mtime
        return self._flags


def logged_scores(events, flag):
    """[(ts, group probability, raised)] for the flag's question in the log's decisions by the flag's run, oldest first:
    no replays (they re-score old decisions), one per distinct state (its latest), choice answers only."""
    latest = {}
    for e in events:
        if e.get("kind") != "decision" or e.get("replay_of") or e.get("run") != flag["run"]: continue
        a = (e.get("answers") or {}).get(flag["question"])
        if not a or a.get("type") != "choice" or not a.get("probabilities"): continue
        key = hashlib.sha256(json.dumps(e.get("state"), sort_keys=True).encode()).hexdigest()
        latest[key] = (e["ts"], group_p(a["probabilities"], flag["options"]), a.get("choice") in flag["options"])
    return sorted(latest.values())


def track(events, flag, refit_at=0):
    """The tracked threshold of a flag with a `track` setting. -> {"n", "t" (the refit flag, or None: too few decisions or
    not due), "rate_recent", "alerts"}. Due when there are at least `window` decisions and `every` more than at the last
    refit (`refit_at`, a count)."""
    tr = flag["track"]; scores = logged_scores(events, flag); n = len(scores)
    out = {"n": n, "t": None, "alerts": []}
    if n >= tr["window"] and n - refit_at >= tr["every"]:
        window = scores[-tr["window"]:]
        t = round(float(np.clip(np.quantile([p for _, p, _ in window], 1 - flag["budget"]), *tr["clamp"])), 4)
        created = datetime.fromtimestamp(window[-1][0], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out["t"] = {**flag, "t": t, "fitted_on": {"kit": "decision-log", "created_max": created, "n": tr["window"]}}
    t_now = out["t"]["t"] if out["t"] else flag["t"]
    recent = scores[-ALERT_WINDOW:]
    if recent:
        out["rate_recent"] = sum(top or p >= t_now for _, p, top in recent) / len(recent)
        if len(recent) == ALERT_WINDOW and out["rate_recent"] > tr["alert_rate"]:
            out["alerts"].append(f"flag rate {out['rate_recent']:.3f} over the last {ALERT_WINDOW} decisions, above {tr['alert_rate']}")
    if any(abs(t_now - c) < 1e-9 for c in tr["clamp"]): out["alerts"].append(f"t {t_now} is at a clamp {tr['clamp']}")
    return out


def _rows(rows, question):
    out = []
    for r in rows:
        if r.get("question") != question or r.get("variant", "clean") != "clean":
            continue
        probs = dict(zip(r["keys"], r["p"]))
        out.append({"probs": probs, "top": r["keys"][int(np.argmax(r["p"]))], "gold": r["keys"][r["label"]],
                    "group": r.get("group") or r.get("id")})
    if not out:
        raise ValueError(f"no rows for question {question!r}")
    return out


def measure(rows, flag):
    """Recall, precision and flag rate of the flag and of the top answer alone on labelled rows (gold in the group =
    a positive)."""
    rs = _rows(rows, flag["question"]); opts = set(flag["options"])
    pos = [r["gold"] in opts for r in rs]
    def stats(on):
        tp = sum(o and p for o, p in zip(on, pos)); n_on = sum(on)
        return {"recall": tp / max(sum(pos), 1), "precision": tp / n_on if n_on else None, "rate": n_on / len(rs), "caught": tp}
    flag_on = [raised(r["probs"], r["top"], flag) for r in rs]
    top_on = [r["top"] in opts for r in rs]
    return {"n": len(rs), "positives": sum(pos), "flag": stats(flag_on), "top": stats(top_on)}


def fit(rows, name, question, options, budget, run, kit, created_max):
    """The flag whose threshold is the (1 - budget) quantile of the group's probability on these rows."""
    if not 0.01 <= budget <= 0.5:
        raise ValueError("budget must be in [0.01, 0.5]")
    rs = _rows(rows, question)
    t = float(np.quantile([group_p(r["probs"], options) for r in rs], 1 - budget))
    return {name: {"question": question, "options": list(options), "t": round(t, 4), "space": "raw", "run": run,
                   "budget": budget, "fitted_on": {"kit": kit, "created_max": created_max, "n": len(rs)}}}


def gate(rows, flag, slack=0.03, n=2000, seed=0):
    """Serve the flag only if, on held-out labelled rows, it catches more positives than the top answer alone, with a
    95% bootstrap interval over groups (e.g. pull requests) entirely above zero, and it raises no more than budget + slack
    of the rows. -> {"serve": bool, "recall_delta", "ci", "rate", "reasons", ...}"""
    rs = _rows(rows, flag["question"]); opts = set(flag["options"])
    positives = [r for r in rs if r["gold"] in opts]
    d = np.array([float(raised(r["probs"], r["top"], flag)) - float(r["top"] in opts) for r in positives])
    groups = np.array([str(r["group"]) for r in positives])
    reasons, lo, hi = [], float("nan"), float("nan")
    if len(d):
        keys = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in keys}; rng = np.random.default_rng(seed)
        boots = np.sort([d[np.concatenate([idx[g] for g in rng.choice(keys, len(keys))])].mean() for _ in range(n)])
        lo, hi = float(boots[int(0.025 * n)]), float(boots[int(0.975 * n) - 1])
    if not len(d) or not lo > 0:
        reasons.append(f"recall not clearly higher than the top answer (delta {d.mean() if len(d) else float('nan'):+.3f}, "
                       f"95% CI [{lo:+.3f}, {hi:+.3f}], {len(d)} positives)")
    rate = sum(raised(r["probs"], r["top"], flag) for r in rs) / len(rs)
    if rate > flag["budget"] + slack:
        reasons.append(f"flag rate {rate:.3f} above the budget {flag['budget']:.2f} + {slack:.2f}")
    return {"serve": not reasons, "recall_delta": float(d.mean()) if len(d) else None, "ci": [lo, hi], "rate": rate,
            "positives": int(len(d)), "reasons": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit a flag's threshold on a labelled kit; prints the flag for learning.json's flags")
    f.add_argument("rows"); f.add_argument("--name", required=True); f.add_argument("--question", required=True)
    f.add_argument("--options", required=True, help="comma-separated, e.g. P0,P1"); f.add_argument("--budget", type=float, default=0.15)
    f.add_argument("--run", required=True, help="the served run the rows were scored with (repo@tag)")
    f.add_argument("--kit", required=True); f.add_argument("--created-max", required=True, help="the newest item's creation time in the kit (ISO)")
    g = sub.add_parser("gate", help="check a fitted flag on a later, held-out labelled kit")
    g.add_argument("rows"); g.add_argument("--flag", required=True, help="the {name: flag} JSON that fit printed")
    a = ap.parse_args(argv)
    rows = read_json(a.rows)
    if a.cmd == "fit":
        flags = fit(rows, a.name, a.question, a.options.split(","), a.budget, a.run, a.kit, a.created_max)
        print(json.dumps(flags)); print(json.dumps(measure(rows, flags[a.name])), file=sys.stderr)
    else:
        (name, flag), = json.loads(a.flag).items()
        print(json.dumps({"flag": name, **measure(rows, flag), "gate": gate(rows, flag)}))


if __name__ == "__main__":
    main()
