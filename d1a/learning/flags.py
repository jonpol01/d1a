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
"""
import argparse
import json
import sys

import numpy as np

from d1a.eval.suite import read_json

FIELDS = ("question", "options", "t", "space", "run", "budget", "fitted_on")


def group_p(probabilities, options):
    """The summed probability of the flag's options in one answer's {option: probability}."""
    return float(sum(probabilities.get(o, 0.0) for o in options))


def raised(probabilities, top, flag):
    """Whether the flag is on for one answer: its top option is in the group, or the group's probability reaches t."""
    return top in flag["options"] or group_p(probabilities, flag["options"]) >= flag["t"]


def annotate(answers, raw_answers, flags, run):
    """The served answers with each configured flag added inside its question's answer as
    answers[q]["flags"][name] = {"on", "p", "t", "space"}, read on raw_answers (the answers before the outcome calibrator)
    and only for flags fitted for `run`. Nothing else in an answer changes; with no flag that applies, `answers` is
    returned as it came."""
    for name, f in (flags or {}).items():
        raw = raw_answers.get(f["question"])
        if f["run"] != run or f["space"] != "raw" or raw is None or raw.get("type") != "choice" or f["question"] not in answers:
            continue
        p = group_p(raw["probabilities"], f["options"])
        answers[f["question"]].setdefault("flags", {})[name] = {"on": raised(raw["probabilities"], raw["choice"], f),
                                                                 "p": round(p, 4), "t": f["t"], "space": "raw"}
    return answers


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
