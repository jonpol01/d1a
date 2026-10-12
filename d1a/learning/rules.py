"""Served decode rules (#252): the served choice of one choice question becomes the rule's option when the model's own
choice is one of `from` and its probability of that option reaches t. A flag (d1a.learning.flags) sits beside an answer and
never changes it; a rule changes the served choice, and only that: the probabilities, the confidence and the decision log's
answers stay the model's, the served answer names the rule (answers[q]["rule"]: name, from, p, t), and the log keeps
what was served beside the model's answers (meta.served, meta.rules).

A rule holds for one served run, the raw probabilities (before d1a.learning.feedback's outcome calibrator) and one request
use case, so a client that does not send that use case never meets it. It is fitted and tested offline on labelled rows,
never by tick, promote or replay. Severity's P2 rule (silver runs 1 and 2, #252): D1A-E4B v0.6 gives P2 less weight than
hermes-agent's written label definitions do, and raising a P3 or P4 answer to P2 when p(P2) >= 0.20 brought its severity
closer to them on held-out pull requests. A client that recomputes the top option from the probabilities disagrees with the
served choice exactly where a rule fired.
"""
import copy

from d1a.learning import flags


def compute(raw_answers, rules, run, use_case):
    """{question: {name: {"from", "to", "p", "t"}}} for the rules that fire: fitted for `run` and the request's `use_case`,
    in the raw space, on a choice question whose choice is in the rule's `from` and whose probability of the rule's option
    reaches t. Read on raw_answers, the answers before the outcome calibrator."""
    out = {}
    for name, r in (rules or {}).items():
        raw = raw_answers.get(r["question"])
        if not use_case or r["use_case"] != use_case or r["run"] != run or r["space"] != "raw" or raw is None or raw.get("type") != "choice":
            continue
        p = raw["probabilities"].get(r["option"])
        if p is not None and raw["choice"] in r["from"] and p >= r["t"]:
            out.setdefault(r["question"], {})[name] = {"from": raw["choice"], "to": r["option"], "p": p, "t": r["t"]}
    return out


def apply(answers, computed):
    """The answers with each fired rule's option as the served choice (one rule per question and use case: settings) and
    the rule beside it, answers[q]["rule"] = {"name", "from", "p", "t"}, so a client tells a rule's choice from the model's;
    nothing else in an answer changes. `answers` itself when no rule fired, else a copy."""
    if not computed:
        return answers
    out = copy.deepcopy(answers)
    for q, fired in computed.items():
        for name, f in fired.items():
            out[q]["choice"] = f["to"]
            out[q]["rule"] = {"name": name, "from": f["from"], "p": f["p"], "t": f["t"]}
    return out


class Configured(flags.Configured):
    """The rules in force for a server: d1a.learning.settings' "rules" (the file D1A_LEARNING names), re-read when the file
    changes, so a rule is served or withdrawn without a restart. No file, or no rules: none."""
    SECTION = "rules"
