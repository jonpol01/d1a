"""d1a.core.api against the module at --ref: rendering, option texts, date facts, confidences, request validation (with the
full error details a 422 returns), records, answers and the JSON schema, on generated inputs.

    uv run python scripts/equivalence/api.py                     # against 34113f45, the last commit before the rewrite
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, arguments, module_at  # noqa: E402

import d1a.core.api as new  # noqa: E402

WORDS = ["a", "", " x ", "Ünïcode", "line\nbreak", "July 4, 2026", "2026-07-01", "2026-02-30", "February 30, 2026", "May 1, 2026",
         "2026-13-01", "December 31, 2025"]


def main():
    a = arguments(__doc__.split("\n")[0], "34113f45", 20000)
    old, rng, check = module_at(a.ref, "d1a/core/api.py"), random.Random(a.seed), Checker()

    def content(depth=0):
        r = rng.random()
        if depth > 3 or r < 0.35:
            return rng.choice([rng.choice(WORDS), rng.randint(-5, 5), rng.random(), True, False, None, 0, 0.0, " ".join(rng.choices(WORDS, k=4))])
        if r < 0.65:
            return [content(depth + 1) for _ in range(rng.randint(0, 3))]
        return {rng.choice(["k", "what", "x y", "", "date"]) + str(i): content(depth + 1) for i in range(rng.randint(0, 3))}

    for _ in range(a.iterations):
        c, indent = content(), rng.randint(0, 3)
        check.same("render", lambda: old.render(c, indent), lambda: new.render(c, indent))
        check.same("with_date_facts", lambda: old.with_date_facts(c), lambda: new.with_date_facts(c))
        name, desc = rng.choice(["k", "yes"]), rng.choice([None, "", c])
        check.same("option_text", lambda: old.option_text(name, desc), lambda: new.option_text(name, desc))
        text = " ".join(rng.choices(WORDS, k=rng.randint(0, 8)))
        check.same("date_facts", lambda: old.date_facts(text), lambda: new.date_facts(text))
        p = [rng.choice([0.0, rng.random(), 1.0, 0.5]) for _ in range(rng.randint(1, 8))]
        if rng.random() < 0.1: p = [0.0] * len(p)
        if rng.random() < 0.02: p[0] = float("nan")
        for f in ("choice_confidence", "score_confidence"):
            check.same(f, lambda: getattr(old, f)(p), lambda: getattr(new, f)(p))
    for _ in range(a.iterations // 4):
        questions = {}
        for i in range(rng.randint(0, 4)):
            kind = rng.choice(["noul", "choice", "score", "bogus"])
            q = {"type": kind, "instructions": content()}
            if kind == "noul" and rng.random() < 0.5: q["criteria"] = rng.choice([{"true": content(), "false": content()}, {"true": None}, {}])
            if kind == "choice": q["criteria"] = rng.choice([{f"o{j}": content() for j in range(rng.randint(0, 4))}, {f"o{j}": None for j in range(256)}])
            if kind == "score": q["criteria"] = [content() for _ in range(rng.choice([0, 1, 3, 256]))]
            questions[f"q{i}"] = q
        body = {"state": content(), "questions": questions, **({"model": "m"} if rng.random() < 0.5 else {})}

        def parse(m):
            try: return m.SystemOneRequest.model_validate(body)
            except Exception as error: return json.dumps(error.errors(include_url=False), default=str)
        old_req, new_req = parse(old), parse(new)
        if isinstance(old_req, str) or isinstance(new_req, str):
            check.equal("validation", old_req, new_req if isinstance(new_req, str) else "accepted")
            continue
        check.equal("model_dump", old_req.model_dump(), new_req.model_dump())
        old_rec, new_rec = old.to_record(old_req), new.to_record(new_req)
        check.equal("to_record", old_rec, new_rec)
        probs = []
        for info in old_rec[1]:
            p = [rng.random() for _ in info["keys"]]
            probs.append([x / sum(p) for x in p])
        check.same("to_answers", lambda: old.to_answers(probs, old_rec[1]), lambda: new.to_answers(probs, old_rec[1]))
    check.equal("json schema", old.SystemOneRequest.model_json_schema(), new.SystemOneRequest.model_json_schema())
    print(f"d1a.core.api: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
