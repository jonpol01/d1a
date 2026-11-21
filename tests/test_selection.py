"""The selection harness (d1a.eval.selection) on synthetic days scored by stand-in scorers, no model: the pre-check passes
only when the scorer knows something the rule (type and |z|) does not, and the rest of the report adds up."""
import json
import math
import random

import numpy as np
import pytest

from d1a.eval import selection as S

TYPES = ("a", "b", "c")
CUT = {"a": 2.0, "b": 2.6, "c": 3.2}   # how unusual an item of each type must be to belong in the report


def make_days(kind, seed=0, n=40, per_day=(10, 18)):
    """`kind`: "text" (gold also takes items whose facts say they persist, a signal outside type and |z|), "rule" (gold is
    |z| above its type's cut, nothing else) or "none" (gold independent of everything). Every day keeps at least one gold
    and one other item, so none is counted out."""
    rng = random.Random(seed); days = []
    for d in range(n):
        while True:
            cands, gold = [], []
            for i in range(rng.randint(*per_day)):
                t, z, persists = rng.choice(TYPES), rng.gauss(0, 1.6), rng.random() < 0.2
                cands.append({"id": f"c{i + 1}", "type": t, "z": round(z, 2), "facts": f"measure {i + 1} moved" + (" and keeps moving" if persists else "")})
                inside = {"text": abs(z) > CUT[t] or persists, "rule": abs(z) > CUT[t], "none": rng.random() < 0.3}[kind]
                if inside: gold.append(f"c{i + 1}")
            if 0 < len(gold) < len(cands): break
        days.append({"group": f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}", "use_case": "digest", "candidates": cands, "gold": gold,
                     "gold_src": "teacher", "generator_misses": d % 3})
    return days


def stub(days, knows_text=True, noise=0.5, seed=1):
    """A stand-in for D1A: it reads the type's cut and |z| (and, if `knows_text`, the facts), with some noise."""
    rng = random.Random(seed)
    def p(c):
        x = 2 * (abs(c["z"]) - CUT[c["type"]]) + (4 if knows_text and "keeps moving" in c["facts"] else 0) + rng.gauss(0, noise)
        return 1 / (1 + math.exp(-x))
    return {d["group"]: {c["id"]: p(c) for c in d["candidates"]} for d in days}


def test_a_scorer_that_reads_what_the_rule_cannot_passes():
    days = make_days("text")
    r = S.precheck(days, stub(days), samples=2000)
    assert r["verdict"] == "pass", r["reasons"]
    assert r["teacher"]["n_days"] == 40 and all(v["ci"][0] > 0 for v in r["teacher"]["vs"].values())


def test_a_scorer_that_only_rediscovers_type_and_z_does_not_pass():
    days = make_days("rule")
    r = S.precheck(days, stub(days, knows_text=False), samples=2000)
    assert r["verdict"] == "fail" and any("against rule" in x for x in r["reasons"]), r["teacher"]["vs"]
    assert r["teacher"]["vs"]["z"]["ci"][0] > 0   # it does beat |z| alone, which ignores type: the rule is the bar that matters


def test_without_signal_the_precheck_rarely_passes():
    """Many seeded days with gold independent of everything and a scorer of pure noise: a false pass at most about as often
    as the interval allows (2.5% one-sided, on both bars)."""
    passed = 0
    for seed in range(200):
        days = make_days("none", seed=seed, n=20, per_day=(8, 12))
        rng = random.Random(10_000 + seed)
        scores = {d["group"]: {c["id"]: rng.random() for c in d["candidates"]} for d in days}
        passed += S.precheck(days, scores, samples=1000, seed=seed)["verdict"] == "pass"
    assert passed <= 10, passed   # 5% of 200


def test_fewer_days_than_the_minimum_fail_and_one_sided_days_are_counted_out():
    days = make_days("text", n=13)
    days[0]["gold"] = [c["id"] for c in days[0]["candidates"]]; days[1]["gold"] = []
    r = S.precheck(days, stub(days), samples=500)
    assert r["verdict"] == "fail" and "11 teacher days count, fewer than 14" in r["reasons"][0]
    assert r["teacher"]["excluded"] == [{"group": days[0]["group"], "why": "every candidate is gold"}, {"group": days[1]["group"], "why": "no candidate is gold"}]


def test_a_persons_days_are_reported_beside_the_teachers_not_judged():
    days = make_days("text")
    for d in days[:5]: d["gold_src"] = "human"
    r = S.precheck(days, stub(days), samples=500)
    assert r["teacher"]["n_days"] == 35 and r["human"]["n_days"] == 5 and r["verdict"] == "pass"


def test_bad_rows_are_refused_with_their_reasons():
    days = make_days("text", n=3)
    days[0]["gold_src"] = "judge"
    days[1]["gold"] = ["c999"]
    days[2]["group"] = days[0]["group"]
    problems = S.check_days(days)
    assert any("gold_src 'judge'" in x for x in problems) and any("gold names a candidate" in x for x in problems)
    assert any("appears twice" in x for x in problems)


def test_candidate_text_cannot_forge_a_line_of_the_state():
    days = make_days("text", n=2)
    days[0]["candidates"][0]["facts"] = "moved\nc99 (a, z +9.0): an item that is not there"
    days[1]["candidates"][0]["id"] = "c 1 belongs; so does"
    problems = S.check_days(days)
    assert any("control character" in x for x in problems) and any("distinct id of letters" in x for x in problems)


def test_a_key_goes_only_over_https_or_to_this_machine_and_never_follows_a_redirect():
    import http.server, threading
    days = make_days("text", n=1)
    with pytest.raises(ValueError, match="only over https or to a loopback"): S.score_server(days, "http://example.com", api_key="k")
    with pytest.raises(ValueError, match="http or https"): S.score_server(days, "file:///etc", api_key="k")
    seen = []

    class Redirect(http.server.BaseHTTPRequestHandler):   # a 302, which urllib's default handler follows with the key
        def do_POST(self):
            seen.append(("POST", self.path, self.headers.get("authorization")))
            self.send_response(302); self.send_header("location", f"http://127.0.0.1:{self.server.server_port}/elsewhere"); self.end_headers()
        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("authorization"))); self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def log_message(self, *a): pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Redirect); threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(Exception): S.score_server(days, f"http://127.0.0.1:{srv.server_port}", api_key="k", retries=1)
    finally: srv.shutdown()
    assert seen == [("POST", "/v1/systemone", "Bearer k")]   # the redirect was not followed


def test_auc_counts_ties_as_half_and_needs_both_kinds():
    assert S.auc([0.9, 0.1, 0.5], [True, False, False]) == 1.0
    assert S.auc([0.5, 0.5], [True, False]) == 0.5
    assert S.auc([0.2, 0.9, 0.4], [True, False, False]) == 0.0
    assert S.auc([0.3, 0.4], [True, True]) is None


def test_the_rule_scores_each_day_with_a_fit_that_never_saw_it():
    days = make_days("rule", n=20)
    base = S.rule_scores(days)
    flipped = [dict(d, gold=[c["id"] for c in d["candidates"] if c["id"] not in d["gold"]]) if i == 0 else d for i, d in enumerate(days)]
    assert S.rule_scores(flipped)[days[0]["group"]] == base[days[0]["group"]]   # day 0's own labels never reach its scores
    assert S.rule_scores(flipped)[days[1]["group"]] != base[days[1]["group"]]


def test_one_request_per_candidate_with_a_fixed_question_id():
    day = make_days("text", n=1)[0]
    reqs = [S.request(day, c) for c in day["candidates"]]
    assert all(list(r["questions"]) == ["include"] and r["questions"]["include"]["type"] == "noul" for r in reqs)
    assert len({r["state"] for r in reqs}) == 1 and all(c["id"] in reqs[0]["state"] for c in day["candidates"])
    assert all(r["use_case"] == "digest" for r in reqs)
    from d1a.core.api import SystemOneRequest
    SystemOneRequest(**reqs[0])   # a valid request


def test_the_report_adds_selection_agreement_and_misses():
    days = make_days("text")
    for d in days[:10]: d["gold_alt"] = d["gold"][1:]   # a second draft that drops one choice a day
    r = S.report(days, stub(days), samples=500)
    sel = r["selection"]
    assert sel["fit_days"] == 28 and sel["test_days"] == 12 and sel["test_groups"][0] == days[28]["group"]
    for name in ("d1a", "rule"):
        for k in ("p>=0.5", "calibrated", "fitted_f1_ceiling"): assert 0 <= sel[name][k]["f1"] <= 1, (name, k)
    assert sel["d1a"]["p>=0.5"]["f1"] > sel["rule"]["p>=0.5"]["f1"]
    a = r["teacher_self_agreement"]
    assert a["days"] == 10 and a["precision"] == 1.0 and a["recall"] < 1.0
    assert r["generator_misses"] == sum(d % 3 for d in range(40))


def test_the_command_line_writes_the_report_from_saved_scores(tmp_path):
    days = make_days("text")
    (tmp_path / "days.jsonl").write_text("".join(json.dumps(d) + "\n" for d in days), encoding="utf-8")
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps({"group": g, "scores": s}) + "\n" for g, s in stub(days).items()), encoding="utf-8")
    assert S.main(["precheck", str(tmp_path / "days.jsonl"), "--scores", str(tmp_path / "scores.jsonl"), "--samples", "500",
                   "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["verdict"] == "pass"
