"""scripts/decide.py, the one-scorecard rule (#202), on synthetic gate folders shaped like quality_gate.py's --out and a
synthetic labeler kit and dump: the BETTER paths, every INCOMPLETE check, each veto, the null and near-miss cases and the
distinct-source rule. No weights and no private text."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("decide", ROOT / "scripts/decide.py")
decide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decide)

CHOICE = {"choice": ("a", "b", "c")}
PR = {"type": ("type/bug", "type/feature", "type/security"),
      "blast": ("review:blast-contained", "review:blast-moderate", "review:blast-broad"), "sev": ("P0", "P1", "P2", "P3", "P4")}
NOISY = ("external_semif-v1", "external_typesafe-v1", "external_wanli-v1", "external_wanli-v2", "ja-jglue_test",
         "routing_factory-development", "routing_generic-development", "ja-jglue_development", "routing_handlabelled-45")


def noise(*used):
    """Balanced churn (18 fixes, 18 breaks of 60) on the non-card, non-monitor, non-PR lines `used` leaves free: a pooled CI wide
    enough that only the wins path can say BETTER."""
    return {s: (18, 18) for s in NOISY if s not in used}


@pytest.fixture(autouse=True)
def fewer_samples(monkeypatch):
    """200 bootstrap samples per suite CI instead of 2,000 (the rule is the same; the suites here are built with margin)."""
    monkeypatch.setattr(decide, "SAMPLES", 200)


# --- synthetic inputs ---------------------------------------------------------------------------------------------------

def answer(row, right):
    k = len(row["keys"]); pick = row["label"] if right else (row["label"] + 1) % k
    p = [0.2 / (k - 1)] * k; p[pick] = 0.8
    return {**row, "p": p, "logits": [math.log(x) for x in p], "inference_temperature": 1.0}


def suite_rows(suite, n):
    """n records; a PR partition's records ask type, blast and sev, every other suite's one question."""
    return [{"id": f"{suite}/{i}", "group": f"{suite}/{i}", "question": q, "source": "synthetic", "task": f"synthetic_{q}",
             "type": "choice", "variant": "clean", "keys": list(keys), "label": (3 * i + len(q)) % len(keys)}
            for i in range(n) for q, keys in (PR if suite.startswith("pr-labels") else CHOICE).items()]


def changed(rows, right, change):
    """The head's right answers: base's, with `fixes` wrong rows made right and `breaks` right rows made wrong (on one
    question when a third item names it)."""
    fixes, breaks, question = (*change, None)[:3] if change else (0, 0, None)
    out, on = list(right), [j for j, r in enumerate(rows) if question in (None, r["question"])]
    for j in [j for j in on if right[j]][:breaks]:
        out[j] = False
    for j in [j for j in on if not right[j]][:fixes]:
        out[j] = True
    return out


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
    return path


def report(**edit):
    groups = {g: {"requests": r, "questions": q, "flips": 0, "max_dp": 0.0} for g, (r, q) in {**decide.DEMOS, "labeler-replay": decide.REPLAY}.items()}
    rep = {"groups": groups, "latency": {"all text": {"ratio": 1.0, "lo": 0.99, "hi": 1.01, "n": 100}},
           "floor": {"all text": {"ratio": 1.0, "lo": 0.99, "hi": 1.01, "n": 100}}, "failures": []}
    rep.update(edit)
    return rep


def make_gate(path, changes=None, sizes=None, suites=decide.EXPECTED, rep=None):
    """A gate folder: base right on 70% of each suite's rows, the head changed by changes[suite] = (fixes, breaks[, question])."""
    for s in suites:
        rows = suite_rows(s, (sizes or {}).get(s, 60))
        right = [j % 10 < 7 for j in range(len(rows))]
        for side, ok in (("base", right), ("head", changed(rows, right, (changes or {}).get(s)))):
            write(path / "suites" / s / side / "rows.json", [answer(r, o) for r, o in zip(rows, ok)])
    if rep is not False:
        write(path / "report.json", rep or report())
    return path


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    """A replay kit (20 decisions, type and blast) and owner labels (10 PRs, type, blast and sev; one sev 'unsure')."""
    d = tmp_path_factory.mktemp("kit")
    write(d / "replay/questions.json", {q: {"type": "choice", "criteria": dict.fromkeys(k)} for q, k in PR.items()})
    write(d / "replay/memory-items.jsonl", "".join(json.dumps({"state": "synthetic", "labels": {
        "type": PR["type"][i % 3], "blast": PR["blast"][i % 3], "other": "x"}}) + "\n" for i in range(20)))
    write(d / "human.json", [{"state": "synthetic", "questions": {q: {} for q in PR},
                              "labels": {"type": PR["type"][i % 3], "blast": PR["blast"][i % 3], "sev": "unsure" if i == 0 else PR["sev"][i % 5]}}
                             for i in range(10)])
    want = decide.kit_labels(d / "replay", d / "human.json")
    assert [len(want["replay"]), len(want["human"])] == [20, 10] and "other" not in want["replay"][0] and "sev" not in want["human"][0]
    return SimpleNamespace(replay=d / "replay", human=d / "human.json", want=want)


def make_dump(kit, live=None):
    """{base, head}: answers to the kit, base right on 80% of the labelled questions, the head changed by live[part]."""
    out = {n: {"runs": "synthetic"} for n in ("base", "head")}
    for part, items in kit.want.items():
        cells = [(i, q) for i, w in enumerate(items) for q in w]
        right = [j % 5 != 4 for j in range(len(cells))]
        for n, ok in (("base", right), ("head", changed([{"question": None}] * len(cells), right, (live or {}).get(part)))):
            out[n][part] = [{} for _ in items]
            for (i, q), o in zip(cells, ok):
                out[n][part][i][q] = [items[i][q] if o else "zz-wrong", items[i][q], 0.8]
    return out


def judge(tmp_path, kit, changes=None, sizes=None, suites=decide.EXPECTED, rep=None, live=None, dump=None, **kw):
    gate = make_gate(tmp_path / "gate", changes, sizes, suites, rep)
    labeler = write(tmp_path / "dump.json", dump if dump is not None else make_dump(kit, live))
    return decide.scorecard(gate, labeler, "base", "head", replay=kit.replay, human=kit.human, **kw)


# --- BETTER ---------------------------------------------------------------------------------------------------------------

def test_better_on_the_pooled_lower_bound_even_with_live_losses(tmp_path, kit):
    """v0.4 -> v0.5's shape: card suites and a D1A suite up, both live lines down; the pooled lower bound > 0 decides."""
    card = judge(tmp_path, kit, {"v7_decision-v7": (10, 0), "hard-v1": (10, 0), "devtools-v1": (10, 0), "ja-jglue_development": (8, 0)},
                 live={"replay": (0, 6), "human": (0, 4)})
    p = card["pooled"]
    assert (p["lines"], p["live"]) == (21, 2) and p["lo"] > 0
    assert "live:replay" in card["losses"]
    assert card["verdict"].startswith("BETTER (pooled") and card["verdict"].endswith("human read of flips pending")


def test_better_on_three_wins_from_distinct_sources(tmp_path, kit):
    wins = {"external_semif-v1": (8, 0), "external_typesafe-v1": (8, 0), "external_wanli-v1": (8, 0)}
    card = judge(tmp_path, kit, {**noise(*wins), **wins})
    assert card["pooled"]["lo"] <= 0 and card["wins"] == ["external_semif-v1", "external_typesafe-v1", "external_wanli-v1"]
    assert not card["losses"] and card["verdict"].startswith("BETTER")


def test_wins_need_a_pooled_delta_of_at_least_0(tmp_path, kit):
    wins = {"external_semif-v1": (8, 0), "external_typesafe-v1": (8, 0), "external_wanli-v1": (8, 0)}
    card = judge(tmp_path, kit, {**{s: (8, 13) for s in NOISY if s not in wins}, **wins})   # six lines down, none significantly
    assert card["pooled"]["delta"] < 0 and len(card["wins"]) == 3 and not card["losses"]
    assert card["verdict"].startswith("NOT BETTER (pooled")


# --- NOT BETTER: null, near miss, distinct sources ------------------------------------------------------------------------

def test_null_head_is_not_better(tmp_path, kit):
    card = judge(tmp_path, kit)
    assert card["pooled"]["delta"] == 0 and not card["wins"] and not card["missing"]
    assert card["verdict"].startswith("NOT BETTER (pooled +0.00") and "near miss" not in card["verdict"]


def test_two_wins_and_no_loss_is_a_near_miss(tmp_path, kit):
    wins = {"external_semif-v1": (8, 0), "external_typesafe-v1": (8, 0)}
    card = judge(tmp_path, kit, {**noise(*wins), **wins})
    assert card["pooled"]["delta"] > 0 and card["pooled"]["lo"] <= 0 and len(card["wins"]) == 2
    assert card["verdict"].startswith("NOT BETTER (near miss:") and "one same-recipe rerun" in card["verdict"]


def test_a_ja_twin_wins_for_its_original_source(tmp_path, kit):
    wins = {"external_semif-v1": (8, 0), "pr-labels_test": (16, 0, "type"), "pr-labels_test-ja": (16, 0, "type")}
    card = judge(tmp_path, kit, {**noise(*wins), **wins})
    assert card["pooled"]["lo"] <= 0 and card["wins"] == ["external_semif-v1", "pr-labels_test", "pr-labels_test-ja"]
    assert card["verdict"].startswith("NOT BETTER (near miss:")


def test_sev_offsets_and_ja_twins_count_as_their_original():
    p = {"delta": 0.5, "lo": -0.5, "hi": 1.5, "lines": 21, "live": 2}
    assert decide.decide([], {}, p, ["hard-v1", "pr-labels_test", "pr-labels_test:sev+offsets"], []).startswith("NOT BETTER (near miss")
    assert decide.decide([], {}, p, ["hard-v1", "pr-labels_development", "pr-labels_development-ja"], []).startswith("NOT BETTER (near miss")
    assert decide.decide([], {}, p, ["hard-v1", "pr-labels_development", "pr-labels_test:sev+offsets"], []).startswith("BETTER")


def test_a_monitor_win_is_not_a_line(tmp_path, kit):
    wins = {"external_semif-v1": (8, 0), "external_typesafe-v1": (8, 0), "night2_dates": (8, 0)}
    card = judge(tmp_path, kit, {**noise(*wins), **wins})
    assert card["suites"]["night2_dates"]["lo"] > 0 and card["wins"] == ["external_semif-v1", "external_typesafe-v1"]
    assert card["verdict"].startswith("NOT BETTER (near miss:")


def test_a_loss_blocks_the_wins_path(tmp_path, kit):
    moved = {"external_semif-v1": (8, 0), "external_typesafe-v1": (8, 0), "external_wanli-v1": (8, 0), "external_wanli-v2": (0, 8)}
    card = judge(tmp_path, kit, {**noise(*moved), **moved})
    assert card["pooled"]["lo"] <= 0 and card["losses"] == ["external_wanli-v2"] and len(card["wins"]) == 3
    assert card["verdict"].startswith("NOT BETTER (pooled") and "significant loss: external_wanli-v2" in card["follow_ups"]


# --- vetoes -----------------------------------------------------------------------------------------------------------------

def test_v1_card_suite_delta_floor(tmp_path, kit):
    card = judge(tmp_path, kit, {"hard-v1": (0, 25)}, sizes={"hard-v1": 1000})
    s = card["suites"]["hard-v1"]
    assert s["delta"] < -2 and s["lo"] >= -4          # only the delta clause fails
    assert card["verdict"] == "VETO: V1"


def test_v1_card_suite_ci_floor(tmp_path, kit):
    card = judge(tmp_path, kit, {"devtools-v1": (20, 21)}, sizes={"devtools-v1": 100})
    s = card["suites"]["devtools-v1"]
    assert s["delta"] >= -2 and s["lo"] < -4          # only the CI clause fails
    assert card["verdict"] == "VETO: V1"


def test_v2_lists_the_flipped_safety_demos_and_never_blocks(tmp_path, kit):
    rep = report()
    rep["groups"]["demo:guardrails"]["flips"] = 2; rep["groups"]["demo:routing"]["flips"] = 3
    card = judge(tmp_path, kit, {"hard-v1": (10, 0), "devtools-v1": (10, 0)}, rep=rep)
    assert card["vetoes"]["V2 human read of flips (demo:guardrails)"] == "NEEDS HUMAN READ"
    assert card["verdict"].startswith("BETTER")


@pytest.mark.parametrize("ratio, verdict", [(1.02, "VETO: V3"), (1.01, "NOT BETTER")])
def test_v3_latency_against_the_floor(tmp_path, kit, ratio, verdict):
    card = judge(tmp_path, kit, rep=report(latency={"all text": {"ratio": ratio, "lo": ratio, "hi": ratio, "n": 100}}))
    assert card["verdict"].startswith(verdict)


@pytest.mark.parametrize("suite", ["ja-jglue_development", "night2_dates"])
def test_v4_a_large_suite_five_points_down(tmp_path, kit, suite):
    card = judge(tmp_path, kit, {suite: (0, 12)}, sizes={suite: 200})
    assert card["suites"][suite]["delta"] == pytest.approx(-6) and card["suites"][suite]["hi"] < 0
    assert card["verdict"] == "VETO: V4"


@pytest.mark.parametrize("change, size", [((0, 6), 60), ((30, 38), 150)], ids=["small suite", "CI reaches 0"])
def test_v4_needs_150_questions_and_a_ci_below_0(tmp_path, kit, change, size):
    card = judge(tmp_path, kit, {"ja-jglue_development": change}, sizes={"ja-jglue_development": size})
    assert card["suites"]["ja-jglue_development"]["delta"] <= -5
    assert card["verdict"].startswith("NOT BETTER")


# --- INCOMPLETE ---------------------------------------------------------------------------------------------------------------

def incomplete(card, why):
    assert card["verdict"].startswith("INCOMPLETE") and why in card["verdict"], card["verdict"]


def test_incomplete_when_a_suite_is_missing(tmp_path, kit):
    incomplete(judge(tmp_path, kit, suites=[s for s in decide.EXPECTED if s != "external_wanli-v2"]), "external_wanli-v2: not in the gate")


def test_incomplete_comes_before_a_veto(tmp_path, kit):
    card = judge(tmp_path, kit, {"hard-v1": (0, 25)}, sizes={"hard-v1": 1000}, suites=decide.EXPECTED[:-1])
    assert any(k.startswith("V1") and v is False for k, v in card["vetoes"].items())
    incomplete(card, f"{decide.EXPECTED[-1]}: not in the gate")


@pytest.mark.parametrize("body, why", [
    (lambda rows: json.dumps(rows)[:5000], "hard-v1: cannot be scored (JSONDecodeError"),
    (lambda rows: [{**rows[0], "label": (rows[0]["label"] + 1) % 3}] + rows[1:], "hard-v1: cannot be scored (ValueError: paired comparison labels"),
    (lambda rows: [{**rows[0], "p": [float("nan")] * 3}] + rows[1:], "hard-v1: cannot be scored (ValueError: head: 1 rows with probabilities"),
    (lambda rows: rows[:-1], "hard-v1: base 60 / head 59 questions do not pair"),
], ids=["truncated", "label differs", "NaN probabilities", "unpaired"])
def test_incomplete_when_a_suite_cannot_be_scored(tmp_path, kit, body, why):
    gate = make_gate(tmp_path / "gate")
    rows = json.loads((gate / "suites/hard-v1/head/rows.json").read_text(encoding="utf-8"))
    write(gate / "suites/hard-v1/head/rows.json", body(rows))
    labeler = write(tmp_path / "dump.json", make_dump(kit))
    incomplete(decide.scorecard(gate, labeler, "base", "head", replay=kit.replay, human=kit.human), why)


def test_incomplete_without_head_rows(tmp_path, kit):
    gate = make_gate(tmp_path / "gate")
    (gate / "suites/hard-v1/head/rows.json").unlink()
    labeler = write(tmp_path / "dump.json", make_dump(kit))
    incomplete(decide.scorecard(gate, labeler, "base", "head", replay=kit.replay, human=kit.human), "hard-v1: no rows on both sides")


def short(group, **counts):
    rep = report()
    if counts:
        rep["groups"][group].update(counts)
    else:
        del rep["groups"][group]
    return rep


@pytest.mark.parametrize("rep, why", [
    (False, "report.json"),
    (short("demo:video"), "demo:video 0/4 requests"),
    (short("demo:video", requests=0, questions=0), "demo:video 0/4 requests"),
    (short("demo:routing", requests=float("nan")), "demo:routing nan/6 requests"),
    (short("demo:routing", requests=6.0), "demo:routing 6.0/6 requests"),
    (short("labeler-replay", questions=273), "labeler-replay 92/92 requests, 273/276 questions"),
    (report(failures=["demo:gate | allow: a deploy / decision: different options"]), "1 quality_gate failures (demo:gate 1)"),
    (report(failures=["labeler-replay | state 7: answered by one side only"]), "1 quality_gate failures (labeler-replay 1)"),
    (report(latency={"all text": None}), "latency: no finite positive"),
    (report(floor={"all text": {"ratio": float("inf")}}), "latency: no finite positive"),
], ids=["no report", "demo group absent", "demo group zero", "NaN count", "float count", "replay short", "different options",
        "unanswered", "no latency", "infinite floor"])
def test_incomplete_when_the_demos_replay_or_latency_are_short(tmp_path, kit, rep, why):
    incomplete(judge(tmp_path, kit, rep=rep), why)


def side(name, edit):
    def go(dump):
        for part in ("replay", "human"):
            dump[name][part] = edit(part, dump[name][part])
        return dump
    return go


@pytest.mark.parametrize("edit, why", [
    (lambda d: {"base": d["base"]}, "labeler dump: no 'head' side"),
    (lambda d: {"head": d["head"]}, "labeler dump: no 'base' side"),
    (side("head", lambda part, items: items[:5]), "labeler dump: 'head' replay has 5 items, the kit 20"),
    (side("head", lambda part, items: [{q: a for q, a in it.items() if q != "blast"} for it in items]), "labeler dump: 'head' replay: 20 of 20 items"),
    (side("head", lambda part, items: [{q: [None, *a[1:]] for q, a in it.items()} for it in items]), "labeler dump: 'head' replay: 20 of 20 items"),
    (side("base", lambda part, items: items[::-1]), "labeler dump: 'base' replay:"),
], ids=["no head side", "no base side", "items short", "question missing", "null choice", "items reordered"])
def test_incomplete_when_the_labeler_dump_is_partial(tmp_path, kit, edit, why):
    incomplete(judge(tmp_path, kit, dump=edit(make_dump(kit))), why)


def test_incomplete_without_a_dump_or_the_owner_labels(tmp_path, kit):
    gate = make_gate(tmp_path / "gate")
    incomplete(decide.scorecard(gate, None, "base", "head", replay=kit.replay, human=kit.human), "labeler dump")
    incomplete(decide.scorecard(gate, tmp_path / "absent.json", "base", "head", replay=kit.replay, human=kit.human), "labeler dump")
    labeler = write(tmp_path / "dump.json", make_dump(kit))
    incomplete(decide.scorecard(gate, labeler, "base", "head", replay=kit.replay, human=None), "no --human")
    incomplete(decide.scorecard(gate, labeler, "base", "head", replay=tmp_path, human=kit.human), "labeler dump or kit unreadable (FileNotFoundError")


# --- what never changes the verdict ----------------------------------------------------------------------------------------

def test_report_only_and_unexpected_suites_never_change_the_verdict(tmp_path, kit):
    base = judge(tmp_path / "a", kit, {"hard-v1": (10, 0)})["verdict"]
    extra = make_gate(tmp_path / "extra", suites=["v9_transfer-v9"], rep=False)
    write(extra / "suites/v9_transfer-v9/head/rows.json", "[")
    card = judge(tmp_path / "b", kit, {"hard-v1": (10, 0)}, extra_gate=extra)
    assert card["verdict"] == base and any("report-only:v9_transfer-v9: cannot be scored" in m for m in card["report_only_missing"])
    gate = make_gate(tmp_path / "c", {"hard-v1": (10, 0), "v9_transfer-v9": (20, 0)}, suites=[*decide.EXPECTED, "v9_transfer-v9"])
    (gate / "suites/.DS_Store").write_bytes(b"\0")
    card = decide.scorecard(gate, write(tmp_path / "dump.json", make_dump(kit)), "base", "head", replay=kit.replay, human=kit.human)
    assert card["verdict"] == base and card["suites"]["unexpected:v9_transfer-v9"]["report_only"] and "v9_transfer-v9" not in card["wins"]


def test_expect_suites_is_a_dry_run(tmp_path, kit):
    suites = [*decide.CARD, "ja-jglue_development"]
    card = judge(tmp_path, kit, suites=suites, expect_suites="ja-jglue:development")
    assert not card["missing"] and card["pooled"]["lines"] == 8
    assert card["verdict"].endswith(f"[DRY RUN: --expect-suites, 6 of the {len(decide.EXPECTED)} --all-suites partitions expected; not a deploy verdict]")


def test_the_cli_prints_the_verdict_and_writes_only_with_out(tmp_path, kit, capsys):
    gate = make_gate(tmp_path / "gate")
    labeler = write(tmp_path / "dump.json", make_dump(kit))
    args = ["--gate", str(gate), "--labeler", str(labeler), "--base-name", "base", "--head-name", "head", "--replay", str(kit.replay), "--human", str(kit.human)]
    decide.main(args)
    assert "VERDICT: NOT BETTER" in capsys.readouterr().out and sorted(p.name for p in gate.iterdir()) == ["report.json", "suites"]
    decide.main([*args, "--out", str(tmp_path / "card.json")])
    assert json.loads((tmp_path / "card.json").read_text(encoding="utf-8"))["verdict"].startswith("NOT BETTER")
