"""recipes/mix/build_mix.py and verify_mix.py (#202): the plan-driven mix builder draws the same bytes for the same plan and
seed (a pinned sha256), stratifies and weights, fills label quotas, keeps only unseen records when asked, screens exact and
near-duplicate eval states (the nearest one by shingles, a PR's title and body), eval and hand-labelled PR ids, the Japanese
twins of eval-like PRs and repeated states (every copy when their labels disagree), refuses a plan that leaves out a
training source, and writes a sidecar d1a.training.train accepts; the verifier catches a changed record. Synthetic records
stand in for the suites (no private rows)."""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recipes/mix"))
_spec = importlib.util.spec_from_file_location("build_mix", ROOT / "recipes/mix/build_mix.py")
bm = importlib.util.module_from_spec(_spec)
sys.modules["build_mix"] = bm
_spec.loader.exec_module(bm)
_spec = importlib.util.spec_from_file_location("verify_mix", ROOT / "recipes/mix/verify_mix.py")
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

VOCAB = [f"w{i}" for i in range(5000)]
SEVS = ("P0", "P1", "P2", "P3")


def words(rng, n=60):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def record(src, i, rng, **extra):
    """A synthetic record of `src`: a unique state, the questions and _meta its source's records carry."""
    fam = src.split(":")[0]
    q = {"label": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}, "label": "a"}}
    meta = {"source": "agnews", "id": i} if fam == "decision-v7" else {"source": f"s{i % 3}", "family": f"f{i % 2}", "id": i}
    r = {"state": f"{src} title {i}\nbody: {words(rng)}", "questions": q, "_meta": meta}
    if fam == "pr-labels":
        r["id"] = f"repo#{src.replace(':', '-')}-{i}"
        r["questions"] = {"sev": {"type": "choice", "instructions": "x", "criteria": dict.fromkeys(SEVS), "label": SEVS[i % 4]}}
    r.update(extra)
    return r


def suites(n=40, seed=0):
    """{plan name: records} for every required source, and {eval partition: records} for a few eval partitions."""
    rng = random.Random(seed)
    train = {name: [record(name, i, rng) for i in range(n)] for name in bm.required()}
    evals = {"pr-labels:test": [record("pr-labels:test", i, rng) for i in range(5)],
             "hard-v1:development": [record("hard-v1:development", i, rng) for i in range(5)]}
    return train, evals


def plan(n=5, **over):
    p = {"seed": 0, "dv7_replay": 0, "sources": [{"source": s, "n": n} for s in bm.required()]}
    p.update(over)
    return p


def spec(p, source):
    return next(s for s in p["sources"] if s["source"] == source)


def test_the_same_plan_and_seed_give_the_same_bytes():
    train, evals = suites()
    a, ia, ra = bm.build(plan(), train, evals)
    b, ib, rb = bm.build(plan(), train, evals)
    assert "".join(map(bm.line, a)) == "".join(map(bm.line, b)) and ia == ib and ra["sha256"] == rb["sha256"]
    assert ra["sha256"] != bm.build(plan(seed=1), train, evals)[2]["sha256"]
    assert len(a) == 5 * len(bm.required()) and {(x["source"], x["row"]) for x in ia} == {(x["source"], x["row"]) for x in ib}


def test_stratify_follows_the_weights_and_natural_shares():
    train, evals = suites(n=80)
    p = plan()
    spec(p, "hard-v1:train").update(n=30, stratify="_meta.family", weights={"f0": 2, "f1": 1})
    spec(p, "pr-labels:train").update(n=20, stratify="q:sev", weights="natural")
    train["pr-labels:train"] = [dict(r, questions={"sev": dict(r["questions"]["sev"], label="P2" if i % 4 else "P0")})
                                for i, r in enumerate(train["pr-labels:train"])]   # P0 a quarter of the pool, P2 the rest
    _, _, report = bm.build(p, train, evals)
    assert report["sources"]["hard-v1:train"]["fill"] == {"f0": 20, "f1": 10}
    assert report["sources"]["pr-labels:train"]["fill"] == {"P0": 5, "P2": 15}


def test_only_unseen_keeps_records_no_earlier_run_trained_on():
    train, evals = suites()
    p = plan()
    spec(p, "documents-v1:train").update(n=20, only_unseen=True)   # more than the 13 unseen: fresh-first alone would add seen ones
    exposure = {(s, i): {"state_sha256": bm.state_hash(r), "seen": {"v0.5 (skills stage)": 1} if s == "documents-v1:train" and i % 3 else {},
                         "phase_c": 1 if s == "documents-v1:train" and i == 3 else 0} for s, rows in train.items() for i, r in enumerate(rows)}
    _, index, report = bm.build(p, train, evals, exposure)
    rows = [x["row"] for x in index if x["source"] == "documents-v1:train"]
    assert sorted(rows) == [i for i in range(0, 40, 3) if i != 3]
    assert report["sources"]["documents-v1:train"]["reuse"] == {"never trained": 13}
    with pytest.raises(ValueError, match="exposure"):
        bm.build(p, train, evals)


def test_quotas_fill_each_label_to_its_count():
    train, evals = suites(n=60)
    p = plan()
    products = ("card", "mortgage", "loan")
    train["documents-v1:train"] = [dict(r, questions={"product": {"type": "choice", "instructions": "x", "criteria": dict.fromkeys(products),
                                                                  "label": products[i % 3]}}) for i, r in enumerate(train["documents-v1:train"])]
    s = spec(p, "documents-v1:train"); del s["n"]
    s["quotas"] = [{"match": {"q:product": "card"}, "n": 7}, {"match": {"q:product": "mortgage"}, "n": 3}]
    rows, index, report = bm.build(p, train, evals)
    got = [r["questions"]["product"]["label"] for r, x in zip(rows, index) if x["source"] == "documents-v1:train"]
    assert sorted(got) == ["card"] * 7 + ["mortgage"] * 3
    assert report["sources"]["documents-v1:train"]["fill"] == {'{"q:product": "card"}': (7, 7), '{"q:product": "mortgage"}': (3, 3)}


def test_exact_eval_states_and_eval_pr_ids_are_screened():
    train, evals = suites()
    e = evals["pr-labels:test"][0]
    train["pr-labels:train"][4] = dict(train["pr-labels:train"][4], state="  " + e["state"].upper() + " ")   # the same state, normalised
    train["pr-labels:train"][6] = dict(train["pr-labels:train"][6], id=evals["pr-labels:test"][1]["id"])  # the same PR, other text
    drop, _ = bm.screen(train, evals)
    assert drop["pr-labels:train"][4] == "exact-eval" and drop["pr-labels:train"][6] == "eval-pr-id"
    _, index, report = bm.build(plan(n=40), train, evals)
    assert report["checks"]["eval_state_overlap"] == 0 and report["checks"]["pr_id_overlap_with_eval"] == 0
    assert sorted(x["row"] for x in index if x["source"] == "pr-labels:train") == [i for i in range(40) if i not in (4, 6)]


def test_near_duplicates_of_eval_states_are_screened():
    train, evals = suites()
    rng = random.Random(7)
    e = evals["hard-v1:development"][0]["state"].split()
    train["hard-v1:train"][2] = dict(train["hard-v1:train"][2], state=" ".join(["other", "start", "here"] + e[3:]))   # Jaccard ~0.9
    half = e[:len(e) // 2] + words(rng, len(e) - len(e) // 2).split()
    train["hard-v1:train"][3] = dict(train["hard-v1:train"][3], state=" ".join(["x", "y", "z"] + half[3:]))         # about 0.3: kept
    drop, pool = bm.screen(train, evals)
    assert drop["hard-v1:train"] == {2: "near-eval-shingle"}
    assert pool["dropped_shingle_ge_0.8"] == {"hard-v1:train": 1}


def test_a_plan_that_leaves_out_a_training_source_is_refused():
    train, evals = suites()
    p = plan()
    p["sources"] = [s for s in p["sources"] if s["source"] not in ("routing:generic-train", "decision-v7:train")]
    with pytest.raises(ValueError, match="routing:generic-train, decision-v7:train"):
        bm.build(p, train, evals)
    assert bm.uncovered(dict(p, dv7_replay=100)) == ["routing:generic-train"]   # the trainer replays decision-v7
    with pytest.raises(ValueError, match="reason"):
        bm.check_plan(dict(p, allow_missing_sources=["routing:generic-train", "evals/v7/decision-v7"]))
    bm.build(dict(p, allow_missing_sources=["routing:generic-train", "evals/v7/decision-v7"], reason="a delta on purpose"), train, evals)
    empty = plan()
    spec(empty, "ja-jglue:train")["max_json_chars"] = 10   # named, but every record is longer: no records drawn
    with pytest.raises(ValueError, match="draws no records from ja-jglue:train"):
        bm.build(empty, train, evals)


def built(tmp_path, train, evals, p=None):
    rows, index, report = bm.build(p or plan(), train, evals)
    out = tmp_path / "mix.jsonl"
    bm.write(out, rows, index, report)
    return out


def test_the_verifier_catches_a_changed_record(tmp_path):
    train, evals = suites()
    out = built(tmp_path, train, evals)
    index = [json.loads(x) for x in Path(f"{out}.index.jsonl").read_text(encoding="utf-8").splitlines()]
    sidecar = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    text = out.read_text(encoding="utf-8")
    assert vm.verify(text, index, sidecar, train, evals)["failures"] == []
    lines = text.split("\n")
    lines[3] = lines[3].replace('"label": "a"', '"label": "b"') if '"label": "a"' in lines[3] else lines[3].replace('"P', '"Q', 1)
    changed = "\n".join(lines)
    sidecar["sha256"] = __import__("hashlib").sha256(changed.encode()).hexdigest()   # only the record check can catch it
    report = vm.verify(changed, index, sidecar, train, evals)
    assert report["equal_to_source"] == len(index) - 1
    assert report["failures"] == ["1 line(s) differ from the source row their index names"]


def test_the_sidecar_passes_the_trainers_source_check(tmp_path):
    from d1a.training import train as trainer
    train, evals = suites()
    out = built(tmp_path, train, evals)
    a = argparse.Namespace(init_from="JohnP1/d1a-e4b", data=str(out), suite=None, replay=0, extra_suites="", allow_missing_sources="", reason="")
    assert trainer.source_coverage(a) == {"covered": trainer.required_sources(), "allowed_missing": [], "reason": None}


def test_the_shingle_screen_finds_the_nearest_eval_state_not_the_ones_sharing_most():
    train, evals = suites()
    rng = random.Random(11)
    t = words(rng, 200).split()
    train["hard-v1:train"][5] = dict(train["hard-v1:train"][5], state=" ".join(t))
    # five long eval states that hold all of the record (Jaccard about 0.09) and one that holds 92% of it and nothing else
    evals["hard-v1:development"] = [{"state": " ".join(words(rng, 2000).split() + t), "questions": {}} for _ in range(5)]
    evals["hard-v1:development"].append({"state": " ".join(t[15:]), "questions": {}})
    drop, _ = bm.screen(train, evals)
    assert drop["hard-v1:train"] == {5: "near-eval-shingle"}


def test_the_japanese_twin_of_an_eval_like_pr_is_dropped():
    train, evals = suites()
    en = train["pr-labels:train"][5]
    evals["pr-labels:test"].append({"state": en["state"], "questions": {}})   # the English PR's text in an eval partition, no id
    ja = train["pr-labels:train-ja"]
    ja[2] = dict(ja[2], id=f"{en['id']}:ja")                              # its Japanese twin: other text, the PR's id with a suffix
    ja[3] = dict(ja[3], id=f"{evals['pr-labels:test'][0]['id']}:ja")      # the twin of an eval PR
    drop, _ = bm.screen(train, evals)
    assert drop["pr-labels:train"] == {5: "exact-eval"}
    assert drop["pr-labels:train-ja"] == {2: "ja-twin-of-eval-like", 3: "eval-pr-id"}


def test_repeated_states_keep_the_first_copy_unless_their_labels_disagree():
    train, evals = suites()
    rows = train["documents-v1:train"]
    rows[7] = dict(rows[7], state=rows[1]["state"])                       # the same label as row 1
    rows[9] = dict(rows[9], state=rows[2]["state"], questions={"label": dict(rows[2]["questions"]["label"], label="b")})
    drop, _ = bm.screen(train, evals)
    assert drop["documents-v1:train"] == {7: "dup", 2: "dup-conflict", 9: "dup-conflict"}


def test_a_pr_with_an_eval_prs_title_and_body_is_dropped():
    train, evals = suites()
    rng = random.Random(3)
    head, body = evals["pr-labels:test"][2]["state"].split("\nbody: ", 1)
    # a long file list between them keeps the whole state's shingle Jaccard near 0.1 and its 300-character prefix apart
    train["pr-labels:train"][8] = dict(train["pr-labels:train"][8], state=f"{head.upper()}\nfiles: {words(rng, 400)}\nbody: {body} more")
    train["pr-labels:train"][9] = dict(train["pr-labels:train"][9], state=f"{head}\nbody: {words(rng)}")   # the title alone: kept
    drop, _ = bm.screen(train, evals)
    assert drop["pr-labels:train"] == {8: "near-eval-title-body"}


def test_hand_labelled_prs_are_screened_by_id_too(tmp_path, monkeypatch):
    train, _ = suites()
    monkeypatch.setattr(bm, "D1A_EVAL_SUITES", ()); monkeypatch.setattr(bm, "FROZEN_EVALS", ())   # the kit alone
    (tmp_path / "evals/d1a").mkdir(parents=True)
    pr = train["pr-labels:train"][4]
    human = [{"id": pr["id"], "state": "the same PR labelled by hand, other text", "labels": {}}]
    (tmp_path / "human.json").write_text(json.dumps(human), encoding="utf-8")
    evals = bm.load_evals(tmp_path, human=tmp_path / "human.json")
    drop, _ = bm.screen(train, evals)
    assert drop["pr-labels:train"] == {4: "eval-pr-id"}


def test_a_fixed_plan_gives_a_pinned_mix():
    train, evals = suites()
    p = plan(n=7)
    spec(p, "hard-v1:train").update(n=12, stratify="_meta.family")
    assert bm.build(p, train, evals)[2]["sha256"] == "39f9dcb3f2f372168393c0d3dc6c17da211d6ad95b2f4b3cf48158f4b5ec457c"
