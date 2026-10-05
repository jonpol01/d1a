"""d1a.data without downloads: build's per-source seeding and provenance (stand-in converters and datasets), the
training-time variations (augment, none_pair), your own JSONL (load_records), and the model's records (materialize)."""
import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from d1a import data as D
from d1a.api import question_keys
from d1a.suite import text_digest

CRITERIA = {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"}


def choice(**extra):
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why?", "criteria": dict(CRITERIA), "label": "size", "src": "t", **extra}}}


# --- build --------------------------------------------------------------------------------------------------------------

def test_each_source_samples_from_its_own_seed(monkeypatch):
    def convert(split, n, src):
        value = src.randrange(10 ** 6)
        src.origins = [{"row": value, "row_sha256": str(value), "text_sha256": str(value)}]
        return [choice()]
    monkeypatch.setattr(D, "SOURCES", {"agnews": (convert, "train", "test"), "mnli": (convert, "train", "test")})
    alone = D.build(1, only=["mnli"])
    together = D.build(1)
    assert alone[0] == next(r for r in together if r["_meta"]["source"] == "mnli")   # adding a source changes no other's
    assert D.build(1, seed=3) != together and D.build(1) == together


def test_build_records_provenance_and_refuses_mismatches(monkeypatch):
    def convert(split, n, src):
        src.origins = [{"row": 7, "row_sha256": "r", "text_sha256": "t"}]
        return [choice()]
    monkeypatch.setattr(D, "SOURCES", {"agnews": (convert, "train", "test")})
    (record,) = D.build(1, split="test", revisions={"fancyzhx/ag_news": "abc"})
    assert record["_meta"] == {"row": 7, "row_sha256": "r", "text_sha256": "t", "source": "agnews", "repo": "fancyzhx/ag_news",
                               "revision": "abc", "split": "test", "id": "agnews/test/7"}
    with pytest.raises(ValueError, match="unknown sources"):
        D.build(1, only=["nope"])
    monkeypatch.setattr(D, "SOURCES", {"agnews": (lambda split, n, src: [choice(), choice()], "train", "test")})
    with pytest.raises(ValueError, match="provenance mismatch"):
        D.build(1)


class Rows:
    def __init__(self, rows):
        self.rows, self.features = rows, {"label": SimpleNamespace(names=["a", "b"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def test_sampling_skips_unlabelled_rows_and_hashes_what_it_keeps():
    src = D.Source(0)
    rows = D._sample(Rows([{"text": "Hello  World", "label": 1}, {"text": "x", "label": -1}, {"premise": "P", "label": 0}]), 10, src)
    assert len(rows) == 2 and -1 not in [r["label"] for r in rows] and len(src.origins) == 2
    hello = next(o for o in src.origins if o["row"] == 0)
    assert hello["text_sha256"] == text_digest("hello world")                 # casefolded, whitespace collapsed
    assert hello["row_sha256"] == hashlib.sha256(json.dumps({"text": "Hello  World", "label": 1}, sort_keys=True).encode()).hexdigest()


def test_converters_are_deterministic_and_vary_the_shape(monkeypatch):
    texts = [f"the customer was charged twice {i}" for i in range(40)]
    monkeypatch.setattr(D, "_dataset", lambda repo, split, src: Rows([{"text": t, "label": i % 2} for i, t in enumerate(texts)]))
    first = D._banking("train", 40, D.Source(5))
    assert first == D._banking("train", 40, D.Source(5))                              # the same seed, the same records
    shapes = {type(r["state"]).__name__ for r in first}
    assert {"str", "dict"} <= shapes                                                  # states arrive in several JSON shapes
    assert all(set(r["questions"]["intent"]["criteria"]) == {"a", "b"} for r in first)


# --- training-time variation --------------------------------------------------------------------------------------------

def test_augment_shuffles_choices_and_leaves_other_types():
    noul = {"type": "noul", "instructions": "i", "label": True, "src": "t"}
    req = {"state": "s", "questions": {"reason": choice()["questions"]["reason"], "n": noul}}
    out = D.augment(req, random.Random(1), 0, 0, 0)
    assert set(out["questions"]["reason"]["criteria"]) == set(CRITERIA) and out["questions"]["reason"]["label"] == "size"
    assert out["questions"]["n"] is noul
    orders = {tuple(D.augment(req, random.Random(i), 0, 0, 0)["questions"]["reason"]["criteria"]) for i in range(20)}
    assert len(orders) > 1


def test_augment_none_options_and_distractors():
    swapped = D.augment(choice(), random.Random(0), 1.0, 0.0, 0.0)["questions"]["reason"]   # the none option becomes the answer
    assert "size" not in swapped["criteria"] and swapped["label"] in swapped["criteria"] and swapped["label"] in dict(D.NONE_OPTIONS)
    wrong = D.augment(choice(), random.Random(0), 0.0, 1.0, 0.0)["questions"]["reason"]     # a none option as a wrong alternative
    assert wrong["label"] == "size" and len(wrong["criteria"]) == 4 and set(wrong["criteria"]) - set(CRITERIA) <= set(dict(D.NONE_OPTIONS))
    distracted = D.augment(choice(), random.Random(0), 0.0, 0.0, 1.0)["questions"]["reason"]
    assert distracted["label"] == "size" and len(set(distracted["criteria"]) & set(D.DISTRACTORS)) == 1
    soft = D.augment(choice(target={"size": 1, "damage": 1}), random.Random(0), 1.0, 0.0, 0.0)["questions"]["reason"]
    assert set(soft["criteria"]) == set(CRITERIA)                                      # a soft target is only shuffled
    two = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}, "label": "a", "src": "t"}}}
    assert set(D.augment(two, random.Random(0), 1.0, 0.0, 0.0)["questions"]["q"]["criteria"]) == {"a", "b"}   # needs 3+ options
    for bad in ((-0.1, 0, 0), (0.5, 0.5, 0.5)):
        with pytest.raises(ValueError, match="probabilities"):
            D.augment(choice(), random.Random(0), *bad)


def test_none_pair_is_minimal_and_relabelled():
    present, absent = D.none_pair(choice(), random.Random(3))
    with_answer, without = present["questions"]["reason"], absent["questions"]["reason"]
    assert len(with_answer["criteria"]) == 4 and with_answer["label"] == "size"
    assert [k for k in with_answer["criteria"] if k != "size"] == list(without["criteria"])   # same order, the answer removed
    assert without["label"] not in CRITERIA and without["criteria"][without["label"]] == with_answer["criteria"][without["label"]]
    D.materialize(present); D.materialize(absent)
    assert D.none_pair({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}, random.Random(0)) == []
    assert D.none_pair(choice(target={"size": 0.5, "damage": 0.5}), random.Random(0)) == []   # a soft target is left alone
    crowded = choice(); crowded["questions"]["reason"]["criteria"].update({k: None for k, _ in D.NONE_OPTIONS})
    _, fallback = D.none_pair(crowded, random.Random(0))
    assert fallback["questions"]["reason"]["label"] == "none_of_these"                 # every none key taken: a fresh one


# --- your own data and the model's records ------------------------------------------------------------------------------

def test_load_records_reads_the_readme_format(tmp_path):
    rows = [{"state": {"subject": "Charged twice", "body": "Two charges for order 4411."},
             "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": None}, "label": "billing"},
                           "angry": {"type": "noul", "instructions": "Is the customer angry?", "label": False},
                           "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "normal", "high"], "label": 1}}},
            {"state": "plain", "questions": {"q": {"type": "noul", "label": True, "src": "mine"}}, "_meta": {"id": "kept"}}]
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1], ensure_ascii=False) + "\n", encoding="utf-8")
    first, second = D.load_records(path, source="tickets")
    assert first["_meta"] == {"source": "tickets", "variant": "clean", "id": "tickets/0", "group_id": "tickets/0", "row": 0, "split": "custom",
                              "text_sha256": text_digest(json.dumps(rows[0]["state"], sort_keys=True, ensure_ascii=False))}
    assert first["questions"]["team"]["src"] == "tickets_choice" and second["questions"]["q"]["src"] == "mine"
    assert second["_meta"]["id"] == "kept" and second["_meta"]["row"] == 2              # given _meta wins; rows count lines
    rec = D.materialize(first)
    assert [q["label"] for q in rec["questions"]] == [0, 0, 1] and [q["keys"] for q in rec["questions"]][1] == question_keys("noul", None)


@pytest.mark.parametrize("line, message", [('{"state": "x", "questions": {"q": {"type": "noul"}}}', "no label"),
                                           ('{"questions": {"q": {"label": 1}}}', "needs a state"), ('{"state": "x", "questions": {}}', "needs a state")])
def test_load_records_refuses_unlabelled_or_empty_records(tmp_path, line, message):
    (tmp_path / "bad.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        D.load_records(tmp_path / "bad.jsonl")
    (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        D.load_records(tmp_path / "empty.jsonl")


def test_what_leaves_the_machine_has_no_answers():
    payload = D.api_request({**choice(target={"size": 1}), "_meta": {"id": "x"}})
    assert set(payload) == {"state", "questions"} and set(payload["questions"]["reason"]) == {"type", "instructions", "criteria"}


def test_materialize_labels_and_soft_targets():
    rec = D.materialize(choice(target={"damage": 3, "color": 1}))
    (q,) = rec["questions"]
    assert q["label"] == 0 and q["qtype"] == "choice" and q["qid"] == "reason" and q["src"] == "t" and q["keys"] == list(CRITERIA)
    assert q["target"] == [0.0, 0.75, 0.25] and q["options"][0] == "size: Wrong size"
    with pytest.raises(ValueError, match="no mass"):
        D.materialize(choice(target={"elsewhere": 1}))
