"""The outcome memory's offline pieces (#149, ported from #158): the blend, its time order, the query vectors' text form,
and the pointer-head queries that d1a.eval.benchmark --queries stores with each row. Synthetic vectors and a bare pointer
head only: no model is loaded."""
import numpy as np
import pytest
import torch

from d1a.core.head import PointerHead, QueryTap
from d1a.eval import predictors
from d1a.eval.benchmark import evaluate_records
from d1a.learning.feedback import blend, decode_vector, encode_vector, memory_probs, prequential

KEYS = ["a", "b"]


def test_the_blend():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    m = memory_probs(KEYS, np.array([1.0, 0.0]), bank, ["a", "b", "b"], k=2, tau=0.5)
    w = np.exp(np.array([1.0, 0.6]) / 0.5); w /= w.sum()                  # the two nearest: [1, 0] ("a") and [.6, .8] ("b")
    assert m == pytest.approx({"a": w[0], "b": w[1]})
    assert blend({"a": 0.9, "b": 0.1}, m, 0.4) == pytest.approx({"a": 0.6 * 0.9 + 0.4 * w[0], "b": 0.6 * 0.1 + 0.4 * w[1]})
    assert blend({"a": 0.9, "b": 0.1}, None, 0.4) == {"a": 0.9, "b": 0.1}
    assert memory_probs(KEYS, np.array([1.0, 0.0]), np.zeros((0, 2)), [], 3, 0.1) is None
    near = memory_probs(KEYS, np.array([0.0, 1.0]), bank, ["a", "b", "b"], k=1, tau=0.1)
    assert max(blend({"a": 0.9, "b": 0.1}, near, 0.8).items(), key=lambda kv: kv[1])[0] == "b"   # the choice can change


def test_memory_uses_only_outcomes_known_before_a_decision():
    rows = [{"ts": t, "outcome_ts": t + 0.5, "q": np.array([1.0, 0.0]), "probs": {"a": 0.5, "b": 0.5}, "label": "b"} for t in (0.0, 1.0, 2.0)]
    out = prequential(rows, rows, np.zeros(2), k=3, tau=0.1, lam=1.0)
    assert out[0] == {"a": 0.5, "b": 0.5}                                 # nothing known yet: the model's answer
    assert out[1] == pytest.approx({"a": 0.0, "b": 1.0}) and out[2] == pytest.approx({"a": 0.0, "b": 1.0})


def test_query_vectors_round_trip_as_text():
    v = np.arange(256, dtype=np.float32) / 7
    assert np.array_equal(decode_vector(encode_vector(v)), v) and decode_vector(encode_vector(v)).dtype == np.float32


def head(d=8, dp=4, seed=0):
    torch.manual_seed(seed)
    h = PointerHead(d, dp).eval(); h.temperature = 1.7
    return h


def test_the_tap_finds_each_questions_query_and_changes_no_logit():
    """Both of the head's entry points, at the head's and at a given temperature: the logits are bit for bit those of an
    untapped head, and each question's matched query is head.q(h_decide). A question whose options are all the same
    vector has a uniform answer whatever its query, so two of them cannot be told apart: None, not a guess."""
    g = torch.Generator().manual_seed(1)
    hd, ho = torch.randn(3, 8, generator=g), [torch.randn(k, 8, generator=g) for k in (3, 5, 4)]
    plain, tapped = head(), head()
    tap = QueryTap(tapped)
    with torch.no_grad():
        for t in (None, 2.5):
            tap.calls.clear()
            zs = [tapped(hd[j], ho[j], t) for j in range(3)]
            assert all(torch.equal(z, plain(hd[j], ho[j], t)) for j, z in enumerate(zs))
            got = tap.match([torch.softmax(z, -1).numpy() for z in zs])
            assert all(np.allclose(q, plain.q(hd[j]).numpy(), atol=1e-6) for j, q in enumerate(got))
        tap.calls.clear()
        owner = torch.tensor([0] * 3 + [1] * 5 + [2] * 4)
        z = tapped.many(hd, torch.cat(ho), owner)
        assert torch.equal(z, plain.many(hd, torch.cat(ho), owner))
        got = tap.match([torch.softmax(z[owner == j], -1).numpy() for j in range(3)])
        assert all(np.allclose(q, plain.q(hd[j]).numpy(), atol=1e-6) for j, q in enumerate(got))
        tap.calls.clear()
        same = torch.randn(1, 8, generator=g).repeat(3, 1)
        zs = [tapped(hd[j], same) for j in range(2)]
        assert tap.match([torch.softmax(z, -1).numpy() for z in zs]) == [None, None]
        assert tap.match([np.array([0.5, 0.5])]) == [None]                  # no call gave it


class MLXLike:
    """What LocalPredictor uses of d1a.backends.mlx.MLXDecisionModel, with a real pointer head over fixed synthetic hidden
    states: encode, and forward calling the head once per question, in reverse order (the tap must not rely on it)."""
    backend, hybrid = "mlx", True

    def __init__(self):
        g = torch.Generator().manual_seed(2)
        self.head = head()
        self.h = {"team": (torch.randn(8, generator=g), torch.randn(2, 8, generator=g)), "urgent": (torch.randn(8, generator=g), torch.randn(2, 8, generator=g))}

    def encode(self, tok, rec, **limits):
        return {"ids": [5, 6, 7], "seg": [0, 1, 1], "pos": [0, 1, 2], "decide_idx": [1], "opt_idx": [[2]]}

    def forward(self, enc):
        return list(reversed([self.head(*self.h[q]) for q in reversed(("team", "urgent"))]))


def local(queries):
    p = object.__new__(predictors.LocalPredictor)               # a loaded checkpoint's attributes, without loading one
    p.tok, p.model, p.device = None, MLXLike(), "cpu"
    p.temperature, p.context = p.model.head.temperature, {"max_state": 64, "max_branch": 64, "max_packed": 64}
    p.tap = QueryTap(p.model.head) if queries else None
    return p


def test_benchmark_rows_carry_each_questions_query(tmp_path, monkeypatch):
    monkeypatch.setattr(predictors, "sync", lambda device: None)
    record = {"state": "s", "_meta": {"id": "r1", "source": "t", "group_id": "r1", "variant": "clean"}, "questions": {
        "team": {"type": "choice", "instructions": "Who?", "criteria": {"billing": "B", "shipping": "S"}, "label": "billing", "src": "t"},
        "urgent": {"type": "noul", "instructions": "Urgent?", "label": True, "src": "t"}}}
    _, plain = evaluate_records([record], local(False), tmp_path / "plain")
    tapped = local(True)
    _, rows = evaluate_records([record], tapped, tmp_path / "tapped")
    assert [r["p"] for r in rows] == [r["p"] for r in plain] and not any("query" in r for r in plain)
    for r in rows:
        want = tapped.model.head.q(tapped.model.h[r["question"]][0]).detach().numpy()
        assert np.allclose(decode_vector(r["query"]), want, atol=1e-6)
