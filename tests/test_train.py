"""d1a.train: the loss, how micro-batches weigh records, and that a run is reproducible from its seed (on the CPU, with
the committed tiny Gemma 4 in tests/golden/tiny-gemma4)."""
import json
import sys
from pathlib import Path

import pytest
import torch

from d1a import train

ROOT = Path(__file__).resolve().parents[1]


def test_the_loss_is_proper():
    """The expected loss under the true distribution is at its minimum there: its gradient is zero."""
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * train.question_loss(logits, {"label": y}, "cpu") for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6
    z = torch.tensor([2.0, 0.0, -2.0])
    soft = train.question_loss(z, {"target": [0.5, 0.5, 0.0], "label": 0}, "cpu")   # a soft target replaces the label
    assert soft.item() == pytest.approx(-(torch.log_softmax(z, -1)[:2] / 2).sum().item())


def test_uneven_micro_batches_weigh_every_record_alike():
    """A step's gradient is the mean over its records whatever the micro-batch sizes, including a short last step."""
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4), (10, 1)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / train.accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2] and gradients[3] == pytest.approx(x.mean().item())
    assert [train.accumulation_records(10, 3, 3, mb) for mb in range(4)] == [9, 9, 9, 1]


def test_a_run_is_reproducible_from_its_seed(tmp_path, monkeypatch):
    """The same seed and data give the same adapter and head, bit for bit; another seed does not (CPU)."""
    rows = [{"state": "the customer is charged twice" + " again" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(8)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.chdir(ROOT)

    def trained(name, seed):
        argv = ["d1a.train", "--base", "tests/golden/tiny-gemma4/base", "--data", str(tmp_path / "data.jsonl"), "--device", "cpu", "--lora", "4",
                "--batch", "2", "--accum", "2", "--p_none_pair", "0.5", "--seed", str(seed), "--out", str(tmp_path / name)]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        from safetensors.torch import load_file
        adapter = load_file(str(tmp_path / name / "adapter_model.safetensors"))
        from d1a.checkpoint import read_meta
        head = read_meta(tmp_path / name).head
        return adapter, head

    first, again, other = trained("a", 0), trained("b", 0), trained("c", 1)
    for x, y in ((first, again),):
        assert x[0].keys() == y[0].keys() and all(torch.equal(x[0][k], y[0][k]) for k in x[0])
        assert all(torch.equal(x[1][k], y[1][k]) for k in x[1])
    assert any(not torch.equal(first[0][k], other[0][k]) for k in first[0])
