"""safe_embeddings (#353): on MPS, PyTorch's embedding kernel (torch < 2.14) returns zero rows past row 2**31 // width in a
table of more than 2**31 elements (pytorch/pytorch#188756); Gemma 4's per-layer table is that big. The guard reads such
tables by indexing on MPS and leaves every other device on the module's own forward."""
from pathlib import Path

import pytest
import torch
from torch import nn

import d1a.backends.device as device
from d1a.backends.device import index_rows, safe_embeddings

BASE = str(Path(__file__).resolve().parent / "golden" / "tiny-gemma4" / "base")


def tables():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    torch.manual_seed(0)
    scaled = Gemma4TextScaledWordEmbedding(50, 4, padding_idx=0, embed_scale=16.0)
    nn.init.normal_(scaled.weight)
    return nn.ModuleDict({"plain": nn.Embedding(50, 4), "scaled": scaled, "small": nn.Embedding(2, 4)})


def test_index_rows_equals_the_forward():
    ids = torch.tensor([[0, 1, 0], [1, 1, 0]])
    for m in tables().values():
        assert torch.equal(index_rows(m, ids), m(ids))
    ids = torch.tensor([[0, 7, 49], [3, 3, 1]])
    for k in ("plain", "scaled"):
        m = tables()[k]
        assert torch.equal(index_rows(m, ids), m(ids))


def test_covers_only_tables_past_the_limit(monkeypatch):
    monkeypatch.setattr(device, "EMBED_LIMIT", 100)          # 50 x 4 = 200 elements: over; 2 x 4: under
    t = tables()
    ids = torch.tensor([1, 49])
    before = {k: t[k](ids) for k in ("plain", "scaled")}
    assert safe_embeddings(t) == 2 and safe_embeddings(t) == 2      # idempotent
    assert t["plain"].indexed_lookup and t["scaled"].indexed_lookup and not hasattr(t["small"], "indexed_lookup")
    assert isinstance(t["scaled"], type(tables()["scaled"]))       # still a Gemma4TextScaledWordEmbedding
    for k in ("plain", "scaled"):                                  # off MPS: the module's own forward, unchanged
        assert torch.equal(t[k](ids), before[k])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_reads_by_indexing(monkeypatch):
    monkeypatch.setattr(device, "EMBED_LIMIT", 100)
    t = tables()
    safe_embeddings(t)
    ids = torch.tensor([0, 7, 49])
    for k in ("plain", "scaled"):
        want = index_rows(t[k], ids)
        assert torch.equal(t[k].to("mps")(ids.to("mps")).cpu(), want)


def test_the_torch_model_is_covered(monkeypatch):
    monkeypatch.setattr(device, "EMBED_LIMIT", 0)            # the tiny fixture's tables are small: cover every one
    from d1a.backends.torch import DecisionModel, load_tokenizer
    m = DecisionModel(BASE, load_tokenizer(BASE), "cpu")
    assert getattr(m.lm.embed_tokens_per_layer, "indexed_lookup", False) and getattr(m.lm.embed_tokens, "indexed_lookup", False)
