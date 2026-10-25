"""Embedding tables too large for PyTorch's MPS embedding kernel (#353), and tables kept in bf16 under an fp32 trunk."""

import torch

# PyTorch's MPS embedding kernel (torch < 2.14) indexes rows with a 32-bit offset: in a table of more than 2**31 elements,
# every row past 2**31 // width comes back as zeros, while plain indexing reads it right (pytorch/pytorch#188756, #353).
# Gemma 4's per-layer embedding table is over that size: E2B is wrong from token id 239,675 (tabs, emoji, the image tokens),
# E4B from 199,729 (every punctuation and digit token).
EMBED_LIMIT = 2**31

_SAFE = {}
_KEPT = {}


def index_rows(emb, input_ids):
    """What nn.Embedding (and Gemma 4's scaled embedding, times its scale in the weight's dtype) returns, read by indexing."""
    rows = emb.weight[input_ids]
    scale = getattr(emb, "embed_scale", None)
    return rows * scale.to(emb.weight.dtype) if scale is not None else rows


def _indexed(cls):
    """A subclass of the embedding class `cls` whose forward reads rows by indexing on MPS and is cls's own elsewhere."""
    if cls not in _SAFE:
        def forward(self, input_ids):
            return index_rows(self, input_ids) if input_ids.device.type == "mps" else cls.forward(self, input_ids)
        _SAFE[cls] = type(cls.__name__, (cls,), {"forward": forward, "__module__": cls.__module__, "indexed_lookup": True})
    return _SAFE[cls]


def safe_embeddings(module):
    """Make every embedding table of more than EMBED_LIMIT elements in `module` look its rows up by indexing on MPS (CPU and
    CUDA keep the module's own forward): its class becomes a subclass that does. Only nn.Embedding and Gemma 4's scaled
    embedding, whose forwards this repeats. Returns how many tables it covers."""
    n = 0
    for m in module.modules():
        if type(m).__name__ not in ("Embedding", "Gemma4TextScaledWordEmbedding") or m.weight.numel() <= EMBED_LIMIT:
            continue
        if not getattr(m, "indexed_lookup", False):
            m.__class__ = _indexed(type(m))
        n += 1
    return n


def _rows_in(cls, dtype):
    """A subclass of the embedding class `cls` that reads rows by indexing and returns them in `dtype`, times Gemma's
    embedding scale in `dtype` (the scale from its Python value, as the all-`dtype` model's buffer holds it)."""
    if (cls, dtype) not in _KEPT:
        def forward(self, input_ids):
            rows = self.weight[input_ids].to(dtype)
            scale = getattr(self, "scalar_embed_scale", None)
            return rows * scale if scale is not None else rows
        _KEPT[cls, dtype] = type(cls.__name__, (cls,), {"forward": forward, "__module__": cls.__module__, "indexed_lookup": True})
    return _KEPT[cls, dtype]


def keep_tables(module, dtype):
    """Cast every floating parameter and buffer of `module` to `dtype` except the embedding tables, which keep the dtype they
    were loaded in and return their rows in `dtype` (read by indexing, so MPS reads every row right). For a checkpoint
    stored in bf16 and loaded in bf16, the forward is then the one of the model loaded in `dtype`, while the tables take half
    the memory (Gemma 4 E4B: 3.5B of its 7.5B text parameters). Returns how many tables it keeps."""
    tables = [m for m in module.modules() if isinstance(m, torch.nn.Embedding)]
    kept = {id(m.weight) for m in tables}
    with torch.no_grad():
        for m in module.modules():
            for p in m.parameters(recurse=False):
                if id(p) not in kept and p.is_floating_point():
                    p.data = p.data.to(dtype)
            for name, b in m.named_buffers(recurse=False):
                if b is not None and b.is_floating_point():
                    m._buffers[name] = b.to(dtype)
    for m in tables:
        m.__class__ = _rows_in(type(m), dtype)
    return len(tables)
