"""The accelerator this process uses: cuda, then mps, then cpu."""
import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    return 0


# PyTorch's MPS embedding kernel (torch < 2.14) indexes rows with a 32-bit offset: in a table of more than 2**31 elements,
# every row past 2**31 // width comes back as zeros, while plain indexing reads it right (pytorch/pytorch#188756, #353).
# Gemma 4's per-layer embedding table is over that size: E2B is wrong from token id 239,675 (tabs, emoji, the image tokens),
# E4B from 199,729 (every punctuation and digit token).
EMBED_LIMIT = 2**31


_SAFE = {}


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
