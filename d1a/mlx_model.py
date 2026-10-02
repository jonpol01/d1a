# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); Gemma 4 bases (sliding-window and KV-shared layers) and quantized MLX exports (export_mlx); model-family details from d1a.backbone; Gemma 4's per-layer embeddings read from the weight files per request (FlashEmbedding).
"""Apple Silicon backend for the Qwen3.5 and Gemma 4 checkpoints: mlx-lm's Metal implementation of the backbone under
D1A's own encoder and pointer head.

MPS has no Gated DeltaNet kernels, so the PyTorch path runs reference code there (Kev-4B ~0.8 s per request). This module
runs the same computation on Metal through mlx-lm and keeps everything Kev-specific unchanged: `d1a.model.encode` builds the
tokens, `rows_of` splits them into one causal row per question (the hybrid form the torch path uses too), and the readout is
the very same `PointerHead` (fp32, with the checkpoint's temperature) applied to the branch hidden states.

Gemma 4 (attention-only, sliding layers with a 512-token window, KV-shared layers) runs the same row form. Its state cache
holds a full KVCache per global layer and a RotatingKVCache per sliding layer (mlx-lm keeps the whole state in a sliding
cache after a multi-token prefill and trims it to the last window - 1 keys when the branch rows arrive, so every branch
token sees exactly the keys the packed torch mask gives it); the KV-shared layers read their source layer's keys, so
they need no cache of their own. `export_mlx` writes a merged (optionally quantized) copy that loads without the base
or the adapter (d1a.checkpoint: EXPORT_CONFIG).

Same contract as `DecisionModel` for serving and scoring: encode / forward / probs / probs_and_prefix / probs_with_prefix /
head / dtype. Selected by `LoadOptions(backend="mlx")` (or "auto" on Apple Silicon) in `d1a.checkpoint`; never by the
benchmark, whose reported numbers stay on the fp32 torch path. Parity against that path is measured in
tests/test_mlx.py (max |dp| and argmax flips on development records, prefix vs full pass, one question vs several).
"""
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.utils import load_model

from .backbone import for_mlx
from .checkpoint import weight_shards
from .model import PointerHead, encode, probs_one, rows_of, rows_per_pass

PLE = "embed_tokens_per_layer"   # Gemma 4's per-layer embedding table
SAFETENSORS_NP = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}   # bf16 is read as its bits
CACHE_LIMIT = 1 << 30   # MLX's buffer cache keeps a buffer per new request shape; Kev-4B on an M5, 50 requests: 1.0 GB cached vs 3.7 GB unbounded, same latency


def merge_lora(lm, adapter_dir, scale=1.0):
    """Fold a PEFT adapter into the mlx-lm model's weights the way the torch path does: W + (B @ A) * alpha / r in fp32,
    rounded once to the backbone dtype. `scale` is LoadOptions.lora_scale (WiSE-FT interpolation). Returns the tensor count."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("trainable_token_indices"):
        raise ValueError("the MLX backend does not carry trained token embeddings (special_embeddings checkpoints); use backend=torch")
    alpha = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(tree_flatten(lm.parameters()))
    merged = {}
    with mx.stream(mx.cpu):   # the GPU's fp32 matmul is a reduced-precision fast path (~1e-3 relative on an M5); the merge is one-time and must be exact
        for name, a in weights.items():
            if not name.endswith(".lora_A.weight"):
                continue
            stem = name[: -len(".lora_A.weight")]
            # peft names the wrapped text model `base_model.model.<layers...>`; mlx-lm nests it as `language_model.model.<layers...>`
            target = stem.replace("base_model.model.", "language_model.model.", 1) + ".weight"
            if target not in params:
                raise ValueError(f"adapter tensor {stem} has no weight in the mlx-lm model (looked for {target})")
            base = params[target]
            delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * (alpha * scale)
            merged[target] = (base.astype(mx.float32) + delta).astype(base.dtype)
            mx.eval(merged[target])   # one tensor at a time: one graph over every target peaks at ~2.9x the base weights and leaves ~2x of them in MLX's buffer cache (25.8 GB RSS for the 4B)
    lm.load_weights(list(merged.items()), strict=False)
    mx.eval(lm.parameters())
    del params, weights   # drop the pre-merge weights and the adapter before clearing the cache, or ~7 GB stay cached
    mx.clear_cache()   # hand the merge transients back to the OS; MLX keeps freed buffers otherwise
    return len(merged)


UNUSED_SHARED_KV = re.compile(r"language_model\.model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm)\.weight")


def safetensors_tensor(path, name):
    """A tensor of a safetensors file as a read-only numpy memmap (bf16 as uint16 bits) and its dtype tag, or None."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little"); header = json.loads(f.read(n))
    t = header.get(name)
    if t is None: return None
    start, end = t["data_offsets"]
    return np.memmap(path, dtype=SAFETENSORS_NP[t["dtype"]], mode="r", offset=8 + n + start, shape=tuple(t["shape"])), t["dtype"]


class FlashEmbedding:
    """Gemma 4's per-layer embedding table read from its safetensors file instead of held in memory (#71).

    Every token looks up one row of it (num_layers x 256 values) and a request needs only its own tokens' rows, yet it is
    the largest tensor of the model: 1.3 GB of E2B's 3.5 GB 8-bit export, 4.7 GB of the bf16 base. The file is memory-
    mapped, so the OS reads the rows a request touches and can evict them again. The values are the in-memory
    (Quantized)Embedding's: the rows are gathered first and dequantized with the same group size, bits and mode."""

    def __init__(self, files, module):
        self.group_size, self.bits, self.mode = getattr(module, "group_size", None), getattr(module, "bits", None), getattr(module, "mode", "affine")
        self.parts = {}
        for part in ("weight", "scales", "biases"):
            for f in files:
                for prefix in ("language_model.model.", "model.language_model."):   # export (mlx-lm) and base (transformers) names
                    found = safetensors_tensor(f, f"{prefix}{PLE}.{part}")
                    if found: self.parts[part] = found; break
                if part in self.parts: break
        if "weight" not in self.parts: raise ValueError(f"no {PLE}.weight in {[str(f) for f in files]}")
        if ("scales" in self.parts) != (self.bits is not None): raise ValueError(f"{PLE}: the file and the module disagree on quantization")

    def _rows(self, part, idx):
        table, dtype = self.parts[part]
        rows = mx.array(np.ascontiguousarray(table[idx]))
        return rows.view(mx.bfloat16) if dtype == "BF16" else rows

    def __call__(self, ids):
        uniq, inv = np.unique(np.asarray(ids).reshape(-1), return_inverse=True)
        rows = self._rows("weight", uniq)
        if self.bits is not None:
            biases = self._rows("biases", uniq) if "biases" in self.parts else None
            rows = mx.dequantize(rows, scales=self._rows("scales", uniq), biases=biases, group_size=self.group_size, bits=self.bits, mode=self.mode)
        return rows[mx.array(inv.astype(np.int32))].reshape(*ids.shape, -1)


def ple_on_flash(lm, files):
    """Swap a Gemma 4 model's in-memory per-layer embedding table for a FlashEmbedding over `files`. Call it on a lazily
    loaded model, before its parameters are evaluated, so the table is never read into memory. -> whether it swapped."""
    text = getattr(getattr(lm, "language_model", None), "model", None)
    if text is None or getattr(text, PLE, None) is None or isinstance(getattr(text, PLE), FlashEmbedding): return False
    module = text.pop(PLE)                      # an mlx Module is a dict of its children: this drops the arrays too
    object.__setattr__(text, PLE, FlashEmbedding(files, module))
    return True


def load_mlx_lm(path, lazy=False):
    """mlx-lm's load_model, strict except for one known gap: Gemma 4 checkpoints store key/value projections for the
    KV-shared layers, which read an earlier layer's keys and never use their own (transformers does not build them, so no
    adapter targets them either); mlx-lm does not build them and its strict load refuses the file. So the load is
    non-strict and this checks what strict would: every parameter is in the file, and the only file tensors left over are
    those projections of layers the model shares. -> (model, config)"""
    path = Path(path)
    lm, config = load_model(path, lazy=lazy, strict=False)
    stored = {}
    for f in weight_shards(path): stored.update(mx.load(str(f)))
    stored = set(lm.sanitize(stored) if hasattr(lm, "sanitize") else stored)
    params = {k for k, _ in tree_flatten(lm.parameters())}
    text = getattr(getattr(lm, "language_model", None), "model", None)
    shared = {i for i, src in enumerate(getattr(text, "previous_kvs", [])) if src != i}   # Gemma 4: layer i reads layer src's keys
    missing = sorted(params - stored)
    extra = sorted(k for k in stored - params if not ((m := UNUSED_SHARED_KV.fullmatch(k)) and int(m[1]) in shared))
    if missing or extra:
        raise ValueError(f"{path}: {len(missing)} parameters not in the weights (e.g. {missing[:2]}), {len(extra)} unexpected tensors (e.g. {extra[:2]})")
    return lm, config


def replicate(cache, n):
    """n copies of a one-sequence attention cache as one batch-n cache of the same kind, scalar offset kept. Every branch
    row continues the same state, so one offset serves the batch and mlx-lm's single-sequence masks (causal, and the
    sliding window of a RotatingKVCache) apply unchanged; the arrays are copied, so the source prefix stays pristine."""
    if isinstance(cache, RotatingKVCache):
        out = RotatingKVCache(max_size=cache.max_size, keep=cache.keep)
        k, v = cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    elif type(cache) is KVCache:
        out, (k, v) = KVCache(), cache.state
    else:
        raise TypeError(f"cannot replicate a {type(cache).__name__}")
    out.keys, out.values = mx.repeat(k, n, axis=0), mx.repeat(v, n, axis=0)
    out.offset = cache.offset
    if isinstance(cache, RotatingKVCache): out._idx = k.shape[2]
    return out


class MLXDecisionModel:
    """Prefill-only scorer: hidden states from mlx-lm, logits from the shared torch PointerHead."""
    backend, device, option_isolation = "mlx", "mlx", False
    ple_flash = False       # set by __init__ when Gemma 4's per-layer embeddings are read from the weight files
    prefix_min_tokens = 0   # d1a.serve caches the state prefix for every request: on Metal the branch-only pass is always the cheaper one

    def __init__(self, base_dir, pad_id, head_dim=256, ple_flash=True):
        """base_dir: an mlx-lm loadable directory: the base snapshot (weights as stored, bf16) or an export_mlx folder
        (merged, optionally quantized; the `quantization` block of its config.json). ple_flash: read Gemma 4's per-layer
        embeddings from the weight files per request instead of holding them in memory (FlashEmbedding)."""
        lm, config = load_mlx_lm(base_dir, lazy=True)
        self.ple_flash = bool(ple_flash) and ple_on_flash(lm, weight_shards(base_dir))
        mx.eval(lm.parameters())
        self._init(lm, pad_id, head_dim, config.get("quantization"))

    @classmethod
    def from_lm(cls, lm, pad_id, head_dim=256, quantization=None):
        """Wrap an already built mlx-lm model (export_mlx, tests)."""
        m = cls.__new__(cls); m._init(lm, pad_id, head_dim, quantization); return m

    def _init(self, lm, pad_id, head_dim, quantization):
        self.lm, self.quantization = lm, quantization
        mx.set_cache_limit(CACHE_LIMIT)
        self.text = self.lm.language_model.model                      # Qwen3_5TextModel / Gemma4TextModel: embeddings -> layers -> final norm = `.model.last_hidden_state`
        # mlx-lm keeps some constants as lazy arrays outside the parameters (Gemma 4's proportional RoPE frequencies). A lazy
        # array belongs to the stream of the thread that built it, and d1a.serve runs the model on its own thread ("There is
        # no Stream(gpu, 1) in current thread"), so they are evaluated here, on the loading thread.
        mx.eval([v for _, module in self.lm.named_modules() for v in module.values() if isinstance(v, mx.array)])   # `_freqs`: in the module's dict, not among its parameters
        self.backbone = for_mlx(make_prompt_cache(self.lm))   # d1a.backbone: hybrid form and how the state cache is copied
        self.hybrid = self.backbone.hybrid
        self.pad_id = pad_id
        self.head = PointerHead(self.text.norm.weight.shape[0], dp=head_dim).eval()   # the final norm's width: a quantized embedding's weight is packed

    @property
    def dtype(self):
        """The activation dtype (the final norm's), prefixed by the weight format of a quantized export ("4bit-g64-bfloat16")."""
        dtype = str(self.text.norm.weight.dtype).removeprefix("mlx.core.")
        q = self.quantization
        return f"{q['bits']}bit-g{q['group_size']}-{dtype}" if q else dtype

    def eval(self):
        self.head.eval(); return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    def _hidden(self, rows, cache=None):
        """[N, L, d] hidden states of right-padded token rows. Pads sit after every real token and both layer kinds are
        causal (attention: causal mask; DeltaNet: a left-to-right recurrence), so no real token sees a pad."""
        L = max(len(r) for r in rows)
        ids = mx.array([r + [self.pad_id] * (L - len(r)) for r in rows], dtype=mx.int32)
        h = self.text(ids, cache=cache)
        mx.eval(h)
        return h

    def _logits(self, h, decide, opts):
        """One question's logits through the fp32 pointer head (temperature included, eval mode)."""
        idx = mx.array([decide, *opts], dtype=mx.int32)
        picked = torch.from_numpy(np.asarray(h[idx].astype(mx.float32)))
        with torch.no_grad():
            return self.head(picked[0], picked[1:])

    def forward_rows(self, enc):
        """Row form, as the torch path computes it: every question is one causal row of state + branch tokens, the state
        recomputed per row. The reference the prefix form is checked against (tests/test_mlx.py); serving uses `forward`."""
        S, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([S + r["ids"] for r in rows]), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            h = self._hidden([S + r["ids"] for r in part])
            out += [self._logits(h[i], len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for i, r in enumerate(part)]
        return out

    # --- state prefix: the state runs once into an mlx-lm prompt cache (KV for the attention layers, conv + recurrent state
    # for the DeltaNet layers); the branches run as one batch on a replicated copy, so the prefix stays pristine and can be
    # reused by the next request with the same state. On Metal this is also the cheapest way to answer a single request
    # (state once instead of once per question), so it is the only path `forward` / `probs` take.

    def prefix(self, enc):
        Ls = enc["seg"].count(0)
        cache = make_prompt_cache(self.lm)
        self._hidden([enc["ids"][:Ls]], cache)
        return Ls, cache

    def _branch_logits(self, enc, cache):
        """Branches as rows on a replicated copy of the state cache, rows_per_pass rows (and cache copies) at a time."""
        _, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([r["ids"] for r in rows], enc["seg"].count(0)), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            merge = self.backbone.mlx_cache_copy == "merge"
            batch = [type(c).merge([c] * len(part)) if merge else replicate(c, len(part)) for c in cache]   # both copy the arrays: `cache` is not mutated
            h = self._hidden([r["ids"] for r in part], batch)
            out += [self._logits(h[i], r["decide"], r["opts"]) for i, r in enumerate(part)]
        return out

    def _branch_probs(self, enc, cache):
        return [F.softmax(z, -1) for z in self._branch_logits(enc, cache)]

    def forward(self, enc):
        """List of logits tensors, one per question."""
        return self._branch_logits(enc, self.prefix(enc)[1])

    def probs(self, enc):
        return self._branch_probs(enc, self.prefix(enc)[1])

    def probs_and_prefix(self, enc):
        prefix = self.prefix(enc)
        return self._branch_probs(enc, prefix[1]), prefix

    def probs_with_prefix(self, enc, prefix):
        Ls, cache = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        return self._branch_probs(enc, cache)

    def probs_batch(self, encs, prefixes, keep):
        """d1a.serve's batch call: one request at a time on Metal."""
        out = [probs_one(self, e, p, k) for e, p, k in zip(encs, prefixes, keep)]
        return [o[0] for o in out], [o[1] for o in out]


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def export_mlx(ck, out, bits=None, group_size=64, embeddings=True, per_layer_bits=None):
    """Write `ck` (a LoRA checkpoint, d1a.checkpoint.Checkpoint) as an MLX export folder that d1a.checkpoint serves
    through this backend without the base or the adapter: the adapter merged into the base in fp32 on the CPU stream
    (merge_lora, the same bits the MLX backend computes at load), then optionally quantized (`bits` per weight, affine,
    `group_size` per scale; `embeddings` also quantizes embed_tokens and the per-layer embeddings, Gemma 4's largest
    tensors; `per_layer_bits` gives the per-layer embeddings their own width: on Gemma 4 E2B they are half the weights and
    take 4 bits well, while the linear layers need 8), saved by mlx-lm (config.json + model*.safetensors), with the pointer head in fp32 (head.safetensors), the
    checkpoint's tokenizer files and d1a_config.json. -> the d1a_config dict."""
    import shutil

    import mlx.nn as nn
    from mlx_lm.utils import quantize_model, save_config, save_model
    from safetensors.torch import save_file

    from .checkpoint import EXPORT_CONFIG, EXPORT_HEAD, export_config, resolve_run
    from .model import layout, load_tokenizer, pad_id

    if ck.export is not None: raise ValueError(f"{ck.path} is already an MLX export")
    if ck.full: raise ValueError("export_mlx merges an adapter; full-weight checkpoints are not supported")
    if ck.meta.option_isolation: raise ValueError("option_isolation needs the packed mask; not available on the MLX backend")
    meta, out = ck.meta, Path(out)
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    lm, config = load_mlx_lm(resolve_run(f"{meta.base}@{meta.base_revision or ''}"))
    merge_lora(lm, ck.path)
    quantization = None
    if bits:
        def widths(path, module):
            if isinstance(module, nn.Embedding) and not embeddings: return False
            if per_layer_bits and path.endswith("embed_tokens_per_layer"): return {"group_size": group_size, "bits": per_layer_bits, "mode": "affine"}
            return True
        lm, config = quantize_model(lm, config, group_size, bits, quant_predicate=widths)
        quantization = {**config["quantization"], "embeddings": bool(embeddings)}
    norm = lm.language_model.model.norm.weight   # read before save_model donates the weights
    hidden, dtype = norm.shape[0], str(norm.dtype).removeprefix("mlx.core.")
    out.mkdir(parents=True, exist_ok=True)
    save_model(out, lm, donate_model=True)
    save_config(config, out / "config.json")
    save_file({k: v.float().contiguous() for k, v in meta.head.items()}, str(out / EXPORT_HEAD))
    for name in TOKENIZER_FILES:
        if ck.file(name).exists(): shutil.copyfile(ck.file(name), out / name)
    leading, delimiters, _ = layout(tok)
    cfg = export_config(ck, leading, delimiters, pad_id(tok), hidden, quantization, dtype)
    (out / EXPORT_CONFIG).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg
