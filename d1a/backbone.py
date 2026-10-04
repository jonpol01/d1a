"""What differs between base-model families, in one place.

d1a.model asks the backbone instead of branching on model details. A new base family is one subclass here (plus its
golden vectors): how it is detected, whether it can run the packed block-causal mask or needs one causal row per
question, its local-attention window, the state cache it needs, any extra LoRA target names, whether the MLX backend
runs it, and its own CUDA serving kernels (Qwen3.5's, loaded only through Qwen35).

    from d1a.backbone import for_config
    bb = for_config(lm.config)      # Gemma4 (sliding layers), Qwen35 (Gated DeltaNet), or Attention (plain)
    bb.hybrid, bb.sliding_window, bb.new_cache(), bb.lora_extra, bb.prefix_min_tokens, bb.mlx, bb.serve_cuda(m, opts, merged)

Delimiters are not here: they follow the tokenizer (d1a.model.layout picks the first delimiter set its vocabulary holds),
so code that only has a tokenizer (encode) needs no backbone.
"""
from transformers import DynamicCache


def layer_types(config):
    return set(getattr(config, "layer_types", None) or [])


class Attention:
    """A plain attention-only backbone: the packed block-causal mask, one KV cache, no extra LoRA targets."""
    name = "attention"
    hybrid = False      # recurrent layers that cannot honour the packed mask: run one causal row per question instead
    lora_extra = ()     # LoRA target names beyond the attention and MLP projections, for the "all" and "attn" presets
    mlx_cache_copy = "replicate"   # d1a.mlx_model: plain and rotating KV caches are copied with their scalar offset
    mlx = False         # d1a.checkpoint: whether the MLX backend runs this family (backend="auto" picks it on Apple Silicon)

    def __init__(self, config):
        self.config = config

    @classmethod
    def matches(cls, config):
        return True

    @property
    def sliding_window(self):
        """The local-attention window of the sliding layers, or None: the packed form then needs a second mask for them
        (d1a.model.branch_masks), since one additive mask would give every layer global reach."""
        return None

    @property
    def prefix_min_tokens(self):
        """d1a.serve caches the state prefix from this many state tokens. Attention-only: 384, below which the branch-only
        pass is not faster than one packed pass on MPS (per-op overhead)."""
        return 384

    def new_cache(self):
        """An empty cache for the state prefix. A sliding backbone must NOT get its config's layer types: a sliding layer's
        cache keeps only the last window, but the packed branch pass hands those layers the full-length mask, so every layer
        caches the whole state."""
        return DynamicCache()

    @staticmethod
    def unwrap(lm):
        """The text model of a multimodal checkpoint (Gemma 4 loads as a wrapper around it), so the vision and audio towers
        are neither held in memory nor matched by LoRA target names; other models unchanged."""
        return getattr(lm, "language_model", lm)

    def serve_cuda(self, m, opts, merged):
        """Swap in this family's CUDA serving kernels (d1a.checkpoint, serving on CUDA). Plain attention has none."""

    def picked_hidden(self, lm, ids, pos, mask, picks):
        """[B, P, d] final hidden states at `picks` [B, P] of a packed batch, when this family can compute them more cheaply
        than every position (d1a.model.forward_batch); None: run the whole sequence."""
        return None


class Gemma4(Attention):
    """Gemma 4: attention-only with sliding layers (a 512-token window), KV-shared layers, reserved <unused0-4> delimiter
    rows and a leading <bos>. Detected by its sliding layers, so another sliding-window family gets the same mask rule."""
    name = "gemma4"

    @classmethod
    def matches(cls, config):
        return "sliding_attention" in layer_types(config)

    @property
    def sliding_window(self):
        return self.config.sliding_window

    @property
    def mlx(self):
        """Gemma 4 only: d1a.mlx_model reads its per-layer embeddings and KV-shared layers, which another sliding family lacks."""
        return self.config.model_type == "gemma4_text"

    def picked_hidden(self, lm, ids, pos, mask, picks):
        """Gemma 4's last num_kv_shared_layers layers take their keys and values from earlier layers (transformers'
        shared_kv_states), so no position's output there feeds any other position: only the positions the head reads need
        those layers. The earlier layers run over the whole packed sequence as transformers runs them; the shared layers run
        over `picks` alone (their queries, rotary angles, mask rows and per-layer inputs). Exact: per-token norms, MLPs and
        gates, and attention over the same keys and values; the same applies to the gradients."""
        from collections import UserDict
        text = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        shared = getattr(text.config, "num_kv_shared_layers", 0) or 0
        if not shared or not hasattr(text, "rotary_emb"): return None
        depth, types = len(text.layers) - shared, text.config.layer_types
        masks = mask if isinstance(mask, dict) else {t: mask for t in set(types)}
        h = text.embed_tokens(ids)
        ple = text.project_per_layer_inputs(h, text.get_per_layer_inputs(ids, h)) if text.hidden_size_per_layer_input else None
        rope = {t: text.rotary_emb(h, pos, t) for t in text.unique_layer_types}
        kv = UserDict()   # transformers fills it from the last non-shared layer of each type; the shared layers read it
        for i in range(depth):
            h = text.layers[i](h, ple[:, :, i] if ple is not None else None, shared_kv_states=kv, position_embeddings=rope[types[i]],
                               attention_mask=masks[types[i]], position_ids=pos)
        take = lambda x: x.gather(1, picks.view(*picks.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))   # rows along dim 1
        h, pos_p, ple = take(h), pos.gather(1, picks), take(ple) if ple is not None else None
        rope = {t: tuple(take(x) for x in cs) for t, cs in rope.items()}
        masks = {t: m.gather(2, picks[:, None, :, None].expand(-1, m.shape[1], -1, m.shape[-1])) for t, m in masks.items()}
        for i in range(depth, len(text.layers)):
            h = text.layers[i](h, ple[:, :, i] if ple is not None else None, shared_kv_states=kv, position_embeddings=rope[types[i]],
                               attention_mask=masks[types[i]], position_ids=pos_p)
        return text.norm(h)


class Qwen35(Attention):
    """Qwen3.5 (Kev's bases): Gated DeltaNet layers, recurrent, so every question runs as its own causal row continuing
    from the state (d1a.model.rows_of); the cache carries DeltaNet conv and recurrent states."""
    name = "qwen35"
    hybrid = True
    # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
    lora_extra = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    mlx_cache_copy = "merge"       # DeltaNet conv + recurrent states: copied by mlx-lm's batch merge
    mlx = True

    @classmethod
    def matches(cls, config):
        return "linear_attention" in layer_types(config)

    @property
    def prefix_min_tokens(self):
        """Always cache: without it the miss path recomputes the state once per question (Kev-0.8B bf16 on MPS, 5
        questions: 1011 -> 413 ms)."""
        return 0

    def new_cache(self):
        return DynamicCache(config=self.config)

    def serve_cuda(self, m, opts, merged):
        """Fused Triton kernels (d1a.fused_qwen35; they need plain, merged weights) and CUDA graphs (d1a.cuda_graphs)."""
        if opts.fused and merged:
            from .fused_qwen35 import fuse
            fuse(m.lm)
        if opts.cuda_graphs:
            from .cuda_graphs import CudaGraphs
            m.graphs = CudaGraphs(m.lm, m.pad_id)


REGISTRY = (Qwen35, Gemma4)   # first match wins; Attention is the fallback


def for_config(config):
    """The backbone for a (text) model config."""
    return next((cls for cls in REGISTRY if cls.matches(config)), Attention)(config)


def for_mlx(caches):
    """The backbone of an mlx-lm model, from the prompt cache it builds (mlx-lm's configs differ from transformers'): any
    cache that is not a plain or rotating KV cache holds recurrent state (Qwen3.5's DeltaNet layers)."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    return (Attention if all(type(c) in (KVCache, RotatingKVCache) for c in caches) else Qwen35)(None)
