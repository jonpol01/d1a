"""What differs between base-model families, in one place.

d1a.model asks the backbone instead of branching on model details. A new base family is one subclass here (plus its
golden vectors): how it is detected, whether it can run the packed block-causal mask or needs one causal row per
question, its local-attention window, the state cache it needs, and any extra LoRA target names.

    from d1a.backbone import for_config
    bb = for_config(lm.config)      # Gemma4 (sliding layers), Qwen35 (Gated DeltaNet), or Attention (plain)
    bb.hybrid, bb.sliding_window, bb.new_cache(), bb.lora_extra, bb.prefix_min_tokens

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


class Qwen35(Attention):
    """Qwen3.5 (Kev's bases): Gated DeltaNet layers, recurrent, so every question runs as its own causal row continuing
    from the state (d1a.model.rows_of); the cache carries DeltaNet conv and recurrent states."""
    name = "qwen35"
    hybrid = True
    # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
    lora_extra = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    mlx_cache_copy = "merge"       # DeltaNet conv + recurrent states: copied by mlx-lm's batch merge

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


REGISTRY = (Qwen35, Gemma4)   # first match wins; Attention is the fallback


def for_config(config):
    """The backbone for a (text) model config."""
    return next((cls for cls in REGISTRY if cls.matches(config)), Attention)(config)


def for_mlx(caches):
    """The backbone of an mlx-lm model, from the prompt cache it builds (mlx-lm's configs differ from transformers'): any
    cache that is not a plain or rotating KV cache holds recurrent state (Qwen3.5's DeltaNet layers)."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    return (Attention if all(type(c) in (KVCache, RotatingKVCache) for c in caches) else Qwen35)(None)
