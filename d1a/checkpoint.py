# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); the MLX backend for Gemma 4 bases and MLX export folders (d1a_config.json); the ple_flash load option (D1A_PLE_FLASH).
"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter (or, for a full-weight run, the whole bf16
backbone), `head.pt` and the tokenizer; or an MLX export of one (d1a_config.json).

Loader rule: `adapter_config.json` present -> a LoRA adapter on `meta.base` at `meta.base_revision`; no adapter and
`config.json` + `model*.safetensors` (save_pretrained of the backbone, `meta.weights == "full"`) -> the backbone is loaded from
the checkpoint directory itself, nothing is merged, in the dtype head.pt's `weights_dtype` names (it must match the `dtype`
save_pretrained wrote to config.json). The tokenizer always comes from the base (both layouts carry a copy).
`d1a_config.json` present (EXPORT_CONFIG, written by d1a.mlx_model.export_mlx / scripts/export_mlx.py) -> an MLX export:
the adapter already merged into the base and saved by mlx-lm (config.json + model*.safetensors, optionally quantized),
the pointer head in fp32 in head.safetensors and the tokenizer files; it loads through the MLX backend only, with its
own tokenizer, and needs neither the base nor the adapter.

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`d1a.serve`, `d1a.benchmark`, `d1a.train --init_from`, `d1a.publish`, the scripts and the Hugging Face Space all go
through it. The Space vendors this file next to `model.py` and `api.py` (scripts/publish_space.sh), so it must not
import the data or suite modules at import time.

    ck = Checkpoint("jaredpalmer/kev-4b")          # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import datetime
import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, is_hybrid, load_tokenizer, pad_id

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")
EXPORT_CONFIG, EXPORT_HEAD = "d1a_config.json", "head.safetensors"   # an MLX export folder (loader rule in the module docstring)
EXPORT_FORMAT, EXPORT_VERSION = "d1a-mlx", 1
MLX_ATTENTION_BASES = ("gemma4_text",)   # attention-only text configs the MLX backend runs, besides the hybrid ones


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local run directory as given, or a Hub repo id like jaredpalmer/kev-4b, optionally pinned to a revision or tag
    with `@` (jaredpalmer/kev-4b@qwen3), downloaded to the HF cache. Returns a str path."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
                             ignore_patterns=["media/*"])   # an MLX export's photo and voice encoders (1 GB): d1a.serve fetches them on the first such request


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    weights: str = "lora"          # "lora": an adapter on the base; "full": the whole backbone is in the checkpoint (d1a.train --full_ft); "mlx": an MLX export (never written to a head.pt)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "holdout", "weights")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # known fields win over a stray key in extra


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


def weight_shards(path):
    """The safetensors weight files of a saved backbone (model.safetensors or model-*-of-*.safetensors), sorted."""
    return sorted(Path(path).glob("model*.safetensors"))


def export_config(ck, leading, delimiters, pad, hidden_size, quantization, dtype):
    """The d1a_config.json of an MLX export of checkpoint `ck`: what the loader needs (base, head size, temperature), the
    encoder's token layout the export was made for (checked against the export's tokenizer at load), the weight format,
    and where it came from."""
    meta = ck.meta
    return {"format": EXPORT_FORMAT, "format_version": EXPORT_VERSION, "base": meta.base, "base_revision": meta.base_revision,
            "source": ck.requested, "source_revision": Path(ck.path).name if is_hub_id(ck.requested) else None,   # a Hub snapshot directory is named by its commit
            "adapter_sha256": ck.weights_sha256(), "lora": meta.lora, "head_dim": meta.head_dim, "hidden_size": hidden_size,
            "temperature": meta.temperature, "leading_ids": list(leading), "bos_id": leading[0] if leading else None,
            "delimiter_ids": list(delimiters), "pad_id": pad, "dtype": dtype, "quantization": quantization, "head": EXPORT_HEAD}


def read_export(path):
    """The checked d1a_config.json of an MLX export folder, or None for any other checkpoint."""
    f = Path(path) / EXPORT_CONFIG
    if not f.exists(): return None
    cfg = json.loads(f.read_text(encoding="utf-8"))
    version = cfg.get("format_version")
    if cfg.get("format") != EXPORT_FORMAT or not isinstance(version, int) or not 1 <= version <= EXPORT_VERSION:
        raise ValueError(f"{f}: format {cfg.get('format')!r} version {version!r}; this D1A reads {EXPORT_FORMAT} versions 1-{EXPORT_VERSION}")
    return cfg


def export_meta(path, cfg):
    """A Meta for an MLX export: the fields its d1a_config.json carries and the fp32 head from head.safetensors."""
    from safetensors.torch import load_file
    return Meta(base=cfg["base"], head=load_file(str(Path(path) / cfg["head"])), base_revision=cfg["base_revision"], lora=cfg["lora"],
                head_dim=cfg["head_dim"], temperature=cfg["temperature"], weights="mlx", extra={"export": cfg})


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses; the fields
    are the same knobs the D1A_* environment variables expose to the command-line tools (see from_env).

    dtype        None = fp32, the exact path every reported number uses (bf16 when the checkpoint was trained with a bf16
                 backbone). d1a.serve defaults to bf16 on CUDA and MPS instead: half the memory, 2-4.5x lower latency on an
                 L4 (Kev-4B: 209 -> 118 ms at 101 tokens, 850 -> 189 ms at 330 tokens), probabilities within ~0.01 and
                 the same argmax on the checks run so far. D1A_DTYPE=fp32 restores the exact path when serving.
    merge        fold the LoRA into the base weights: the delta is computed from the fp32 adapter and added in fp32 with
                 one rounding to the load dtype, so a bf16 model holds exactly round(W + delta), the same bits as merging
                 an fp32 copy and casting, without the fp32 copy (Kev-9B needed 36 GB of GPU memory to load for that).
                 Identical because the Qwen bases are stored in bf16; a base stored in fp32 would be rounded twice.
                 Exact in fp32; in bf16 it is faster (~15%) and closer to the fp32 numbers than the unmerged adapter
                 (kev-4b, 24 dev records: max |dp| 0.017 vs 0.029, 0 vs 1 argmax flips). Ignored for adapters that carry
                 trained token embeddings. A checkpoint trained on a bf16 backbone (--weights_dtype bf16: Kev-27B) keeps
                 its adapter unmerged, as it was trained, unless `fused` is asked for (serving); then it is folded the same
                 way (Kev-27B served against unmerged: max |dp| 0.009, 0 flips in 280 questions, runs/fused-27b-h200).
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    backend      None = torch, the path every reported number uses. "mlx" = d1a.mlx_model (Metal kernels through mlx-lm
                 for the hybrid Qwen3.5 and the Gemma 4 backbones; the pointer head and encoder are shared; refused for
                 other attention-only bases). "auto" = mlx when the device is mps, the base is one of those, mlx-lm is
                 installed and fp32 was not asked for (an explicit dtype=float32 means "the exact path"), else torch;
                 d1a.serve uses auto. The MLX path always merges the adapter and ignores `attn` and `dtype` (the backbone
                 runs as stored, bf16). An MLX export (d1a_config.json) runs on mlx whatever None/"auto"/"mlx" says, and
                 refuses "torch".
    cuda_graphs  replay the serving passes of a hybrid backbone on CUDA (state prefix, question rows on a cached state) as
                 CUDA graphs, batched across requests (d1a.cuda_graphs, DecisionModel.probs_batch). None = off, the eager
                 path every reported number uses; d1a.serve turns it on for CUDA. Exact up to floating-point
                 reassociation, not bit for bit (the passes are padded to buckets).
    ple_flash    on the MLX backend, read Gemma 4's per-layer embeddings from the weight files per request instead of holding
                 them in memory (d1a.mlx_model.FlashEmbedding; same values). None = on; False (D1A_PLE_FLASH=0) keeps them in memory.
    fused        rewrite a merged hybrid backbone on CUDA with fused Triton kernels (d1a.fused_qwen35; needs
                 flash-linear-attention fused_qwen35.FLA_VERSION and refuses any other). None = off; d1a.serve turns it on
                 for CUDA when fused_available() (D1A_FUSED=0 to decline). Equal to the reference layers up to bf16 rounding.
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    backend: str | None = None
    cuda_graphs: bool | None = None
    fused: bool | None = None
    ple_flash: bool | None = None

    BACKENDS = (None, "torch", "mlx", "auto")

    @classmethod
    def from_env(cls, env=os.environ):
        """D1A_DTYPE=bf16|fp16|fp32, D1A_MERGE=0, D1A_ATTN=sdpa|eager, D1A_LORA_SCALE, D1A_TEMPERATURE, D1A_BACKEND=torch|mlx|auto,
        D1A_CUDA_GRAPHS=0|1, D1A_FUSED=0|1, D1A_PLE_FLASH=0|1.
        For command-line entry points only; library code passes an explicit LoadOptions. Explicit values that equal a
        library default are kept (fp32 as torch.float32, "torch" as a string) so a caller with its own default, like
        d1a.serve, can tell "asked for it" from "did not say"."""
        backend = env.get("D1A_BACKEND") or None
        if backend not in cls.BACKENDS: raise ValueError(f"D1A_BACKEND must be one of torch, mlx, auto; got {backend!r}")
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(env.get("D1A_DTYPE", "")),
                   merge=env.get("D1A_MERGE", "1") != "0", attn=env.get("D1A_ATTN") or None,
                   lora_scale=float(env.get("D1A_LORA_SCALE", "1")),
                   temperature=float(env["D1A_TEMPERATURE"]) if env.get("D1A_TEMPERATURE") else None, backend=backend,
                   cuda_graphs={"0": False, "1": True}.get(env.get("D1A_CUDA_GRAPHS", "")),
                   fused={"0": False, "1": True}.get(env.get("D1A_FUSED", "")),
                   ple_flash={"0": False, "1": True}.get(env.get("D1A_PLE_FLASH", "")))


def mlx_available():
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def fused_available():
    """Whether d1a.fused_qwen35 can run: flash-linear-attention importable at its FLA_VERSION (d1a.serve's CUDA default;
    not in the serve extra, pinned in the Modal images). An explicit fused=True skips this and fails loudly instead."""
    if importlib.util.find_spec("fla") is None: return False
    try:
        import fla
        from .fused_qwen35 import FLA_VERSION
    except ImportError:
        return False
    return fla.__version__ == FLA_VERSION


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.export = read_export(self.path)         # d1a_config.json of an MLX export, else None
        self.meta = export_meta(self.path, self.export) if self.export else read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    @property
    def full(self):
        """The loader rule (module docstring): True for a full-weight checkpoint, False for a LoRA adapter; head.pt's
        `weights` must agree with the files. An MLX export is neither (its weights are mlx-lm's): False."""
        if self.export is not None: return False
        found = "lora" if self.file("adapter_config.json").exists() else "full" if self.file("config.json").exists() and self.shards() else None
        if found != self.meta.weights:
            raise ValueError(f"{self.path}: head.pt says weights={self.meta.weights!r} but the directory holds "
                             f"{ {'lora': 'an adapter', 'full': 'backbone weights'}.get(found, 'neither an adapter nor backbone weights') }")
        return found == "full"

    def shards(self):
        """The backbone's safetensors files of a full-weight checkpoint (model.safetensors or model-*-of-*.safetensors)."""
        return weight_shards(self.path)

    def weights_sha256(self):
        """What a run's provenance pins: the adapter file's sha256, or for full weights and MLX exports the sha256 over
        every shard's."""
        from .suite import digest   # lazy: the Space vendors this module without d1a/suite.py
        if self.export is None and not self.full: return digest(self.file("adapter_model.safetensors"))
        import hashlib
        return hashlib.sha256("".join(f"{p.name}:{digest(p)}\n" for p in self.shards()).encode()).hexdigest()

    def release_date(self):
        """ISO date for the TypeSafe model card: the Hub commit date for a Hub checkpoint (falls back to the cached file's
        date offline), the time head.pt was written for a local run."""
        if is_hub_id(self.requested):
            from huggingface_hub import HfApi
            repo, _, revision = self.requested.partition("@")
            try:
                return HfApi().model_info(repo, revision=revision or None).last_modified.date().isoformat()
            except Exception:
                pass
        return datetime.date.fromtimestamp(self.file(EXPORT_CONFIG if self.export else "head.pt").stat().st_mtime).isoformat()

    def text_config(self):
        """The base's text config, read without loading weights."""
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(self.meta.base, revision=self.meta.base_revision).get_text_config()

    def hybrid_base(self):
        """Whether the base has Gated DeltaNet layers (Qwen3.5)."""
        return is_hybrid(self.text_config())

    def mlx_base(self):
        """Whether the MLX backend runs this base: the hybrid Qwen3.5 bases and the attention-only MLX_ATTENTION_BASES
        (Gemma 4)."""
        cfg = self.text_config()
        return is_hybrid(cfg) or cfg.model_type in MLX_ATTENTION_BASES

    def backend(self, device, opts=LoadOptions()):
        """The backend `load` will use: LoadOptions.backend resolved ("auto" -> mlx only where it pays and is installed)."""
        if opts.backend not in LoadOptions.BACKENDS: raise ValueError(f"unknown backend {opts.backend!r}")
        if self.export is not None:
            if opts.backend == "torch":
                raise ValueError(f"{self.requested} is an MLX export: it runs on backend=mlx only; load its source checkpoint {self.export['source']} for torch")
            return "mlx"
        if opts.backend != "auto": return opts.backend or "torch"
        exact = opts.dtype is torch.float32   # D1A_DTYPE=fp32: the caller wants the reported-numbers path, not a faster one
        return "mlx" if str(device) == "mps" and not exact and mlx_available() and not self.full and self.mlx_base() else "torch"

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, model) in eval mode with the LoRA applied (or the full backbone loaded) and the pointer head loaded. The model is a
        DecisionModel (torch) or an MLXDecisionModel (backend mlx); both expose the same scoring interface."""
        meta = self.meta
        tok = self._export_tokenizer() if self.export else load_tokenizer(meta.base, revision=meta.base_revision)
        m = self._load_mlx(tok, opts) if self.backend(device, opts) == "mlx" else self._load_torch(tok, device, opts)
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    def _export_tokenizer(self):
        """An MLX export's own tokenizer, checked against the token layout the export was made for."""
        from .model import layout
        tok, cfg = load_tokenizer(self.path), self.export
        leading, delimiters, _ = layout(tok)
        if (list(leading), list(delimiters), pad_id(tok)) != (cfg["leading_ids"], cfg["delimiter_ids"], cfg["pad_id"]):
            raise ValueError(f"{self.path}: the tokenizer encodes leading {leading} / delimiters {delimiters} / pad {pad_id(tok)}, "
                             f"but the export was made for {cfg['leading_ids']} / {cfg['delimiter_ids']} / {cfg['pad_id']}")
        return tok

    def _load_mlx(self, tok, opts):
        if self.full: raise ValueError("the MLX backend merges an adapter into the base; full-weight checkpoints run on backend=torch")
        if not mlx_available(): raise ValueError("the MLX backend needs mlx-lm on Apple Silicon (the `mlx` extra)")
        from .mlx_model import MLXDecisionModel, merge_lora   # after the refusal: without mlx-lm the import would hide it
        if not opts.merge: raise ValueError("the MLX backend always merges the adapter (D1A_MERGE=0 needs backend=torch)")
        if self.meta.option_isolation: raise ValueError("option_isolation needs the packed mask; not available on the MLX backend")
        if self.export is not None:
            if opts.lora_scale != 1: raise ValueError(f"an MLX export holds merged weights; lora_scale needs its source checkpoint {self.export['source']}")
            return MLXDecisionModel(self.path, pad_id(tok), head_dim=self.meta.head_dim, ple_flash=opts.ple_flash is not False)
        if not self.mlx_base(): raise ValueError(f"the MLX backend is for the hybrid (Qwen3.5) and Gemma 4 bases; {self.meta.base} is another attention-only base and runs on MPS with backend=torch")
        base_dir = resolve_run(f"{self.meta.base}@{self.meta.base_revision or ''}")   # the base snapshot the torch path already cached
        m = MLXDecisionModel(base_dir, pad_id(tok), head_dim=self.meta.head_dim, ple_flash=opts.ple_flash is not False)
        merge_lora(m.lm, self.path, opts.lora_scale)
        return m

    def _load_torch(self, tok, device, opts):
        m, merged = self._full_torch(tok, device, opts) if self.full else self._adapted_torch(tok, device, opts)
        serving = str(device).startswith("cuda") and m.hybrid
        if opts.fused and serving and merged:   # fused projections need plain (merged or full) weights
            from .fused_qwen35 import fuse
            fuse(m.lm)
        if opts.cuda_graphs and serving:
            from .cuda_graphs import CudaGraphs
            m.graphs = CudaGraphs(m.lm, m.pad_id)
        return m

    SAVED_DTYPES = {"bf16": "bfloat16", "fp32": "float32"}   # head.pt weights_dtype -> the dtype save_pretrained writes to config.json

    def _full_torch(self, tok, device, opts):
        """-> (model, True). Full weights load in the dtype head.pt's `weights_dtype` names (bf16 for every d1a.train
        --full_ft run: the dtype they were trained in), which must be the dtype save_pretrained recorded in config.json;
        otherwise a mislabelled export would be silently cast (fp32 weights rounded to bf16, or bf16 upcast to twice the
        memory). An explicit dtype still casts on purpose (fp32: the same values computed in fp32). Nothing to merge."""
        if opts.lora_scale != 1: raise ValueError("lora_scale interpolates an adapter; a full-weight checkpoint has none")
        meta = self.meta
        cfg = json.loads(self.file("config.json").read_text(encoding="utf-8"))
        expected, saved = self.SAVED_DTYPES.get(meta.weights_dtype), cfg.get("dtype") or cfg.get("torch_dtype")
        if expected is None or saved not in (None, expected):
            raise ValueError(f"{self.path}: config.json records the weights as {saved} but head.pt says weights_dtype={meta.weights_dtype!r}")
        return DecisionModel(meta.base, tok, device, head_dim=meta.head_dim, option_isolation=meta.option_isolation,
                             dtype=opts.dtype or getattr(torch, expected), attn=opts.attn, weights=self.path), True

    def _adapted_torch(self, tok, device, opts):
        """-> (model, whether the adapter was merged): the base with this checkpoint's LoRA."""
        from peft import PeftModel
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # trained with a bf16 backbone (--weights_dtype bf16: Kev-27B, the 35B-A3B MoE whose fused experts need bf16):
            # load it the same way. The exact path keeps the fp32 adapter unmerged; the fused serving path folds it in
            # (one rounding of W + delta, as for every served Kev; parity in runs/serving-27b-*).
            dtype, merge = torch.bfloat16, merge and bool(opts.fused)
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # token-trained adapters stay unmerged
        m = DecisionModel(meta.base, tok, device, lora=None, revision=meta.base_revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, dtype=dtype, attn=opts.attn)
        m.lm = PeftModel.from_pretrained(m.lm, self.path, torch_device=str(device)).to(device)   # trainable token embeddings, if any, live in the adapter
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge: m.lm = m.lm.merge_and_unload()     # W += delta: fp32 math, one rounding (see LoadOptions.merge)
        if dtype != torch.float32: m.lm = m.lm.to(dtype)
        return m, merge

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights")

    def warm_start(self, model, ours):
        """Delta training: load this checkpoint's weights (adapter, or full backbone) and pointer head into `model` (a fresh
        DecisionModel built the way `ours` says: with LoRA, or for full-weight training without). `ours` is the Meta the new
        run will save; every architecture field is compared BEFORE loading, because peft and load_state_dict(strict=False)
        load matching keys silently and a half-loaded model still trains and still reports a loss. Returns provenance."""
        from .suite import digest   # lazy: the Space vendors this module without d1a/suite.py
        if self.export is not None: raise ValueError(f"--init_from {self.path} is an MLX export; start from its source checkpoint {self.export['source']}")
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        tensors = self._load_backbone_into(model.lm) if self.full else self._load_adapter_into(model.lm)
        model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "weights_sha256": self.weights_sha256(),
                "head_sha256": digest(self.file("head.pt")), "tensors": tensors}

    def _load_backbone_into(self, lm):
        """Copy the saved backbone over `lm` shard by shard (the base's copy in memory is replaced, never merged); every
        tensor of `lm` must be covered exactly once. -> tensor count."""
        from safetensors.torch import load_file
        have, seen = set(lm.state_dict()), set()
        for shard in self.shards():
            part = load_file(shard)
            unexpected = sorted(set(part) - have)
            if unexpected: raise ValueError(f"--init_from {self.path}: {shard.name} carries tensors this backbone does not have (e.g. {unexpected[:2]})")
            lm.load_state_dict(part, strict=False); seen |= set(part)
        missing = sorted(have - seen)
        if missing: raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this backbone's tensors (e.g. {missing[:2]})")
        return len(seen)

    def _load_adapter_into(self, lm):
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        weights = load_peft_weights(self.path, device="cpu")
        have = set(get_peft_model_state_dict(lm))
        unexpected, missing = sorted(set(weights) - have), sorted(have - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(lm, weights)
        return len(weights)


def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
