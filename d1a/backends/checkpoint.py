"""Trained checkpoints, and the one place that turns one into a model.

Two layouts, told apart by their files:
- a training run (a local directory or a Hub repo, `owner/name@revision`): a LoRA adapter (adapter_config.json,
  adapter_model.safetensors) for the base its metadata names (`meta.base` at `meta.base_revision`, whose tokenizer it
  uses), the pointer head and the run's metadata (Meta). A run is saved as d1a_config.json (format "d1a-torch",
  versioned; the metadata, read without unpickling anything) with head.safetensors (the head), and, for D1A 0.4 only,
  also as the head.pt older versions read; runs saved before 0.4 have head.pt alone, which stays readable;
- an MLX export (d1a_config.json, written by scripts/export_mlx.py): the adapter already merged into the base and saved
  by mlx-lm (config.json + model*.safetensors, perhaps quantized), the head in fp32 in head.safetensors and the
  tokenizer. It needs neither the base nor the adapter and runs on the MLX backend only.

d1a.serving.serve, d1a.eval.benchmark, d1a.training.train --init_from and the scripts all load through Checkpoint. The Hugging Face Space
vendors this file next to model.py and api.py, so it imports nothing from the data or suite modules at import time.

    ck = Checkpoint("JohnP1/d1a-e2b@v0.4")          # or a local run directory
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
    ck.meta.use_case_temperatures                   # {use case: temperature} a request may select instead (#209)
"""
import datetime
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from d1a.backends.backbone import for_config
from d1a.backends.torch import DecisionModel
from d1a.core.encoding import load_tokenizer, pad_id

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")
EXPORT_CONFIG, EXPORT_HEAD = "d1a_config.json", "head.safetensors"   # an MLX export folder's two D1A files, and a training run's
EXPORT_FORMAT, EXPORT_VERSION = "d1a-mlx", 1
TORCH_FORMAT, TORCH_VERSION = "d1a-torch", 1
HEAD_PT_LAST_WRITTEN = "0.4"   # the last release that also writes head.pt (CHANGELOG, Deprecated); reading it has no end
USE_CASE_TEMPERATURES, USE_CASE_FITS = "use_case_temperatures", "use_case_temperature_fits"   # Meta.extra keys (#209)
CALIBRATION = ("temperature_fit", USE_CASE_TEMPERATURES, USE_CASE_FITS)   # the extra fields an MLX export carries over


# --- where a checkpoint is ----------------------------------------------------------------------------------------------

def is_hub_id(run):
    """Whether `run` names a Hub repo ("owner/name", optionally "@revision") rather than a local directory."""
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """A local directory as given, or a Hub repo ("owner/name@revision", the revision optional) downloaded to the HF cache.
    -> the path as a str. An MLX export's photo and voice encoders (media/, ~1 GB) are left for d1a.serving.serve to fetch on the
    first such request."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
                             ignore_patterns=["media/*"])


# --- what a checkpoint says about itself --------------------------------------------------------------------------------

@dataclass
class Meta:
    """A run's metadata and head: d1a_config.json's fields (or head.pt's, for older runs). Fields older checkpoints did not
    write read as these defaults everywhere; `extra` keeps the rest (training arguments, suite hash, init provenance, the
    temperature fit), so reading and rewriting it loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False      # Kev's isolated options, still read so such a checkpoint can be refused
    special_embeddings: bool = False    # an adapter that trained the delimiter embeddings: loads (they live in the adapter)
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    weights: str = "lora"               # "lora": an adapter on the base; "mlx": an MLX export (never written to a head.pt)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "holdout", "weights")

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.KNOWN if k in d}
        return cls(**known, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # a known field wins over a stray key in extra

    @property
    def use_case_temperatures(self):
        """{use case: temperature}, written by d1a.training.calibrate --use-case: a request naming one of these use cases
        (SystemOneRequest.use_case) is served at its temperature instead of `temperature`; empty for most checkpoints. Kept
        in `extra` with its fits (USE_CASE_FITS), so a D1A that predates it keeps it when it rewrites the run."""
        return checked_use_case_temperatures(self.extra.get(USE_CASE_TEMPERATURES, {}))


def checked_use_case_temperatures(table):
    """`table` as {non-empty name: finite positive float}, names stripped (" routing " would never match a request), or
    ValueError naming what is wrong (a bad entry would otherwise fail or skew one use case's answers only, long after the
    checkpoint loaded)."""
    if not isinstance(table, dict):
        raise ValueError(f"{USE_CASE_TEMPERATURES} must be an object of use case -> temperature, not {table!r}")
    out = {}
    for name, t in table.items():
        if not (isinstance(name, str) and name.strip()) or isinstance(t, bool) or not isinstance(t, (int, float)) or not (math.isfinite(t) and t > 0):
            raise ValueError(f"{USE_CASE_TEMPERATURES}: {name!r} -> {t!r}; each entry must name a use case and a finite positive temperature")
        if name.strip() in out:
            raise ValueError(f"{USE_CASE_TEMPERATURES}: {name!r} names {name.strip()!r} twice")
        out[name.strip()] = float(t)
    return out


def _read_head_pt(run):
    # weights_only: head.pt holds tensors and plain data, and may come from any Hub repo, so nothing in it is ever executed
    # (torch >= 2.6 defaults to this; explicit, it also holds under TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu", weights_only=True))


def read_torch_config(run):
    """A training run's checked d1a_config.json, or None when it has none (a run saved before D1A 0.4) or is an MLX export."""
    path = Path(run) / EXPORT_CONFIG
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("format") == EXPORT_FORMAT:
        return None
    version = cfg.get("format_version")
    if cfg.get("format") != TORCH_FORMAT or not isinstance(version, int) or not 1 <= version <= TORCH_VERSION:
        raise ValueError(f"{path}: format {cfg.get('format')!r} version {version!r}; this D1A reads {TORCH_FORMAT} versions 1-{TORCH_VERSION} and {EXPORT_FORMAT}")
    return cfg


def read_meta(run):
    """A training run's Meta: from d1a_config.json and head.safetensors when it has them, else from head.pt. A run that has
    both must say the same in both (an older D1A recalibrating it rewrote head.pt alone, for example)."""
    cfg = read_torch_config(run)
    if cfg is None:
        return _read_head_pt(run)
    from safetensors.torch import load_file
    head = load_file(str(Path(run) / cfg["head"])) if cfg.get("head") else None   # None: a run saved without a head (metadata only)
    meta = Meta.from_dict({**{k: cfg[k] for k in Meta.KNOWN if k in cfg and k != "head"}, **cfg.get("extra", {}), "head": head})
    if (Path(run) / "head.pt").exists():
        old = _read_head_pt(run)
        for name in Meta.KNOWN:
            a, b = getattr(meta, name), getattr(old, name)
            same = (a is b is None or a is not None and b is not None and a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)) if name == "head" else a == b
            if not same:
                raise ValueError(f"{run}: {name} differs between {EXPORT_CONFIG} and head.pt" + ("" if name == "head" else f" ({a!r} and {b!r})")
                                 + f"; {EXPORT_CONFIG} is the one D1A writes, so rewrite the run with d1a.backends.checkpoint.write_meta, or remove the stale file")
    return meta


def _json_safe(value, where):
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str): raise ValueError(f"{where}: key {k!r} is not a string; {EXPORT_CONFIG} holds JSON only")
            _json_safe(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): _json_safe(v, f"{where}[{i}]")
    elif not (value is None or isinstance(value, (str, bool, int, float))):
        raise ValueError(f"{where} is a {type(value).__name__}; {EXPORT_CONFIG} holds JSON only, so convert it before saving")


def torch_config(meta):
    """The d1a_config.json of a training run (format d1a-torch): every known field, the head's file, and `extra`."""
    if meta.weights != "lora":
        raise ValueError(f"weights {meta.weights!r}: D1A saves LoRA runs only (full-weight training was removed, docs/removed-tools.md)")
    _json_safe(meta.extra, "extra")
    known = {k: getattr(meta, k) for k in Meta.KNOWN if k != "head"}
    _json_safe(known, "meta")
    return {"format": TORCH_FORMAT, "format_version": TORCH_VERSION, **known, "head": EXPORT_HEAD if meta.head is not None else None, "extra": meta.extra}


def write_meta(run, meta):
    """Save a run's metadata and head: d1a_config.json and head.safetensors, and head.pt for D1A <= 0.3 readers (written
    through HEAD_PT_LAST_WRITTEN). Everything is checked before any file is written."""
    from safetensors.torch import save_file
    cfg = torch_config(meta)
    if meta.head is not None:
        save_file({k: v.contiguous() for k, v in meta.head.items()}, str(Path(run) / EXPORT_HEAD))
    (Path(run) / EXPORT_CONFIG).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    torch.save(meta.to_dict(), f"{run}/head.pt")


def weight_shards(path):
    """A saved backbone's safetensors files (model.safetensors, or model-*-of-*.safetensors), sorted."""
    return sorted(Path(path).glob("model*.safetensors"))


def export_config(ck, leading, delimiters, pad, hidden_size, quantization, dtype):
    """The d1a_config.json of an MLX export of checkpoint `ck`: what loading needs (the base, the head's size, the
    temperature, the use-case temperatures, and where they were fitted), the token layout the export was made for
    (checked against its tokenizer when it loads), the weight format, and where it came from."""
    meta = ck.meta
    source_revision = Path(ck.path).name if is_hub_id(ck.requested) else None   # a Hub snapshot directory is named by its commit
    return {"format": EXPORT_FORMAT, "format_version": EXPORT_VERSION, "base": meta.base, "base_revision": meta.base_revision,
            "source": ck.requested, "source_revision": source_revision,
            "adapter_sha256": ck.weights_sha256(), "lora": meta.lora, "head_dim": meta.head_dim, "hidden_size": hidden_size,
            "temperature": meta.temperature, "leading_ids": list(leading), "bos_id": leading[0] if leading else None,
            "delimiter_ids": list(delimiters), "pad_id": pad, "dtype": dtype, "quantization": quantization, "head": EXPORT_HEAD,
            "temperature_fit": meta.extra.get("temperature_fit"), USE_CASE_TEMPERATURES: meta.use_case_temperatures,
            USE_CASE_FITS: meta.extra.get(USE_CASE_FITS, {})}


def read_export(path):
    """An MLX export folder's checked d1a_config.json, or None when the folder is not an export."""
    config = Path(path) / EXPORT_CONFIG
    if not config.exists():
        return None
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if cfg.get("format") == TORCH_FORMAT:   # a training run's metadata (read_torch_config), not an export
        return None
    version = cfg.get("format_version")
    if cfg.get("format") != EXPORT_FORMAT or not isinstance(version, int) or not 1 <= version <= EXPORT_VERSION:
        raise ValueError(f"{config}: format {cfg.get('format')!r} version {version!r}; this D1A reads {EXPORT_FORMAT} versions 1-{EXPORT_VERSION}")
    return cfg


def export_meta(path, cfg):
    """An MLX export's Meta: what its d1a_config.json says, and the fp32 head from head.safetensors. The calibration
    records (CALIBRATION) sit in `extra` where a training run keeps them; an export written before they were carried has none."""
    from safetensors.torch import load_file
    return Meta(base=cfg["base"], head=load_file(str(Path(path) / cfg["head"])), base_revision=cfg["base_revision"], lora=cfg["lora"],
                head_dim=cfg["head_dim"], temperature=cfg["temperature"], weights="mlx",
                extra={"export": cfg, **{k: cfg[k] for k in CALIBRATION if cfg.get(k)}})


# --- how to load it -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint becomes a model. The defaults are the exact path every reported number uses; the fields are the
    knobs the D1A_* environment variables set for the command-line tools (from_env).

    dtype        None: fp32, the exact path (bf16 when the checkpoint was trained on a bf16 backbone). d1a.serving.serve defaults
                 to bf16 on CUDA and MPS: half the memory and 2-4.5x lower latency on an L4 (Kev-4B: 209 -> 118 ms at
                 101 tokens, 850 -> 189 ms at 330), probabilities within ~0.01 with the same argmax so far; D1A_DTYPE=fp32
                 restores the exact path.
    merge        fold the LoRA into the base weights. The delta is computed from the fp32 adapter and added in fp32, then
                 rounded once to the load dtype, so a bf16 model holds exactly round(W + delta): the bits of merging an
                 fp32 copy and casting, without the fp32 copy (Kev-9B needed 36 GB of GPU memory to load for that). Exact
                 because the Qwen bases are stored in bf16 (a base stored in fp32 would be rounded twice). Exact in fp32;
                 in bf16 faster (~15%) and closer to fp32 than the unmerged adapter (kev-4b, 24 dev records: max |dp|
                 0.017 vs 0.029, 0 vs 1 argmax flips). Ignored for adapters with trained token embeddings. A checkpoint
                 trained on a bf16 backbone (--weights_dtype bf16: Kev-27B) stays unmerged, as it was trained, unless
                 `fused` is asked for (serving); then it is folded the same way (Kev-27B against unmerged: max |dp| 0.009,
                 0 flips in 280 questions, runs/fused-27b-h200).
    attn         the attention backend; None: the model's default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS
                 matched eager and is a few percent faster.
    lora_scale   WiSE-FT interpolation at inference between the base (0) and the fine-tuned weights (1).
    temperature  None: the checkpoint's own (fitted by d1a.training.calibrate), and its use-case temperatures for the requests
                 that name a use case; a value (1.0: raw logits) serves every request at it, use cases included.
    backend      None: torch, the path every reported number uses. "mlx": d1a.backends.mlx (Metal kernels through mlx-lm for
                 the hybrid Qwen3.5 and the Gemma 4 backbones; the pointer head and encoder are shared; refused for other
                 attention-only bases). "auto": mlx when the device is mps, the base is one of those, mlx-lm is installed
                 and fp32 was not asked for (an explicit dtype=float32 means the exact path), else torch; d1a.serving.serve uses
                 auto. The MLX path always merges the adapter and ignores `attn` and `dtype` (the backbone runs as stored,
                 bf16). An MLX export runs on mlx whatever None, "auto" or "mlx" says, and refuses "torch".
    cuda_graphs  replay a hybrid backbone's serving passes on CUDA (the state prefix, the question rows on a cached state)
                 as CUDA graphs, batched across requests (d1a.backends.cuda_graphs, DecisionModel.probs_batch). None: off, the eager
                 path; d1a.serving.serve turns it on for CUDA. Exact up to floating-point reassociation (passes are padded to buckets).
    ple_flash    on the MLX backend, read Gemma 4's per-layer embeddings from the weight files per request instead of
                 holding them in memory (d1a.backends.mlx.FlashEmbedding; the same values). None: on; False (D1A_PLE_FLASH=0)
                 keeps them in memory.
    fused        rewrite a merged hybrid backbone on CUDA with fused Triton kernels (d1a.backends.fused_qwen35; needs
                 flash-linear-attention at fused_qwen35.FLA_VERSION and refuses any other). None: off; d1a.serving.serve turns it
                 on for CUDA when fused_available() (D1A_FUSED=0 declines). Equal to the reference layers up to bf16 rounding.
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
    DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    SWITCH = {"0": False, "1": True}

    @classmethod
    def from_env(cls, env=os.environ):
        """D1A_DTYPE=bf16|fp16|fp32, D1A_MERGE=0, D1A_ATTN=sdpa|eager, D1A_LORA_SCALE, D1A_TEMPERATURE, D1A_BACKEND=torch|mlx|auto,
        D1A_CUDA_GRAPHS=0|1, D1A_FUSED=0|1, D1A_PLE_FLASH=0|1. For the command-line entry points only (library code passes
        its own LoadOptions). An explicit value equal to a library default is kept as given (fp32 as torch.float32,
        "torch" as a string), so a caller with defaults of its own, like d1a.serving.serve, can tell "asked for it" from "said nothing"."""
        backend = env.get("D1A_BACKEND") or None
        if backend not in cls.BACKENDS:
            raise ValueError(f"D1A_BACKEND must be one of torch, mlx, auto; got {backend!r}")
        return cls(dtype=cls.DTYPES.get(env.get("D1A_DTYPE", "")), merge=env.get("D1A_MERGE", "1") != "0", attn=env.get("D1A_ATTN") or None,
                   lora_scale=float(env.get("D1A_LORA_SCALE", "1")),
                   temperature=float(env["D1A_TEMPERATURE"]) if env.get("D1A_TEMPERATURE") else None, backend=backend,
                   cuda_graphs=cls.SWITCH.get(env.get("D1A_CUDA_GRAPHS", "")), fused=cls.SWITCH.get(env.get("D1A_FUSED", "")),
                   ple_flash=cls.SWITCH.get(env.get("D1A_PLE_FLASH", "")))


def mlx_available():
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def fused_available():
    """Whether d1a.backends.fused_qwen35 can run: flash-linear-attention importable at its FLA_VERSION (d1a.serving.serve's CUDA default;
    not in the serve extra). An explicit fused=True skips this check and fails loudly instead."""
    if importlib.util.find_spec("fla") is None:
        return False
    try:
        import fla
        from d1a.backends.fused_qwen35 import FLA_VERSION
    except ImportError:
        return False
    return fla.__version__ == FLA_VERSION


# --- the checkpoint -----------------------------------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # as the caller named it (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.export = read_export(self.path)         # an MLX export's d1a_config.json, else None
        self.meta = export_meta(self.path, self.export) if self.export else read_meta(self.path)
        if self.meta.option_isolation:   # Kev's isolated option spans: encode() no longer builds them, so its answers would be wrong
            raise ValueError(f"{self.requested} was trained with option_isolation, which D1A no longer supports (docs/UPSTREAM.md)")
        if self.meta.weights == "full":   # Kev's full-weight runs (d1a.training.train --full_ft, removed)
            raise ValueError(f"{self.requested} is a full-weight checkpoint, which D1A no longer loads (docs/removed-tools.md)")
        try: self.meta.use_case_temperatures   # checked now, so a bad entry fails the load, not one use case's requests
        except ValueError as e: raise ValueError(f"{self.requested}: {e}") from None

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def shards(self):
        """An MLX export's weight files."""
        return weight_shards(self.path)

    def weights_sha256(self):
        """What a run's provenance pins: the adapter file's sha256, or for an MLX export one sha256 over every shard's."""
        from d1a.eval.suite import digest   # lazy: the Space vendors this module without d1a/eval/suite.py
        if self.export is None:
            return digest(self.file("adapter_model.safetensors"))
        import hashlib
        return hashlib.sha256("".join(f"{shard.name}:{digest(shard)}\n" for shard in self.shards()).encode()).hexdigest()

    def release_date(self):
        """The ISO date on the TypeSafe model card: the Hub commit date for a Hub checkpoint (the cached file's date when
        offline), else the date its d1a_config.json (or a run's head.pt, before D1A 0.4) was written."""
        if is_hub_id(self.requested):
            from huggingface_hub import HfApi
            repo, _, revision = self.requested.partition("@")
            try:
                return HfApi().model_info(repo, revision=revision or None).last_modified.date().isoformat()
            except Exception:
                pass
        written = self.file(EXPORT_CONFIG if self.file(EXPORT_CONFIG).exists() else "head.pt").stat().st_mtime
        return datetime.date.fromtimestamp(written).isoformat()

    def text_config(self):
        """The base's text config, read without loading weights."""
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(self.meta.base, revision=self.meta.base_revision).get_text_config()

    def hybrid_base(self):
        """Whether the base has Gated DeltaNet layers (Qwen3.5)."""
        return for_config(self.text_config()).hybrid

    def mlx_base(self):
        """Whether the MLX backend runs this base (d1a.backends.backbone: Gemma 4 and the hybrid Qwen3.5 bases)."""
        return for_config(self.text_config()).mlx

    def backend(self, device, opts=LoadOptions()):
        """The backend load() uses: LoadOptions.backend resolved ("auto" picks mlx only where it pays and is installed)."""
        if opts.backend not in LoadOptions.BACKENDS:
            raise ValueError(f"unknown backend {opts.backend!r}")
        if self.export is not None:
            if opts.backend == "torch":
                raise ValueError(f"{self.requested} is an MLX export: it runs on backend=mlx only; load its source checkpoint {self.export['source']} for torch")
            return "mlx"
        if opts.backend != "auto":
            return opts.backend or "torch"
        exact = opts.dtype is torch.float32   # D1A_DTYPE=fp32: the caller wants the reported-numbers path, not a faster one
        return "mlx" if str(device) == "mps" and not exact and mlx_available() and self.mlx_base() else "torch"

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, model): the model in eval mode with the adapter applied and the pointer head and temperature
        loaded. A DecisionModel (torch) or an MLXDecisionModel (mlx), which share one scoring interface."""
        meta = self.meta
        tok = self._export_tokenizer() if self.export else load_tokenizer(meta.base, revision=meta.base_revision)
        model = self._load_mlx(tok, opts) if self.backend(device, opts) == "mlx" else self._load_torch(tok, device, opts)
        model.head.load_state_dict(meta.head)
        model.eval()
        model.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        model.head.use_case_temperatures = meta.use_case_temperatures if opts.temperature is None else {}
        return tok, model

    def _export_tokenizer(self):
        """An MLX export's own tokenizer, checked against the token layout the export was made for."""
        from d1a.core.encoding import layout
        tok, cfg = load_tokenizer(self.path), self.export
        leading, delimiters, _ = layout(tok)
        if (list(leading), list(delimiters), pad_id(tok)) != (cfg["leading_ids"], cfg["delimiter_ids"], cfg["pad_id"]):
            raise ValueError(f"{self.path}: the tokenizer encodes leading {leading} / delimiters {delimiters} / pad {pad_id(tok)}, "
                             f"but the export was made for {cfg['leading_ids']} / {cfg['delimiter_ids']} / {cfg['pad_id']}")
        return tok

    def _load_mlx(self, tok, opts):
        if not mlx_available():
            raise ValueError("the MLX backend needs mlx-lm on Apple Silicon (the `mlx` extra)")
        from d1a.backends.mlx import MLXDecisionModel, merge_lora   # after the refusal: without mlx-lm this import would hide it
        if not opts.merge:
            raise ValueError("the MLX backend always merges the adapter (D1A_MERGE=0 needs backend=torch)")
        flash = opts.ple_flash is not False
        if self.export is not None:
            if opts.lora_scale != 1:
                raise ValueError(f"an MLX export holds merged weights; lora_scale needs its source checkpoint {self.export['source']}")
            return MLXDecisionModel(self.path, pad_id(tok), head_dim=self.meta.head_dim, ple_flash=flash)
        if not self.mlx_base():
            raise ValueError(f"the MLX backend is for the hybrid (Qwen3.5) and Gemma 4 bases; {self.meta.base} is another attention-only base and runs on MPS with backend=torch")
        base_dir = resolve_run(f"{self.meta.base}@{self.meta.base_revision or ''}")   # the snapshot the torch path already cached
        model = MLXDecisionModel(base_dir, pad_id(tok), head_dim=self.meta.head_dim, ple_flash=flash)
        merge_lora(model.lm, self.path, opts.lora_scale)
        return model

    def _load_torch(self, tok, device, opts):
        model, merged = self._adapted_torch(tok, device, opts)
        if str(device).startswith("cuda"):
            model.backbone.serve_cuda(model, opts, merged)   # Qwen3.5: fused kernels and CUDA graphs
        return model

    def _adapted_torch(self, tok, device, opts):
        """-> (the base with this checkpoint's adapter, whether the adapter was merged into it)."""
        from peft import PeftModel
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # trained on a bf16 backbone (--weights_dtype bf16: Kev-27B, the 35B-A3B MoE whose fused experts need bf16):
            # loaded the same way. The exact path keeps the fp32 adapter unmerged; the fused serving path folds it in
            # (one rounding of W + delta, as for every served Kev; parity in runs/serving-27b-*).
            dtype, merge = torch.bfloat16, merge and bool(opts.fused)
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # an adapter that trained token embeddings stays unmerged
        model = DecisionModel(meta.base, tok, device, lora=None, revision=meta.base_revision, head_dim=meta.head_dim, dtype=dtype, attn=opts.attn)
        model.lm = PeftModel.from_pretrained(model.lm, self.path, torch_device=str(device)).to(device)   # trained token embeddings, if any, are in the adapter
        if opts.lora_scale != 1:
            for module in model.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for adapter in module.scaling:
                        module.scaling[adapter] *= opts.lora_scale
            model.lora_scale = opts.lora_scale
        if merge:
            model.lm = model.lm.merge_and_unload()   # W += delta in fp32, one rounding (LoadOptions.merge)
        if dtype != torch.float32:
            model.lm = model.lm.to(dtype)
        return model, merge

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights")

    def warm_start(self, model, ours):
        """Delta training (d1a.training.train --init_from): this checkpoint's adapter and pointer head into `model`, a fresh
        DecisionModel with LoRA built as `ours` (the Meta the new run will save) says. Every architecture field is compared
        first, because peft and load_state_dict(strict=False) load matching keys silently, and a half-loaded model still
        trains and reports a loss. -> provenance."""
        from d1a.eval.suite import digest   # lazy: the Space vendors this module without d1a/eval/suite.py
        if self.export is not None:
            raise ValueError(f"--init_from {self.path} is an MLX export; start from its source checkpoint {self.export['source']}")
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        tensors = self._load_adapter_into(model.lm)
        model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "weights_sha256": self.weights_sha256(),
                "head_sha256": digest(self.file("head.pt") if self.file("head.pt").exists() else self.file(EXPORT_HEAD)), "tensors": tensors}

    def _load_adapter_into(self, lm):
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        weights = load_peft_weights(self.path, device="cpu")
        expected = set(get_peft_model_state_dict(lm))
        unexpected, missing = sorted(set(weights) - expected), sorted(expected - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(lm, weights)
        return len(weights)


def load(run, device, opts=LoadOptions()):
    """Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
