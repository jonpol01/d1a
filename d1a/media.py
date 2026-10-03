"""Decisions about a photo, a voice clip or a video, for the playground's Photo check and Voice triage demos.

POST /v1/systemone/media takes a /v1/systemone request plus {"media": {"type": "image" | "audio" | "video", "data": <base64>}} and
answers in the same shape. The checkpoint is the text-trained one: Gemma 4 keeps its vision and audio encoders, and the
media's soft tokens go between <state> and the state text. Zero-shot (issue #78) it read damage and drop-off place off
delivery photos and intent off EN/JA voice notes, no speech-to-text.

Two ways to serve it:
- d1a.serve, the same model as every other request (MediaEncoder + with_media): the encoders alone (~1 GB for E4B, the
  media/ folder of an MLX export, export_media) turn the media into soft tokens, and the text model, MLX included, reads
  them in place of its placeholder tokens. Nothing else is loaded, and --idle-unload frees both.
- python -m d1a.media, for a PyTorch checkpoint (MediaModel): the full multimodal base (bf16, ~10 GB for E2B) with the
  adapter merged, in its own process. The media is encoded once per request; each question then runs as its own row on a
  copy of that prefix cache (rows_of: the same tokens and positions the packed form gives a question).
    uv run --extra serve --extra media python -m d1a.media --run JohnP1/d1a-e2b --port 8010
  It starts without the model, loads it on the first request (about 20-30 s) and drops it again after --idle-unload
  seconds without one (default 600; 0 keeps it loaded), so an idle server holds almost no memory.
"""
import argparse, base64, copy, gc, io, sys, threading, time
from contextlib import contextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from .api import SystemOneRequest, output_tokens, to_answers, to_record
from .checkpoint import Checkpoint
from .device import default_device, empty_cache
from .model import PointerHead, encode, layout, rows_of

SAMPLE_RATE = 16_000             # what Gemma 4's audio feature extractor expects
MAX_AUDIO_S = 30                 # Gemma 4's audio encoder limit
MAX_MEDIA_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MAX_VIDEO_FRAMES = 16            # sampled evenly over a clip, ~63 tokens each (Gemma 4's processor default is 32)


class Media(BaseModel):
    type: Literal["image", "audio", "video"]
    data: str = Field(description="base64 of the file: an image PIL reads (JPEG, PNG, WebP), a WAV/FLAC/OGG clip, or a video PyAV reads (MP4, MOV, WebM)")


class MediaRequest(SystemOneRequest):
    state: str | dict | list | None = None   # optional text next to the media (a driver's note, a caption)
    media: Media


def decode_audio(raw):
    """-> mono float32 samples at SAMPLE_RATE. Linear resampling: these clips are speech, and the feature extractor
    computes its own mel bins."""
    import numpy as np, soundfile as sf
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = round(len(wav) * SAMPLE_RATE / sr)
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
    if len(wav) > MAX_AUDIO_S * SAMPLE_RATE: raise ValueError(f"audio longer than {MAX_AUDIO_S} s")
    return wav


def decode_video(raw, num_frames=MAX_VIDEO_FRAMES):
    """-> (frames [n, H, W, 3] uint8, transformers VideoMetadata), at most num_frames sampled evenly over the clip. The
    frames only: Gemma 4 reads a video as timestamped images, and a clip's sound track is not used."""
    import tempfile
    import numpy as np
    from transformers.video_utils import load_video
    pick = lambda metadata, **kw: np.linspace(0, metadata.total_num_frames - 1, min(num_frames, metadata.total_num_frames), dtype=int)
    with tempfile.NamedTemporaryFile(suffix=".video") as f:   # PyAV opens files, not bytes
        f.write(raw); f.flush()
        try: return load_video(f.name, backend="pyav", sample_indices_fn=pick)
        except Exception as e: raise ValueError(f"cannot decode the video: {e}") from e


def processor_inputs(proc, media):
    """The processor's output for one photo, voice clip or video: its token ids (a placeholder per soft token, Gemma 4's
    timestamps between a video's frames) and the encoder inputs."""
    raw = base64.b64decode(media.data, validate=True)
    limit = MAX_VIDEO_BYTES if media.type == "video" else MAX_MEDIA_BYTES
    if len(raw) > limit: raise ValueError(f"{media.type} larger than {limit // 2**20} MB")
    if media.type == "image":
        return proc(text=proc.image_token, images=decode_image(raw), return_tensors="pt")
    if media.type == "audio":
        return proc(text=proc.audio_token, audio=decode_audio(raw), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    frames, meta = decode_video(raw)
    return proc(text=proc.video_token, videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")


def decode_image(raw):
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


class MediaModel:
    """A LoRA checkpoint on a Gemma 4 base, loaded with the base's vision and audio encoders."""

    def __init__(self, run, device, dtype=torch.bfloat16):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.checkpoint = ck = Checkpoint(run); meta = ck.meta
        if ck.export is not None or meta.weights != "lora":
            raise SystemExit(f"{run}: the media server needs a LoRA checkpoint (torch), not an MLX export or full weights")
        if meta.option_isolation: raise SystemExit(f"{run}: option_isolation needs the packed mask; the media rows are plain causal rows")
        self.device = device
        self.proc = AutoProcessor.from_pretrained(meta.base, revision=meta.base_revision)
        self.tok = self.proc.tokenizer
        # straight onto the device: loading on the CPU and moving it holds two copies at once (15 GB peak for E2B on a Mac)
        full = AutoModelForImageTextToText.from_pretrained(meta.base, revision=meta.base_revision, dtype=dtype, attn_implementation="sdpa", device_map=device)
        inner = full.model
        if not (hasattr(inner, "vision_tower") and hasattr(inner, "audio_tower")):
            raise SystemExit(f"{meta.base} has no vision and audio encoders; the media server is for Gemma 4 bases")
        inner.language_model = PeftModel.from_pretrained(inner.language_model, ck.path).merge_and_unload()
        self.model = inner.eval()
        self.head = PointerHead(full.config.get_text_config().hidden_size, dp=meta.head_dim).to(device).eval()
        self.head.load_state_dict(meta.head)
        self.head.temperature = meta.temperature
        self.lock = threading.Lock()   # one forward at a time on the device

    def media_inputs(self, media: Media):
        """-> (media token ids, mm_token_type_ids for them, the encoder inputs)."""
        mm = processor_inputs(self.proc, media)
        leading = layout(self.tok)[0]
        ids, types = mm["input_ids"][0].tolist(), mm["mm_token_type_ids"][0].tolist()
        if ids[:len(leading)] == leading: ids, types = ids[len(leading):], types[len(leading):]
        extra = {k: v.to(self.device) for k, v in mm.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")}
        return ids, types, extra

    @torch.no_grad()
    def probs(self, rec, media: Media):
        """-> (probabilities per question, input token count) for one record whose state sits after the media."""
        media_ids, media_types, extra = self.media_inputs(media)
        state_ids, _, rows = rows_of(encode(self.tok, rec))
        n_head = len(layout(self.tok)[0]) + 1                  # leading ids + <state>
        prefix = state_ids[:n_head] + media_ids + state_ids[n_head:]
        types = [0] * n_head + media_types + [0] * (len(state_ids) - n_head)
        with self.lock:
            out = self.model(input_ids=torch.tensor([prefix], device=self.device), mm_token_type_ids=torch.tensor([types], device=self.device),
                             use_cache=True, **extra)
            ps = []
            for row in rows:
                h = self.model(input_ids=torch.tensor([row["ids"]], device=self.device), past_key_values=copy.deepcopy(out.past_key_values),
                               use_cache=True).last_hidden_state[0].float()
                z = self.head(h[row["decide"]], h[torch.tensor(row["opts"], device=self.device)])
                ps.append(torch.softmax(z, -1).tolist())   # the head applies the temperature in eval mode
        return ps, len(prefix) + sum(len(r["ids"]) for r in rows)

    def answer(self, req: MediaRequest):
        t0 = time.perf_counter()
        rec, meta = to_record(req)
        ps, n_in = self.probs(rec, req.media)
        answers = to_answers(ps, meta)
        return {"model": req.model, "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": output_tokens(self.tok, answers)},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


TOWERS = ("vision_tower", "audio_tower", "embed_vision", "embed_audio")   # Gemma 4's encoders and their projections into the text model
MEDIA_DIR = "media"   # the encoders inside an MLX export folder (export_media)


def export_media(base, revision, out):
    """Write Gemma 4's vision and audio encoders (~1 GB bf16 for E4B, of a 16 GB base) as `out`/media, a folder that
    MediaEncoder loads without the base: config.json, the processor files and model.safetensors with the encoder weights
    only. They are the base's own weights: the LoRA adapter only touches the text model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoProcessor
    from .checkpoint import weight_shards
    import os, shutil
    folder = os.path.dirname(hf_hub_download(base, "config.json", revision=revision))
    dest = os.path.join(out, MEDIA_DIR); os.makedirs(dest, exist_ok=True)
    tensors = {}
    for shard in weight_shards(folder) or [hf_hub_download(base, "model.safetensors", revision=revision)]:
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                name = k.removeprefix("model.")
                if name.startswith(TOWERS): tensors[name] = f.get_tensor(k).to(torch.bfloat16)
    if not tensors: raise SystemExit(f"{base} has no vision or audio encoders")
    save_file(tensors, os.path.join(dest, "model.safetensors"))
    shutil.copy(os.path.join(folder, "config.json"), dest)
    AutoProcessor.from_pretrained(base, revision=revision).save_pretrained(dest)
    return dest


class MediaEncoder:
    """Gemma 4's vision and audio encoders on their own (torch): media -> the soft tokens the text model reads in place of
    its image or audio placeholder tokens. With these the text model itself, MLX included, answers about a photo or a
    voice clip, so no second copy of the model is needed. Loaded from an export's media folder (export_media)."""

    def __init__(self, folder, device):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoProcessor
        from transformers.initialization import no_init_weights
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model, Gemma4MultimodalEmbedder
        cfg = AutoConfig.from_pretrained(folder)
        self.proc = AutoProcessor.from_pretrained(folder)
        self.tok, self.device = self.proc.tokenizer, device
        with no_init_weights():   # every weight comes from the file
            towers = torch.nn.Module()
            towers.vision_tower, towers.audio_tower = AutoModel.from_config(cfg.vision_config), AutoModel.from_config(cfg.audio_config)
            towers.embed_vision = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config)
            towers.embed_audio = Gemma4MultimodalEmbedder(cfg.audio_config, cfg.text_config)
        towers.load_state_dict(load_file(f"{folder}/model.safetensors"), strict=True)
        towers.config = cfg
        self.towers = towers.to(device, torch.bfloat16).eval()
        # transformers' own feature code (pooling, padding removal, projection) on the towers alone, not a copy of it
        self._image = lambda **kw: Gemma4Model.get_image_features.__wrapped__(self.towers, **kw)
        self._audio = lambda **kw: Gemma4Model.get_audio_features.__wrapped__(self.towers, **kw)
        self._video = lambda **kw: Gemma4Model.get_video_features.__wrapped__(self.towers, **kw)
        self.placeholders = {cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id}
        self.lock = threading.Lock()   # one encoder pass at a time on the device

    @torch.no_grad()
    def encode(self, media: Media):
        """-> (token ids of the media span, the indices within it that are placeholders, their soft tokens [n, d] fp32)."""
        mm = processor_inputs(self.proc, media)
        leading = layout(self.tok)[0]
        ids = mm["input_ids"][0].tolist()
        if ids[:len(leading)] == leading: ids = ids[len(leading):]
        at = [i for i, t in enumerate(ids) if t in self.placeholders]
        dev = lambda k: mm[k].to(self.device)
        with self.lock:
            if media.type == "image":
                out = self._image(pixel_values=dev("pixel_values").to(torch.bfloat16), image_position_ids=dev("image_position_ids"))
                feats = torch.cat(out.pooler_output, dim=0)
            elif media.type == "video":
                out = self._video(pixel_values_videos=dev("pixel_values_videos").to(torch.bfloat16), video_position_ids=dev("video_position_ids"))
                feats = torch.cat(out.pooler_output, dim=0)
            else:
                out = self._audio(input_features=dev("input_features").to(torch.bfloat16), input_features_mask=dev("input_features_mask"))
                feats = out.pooler_output[out.attention_mask]
        if feats.shape[0] != len(at): raise ValueError(f"{len(at)} media placeholders but {feats.shape[0]} soft tokens")
        return ids, at, feats.float().cpu().numpy()


def with_media(enc, n_head, ids, at, embeds):
    """An encoding (d1a.model.encode) with a media span inserted at state index n_head (after <state>): the span's ids
    join the state and every later position and index shifts by its length. enc["media"] = (absolute placeholder
    indices, their soft tokens) for the backend's state pass."""
    n = len(ids)
    out = dict(enc)
    out["ids"] = enc["ids"][:n_head] + ids + enc["ids"][n_head:]
    out["seg"] = enc["seg"][:n_head] + [0] * n + enc["seg"][n_head:]
    out["opt"] = enc["opt"][:n_head] + [enc["opt"][0]] * n + enc["opt"][n_head:]
    out["pos"] = enc["pos"][:n_head] + list(range(n_head, n_head + n)) + [p + n for p in enc["pos"][n_head:]]
    out["decide_idx"] = [d + n for d in enc["decide_idx"]]
    out["opt_idx"] = [[o + n for o in oi] for oi in enc["opt_idx"]]
    out["media"] = ([n_head + i for i in at], embeds)
    return out


class OnDemand:
    """Builds a model on first use and drops it after `idle_s` seconds unused; never while a request holds it. A model
    with a close() method is closed when dropped."""

    def __init__(self, load, idle_s, device):
        self.load, self.idle_s, self.device = load, idle_s, device
        self.model, self.busy, self.last = None, 0, 0.0
        self.lock = threading.Lock()   # held while loading, so concurrent first requests wait for one load

    def acquire(self):
        """-> the model, loading it if needed; pair with release(). use() is the context-manager form."""
        with self.lock:
            if self.model is None: self.model = self.load()
            self.busy += 1
            return self.model

    def release(self):
        with self.lock:
            self.busy -= 1; self.last = time.monotonic()

    @contextmanager
    def use(self):
        model = self.acquire()
        try:
            yield model
        finally:
            self.release()

    def reap(self, now=None):
        """Drop the model when it has been idle for idle_s. Returns whether it did."""
        with self.lock:
            idle = (time.monotonic() if now is None else now) - self.last
            if self.model is None or self.busy or self.idle_s <= 0 or idle < self.idle_s: return False
            model, self.model = self.model, None
        if hasattr(model, "close"): model.close()
        del model
        gc.collect(); empty_cache(self.device)
        if "mlx.core" in sys.modules: sys.modules["mlx.core"].clear_cache()   # MLX keeps freed buffers for reuse until told otherwise
        return True

    def run_reaper(self):
        def loop():
            while True:
                time.sleep(min(30, self.idle_s)); self.reap()
        if self.idle_s > 0: threading.Thread(target=loop, daemon=True).start()


app = FastAPI(title="d1a-media")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/systemone/media")
def systemone_media(req: MediaRequest):
    try:
        with app.state.models.use() as m:
            return m.answer(req)
    except ValueError as e:   # undecodable base64, image or audio; too long or too large
        raise HTTPException(422, str(e))


@app.get("/v1/models")
def models():
    od = app.state.models; meta = app.state.checkpoint.meta   # reports without loading the model
    return {"models": [{"name": "d1a-media", "run": app.state.checkpoint.requested, "base": meta.base, "device": od.device, "backend": "torch",
                        "dtype": "bfloat16", "temperature": meta.temperature, "calibrated": meta.temperature != 1.0, "media": ["image", "audio"],
                        "loaded": od.model is not None, "idle_unload_s": od.idle_s}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="JohnP1/d1a-e2b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument("--idle-unload", type=int, default=600, help="seconds without a request before the model is dropped from memory; 0 keeps it loaded")
    a = ap.parse_args()
    dev = default_device() if a.device == "auto" else a.device
    app.state.checkpoint = Checkpoint(a.run)   # fetches the adapter and head now, so a bad --run fails at startup
    app.state.models = OnDemand(lambda: MediaModel(a.run, dev), a.idle_unload, dev)
    app.state.models.run_reaper()
    print(f"serving images and audio with {a.run} on {dev} at {a.host}:{a.port}; the model loads on the first request"
          + (f" and unloads after {a.idle_unload} s idle" if a.idle_unload > 0 else ""), flush=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, timeout_keep_alive=75)


if __name__ == "__main__":
    main()
