"""Decisions about a photo or a voice clip: the media server for the playground's Photo check and Voice triage demos.

Run: uv run --extra serve --extra media python -m d1a.media --run JohnP1/d1a-e2b --port 8010

POST /v1/systemone/media takes a /v1/systemone request plus {"media": {"type": "image" | "audio", "data": <base64>}} and
answers in the same shape. The checkpoint is the text-trained one: Gemma 4 keeps its vision and audio encoders, the
adapter is merged into the language model, and the media's soft tokens go between <state> and the state text. Zero-shot
(issue #78) it read damage and drop-off place off delivery photos and intent off EN/JA voice notes, no speech-to-text.

A separate process from d1a.serve on purpose: the text server stays on its fast backend (MLX on Apple Silicon) and the
multimodal base (bf16 torch, ~10 GB for E2B) is loaded only where these demos run. The media is encoded once per
request; each question then runs as its own row on a copy of that prefix cache (rows_of: the same tokens and positions
the packed form gives a question).
"""
import argparse, base64, copy, io, threading, time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from .api import SystemOneRequest, output_tokens, to_answers, to_record
from .checkpoint import Checkpoint
from .device import default_device
from .model import PointerHead, encode, layout, rows_of

SAMPLE_RATE = 16_000             # what Gemma 4's audio feature extractor expects
MAX_AUDIO_S = 30                 # Gemma 4's audio encoder limit
MAX_MEDIA_BYTES = 12 * 1024 * 1024


class Media(BaseModel):
    type: Literal["image", "audio"]
    data: str = Field(description="base64 of the file: an image PIL reads (JPEG, PNG, WebP) or a WAV/FLAC/OGG clip")


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
        full = AutoModelForImageTextToText.from_pretrained(meta.base, revision=meta.base_revision, dtype=dtype, attn_implementation="sdpa")
        inner = full.model
        if not (hasattr(inner, "vision_tower") and hasattr(inner, "audio_tower")):
            raise SystemExit(f"{meta.base} has no vision and audio encoders; the media server is for Gemma 4 bases")
        inner.language_model = PeftModel.from_pretrained(inner.language_model, ck.path).merge_and_unload()
        self.model = inner.to(device).eval()
        self.head = PointerHead(full.config.get_text_config().hidden_size, dp=meta.head_dim).to(device).eval()
        self.head.load_state_dict(meta.head)
        self.head.temperature = meta.temperature
        self.lock = threading.Lock()   # one forward at a time on the device

    def media_inputs(self, media: Media):
        """-> (media token ids, mm_token_type_ids for them, the encoder inputs)."""
        raw = base64.b64decode(media.data, validate=True)
        if len(raw) > MAX_MEDIA_BYTES: raise ValueError(f"media larger than {MAX_MEDIA_BYTES // 2**20} MB")
        if media.type == "image":
            mm = self.proc(text=self.proc.image_token, images=decode_image(raw), return_tensors="pt")
        else:
            mm = self.proc(text=self.proc.audio_token, audio=decode_audio(raw), sampling_rate=SAMPLE_RATE, return_tensors="pt")
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


app = FastAPI(title="d1a-media")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/systemone/media")
def systemone_media(req: MediaRequest):
    try:
        return app.state.model.answer(req)
    except ValueError as e:   # undecodable base64, image or audio; too long or too large
        raise HTTPException(422, str(e))


@app.get("/v1/models")
def models():
    m = app.state.model; meta = m.checkpoint.meta
    return {"models": [{"name": "d1a-media", "run": m.checkpoint.requested, "base": meta.base, "device": m.device, "backend": "torch",
                        "dtype": str(next(m.model.parameters()).dtype).removeprefix("torch."), "temperature": m.head.temperature,
                        "calibrated": m.head.temperature != 1.0, "media": ["image", "audio"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="JohnP1/d1a-e2b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    a = ap.parse_args()
    dev = default_device() if a.device == "auto" else a.device
    app.state.model = MediaModel(a.run, dev)
    print(f"serving images and audio with {a.run} on {dev} at {a.host}:{a.port}", flush=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, timeout_keep_alive=75)


if __name__ == "__main__":
    main()
