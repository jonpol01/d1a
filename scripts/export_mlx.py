"""Export a D1A checkpoint for the MLX backend: the LoRA merged into the base in fp32, optionally quantized, served by
d1a.backends.checkpoint / d1a.serving.serve from the folder (or its Hub copy) without the base or the adapter. Apple Silicon only.

    uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 4 --q-group-size 64 --out runs/exports/d1a-e2b-mlx-4bit
    uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 8 --q-per-layer-bits 4 --out runs/exports/d1a-e2b-mlx-8bit-ple4
    uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --out runs/exports/d1a-e2b-mlx-bf16    # unquantized (bf16)

--no-quantize-embeddings keeps embed_tokens and the per-layer embeddings in bf16 (on Gemma 4 E2B they are ~60% of the
weights). The folder holds mlx-lm's config.json + model*.safetensors, head.safetensors (the fp32 pointer head), the
tokenizer files and d1a_config.json (d1a.backends.checkpoint.export_config).

--media adds media/: Gemma 4's vision and audio encoders from the base (~1 GB for E4B), so d1a.serving.serve answers about
photos and voice clips with this same model (POST /v1/systemone/media). Run it alone on an existing folder with --media-only.
"""
import argparse, json, time

from d1a.backends.checkpoint import Checkpoint
from d1a.backends.mlx import export_mlx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a LoRA checkpoint: run directory or Hub id (optionally @revision)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-bits", type=int, default=0, help="bits per weight (4 or 8); 0 = no quantization (bf16)")
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--q-per-layer-bits", type=int, default=0, help="bits for Gemma 4's per-layer embeddings; 0 = the same as --q-bits")
    ap.add_argument("--no-quantize-embeddings", action="store_true")
    ap.add_argument("--media", action="store_true", help="also write media/: the base's vision and audio encoders (Gemma 4)")
    ap.add_argument("--media-only", action="store_true", help="only add media/ to an existing export folder --out")
    a = ap.parse_args()
    t = time.time()
    if not a.media_only:
        cfg = export_mlx(Checkpoint(a.run), a.out, bits=a.q_bits or None, group_size=a.q_group_size, embeddings=not a.no_quantize_embeddings,
                         per_layer_bits=a.q_per_layer_bits or None)
        print(json.dumps(cfg, indent=2))
    if a.media or a.media_only:
        from d1a.serving.media import export_media
        meta = Checkpoint(a.run).meta
        print("wrote", export_media(meta.base, meta.base_revision, a.out))
    print(f"exported {a.run} to {a.out} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
