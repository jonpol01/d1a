"""Add the media pad row (#386) to an MLX export made before export_mlx wrote one: the pad id's row of Gemma 4's per-layer
embedding table, as stored in the checkpoint the export quantized it from. The row is taken only if quantizing it with the
export's own bits and group size gives the export's stored row bit for bit, so a row from another checkpoint (the base
instead of the QAT one, another revision) is refused. Writes media_pad_row.safetensors and names it in d1a_config.json;
the weights are not touched.
  uv run --extra mlx python scripts/media_pad_row.py <export dir> --from <checkpoint folder | Hub id@revision>
A Hub source is read with HTTP range requests (the one row, about 20 KB), not downloaded."""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d1a.backends.checkpoint import EXPORT_CONFIG, weight_shards  # noqa: E402
from d1a.backends.mlx import MEDIA_PAD_ROW_KEY, PLE, safetensors_tensor, write_media_pad_row  # noqa: E402

KEY = f"model.language_model.{PLE}.weight"   # the name in a transformers checkpoint (the base, or a compressed-tensors QAT one)


def hub_row(spec, row):
    """Row `row` of KEY in a Hub checkpoint, read by range requests -> (bf16 bits as uint16, the commit read)."""
    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session
    repo, _, rev = spec.partition("@")
    rev = HfApi().model_info(repo, revision=rev or None).sha
    url = hf_hub_url(repo, get_safetensors_metadata(repo, revision=rev).weight_map[KEY], revision=rev)
    s, h = get_session(), build_hf_headers()
    get = lambda a, b: s.get(url, headers={**h, "Range": f"bytes={a}-{b}"}).content
    n = struct.unpack("<Q", get(0, 7))[0]
    t = json.loads(get(8, 7 + n))[KEY]
    if t["dtype"] != "BF16": raise SystemExit(f"{spec}: {KEY} is {t['dtype']}, not bf16")
    width = int(np.prod(t["shape"][1:])); start = 8 + n + t["data_offsets"][0] + row * width * 2
    data = get(start, start + width * 2 - 1)
    if len(data) != width * 2: raise SystemExit(f"{spec}: read {len(data)} bytes of row {row}, expected {width * 2}")
    return np.frombuffer(data, np.uint16).copy(), f"{repo}@{rev}"


def local_row(folder, row):
    for f in weight_shards(folder):
        if (found := safetensors_tensor(f, KEY)) is not None:
            table, dtype = found
            if dtype != "BF16": raise SystemExit(f"{folder}: {KEY} is {dtype}, not bf16")
            return np.array(table[row]), str(folder)
    raise SystemExit(f"{folder}: no {KEY}")


def main():
    import mlx.core as mx
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("export"); p.add_argument("--from", dest="source", required=True)
    a = p.parse_args()
    out = Path(a.export); cfg = json.loads((out / EXPORT_CONFIG).read_text(encoding="utf-8"))
    if cfg.get(MEDIA_PAD_ROW_KEY): raise SystemExit(f"{out}: already names a media pad row ({cfg[MEDIA_PAD_ROW_KEY]})")
    q = (cfg.get("quantization") or {}).get(f"language_model.model.{PLE}")
    if not q: raise SystemExit(f"{out}: its per-layer embeddings are not quantized, so media already read the pad row as stored")
    pad = cfg["pad_id"]
    bits, row_ref = local_row(a.source, pad) if Path(a.source).is_dir() else hub_row(a.source, pad)
    row = mx.array(bits).view(mx.bfloat16)
    stored = {part: safetensors_tensor(f, f"language_model.model.{PLE}.{part}") for f in weight_shards(out) for part in ("weight", "scales", "biases")}
    stored = {k: v for k, v in stored.items() if v is not None}
    w, s, b = mx.quantize(row[None], group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"))
    for part, mine in (("weight", w), ("scales", s), ("biases", b)):
        table, dtype = stored[part]
        theirs = np.asarray(table[pad])
        ours = np.asarray(mine[0].view(mx.uint16) if dtype == "BF16" else mine[0])
        if not np.array_equal(ours, theirs):
            raise SystemExit(f"{a.source}: its pad row does not quantize to {out}'s ({part} differs); it is not the checkpoint the export was made from")
    cfg[MEDIA_PAD_ROW_KEY] = write_media_pad_row(out, row)
    (out / EXPORT_CONFIG).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export": str(out), "from": row_ref, "pad_id": pad, "values": int(row.size), "file": cfg[MEDIA_PAD_ROW_KEY]}))


if __name__ == "__main__":
    main()
