"""How POST /v1/systemone/media reads a photo (d1a.serving.media.decode_image): JPEG, PNG and WebP decode exactly as before,
upright by their EXIF orientation; any other format is refused before its decoder runs, and so is a photo that does not
decode, each as a ValueError, which the server answers with a 422 instead of a 500. No model: images are made here."""
import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageOps, features

from d1a.serving.media import Media, decode_image, processor_inputs


def picture(fmt, size=(48, 32), **save):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    out = io.BytesIO()
    img.save(out, format=fmt, **save)
    return out.getvalue()


def rotated_jpeg():
    """A JPEG whose EXIF says it is turned 90 degrees, as phones write them."""
    exif = Image.Exif()
    exif[0x0112] = 6
    return picture("JPEG", exif=exif.tobytes())


def before(raw):
    """What decode_image returned before the formats were limited."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


ALLOWED = [("JPEG", picture("JPEG")), ("rotated JPEG", rotated_jpeg()), ("PNG", picture("PNG"))]
if features.check("webp"): ALLOWED.append(("WebP", picture("WEBP", lossless=True)))


@pytest.mark.parametrize("name,raw", ALLOWED, ids=[n for n, _ in ALLOWED])
def test_the_documented_formats_decode_exactly_as_before(name, raw):
    got, was = decode_image(raw), before(raw)
    assert got.mode == "RGB" and got.size == was.size
    assert np.array_equal(np.asarray(got), np.asarray(was))
    if name == "rotated JPEG": assert got.size == (32, 48)   # turned upright


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF", "ICO", "PPM", "TGA"])
def test_another_format_is_refused_before_its_decoder_runs(fmt):
    raw = picture(fmt)
    assert before(raw).mode == "RGB"   # Pillow alone would have decoded it
    with pytest.raises(ValueError, match="must be a JPEG, PNG or WebP"):
        decode_image(raw)


def test_a_photo_that_does_not_decode_is_a_value_error(monkeypatch):
    with pytest.raises(ValueError, match="cannot decode the image"):
        decode_image(picture("JPEG")[:400])   # cut short in its header
    png = picture("PNG")
    Image.open(io.BytesIO(png[:len(png) // 2]))   # cut short in its pixels: it opens, and fails only when decoded
    with pytest.raises(ValueError, match="cannot decode the image"):
        decode_image(png[:len(png) // 2])
    with pytest.raises(ValueError, match="must be a JPEG, PNG or WebP"):
        decode_image(b"not an image at all")
    for fmt in ["PNG", "WEBP"] if features.check("webp") else ["PNG"]:   # an EXIF block that is no TIFF: Pillow's SyntaxError
        with pytest.raises(ValueError, match="cannot decode the image"):
            decode_image(picture(fmt, exif=b"Exif\x00\x00not a tiff header at all"))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)   # past twice the limit: Pillow's decompression-bomb error
    with pytest.raises(ValueError, match="cannot decode the image"):
        decode_image(picture("PNG"))


def test_the_endpoint_input_refuses_another_format_as_a_value_error():
    """processor_inputs is what both the torch media model and the MLX media encoder call: its ValueError is the server's 422."""
    def proc(**kw): raise AssertionError("the processor ran on a refused image")
    proc.image_token = "<image>"
    with pytest.raises(ValueError, match="must be a JPEG, PNG or WebP"):
        processor_inputs(proc, Media(type="image", data=base64.b64encode(picture("TIFF")).decode()))


@pytest.mark.parametrize("error", [MemoryError, RuntimeError, TypeError])
def test_a_failure_that_is_not_the_photos_stays_a_server_error(monkeypatch, error):
    """Only what a broken file raises is the client's 422: running out of memory, or a bug, stays a 500, logged as one."""
    def fail(image): raise error("not the photo")
    monkeypatch.setattr(ImageOps, "exif_transpose", fail)
    with pytest.raises(error):
        decode_image(picture("JPEG"))
