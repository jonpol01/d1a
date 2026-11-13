"""How POST /v1/systemone/media reads a photo (d1a.serving.media.decode_image): JPEG, PNG and WebP decode exactly as before,
upright by their EXIF orientation; any other format is refused before its decoder runs, and so is a photo that does not
decode, each as a ValueError, which the server answers with a 422 instead of a 500. No model: images are made here."""
import base64
import io
import logging
import zlib

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


def test_a_photo_that_does_not_decode_is_a_value_error(monkeypatch, caplog):
    """Each with Pillow's reason in the 422, and nothing logged: these are broken files, not the server's failures."""
    with pytest.raises(ValueError, match="cannot decode the image"):
        decode_image(picture("JPEG")[:400])   # cut short in its header
    png = picture("PNG")
    Image.open(io.BytesIO(png[:len(png) // 2]))   # cut short in its pixels: it opens, and fails only when decoded
    with pytest.raises(ValueError, match="cannot decode the image"):
        decode_image(png[:len(png) // 2])
    with pytest.raises(ValueError, match="must be a JPEG, PNG or WebP"):
        decode_image(b"not an image at all")
    for fmt in ["PNG", "WEBP"] if features.check("webp") else ["PNG"]:   # an EXIF block that is no TIFF: Pillow's SyntaxError
        with pytest.raises(ValueError, match="cannot decode the image: not a TIFF file"):
            decode_image(picture(fmt, exif=b"Exif\x00\x00not a tiff header at all"))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)   # past twice the limit: Pillow's decompression-bomb error
    with pytest.raises(ValueError, match="cannot decode the image: Image size"):
        decode_image(picture("PNG"))
    assert caplog.records == []


def test_the_endpoint_input_refuses_another_format_as_a_value_error():
    """processor_inputs is what both the torch media model and the MLX media encoder call: its ValueError is the server's 422."""
    def proc(**kw): raise AssertionError("the processor ran on a refused image")
    proc.image_token = "<image>"
    with pytest.raises(ValueError, match="must be a JPEG, PNG or WebP"):
        processor_inputs(proc, Media(type="image", data=base64.b64encode(picture("TIFF")).decode()))


def test_running_out_of_memory_stays_a_server_error(monkeypatch):
    def fail(image): raise MemoryError()
    monkeypatch.setattr(ImageOps, "exif_transpose", fail)
    with pytest.raises(MemoryError):
        decode_image(picture("JPEG"))
    monkeypatch.setattr(Image, "open", lambda *a, **kw: fail(None))
    with pytest.raises(MemoryError):
        decode_image(picture("JPEG"))


@pytest.mark.parametrize("error", [KeyError, IndexError, TypeError, zlib.error, RuntimeError])
@pytest.mark.parametrize("stage", ["open", "decode"])
def test_an_unexpected_failure_is_a_422_that_keeps_its_text_in_the_log(monkeypatch, caplog, error, stage):
    """A parser error no tested file raised is still the photo's 422, with no detail from inside the server; the log has it, with
    its traceback, so a bug still shows."""
    def fail(*a, **kw): raise error("internal detail")
    if stage == "open": monkeypatch.setattr(Image, "open", fail)
    else: monkeypatch.setattr(ImageOps, "exif_transpose", fail)
    with caplog.at_level(logging.WARNING, logger="d1a.serving.media"), pytest.raises(ValueError) as refused:
        decode_image(picture("JPEG"))
    assert str(refused.value) == "cannot decode the image"
    assert [r.exc_info[0] for r in caplog.records] == [error] and error.__name__ in caplog.text
