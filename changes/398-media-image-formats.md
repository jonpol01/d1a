### Security

- `POST /v1/systemone/media` decodes a photo as JPEG, PNG or WebP only, the formats the API documents (#398). Before,
  Pillow tried every decoder it has (TIFF, GIF, BMP, JPEG 2000, EPS through Ghostscript, ...) on bytes from the network,
  and those decoders carry most of its published advisories. A server that pins Pillow older than 13.0.0 should still
  upgrade it. A client that sent another format (AVIF or HEIC from a phone, GIF, BMP, TIFF) gets a 422 now: convert the
  photo to JPEG first, as the playground does.

### Fixed

- A photo in another format, or one that does not decode (cut short, a broken PNG chunk or EXIF block, past Pillow's
  decompression-bomb limit), gets a 422 with the reason, instead of a 500 (#398).
