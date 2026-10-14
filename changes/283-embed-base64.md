### Fixed

- `POST /v1/embeddings` honours `encoding_format: "base64"` (#282): each embedding comes as base64 of little-endian
  float32, as the OpenAI API defines it. The openai Node SDK (n8n's embeddings node) always asks for it, and decoded the
  float lists it got before into garbage (192 numbers instead of 768).
