### Added

- OpenAI-compatible chat and embeddings on `d1a.serving.serve` (#274), so apps written for the OpenAI API (mem0, n8n,
  chat UIs) can use D1A's model instead of a separate chat server.
  - `POST /v1/chat/completions` answers with the served model's adapter switched off: an export made with `--unmerged` on an
    instruction-tuned base (D1A-E2B-it), whose base is then a chat model. It runs on the model thread between decision
    requests and supports streaming (token by token), tools (Gemma 4 tool calls as OpenAI `tool_calls`), `stop` and JSON mode.
  - `POST /v1/embeddings` serves a GGUF embedder through llama.cpp (`D1A_EMBED_GGUF`, the new `embed` extra).
  - `/v1/models` also lists the models in OpenAI form beside the TypeSafe listing.
  - New settings: `D1A_CHAT_NAME`, `D1A_EMBED_GGUF`, `D1A_EMBED_NAME`. Guide: "Chat with D1A" in docs/USAGE.md.
  - An `--unmerged` export also carries the base's `generation_config.json`, so chat stops at its end-of-turn tokens.
