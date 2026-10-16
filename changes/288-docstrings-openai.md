### Changed

- Docstrings of `d1a.serving.openai_compat` and `d1a.serving.serve` brought up to date (#288): streaming is token by token
  (one chunk only with `tools`), a tool call cut off by `max_tokens` is dropped (#278), and `/v1/models` lists the chat
  model whenever the checkpoint can chat (#280). No behaviour change.
