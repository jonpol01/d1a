### Changed

- `docs/MODELS.md` opens with a capabilities table of every D1A model line (#273): what D1A-E4B v0.6.1, D1A-E2B v0.6.1,
  D1A-E2B-it v0.6 and the planned D1A-12B can do (decisions, PR labeling, media, routing temperature, self-learning,
  chat, embeddings), their builds and where each runs. Chat is not served by any build yet: D1A-E2B-it can provide it
  with its adapter off, which `d1a.serving.serve` will expose in a later release.
