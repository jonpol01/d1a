### Changed

- docs/MODELS.md and the D1A-E4B-it model cards (v0.6 and QAT, every listing) record a known limit (#361): on transfer-v4,
  mean confidence runs above accuracy (QAT ECE 0.105, v0.6 0.081), and one global temperature does not fix it, since the
  temperature that suite alone wants makes 15 of the other 21 suites worse.
