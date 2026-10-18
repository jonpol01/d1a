### Added

- Self-learning settings, all off by default (#294): `promotion.form = "vector"` (per-option recalibration of choice
  questions), `promotion.split = "time"` (the gate holds out the newest outcomes), and `promotion.serve_choice` with
  `promotion.hold` (serve the recalibrated top option, but keep the model's own answer for held options such as severity
  P0/P1). Flags and rules may name a local-path run.
