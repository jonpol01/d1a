### Changed

- D1A-E4B and D1A-E2B v0.6.1 are the default checkpoints (`d1a.core.versions`) (#264). They have v0.6's weights plus a
  routing temperature: requests sent with `"use_case": "routing"` are served at T 1.18 (E4B) and 0.95 (E2B) instead
  of 1.95 and 1.45. Factory routing calibration error falls 0.093 → 0.026 (E4B) and 0.091 → 0.035 (E2B), and no
  answer changes. CONTRIBUTING.md's release steps now include fitting the per-use-case temperatures.

### Fixed

- `d1a.training.calibrate --transfer` reports rows saved at a served temperature from their raw logits (#264). It used to
  apply the fitted temperature on top of the served one, so factory routing read ECE 0.091 → 0.120 for a temperature
  that brings it to 0.023.
