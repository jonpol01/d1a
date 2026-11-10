### Fixed

- `scripts/quality_gate.py` reads every checkpoint's files once before its server starts, in the floor and the gate alike
  (#388). With `--head-run`, only the head's weight file used to be read cold in the measured pass, so a byte copy of base
  failed latency against itself (1.030 against a floor of 1.007). It now passes (1.002 against 0.997). A checkpoint that
  cannot be read is reported as `not warmed`, and `report.json` records `"warmed": true`.
