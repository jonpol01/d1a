### Fixed

- `scripts/quality_gate.py` gates versions from before the package layout (#60) again. Each checkout's servers and
  benchmarks run its own module paths (`d1a.serving.serve`, or `d1a.serve` before #166), so it can gate a new main
  against the Mac mini's pin.
