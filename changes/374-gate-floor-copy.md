### Fixed

- `scripts/quality_gate.py --head-run`: the latency floor now serves base against a byte copy of base (two sets of
  files, as the gate itself serves for a new checkpoint), not the same files twice (#371). Serving two different files
  measured 3.2% slower than one on its own, so the old floor vetoed new checkpoints for the setup, not the model. The
  copy needs the checkpoint's size in free disk for the floor's duration; the report names the floor's setup.
