### Added

- `scripts/quality_gate.py --sequential` (#286): the base server answers every request, stops, then the head server does, so
  only one model is in memory at a time. The Mac mini (32 GB, with its other services) cannot hold two, which stopped the
  #285 baseline gate there. The latency floor is measured the same way, so the latency check still compares like with like.
