### Fixed

- `scripts/compare_checkpoints.py` counts both records of an id a frozen suite holds twice (devtools-v1), where it
  silently kept one, and `scripts/decide.py` pairs them by their request instead of the order a run lists them
  (#418, #417). No published number changes.
