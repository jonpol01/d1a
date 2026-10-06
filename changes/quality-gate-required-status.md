### Added

- A required `quality-gate` status on pull requests. CI (`.github/workflows/quality-gate.yml`) leaves it pending on a
  pull request that changes a module the model server loads, or the dependencies (`scripts/gate_required.py`, read
  from the server's own imports). `scripts/quality_gate.py --post-status` then sets it to the verdict, on the exact
  commit it tested; it refuses a head checkout with uncommitted changes. Any other pull request gets the status as
  success, with the reason. A new head starts pending again. It ships switched off: the workflow is disabled and the
  check is not required. AGENTS.md (Quality bar) has the two steps to turn it on.
