### Added

- `recipes/swe-verifier`: OpenHands runs (nebius/SWE-rebench-openhands-trajectories, CC-BY-4.0) can be scored, through
  `from_openhands.py` and `zero_shot.py` / `self_improve.py --source openhands` (#248). Also new: `--keep-issue-md` (keep
  the harness's issue.md diff, to measure that shortcut), `--exclude` (a sample disjoint from an earlier draw) and
  `--issues`. The default source, the SWE-agent runs, gives the same rows as before.
