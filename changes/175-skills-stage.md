### Added

- `recipes/skills/`: the skills stage D1A-E4B never had (#175).
  - The mix: every hard-v1 and devtools-v1 training record plus replay, 15,320 records with the trainer's decision-v7
    replay.
  - An HF Jobs script. It first runs 5 steps on the 40 longest states to test memory, then trains and calibrates as
    v0.4 was.
  - The #167 B0 pilot (300 steps) already raised hard-v1 and devtools-v1 by about 5 points each.
