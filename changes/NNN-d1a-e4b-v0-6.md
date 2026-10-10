### Changed

- The newest D1A-E4B is `v0.6` (`JohnP1/d1a-e4b@v0.6` and `JohnP1/d1a-e4b-mlx-q8@v0.6`, #NNN): v0.5 trained further on
  current pull-request labels (479 hermes-agent PRs created 2026-10-02 to 10-05, #198), with documents-v1, routing and low
  replay of every earlier skill. On 487 PRs created after all its training, MLX 8-bit: severity 74.7% → 79.1% (+4.3
  [+0.8, +7.8]), change type 86.7% → 90.6% (+3.9 [+1.2, +6.6]), P1 recall 0.22 → 0.35. Also documents-v1 +2.0 and
  factory routing +3.4, with lower calibration error on most suites (T 1.95). It is an explicit release exception:
  `scripts/decide.py` vetoes it against v0.5 (V1, hard-v1 −2.7 [−4.9, −0.5]; change type on the older frozen PR test set
  −4.1 [−6.0, −2.3]; pooled −0.20 [−0.98, +0.52]), and it ships because the Mac mini's job is labelling current PRs.
  Pin `@v0.5` for hard-v1-style reasoning or the older PR-label conventions. v0.6 carries no use-case temperatures, so a
  `"use_case": "routing"` request is read at T 1.95; a refitted routing temperature follows as v0.6.1.
  `d1a.core.versions.latest()`, and the scripts and demos that take their default from it, now resolve to v0.6.
