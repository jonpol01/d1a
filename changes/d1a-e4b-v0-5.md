### Changed

- The newest D1A-E4B is `v0.5` (`JohnP1/d1a-e4b@v0.5` and `JohnP1/d1a-e4b-mlx-q8@v0.5`, #175): Kev's skills trained back on
  top of v0.4. On held-out data, MLX 8-bit: hard decisions 55% → 71%, developer tools 64% → 70%, transfer-v4 +2.7, JGLUE
  +1.1, routing +1.5, PR change type +2.0. Severity is 2 points lower, and blast radius on the owner's own repositories
  drops (87% → 74% on 39 hand-checked PRs); v0.4 stays the better PR labeler for those. `d1a.core.versions.latest()`,
  and the scripts and demos that take their default from it, now resolve to v0.5.
