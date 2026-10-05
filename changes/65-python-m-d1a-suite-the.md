### Removed

- `python -m d1a.suite`, the suite freezer inherited from Kev, with `d1a/contrastive.py` and the record generator of
  `d1a/composition.py`. D1A loads its frozen suites as data and never rebuilds them; `paired_flip` moved to
  `d1a.benchmark`, and the rule shapes that `d1a.suite.validate_training` checks stay. See docs/removed-tools.md (#65).
