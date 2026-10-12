### Changed

- `tests/test_decide.py` bootstraps each distinct scorecard input once per test session (#255), through a memo keyed on
  every input `decide.paired` reads: the file runs in 205 s instead of 525 s. Test-only; `scripts/decide.py` is unchanged.
