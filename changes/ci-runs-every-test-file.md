### Changed

- CI runs every test file (`pytest tests --unit`) except the few `tests/conftest.py`'s `OUTSIDE_UNIT_TESTS` lists with
  why, instead of a list naming each file: a new test file runs without being listed, and two pull requests adding test
  files no longer conflict on that list. The release runs the same command. CI keeps the Hugging Face cache (the pinned
  tokenizers the tests read) between runs. `docs/UPSTREAM.md` lists rewritten files one per line.
