### Added

- Pull-request checks adapted from NousResearch/hermes-agent (MIT): `blob size` (no new file, and no changed binary, over 500 KiB;
  `scripts/check_blob_size.py`, exemptions in `.github/blob-size-allowlist.txt`), `shared history` (the branch shares
  history with main) and `uv.lock in sync` (`uv lock --check`, uv pinned), plus a weekly OSV-Scanner run on `uv.lock`
  (Security tab, detection only) (#269).

### Fixed

- `uv.lock` matches `pyproject.toml` again: it still named d1a 0.3.0, and newer uv narrows the CUDA wheels to Linux (#269).
