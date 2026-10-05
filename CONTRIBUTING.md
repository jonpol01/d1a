# Contributing to D1A

Run the unit tests (`uv run python -m pytest tests --unit -q`, as CI does) and `python scripts/check_license.py` before opening a pull
request. Contributions are accepted under the Apache License 2.0.

## License and provenance

D1A is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0), and Apache-2.0 §4 requires that
modified files carry a prominent notice of the change. So a file taken from Kev and modified keeps the two-line header
("Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0." and "Changes for D1A
Copyright 2026 John Soliva: ...") and stays in the "Derived files" list of [docs/UPSTREAM.md](docs/UPSTREAM.md); when
you change such a file, keep its header and, if the change is notable, extend its one-line summary. A Kev file that
cannot hold a comment (JSON, lockfiles) is listed under "Derived files without a header" instead, and an unchanged copy
under "Unmodified copies". A file rewritten from scratch drops the header and leaves the lists. `scripts/check_license.py`
enforces all of this in CI: LICENSE and NOTICE keep Kev's attribution, the header set and the Derived list match exactly,
every file still substantially similar to its Kev original is listed, and nothing user-facing is branded "Kev".

## Releasing

D1A's code follows [Semantic Versioning](https://semver.org); model checkpoints are versioned separately as Hugging
Face tags. To release:

1. Write the release notes by hand in `CHANGELOG.md`: move the *Unreleased* items into a new `## [X.Y.Z] - YYYY-MM-DD`
   section with a short *What's new* in plain language, then *Added / Changed / Fixed / Removed* with PR links, the
   checkpoints the release was tested with, and *Upgrade notes* for anything that changes how D1A is used.
2. Set `version = "X.Y.Z"` in `pyproject.toml` (a unit test fails while the version has no changelog section).
3. Merge, then tag the merge commit and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The release workflow checks that the tag, `pyproject.toml` and `CHANGELOG.md` agree, runs the unit tests, builds the
wheel and sdist, and publishes a GitHub release whose text is that version's changelog section.
