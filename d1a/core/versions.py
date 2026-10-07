"""The newest released version of each D1A model, the one place scripts and demos take their default checkpoint from.

    from d1a.core.versions import latest
    latest("JohnP1/d1a-e4b-mlx-q8")   # -> "JohnP1/d1a-e4b-mlx-q8@v0.5"

A release adds its tag here and a row to CHANGELOG.md's "Model versions" table; tests/test_versions.py fails when the
two disagree, so a new version cannot leave a default behind. Results should record the pinned id they ran with.
"""

LATEST = {"JohnP1/d1a-e4b": "v0.5", "JohnP1/d1a-e2b": "v0.2"}
FORMATS = ("-mlx-q8",)   # build repos that carry the same tags as their source repo


def latest(repo):
    """`repo@tag` for the newest version of a D1A model repo or one of its builds (an explicit @tag is kept)."""
    if "@" in repo: return repo
    base = next((repo[: -len(f)] for f in FORMATS if repo.endswith(f)), repo)
    if base not in LATEST: raise KeyError(f"{repo} is not a released D1A model: {sorted(LATEST)}")
    return f"{repo}@{LATEST[base]}"
