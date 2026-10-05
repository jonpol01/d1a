import re
from pathlib import Path

from d1a.core.versions import LATEST, latest

ROOT = Path(__file__).resolve().parent.parent


def test_latest_matches_the_newest_row_of_the_changelog_versions_table():
    """CHANGELOG lists versions newest first; its first row per model must be d1a.core.versions.LATEST, so a release that
    updates one and not the other fails here."""
    table = ROOT.joinpath("CHANGELOG.md").read_text(encoding="utf-8").split("## Model versions", 1)[1]
    newest = {}
    for repo, tag in re.findall(r"^\| (JohnP1/[\w.-]+) \| `([^`]+)` \|", table, re.MULTILINE):
        newest.setdefault(repo, tag)
    assert newest == LATEST
    assert latest("JohnP1/d1a-e4b-mlx-q8") == f"JohnP1/d1a-e4b-mlx-q8@{LATEST['JohnP1/d1a-e4b']}"
    assert latest("JohnP1/d1a-e2b@v0.1") == "JohnP1/d1a-e2b@v0.1"
