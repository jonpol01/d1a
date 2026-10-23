"""d1a.core.hygiene and the pull-request checks (scripts/check_hygiene.py, scripts/release_notes.py --check-pr): AI
attribution and local machine paths are refused, ordinary lines and other people's paths are not, and every pull request
carries a changes fragment named for it. Paths here are built from pieces, so this file passes the check it tests."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from d1a.core.hygiene import attribution, local_paths_in_diff, refuse_local_paths

ROOT = Path(__file__).resolve().parents[1]
HOME = "/Users/" + "tester/"


@pytest.fixture(autouse=True)
def tester_is_one_of_ours(monkeypatch):
    """The real list holds the owners' user names as hashes; the tests use a made-up one."""
    from d1a.core import hygiene
    monkeypatch.setattr(hygiene, "OWN_HOMES", hygiene.OWN_HOMES | {hashlib.sha256(b"tester").hexdigest()})
TMP = "/private/tmp/" + "claude-501/x"


def test_ai_attribution_is_found_in_commits_and_bodies():
    bad = "Release 0.5.0\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    assert attribution(bad) and attribution("body\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)")
    assert not attribution("Co-Authored-By: Jane Doe <jane@example.com>\nDocs: mention Claude Code's config page")


def test_local_paths_are_found_only_on_added_lines_and_only_ours():
    diff = "\n".join(["+++ b/docs/a.md", f"+run from {HOME}Develop", f"-was {TMP}", "+++ b/data/x.jsonl",
                      "+/Users/" + "Shared/d1a/labeler-decisions.jsonl", "+/Users/" + "seb/project/file.py", f"+see {TMP}/out"])
    assert [f for f, _ in local_paths_in_diff(diff)] == ["docs/a.md", "data/x.jsonl"]


def test_an_own_home_is_found_wherever_and_however_it_is_written():
    # inside another path, in capitals (macOS paths ignore case), and with no slash after it
    for line in ("+cp /Users/x" + HOME + "a b", "+see " + HOME.upper() + "Develop", "+cd " + HOME.rstrip("/")):
        assert local_paths_in_diff("+++ b/a.md\n" + line), line
    assert not local_paths_in_diff("+++ b/a.md\n+cd " + HOME.rstrip("/") + "ny/x")   # another user whose name starts the same


def test_the_hygiene_module_passes_its_own_check():
    src = (ROOT / "d1a" / "core" / "hygiene.py").read_text(encoding="utf-8")
    assert not local_paths_in_diff("+++ b/d1a/core/hygiene.py\n" + "\n".join("+" + l for l in src.splitlines()))


def test_an_upload_with_a_local_path_is_refused(tmp_path):
    (tmp_path / "ok").mkdir(); (tmp_path / "ok" / "d1a_config.json").write_text(json.dumps({"base": "google/gemma-4-E4B"}), encoding="utf-8")
    refuse_local_paths(tmp_path / "ok")
    (tmp_path / "bad").mkdir(); (tmp_path / "bad" / "d1a_config.json").write_text(json.dumps({"data": HOME + "runs/mix.jsonl"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to upload"):
        refuse_local_paths(tmp_path / "bad")
    with pytest.raises(SystemExit):
        refuse_local_paths(tmp_path / "bad" / "d1a_config.json")


def release_notes():
    spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.mark.parametrize("diff,author,ok", [
    ("A\tchanges/301-x.md\nM\td1a/a.py", "jonpol01", True),
    ("M\tchanges/301-x.md", "jonpol01", True),
    ("M\td1a/a.py", "jonpol01", False),                                  # no fragment
    ("A\tchanges/301-x.md\nA\tchanges/x.md", "jonpol01", False),        # an unnumbered one
    ("A\tchanges/7-x.md", "jonpol01", False),                           # another PR's number only
    ("M\tuv.lock", "dependabot[bot]", True),                            # dependency bumps
    ("D\tchanges/250-prune.md\nM\tCHANGELOG.md", "jonpol01", True),     # a release assembles them
])
def test_every_pull_request_carries_its_fragment(diff, author, ok):
    assert (release_notes().check_pr(301, author, diff) == []) is ok
