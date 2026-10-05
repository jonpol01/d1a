"""scripts/release_notes.py's change fragments (#163): every file in changes/ parses, assembling them loses nothing, and a
release with fragments left over is stopped."""
import collections
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts/release_notes.py")
rn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rn)


def put(path, text):
    """A fragment file (here, not inline: the conventions scan cuts each line at its first '#')."""
    path.write_text(text, encoding="utf-8")


def test_every_fragment_parses():
    frags, errors = rn.fragments()
    assert errors == []
    assert all(sections for _, sections in frags)


def test_assembling_round_trips():
    """The fragments in changes/ and CHANGELOG.md's Unreleased section become one new version section holding exactly
    their items, each under its own section, and Unreleased is left empty."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    frags, _ = rn.fragments()
    if not frags:   # just after a release (the tagged commit itself): everything is assembled, nothing to round-trip
        with pytest.raises(ValueError, match="nothing to release"):
            rn.assemble("99.0.0", "2099-01-01", text, frags)
        return
    new_text, body = rn.assemble("99.0.0", "2099-01-01", text, frags)
    back, errors = rn.parse(rn.section("99.0.0", new_text), "assembled")
    want = collections.defaultdict(list)
    for _, sections in frags:
        for s, items in sections.items():
            want[s] += items
    assert errors == [] and {s: collections.Counter(v) for s, v in back.items()} == {s: collections.Counter(v) for s, v in want.items()}
    assert rn.section("0.3.0", new_text) == rn.section("0.3.0", text)                      # older releases untouched
    after = new_text[new_text.index("## [Unreleased]"):]
    assert after.startswith("## [Unreleased]\n\n## [99.0.0] - 2099-01-01\n\n")


@pytest.mark.parametrize("text, message", [
    ("### Improved\n\n- x\n", "unknown section 'Improved'"),
    ("- x\n", "stray text"),
    ("### Added\n\nsome prose\n", "stray text"),
    ("### Added\n", "'### Added' has no items"),
    ("### Added\n\n- x\n\n### Added\n\n- y\n", "appears twice"),
    ("", "no '### Section' with items"),
])
def test_malformed_fragments_are_refused(text, message):
    _, errors = rn.parse(text, "f.md")
    assert errors and message in errors[0]


def test_items_keep_their_wrapped_lines_and_order(tmp_path):
    put(tmp_path / "7-old.md", "### Fixed\n\n- old fix\n")
    put(tmp_path / "12-new.md", "### Added\n\n- new thing\n  wrapped (#12)\n\n### Fixed\n\n- new fix\n")
    put(tmp_path / "docs-tweak.md", "### Changed\n\n- docs\n")
    put(tmp_path / "README.md", "not a fragment")
    frags, errors = rn.fragments(tmp_path)
    assert errors == [] and [p.name for p, _ in frags] == ["12-new.md", "7-old.md", "docs-tweak.md"]   # newest PR first
    _, body = rn.assemble("1.1.0", "2026-10-06", "# C\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n\n- a\n", frags)
    assert body == "### Added\n\n- new thing\n  wrapped (#12)\n\n### Changed\n\n- docs\n\n### Fixed\n\n- new fix\n- old fix"


def test_assembly_and_release_refusals(tmp_path, monkeypatch):
    text = "# C\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n\n- a\n"
    with pytest.raises(ValueError, match="already has a section"):
        rn.assemble("1.0.0", "2026-10-06", text, [])
    with pytest.raises(ValueError, match="nothing to release"):
        rn.assemble("1.1.0", "2026-10-06", text, [])
    monkeypatch.setattr(rn, "CHANGES", tmp_path)
    monkeypatch.setattr(rn, "check", lambda tag: [])
    assert rn.release_check("v1.1.0") == []
    put(tmp_path / "5-x.md", "### Added\n\n- x\n")
    assert "still holds 1 fragments" in rn.release_check("v1.1.0")[0]
