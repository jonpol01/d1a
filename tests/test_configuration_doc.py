"""docs/CONFIGURATION.md lists every setting: each D1A_* name the package reads and each flag of d1a.serving.serve. A new
setting fails this test until the page documents it, so the page a newcomer (or their coding agent) reads stays complete.
    uv run python -m pytest tests/test_configuration_doc.py -q
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CONFIGURATION.md"


def env_names(root=ROOT / "d1a"):
    return {n for f in root.rglob("*.py") for n in re.findall(r"\bD1A_[A-Z0-9_]+\b", f.read_text(encoding="utf-8"))}


def serve_flags(path=ROOT / "d1a" / "serving" / "serve.py"):
    return set(re.findall(r"add_argument\(\"(--[a-z][a-z0-9-]*)\"", path.read_text(encoding="utf-8")))


def doc():
    return DOC.read_text(encoding="utf-8")


def missing(doc_text, names):
    return sorted(n for n in names if f"`{n}`" not in doc_text)


def test_every_env_variable_is_documented():
    from d1a.core.settings import SETTINGS
    assert len(SETTINGS) >= 21, sorted(SETTINGS)
    gaps = missing(doc(), set(SETTINGS))
    assert not gaps, f"add these to docs/CONFIGURATION.md: {gaps}"


def test_every_env_variable_the_package_names_is_registered_and_read_through_the_registry():
    """d1a.core.settings is the one list: a D1A_* name anywhere in d1a/ must be in it, and no module reads one directly
    (os.environ / os.getenv), so a default lives in one place and --show-config shows what is really in force."""
    from d1a.core.settings import SETTINGS
    unregistered = sorted(env_names() - set(SETTINGS))
    assert not unregistered, f"register these in d1a/core/settings.py: {unregistered}"
    direct = [f"{f.relative_to(ROOT)}:{i}" for f in (ROOT / "d1a").rglob("*.py") if f.name != "settings.py" or f.parent.name != "core"
              for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
              if re.search(r"(environ(\.get)?\s*[\[(]|getenv\()\s*[\"']D1A_", line)]
    assert not direct, f"read these through d1a.core.settings.get: {direct}"


def test_show_config_masks_secrets_and_says_where_each_value_comes_from():
    from d1a.core.settings import effective, show
    rows = {r[0]: r for r in effective({"D1A_API_KEY": "s3cret-value", "D1A_PREFIX_CACHE": "2"})}
    assert rows["D1A_API_KEY"][1:3] == ("(set, hidden)", "set") and rows["D1A_PREFIX_CACHE"][1:3] == ("2", "set")
    assert rows["D1A_PREFIX_MAX_TOKENS"][1:3] == ("65536", "default")
    assert "s3cret-value" not in show({"D1A_API_KEY": "s3cret-value", "D1A_REMOTE_API_KEY": "s3cret-value"})


def test_every_server_flag_is_documented():
    flags = serve_flags()
    assert {"--run", "--host", "--port", "--idle-unload"} <= flags, flags
    gaps = missing(doc(), flags)
    assert not gaps, f"add these to docs/CONFIGURATION.md: {gaps}"


def test_a_missing_entry_is_caught():
    text = doc()
    assert missing(text.replace("`D1A_PREFIX_CACHE`", "PREFIX_CACHE"), {"D1A_PREFIX_CACHE"}) == ["D1A_PREFIX_CACHE"]
    assert missing(text.replace("`--idle-unload`", "idle unload"), {"--idle-unload"}) == ["--idle-unload"]
