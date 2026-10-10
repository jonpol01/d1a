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
    names = env_names()
    assert len(names) >= 20, names   # the scan still finds the settings
    gaps = missing(doc(), names)
    assert not gaps, f"add these to docs/CONFIGURATION.md: {gaps}"


def test_every_server_flag_is_documented():
    flags = serve_flags()
    assert {"--run", "--host", "--port", "--idle-unload"} <= flags, flags
    gaps = missing(doc(), flags)
    assert not gaps, f"add these to docs/CONFIGURATION.md: {gaps}"


def test_a_missing_entry_is_caught():
    text = doc()
    assert missing(text.replace("`D1A_PREFIX_CACHE`", "PREFIX_CACHE"), {"D1A_PREFIX_CACHE"}) == ["D1A_PREFIX_CACHE"]
    assert missing(text.replace("`--idle-unload`", "idle unload"), {"--idle-unload"}) == ["--idle-unload"]
