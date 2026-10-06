"""The package layout (#60). For one release every module of D1A 0.3 still imports from its old path, as the very module
at its new one, with a DeprecationWarning; `python -m d1a.<old>` still runs; and nothing in this repository uses an old
path any more, so removing the shims in 0.5 breaks no caller of ours."""
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from d1a._layout import MOVED

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("serve", "train", "benchmark", "calibrate", "feedback", "suites", "recipe", "media")   # mcp_server is a stdio server, not a command line


def import_or_skip(name):
    """The module, or a skip when an optional extra it needs (mlx, mcp, ...) is not installed here."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("d1a"):
            pytest.skip(f"{name} needs {e.name}, an optional extra not installed here")
        raise


@pytest.mark.parametrize("old", sorted(MOVED))
def test_an_old_import_path_is_the_new_module(old):
    new = import_or_skip(f"d1a.{MOVED[old]}")
    sys.modules.pop(f"d1a.{old}", None)
    with pytest.warns(DeprecationWarning, match=rf"d1a\.{old} moved to d1a\.{re.escape(MOVED[old])}; the old path is removed in D1A 0\.5"):
        shimmed = importlib.import_module(f"d1a.{old}")
    assert shimmed is new


@pytest.mark.parametrize("old", COMMANDS)
def test_an_old_command_still_runs(old):
    import_or_skip(f"d1a.{MOVED[old]}")
    out = subprocess.run([sys.executable, "-m", f"d1a.{old}", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "usage" in out.stdout.lower()


def test_nothing_here_uses_an_old_path():
    names = "|".join(sorted(MOVED, key=len, reverse=True))
    old = re.compile(rf"\bd1a\.({names})\b(?!\.py)|^\s*from d1a import ({names})\b|^\s*from \.({names}) import|\bd1a/({names})\.py\b", re.M)
    exempt = {f"d1a/{name}.py" for name in MOVED} | {"d1a/_layout.py", "tests/test_layout.py", "CHANGELOG.md"}
    files = subprocess.run(["git", "ls-files", "*.py", "*.md", "*.sh", "*.yml", "*.toml"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    hits = [f"{f}:{text.count(chr(10), 0, m.start()) + 1}: {m[0]}" for f in files
            if f not in exempt and not f.startswith(("changes/", "docs/reports/", "evals/")) and (ROOT / f).exists()
            for text in [(ROOT / f).read_text(encoding="utf-8")] for m in old.finditer(text)]
    assert hits == [], "old module paths still used:\n" + "\n".join(hits[:20])
