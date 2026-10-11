"""The package layout (#60). D1A 0.4 kept a shim at each module's 0.3 path for one release; 0.5 removed them. The map of
old to new paths (d1a/_layout.py) stays for scripts/kev_share.py and scripts/check_license.py, so it must still name
modules that exist; and nothing in this repository may use an old path."""
import importlib
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from d1a._layout import MOVED

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("old", sorted(MOVED))
def test_an_old_path_is_gone_and_its_new_module_exists(old):
    assert not (ROOT / "d1a" / f"{old}.py").exists()
    spec = importlib.util.find_spec(f"d1a.{old}")   # an editable install of another checkout may still have it: not ours
    assert spec is None or not str(spec.origin).startswith(str(ROOT)), spec
    assert (ROOT / "d1a" / (MOVED[old].replace(".", "/") + ".py")).exists(), MOVED[old]


def test_nothing_here_uses_an_old_path():
    names = "|".join(sorted(MOVED, key=len, reverse=True))
    old = re.compile(rf"\bd1a\.({names})\b(?!\.py)|^\s*from d1a import ({names})\b|^\s*from \.({names}) import|\bd1a/({names})\.py\b", re.M)
    exempt = {"d1a/_layout.py", "tests/test_layout.py", "CHANGELOG.md",
              # the gate starts OLD checkouts (the Mac mini's pin) by their own flat module paths
              "scripts/quality_gate.py", "tests/test_quality_gate.py"}
    files = subprocess.run(["git", "ls-files", "*.py", "*.md", "*.sh", "*.yml", "*.toml"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    hits = [f"{f}:{text.count(chr(10), 0, m.start()) + 1}: {m[0]}" for f in files
            if f not in exempt and not f.startswith(("changes/", "docs/reports/", "evals/")) and (ROOT / f).exists()
            for text in [(ROOT / f).read_text(encoding="utf-8")] for m in old.finditer(text)]
    assert hits == [], "old module paths still used:\n" + "\n".join(hits[:20])
