"""How much of D1A is still Kev: compare this checkout with Kev at the import base, line by line, by area.

    uv run python scripts/kev_share.py            # table for the working tree
    uv run python scripts/kev_share.py --json     # the same numbers as JSON (for reports)

Kev's tree at the import base (docs/UPSTREAM.md: upstream 0fe8fc9) is downloaded once from GitHub into a cache. The
kev -> d1a rename (package, KEV_* variables, the name) is undone before comparing, so a renamed line counts as Kev's.
A line is "kept" when it survives in the matching Kev file (difflib matching blocks); files with no Kev counterpart are
"new"; Kev files with no D1A counterpart are "pruned".
"""
import argparse, ast, difflib, io, json, os, re, tarfile, urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "0fe8fc9"
ROOT = Path(__file__).resolve().parents[1]
# the modules #60 moved into subpackages, by the name Kev gave them (d1a/_layout.py, read without importing d1a)
_MOVED = ast.literal_eval((ROOT / "d1a/_layout.py").read_text(encoding="utf-8").split("MOVED = ", 1)[1])
KEV_NAME = {f"d1a/{new.replace('.', '/')}.py": f"kev/{old}.py" for old, new in _MOVED.items()}
SHIMS = {f"d1a/{old}.py" for old in _MOVED}
CACHE = Path.home() / ".cache" / "d1a" / f"kev-{BASE}"
TEXT = (".py", ".ts", ".tsx", ".js", ".md", ".toml", ".yml", ".yaml", ".json", ".css", ".txt", ".html", ".sh")
# generated test data (tests/golden: golden vectors, a committed checkpoint): shown, never counted in TOTAL
FIXTURES = "test fixtures (generated)"
AREAS = [("d1a/", "model code (d1a/)"), ("tests/golden/", FIXTURES), ("tests/", "tests"), ("scripts/", "scripts"), ("evals/", "frozen eval suites")]


def kev_tree():
    if not CACHE.exists():
        url = f"https://codeload.github.com/jaredpalmer/kev/tar.gz/{BASE}"
        with urllib.request.urlopen(url) as r, tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz") as t:
            t.extractall(CACHE.parent / f".kev-{BASE}-tmp", filter="data")
        (top,) = list((CACHE.parent / f".kev-{BASE}-tmp").iterdir()); top.rename(CACHE)
    return CACHE


def files(root):
    out = {}
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "node_modules", ".claude", "__pycache__", "runs")]
        for f in fs:
            p = os.path.relpath(os.path.join(dp, f), root)
            if p.endswith(TEXT): out[p] = os.path.join(dp, f)
    return out


def norm(text):
    return re.sub(r"\bKEV_", "D1A_", re.sub(r"\bkev\b", "d1a", text)).replace("Kev", "D1A")


# #60 moved modules into subpackages and made their imports absolute: on both sides, every module reference is written
# as the flat d1a.<old name> before comparing, so a moved file is not counted as rewritten for its import lines
_OLD = {new: old for old, new in _MOVED.items()}
_NEW_DOTTED = re.compile(r"\bd1a\.(" + "|".join(re.escape(n) for n in sorted(_OLD, key=len, reverse=True)) + r")\b")
_FROM_PKG = re.compile(r"\bfrom d1a\.(\w+) import (\w+)(?: as (\w+))?")


def canonical(text):
    text = re.sub(r"^(\s*)from \.(\w+) import", r"\1from d1a.\2 import", text, flags=re.M)   # relative -> absolute
    def flat(m):
        old = _OLD.get(f"{m[1]}.{m[2]}")
        if old is None:
            return m[0]
        return f"from d1a import {old}" + (f" as {m[3]}" if m[3] and m[3] != old else "")
    text = _FROM_PKG.sub(flat, text)
    return _NEW_DOTTED.sub(lambda m: f"d1a.{_OLD[m[1]]}", text)


def area(p):
    return next((name for prefix, name in AREAS if p.startswith(prefix)), "docs / config / other")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    kev, d1a = files(kev_tree()), files(ROOT)
    stats = defaultdict(lambda: defaultdict(int))
    for p, path in d1a.items():
        if os.path.getsize(path) > 3_000_000: continue
        if p in SHIMS: continue   # one-release import shims (d1a/<old>.py), not code
        s = stats[area(p)]; lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        kp = KEV_NAME.get(p) or ("kev/" + p[4:] if p.startswith("d1a/") else p)
        s["lines"] += len(lines)
        if kp not in kev:
            s["new"] += len(lines); s["new_files"] += 1; continue
        old = canonical(norm(open(kev[kp], encoding="utf-8", errors="replace").read())).splitlines()
        kept = sum(b.size for b in difflib.SequenceMatcher(None, old, canonical("\n".join(lines)).splitlines(), autojunk=False).get_matching_blocks())
        s["kept"] += kept; s["changed"] += len(lines) - kept; s["files_from_kev"] += 1
    ours = {KEV_NAME.get(p) or ("kev/" + p[4:] if p.startswith("d1a/") else p) for p in d1a}
    pruned = sum(1 for p in kev if p not in ours)
    total = defaultdict(int)
    for name, s in stats.items():
        if name == FIXTURES: continue
        for k, v in s.items(): total[k] += v
    if a.json:
        print(json.dumps({"base": BASE, "areas": stats, "total": total, "pruned_kev_files": pruned}, indent=1)); return
    print(f"{'area':24s} {'lines':>7s} {'Kev kept':>9s} {'changed':>8s} {'new':>7s} {'% Kev':>6s}")
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["lines"]):
        print(f"{name:24s} {s['lines']:7d} {s['kept']:9d} {s['changed']:8d} {s['new']:7d} {100 * s['kept'] / max(1, s['lines']):5.0f}%")
    print(f"{'TOTAL':24s} {total['lines']:7d} {total['kept']:9d} {total['changed']:8d} {total['new']:7d} {100 * total['kept'] / total['lines']:5.0f}%")
    print(f"Kev text files pruned from D1A: {pruned}")


if __name__ == "__main__":
    main()
