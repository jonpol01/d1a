"""Refuse a pull request that adds a file over 500 KiB, or changes a binary over 500 KiB. Adapted from
NousResearch/hermes-agent's blob-size-check (MIT).

Every blob a merge adds stays in every clone forever, even after a later commit deletes it, so a model file, a checkpoint
or a dataset dump has to be stopped at the pull request. Edits to existing text files pass (the frozen eval suites are
large text). Path patterns in .github/blob-size-allowlist.txt (fnmatch; `*` spans directories) are exempt.

    python scripts/check_blob_size.py origin/main     # the pull request's base; CI passes origin/<base branch>
"""
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

LIMIT = 500 * 1024
ALLOWLIST = Path(".github/blob-size-allowlist.txt")


def allowed_patterns(path=ALLOWLIST):
    if not path.exists(): return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]


def violations(changes, sizes, binary, patterns, limit=LIMIT):
    """changes: [(status A|M, path)]; sizes: {path: bytes}; binary: {paths git sees as binary}. -> [(path, bytes, why)]."""
    out = []
    for status, path in changes:
        size = sizes.get(path, 0)
        if size <= limit or any(fnmatch.fnmatch(path, p) for p in patterns): continue
        if status == "A": out.append((path, size, "new file"))
        elif status == "M" and path in binary: out.append((path, size, "changed binary"))
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout
    changes = [tuple(l.split("\t", 1)) for l in git("diff", "--name-status", "--diff-filter=AM", f"{base}...HEAD").splitlines()]
    binary = {l.split("\t")[2] for l in git("diff", "--numstat", f"{base}...HEAD").splitlines() if l.startswith("-\t-\t")}
    sizes = {p: os.path.getsize(p) for _, p in changes if os.path.exists(p)}
    bad = violations(changes, sizes, binary, allowed_patterns())
    for path, size, why in bad:
        print(f"::error file={path}::{why} of {size / 1024:.0f} KiB (limit {LIMIT // 1024} KiB): keep it out of git (Hugging Face for weights and data), or add an exact path to {ALLOWLIST}")
    print(f"blob size: {len(changes)} added or changed files checked, {len(bad)} over the limit")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
