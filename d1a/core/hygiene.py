"""What must never be published: a local path of the machines D1A is built on, and an AI-attribution line (the maintainer's rule:
commits and pull requests carry no Co-Authored-By: Claude and no "Generated with Claude Code"). Used by CI
(scripts/check_hygiene.py) and before every Hugging Face upload, since a local path once reached a public model repo.
"""
import hashlib
import re
from pathlib import Path

# The home folders of the machines D1A is built on, by the sha256 of their user name, so this public file names no one;
# other people's paths (/Users/Shared, a contributor's own) are left alone. The other patterns are split so this file never
# holds one of the paths it looks for (CI scans every added line).
OWN_HOMES = {"96d9632f363564cc3032521409cf22a852f2032eec099ed5967c0d000cec607a",
             "b2b5d2b6d117e164f8f6e551cccdbdca2e3d5014432008463b924c047d83ff0a"}
HOME_PATH = re.compile(r"(?=/Users/([^/\s\"'`]+))", re.I)   # a lookahead: every /Users/ is read, even one inside another path
LOCAL_OTHER = re.compile(r"/private/tmp/" r"claude-" "|" r"/Vol" r"umes/")


def is_local_path(line):
    """Whether `line` holds a path that exists only on the machines D1A is built on (macOS paths ignore case, so the
    name is lowercased before it is hashed)."""
    return bool(LOCAL_OTHER.search(line)) or any(hashlib.sha256(m.group(1).lower().encode()).hexdigest() in OWN_HOMES
                                                 for m in HOME_PATH.finditer(line))
ATTRIBUTION = re.compile(r"(?im)^\s*co-authored-by:.*\b(claude|anthropic)\b|generated with \[?claude code")
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".cfg", ".toml", ".py", ".sh", ""}


def attribution(text):
    """The AI-attribution lines in `text` (commit messages, a pull request's body)."""
    return [m.group(0).strip() for m in ATTRIBUTION.finditer(text or "")]


def local_paths_in_diff(diff):
    """[(file, line)] for each added line of a unified diff (git diff -U0) that holds a local path."""
    out, path = [], None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++") and is_local_path(line):
            out.append((path, line[1:].strip()[:160]))
    return out


def local_paths_in_folder(folder, max_bytes=50_000_000):
    """[(file, line)] for each text file under `folder` (or `folder` itself, a file) that an upload would publish and that
    holds a local path."""
    out, root = [], Path(folder)
    for f in ([root] if root.is_file() else sorted(root.rglob("*"))):
        if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIXES or f.stat().st_size > max_bytes:
            continue
        try: text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError): continue
        out += [(str(f), line.strip()[:160]) for line in text.splitlines() if is_local_path(line)]
    return out


def refuse_local_paths(folder):
    """Raise before an upload (a folder or one file) whose files hold a local path."""
    hits = local_paths_in_folder(folder)
    if hits:
        raise SystemExit("refusing to upload: local paths in " + "; ".join(f"{f}: {l}" for f, l in hits[:5])
                         + (f" (and {len(hits) - 5} more)" if len(hits) > 5 else ""))
