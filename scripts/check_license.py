"""License and provenance check: D1A is built on Kev (Apache-2.0), and these rules keep that compliance mechanical.

    python scripts/check_license.py                        # full check; fetches Kev at UPSTREAM_COMMIT (needs git + network)
    python scripts/check_license.py --upstream-dir PATH    # use an existing checkout of Kev at UPSTREAM_COMMIT
    python scripts/check_license.py --skip-upstream        # offline: every rule except the Kev-origin similarity scan

Rules (each violation is printed on its own line; exit status 1 if there is any):
 1. LICENSE is the Apache-2.0 text and keeps "Copyright 2026 Jared Palmer".
 2. NOTICE credits Kev, Jared Palmer, the Kev URL, the Apache License, Gemma and John Soliva.
 3. pyproject.toml lists Jared Palmer among the authors, and the project is not named "kev".
 4. docs/UPSTREAM.md exists and names the upstream base commit.
 5. The files whose first lines carry the "Modified from Kev" header are exactly the "Derived files" of docs/UPSTREAM.md.
 6. Every file still substantially similar to its Kev original (difflib ratio >= SIMILARITY on normalised lines) is listed
    in docs/UPSTREAM.md, and a file listed as an unmodified copy is byte-identical to Kev's.
 7. Branding: the README's title is not Kev's, the README credits Kev and disclaims endorsement, and no package, CLI or
    UI name is "kev" (ALLOWED lists the exceptions and why).
Standard library only.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/jaredpalmer/kev"
UPSTREAM_COMMIT = "0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91"
HEADER = re.compile(r"^\s*(#|//|<!--|/\*|\*)\s*Modified from Kev \(https://github\.com/jaredpalmer/kev\)")
HEADER_LINES = 15
SIMILARITY = 0.5
PATH_MAP = [("d1a/", "kev/")]   # D1A path -> Kev path (renames)
SECTIONS = {"derived": "Derived files", "derived_noheader": "Derived files without a header", "copies": "Unmodified copies"}

# "kev" is allowed in user-facing files only inside these, each for a reason:
ALLOWED = [
    (r"kev-latest", "compatibility alias: clients written against Kev send it, and the server keeps accepting it"),
    (r"jaredpalmer/kev[\w./-]*", "credit and upstream links, and Kev's Hub ids (the kev-suites dataset, Kev checkpoints)"),
    (r"jonpol01/kev[\w./-]*", "John's Kev fork and the kev-usecases-poc demo repo, both named before D1A existed"),
    (r"JohnP1/kev-gemma4-e2b", "the prototype checkpoint's Hub id, published before the rename"),
    (r"Built on Kev|built on Kev|Modified from Kev|from Kev|by the Kev authors|Kev authors", "credit lines"),
]
USER_FACING = ["clients/js/package.json", "clients/python/pyproject.toml", "pyproject.toml"]


def git_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return sorted(f for f in set(out.splitlines()) if (ROOT / f).is_file())


def is_binary(data):
    return b"\0" in data[:8192]


def has_header(path):
    try:
        head = path.read_text(encoding="utf-8").splitlines()[:HEADER_LINES]
    except (UnicodeDecodeError, OSError):
        return False
    return any(HEADER.search(line) for line in head)


def read_lists(errors):
    """{section key: set of paths} from docs/UPSTREAM.md's fenced blocks under the SECTIONS headings."""
    path = ROOT / "docs/UPSTREAM.md"
    if not path.is_file():
        errors.append("docs/UPSTREAM.md is missing (rule 4)")
        return None, ""
    text = path.read_text(encoding="utf-8")
    lists = {}
    for key, title in SECTIONS.items():
        m = re.search(rf"^## {re.escape(title)}\s*$.*?^```[a-z]*\n(.*?)^```", text, re.M | re.S)
        if not m:
            errors.append(f"docs/UPSTREAM.md has no fenced list under '## {title}' (rule 5)")
            lists[key] = set()
            continue
        lists[key] = {line.strip() for line in m.group(1).splitlines() if line.strip() and not line.startswith("#")}
    return lists, text


def listed(path, lists):
    """Which list names a path: derived, derived_noheader, copies (directory entries end in '/'), or None."""
    for key in ("derived", "derived_noheader"):
        if path in lists[key]:
            return key
    if path in lists["copies"] or any(e.endswith("/") and path.startswith(e) for e in lists["copies"]):
        return "copies"
    return None


def upstream_path(path):
    for ours, theirs in PATH_MAP:
        if path == ours or (ours.endswith("/") and path.startswith(ours)):
            return theirs + path[len(ours):]
    return path


def normalised(data):
    lines = data.decode("utf-8", "replace").splitlines()
    return [line.strip() for line in lines if line.strip() and not HEADER.search(line) and "Changes for D1A" not in line]


def similarity(ours, theirs):
    if ours == theirs:
        return 1.0
    if is_binary(ours) or is_binary(theirs):
        return 0.0
    matcher = difflib.SequenceMatcher(None, normalised(ours), normalised(theirs), autojunk=False)
    return matcher.ratio() if matcher.real_quick_ratio() >= SIMILARITY and matcher.quick_ratio() >= SIMILARITY else 0.0


def fetch_upstream(target):
    for cmd in (["git", "init", "-q", str(target)],
                ["git", "-C", str(target), "fetch", "-q", "--depth", "1", UPSTREAM_URL, UPSTREAM_COMMIT],
                ["git", "-C", str(target), "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, check=True)
    return target


def check_legal_files(errors):
    lic = ROOT / "LICENSE"
    text = lic.read_text(encoding="utf-8") if lic.is_file() else ""
    if not text:
        errors.append("LICENSE is missing (rule 1)")
    else:
        for needle in ("Apache License", "Version 2.0, January 2004", "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
                       "Copyright 2026 Jared Palmer"):
            if needle not in text:
                errors.append(f"LICENSE does not contain {needle!r} (rule 1)")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8") if (ROOT / "NOTICE").is_file() else None
    if notice is None:
        errors.append("NOTICE is missing (rule 2)")
    else:
        for needle in ("Kev", "Jared Palmer", UPSTREAM_URL, "Apache License", "Gemma", "John Soliva"):
            if needle not in notice:
                errors.append(f"NOTICE does not mention {needle!r} (rule 2)")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    if not any(a.get("name") == "Jared Palmer" for a in project.get("authors", [])):
        errors.append("pyproject.toml [project].authors does not include Jared Palmer (rule 3)")
    if str(project.get("name", "")).lower() == "kev":
        errors.append("pyproject.toml [project].name is 'kev' (rule 3)")
    for script in project.get("scripts", {}):
        if "kev" in script.lower():
            errors.append(f"pyproject.toml exposes a CLI named {script!r} (rule 7)")


def check_provenance(files, lists, upstream_text, errors):
    if UPSTREAM_COMMIT[:7] not in upstream_text:
        errors.append(f"docs/UPSTREAM.md does not name the upstream base commit {UPSTREAM_COMMIT[:7]} (rule 4)")
    headed = {f for f in files if has_header(ROOT / f)}
    for f in sorted(lists["derived"] - headed):
        errors.append(f"{f}: listed under 'Derived files' but has no 'Modified from Kev' header in its first {HEADER_LINES} lines (rule 5)")
    for f in sorted(headed - lists["derived"]):
        errors.append(f"{f}: carries the 'Modified from Kev' header but is not listed under 'Derived files' (rule 5)")
    present = set(files)
    for key in SECTIONS:
        for f in sorted(lists[key]):
            if not f.endswith("/") and f not in present:
                errors.append(f"docs/UPSTREAM.md lists {f} under '{SECTIONS[key]}', but it is not in the repository")


def check_origin(files, lists, upstream, errors):
    for f in files:
        theirs = upstream / upstream_path(f)
        if not theirs.is_file():
            continue
        ours_bytes, theirs_bytes = (ROOT / f).read_bytes(), theirs.read_bytes()
        where = listed(f, lists)
        if where == "copies":
            if ours_bytes != theirs_bytes:
                errors.append(f"{f}: listed as an unmodified copy but differs from Kev's {upstream_path(f)}; add the header and list it "
                              "under 'Derived files' (rule 6)")
            continue
        if where is None and similarity(ours_bytes, theirs_bytes) >= SIMILARITY:
            errors.append(f"{f}: still substantially similar to Kev's {upstream_path(f)} but not listed in docs/UPSTREAM.md (rule 6)")


def check_branding(files, errors):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    title = next((line for line in readme.splitlines() if line.startswith("# ")), "")
    if "kev" in title.lower():
        errors.append(f"README.md's title mentions Kev: {title!r} (rule 7)")
    if "Built on Kev" not in readme or UPSTREAM_URL not in readme:
        errors.append(f"README.md lacks a 'Built on Kev' credit linking {UPSTREAM_URL} (rule 7)")
    if not re.search(r"not affiliated with", readme) or "endorsed" not in readme:
        errors.append("README.md lacks the statement that D1A is not affiliated with or endorsed by the Kev authors (rule 7)")
    allowed = re.compile("|".join(f"(?:{p})" for p, _ in ALLOWED))
    for f in files:
        if not any(f == u or f.startswith(u.rstrip("/") + "/") for u in USER_FACING):
            continue
        try:
            text = (ROOT / f).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if HEADER.search(line) or "Changes for D1A" in line:
                continue
            if re.search(r"kev", allowed.sub("", line), re.I):
                errors.append(f"{f}:{n}: user-facing 'kev' outside the allowlist: {line.strip()[:120]} (rule 7)")
    for name in ("clients/js/package.json",):
        if (ROOT / name).is_file() and "kev" in json.loads((ROOT / name).read_text(encoding="utf-8")).get("name", "").lower():
            errors.append(f"{name}: package name contains 'kev' (rule 7)")
    if (ROOT / "kev").exists():
        errors.append("a top-level 'kev' package directory exists (rule 7)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-upstream", action="store_true", help="skip the Kev-origin similarity scan (no network)")
    ap.add_argument("--upstream-dir", help="an existing checkout of Kev at UPSTREAM_COMMIT instead of fetching it")
    a = ap.parse_args()
    errors, files = [], git_files()
    check_legal_files(errors)
    lists, upstream_text = read_lists(errors)
    if lists is not None:
        check_provenance(files, lists, upstream_text, errors)
        if not a.skip_upstream:
            if a.upstream_dir:
                check_origin(files, lists, Path(a.upstream_dir), errors)
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    check_origin(files, lists, fetch_upstream(Path(tmp) / "kev"), errors)
    check_branding(files, errors)
    for e in errors:
        print(f"license check: {e}")
    print(f"license check: {'FAILED, ' + str(len(errors)) + ' violation(s)' if errors else 'ok'}"
          f"{' (upstream scan skipped)' if a.skip_upstream else ''}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
