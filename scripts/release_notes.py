"""Release notes for a version, from CHANGELOG.md, and the checks the release workflow runs.

    python scripts/release_notes.py 0.2.0             # print that version's section as the GitHub release text
    python scripts/release_notes.py --check v0.2.0    # the tag, pyproject.toml and CHANGELOG.md agree

A section starts at `## [X.Y.Z] - YYYY-MM-DD` and ends at the next `## ` heading. CHANGELOG.md is wrapped for reading
as a file; GitHub renders every line break of a release text, so the release text joins wrapped lines back together.
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$", re.M)


def package_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def section(version, text=None):
    """The body of `version`'s changelog section, without its heading; None when there is none."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8") if text is None else text
    for m in HEADING.finditer(text):
        if m[1] == version:
            end = text.find("\n## ", m.end())
            body = text[m.end(): end if end != -1 else len(text)]
            return re.sub(r"\n\[[^\]]+\]: \S+", "", body).strip()   # the link references at the end of the file
    return None


STARTS_BLOCK = re.compile(r"^(#|\||```|[-*] |\d+\. )")   # a line that starts its own block, never joined to the one above


def unwrap(text):
    """Join hard-wrapped lines back into their paragraph or list item; headings, table rows, code fences and blank lines
    are left as they are."""
    out = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        joinable = (line.strip() and prev.strip() and not STARTS_BLOCK.match(line.lstrip())
                    and not prev.lstrip().startswith(("#", "|", "```")))
        if joinable:
            out[-1] = prev.rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def check(tag):
    """Errors that should stop a release of `tag` (vX.Y.Z)."""
    errors, version = [], tag.removeprefix("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version): errors.append(f"tag {tag!r} is not vMAJOR.MINOR.PATCH")
    if package_version() != version: errors.append(f"pyproject.toml says {package_version()}, the tag says {version}")
    body = section(version)
    if not body: errors.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section, or it is empty")
    return errors


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        problems = check(sys.argv[2])
        print("\n".join(problems) or f"release {sys.argv[2]}: tag, pyproject.toml and CHANGELOG.md agree")
        sys.exit(1 if problems else 0)
    notes = section(sys.argv[1].removeprefix("v"))
    if notes is None: sys.exit(f"no CHANGELOG.md section for {sys.argv[1]}")
    print(unwrap(notes))
