"""Release notes for a version, from CHANGELOG.md and the change fragments, and the checks the release workflow runs.

    python scripts/release_notes.py 0.2.0                    # print that version's section as the GitHub release text
    python scripts/release_notes.py --check v0.2.0           # the tag, pyproject.toml and CHANGELOG.md agree
    python scripts/release_notes.py --check-fragments        # every file in changes/ parses
    python scripts/release_notes.py assemble 0.4.0 [--date YYYY-MM-DD] [--write]   # the version's section from changes/
    python scripts/release_notes.py --release-check v0.4.0   # --check, and no fragment left unassembled (release.yml)

Each change adds one fragment, changes/<PR number>-<slug>.md, instead of editing CHANGELOG.md, so pull requests never
conflict on its Unreleased lines. A fragment is `### Section` headings (Added, Changed, Deprecated, Removed, Fixed,
Security) with `- ` list items under them, wrapped lines indented two spaces. At release, `assemble --write` moves every
fragment into a new `## [X.Y.Z] - date` section (newest PR first) and deletes them; the release text is then written by
hand around it (What's new, checkpoints, Upgrade notes; CONTRIBUTING.md).

A section starts at `## [X.Y.Z] - YYYY-MM-DD` and ends at the next `## ` heading. CHANGELOG.md is wrapped for reading
as a file; GitHub renders every line break of a release text, so the release text joins wrapped lines back together.
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGES = ROOT / "changes"
SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")   # Keep a Changelog's order
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


# --- change fragments ---------------------------------------------------------------------------------------------------

def parse(text, name):
    """({section: [item, ...]}, errors) of one fragment, or of an Unreleased section, which has the same shape."""
    out, current, errors = {}, None, []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            if title not in SECTIONS:
                errors.append(f"{name}:{n}: unknown section {title!r}; use one of {', '.join(SECTIONS)}")
                current = None
            elif title in out:
                errors.append(f"{name}:{n}: '### {title}' appears twice")
                current = out[title]
            else:
                current = out[title] = []
        elif line.startswith("- ") and current is not None:
            current.append(line.rstrip())
        elif line.startswith("  ") and current:
            current[-1] += "\n" + line.rstrip()
        else:
            errors.append(f"{name}:{n}: stray text; items start with '- ' under a '### Section' heading, wrapped lines with two spaces")
    errors += [f"{name}: '### {s}' has no items" for s, items in out.items() if not items]
    if not out and not errors:
        errors.append(f"{name}: no '### Section' with items")
    return out, errors


def order(path):
    """Newest first: by the PR number the file name starts with, then fragments without one, by name."""
    m = re.match(r"(\d+)", path.name)
    return (0, -int(m[1]), path.name) if m else (1, 0, path.name)


def fragments(directory=None):
    """[(path, {section: [item]})] in assembly order, and the parse errors of every file."""
    out, errors = [], []
    for path in sorted((p for p in Path(directory or CHANGES).glob("*.md") if p.name != "README.md"), key=order):
        sections, errs = parse(path.read_text(encoding="utf-8"), path.name)
        out.append((path, sections)); errors += errs
    return out, errors


def render(sections):
    return "\n\n".join(f"### {s}\n\n" + "\n".join(sections[s]) for s in SECTIONS if sections.get(s))


def assemble(version, date, text, frags):
    """CHANGELOG text with the Unreleased section's items and every fragment's moved into a new `## [version] - date`
    section under `## [Unreleased]`, which is left empty. -> (new text, the new section's body)."""
    if section(version, text) is not None:
        raise ValueError(f"CHANGELOG.md already has a section for {version}")
    start = text.index("## [Unreleased]") + len("## [Unreleased]")
    end = text.find("\n## ", start)
    unreleased, errors = parse(text[start:end], "CHANGELOG.md [Unreleased]") if text[start:end].strip() else ({}, [])
    if errors:
        raise ValueError("\n".join(errors))
    merged = {s: [i for _, f in frags for i in f.get(s, [])] + unreleased.get(s, []) for s in SECTIONS}
    body = render(merged)
    if not body:
        raise ValueError("nothing to release: no fragments in changes/ and an empty Unreleased section")
    return text[:start] + f"\n\n## [{version}] - {date}\n\n{body}\n" + text[end:], body


def release_check(tag):
    """check(), and no fragment left in changes/: one would otherwise ship in the next release's notes instead."""
    left = [p.name for p, _ in fragments()[0]]
    return check(tag) + ([f"changes/ still holds {len(left)} fragments ({', '.join(left[:5])}...): run assemble --write and commit"] if left else [])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check-fragments"]:
        frags, errors = fragments()
        print("\n".join(errors) or f"{len(frags)} fragments in changes/ parse")
        sys.exit(1 if errors else 0)
    if sys.argv[1:2] == ["--release-check"]:
        problems = release_check(sys.argv[2])
        print("\n".join(problems) or f"release {sys.argv[2]}: tag, pyproject.toml and CHANGELOG.md agree, and every fragment is assembled")
        sys.exit(1 if problems else 0)
    if sys.argv[1:2] == ["assemble"]:
        import argparse
        import datetime
        ap = argparse.ArgumentParser(prog="release_notes.py assemble")
        ap.add_argument("cmd"); ap.add_argument("version"); ap.add_argument("--date", default=datetime.date.today().isoformat())
        ap.add_argument("--write", action="store_true", help="rewrite CHANGELOG.md and delete the assembled fragments")
        a = ap.parse_args()
        frags, errors = fragments()
        if errors:
            sys.exit("\n".join(errors))
        text, body = assemble(a.version, a.date, (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), frags)
        print(f"## [{a.version}] - {a.date}\n\n{body}")
        if a.write:
            (ROOT / "CHANGELOG.md").write_text(text, encoding="utf-8")
            for path, _ in frags:
                path.unlink()
        sys.exit(0)
    if sys.argv[1:2] == ["--check"]:
        problems = check(sys.argv[2])
        print("\n".join(problems) or f"release {sys.argv[2]}: tag, pyproject.toml and CHANGELOG.md agree")
        sys.exit(1 if problems else 0)
    notes = section(sys.argv[1].removeprefix("v"))
    if notes is None: sys.exit(f"no CHANGELOG.md section for {sys.argv[1]}")
    print(unwrap(notes))
