"""CI checks on a pull request (see d1a/core/hygiene.py):
    gh api ... commit messages | python scripts/check_hygiene.py attribution    # plus the PR body via $BODY
    git diff -U0 origin/main...HEAD | python scripts/check_hygiene.py paths
Exit 1 with a line per finding."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d1a.core.hygiene import attribution, local_paths_in_diff  # noqa: E402


def main(argv):
    text = sys.stdin.read()
    if argv[:1] == ["attribution"]:
        hits = attribution(text + "\n" + os.environ.get("BODY", ""))
        for h in hits: print(f"::error::AI attribution: {h}")
        if hits: print("::error::commits and pull requests carry no AI attribution (CONTRIBUTING.md)")
    elif argv[:1] == ["paths"]:
        hits = local_paths_in_diff(text)
        for f, line in hits: print(f"::error file={f}::a local path in an added line: {line}")
        if hits: print("::error::no local machine paths in the repository (d1a/core/hygiene.py)")
    else:
        sys.exit("usage: check_hygiene.py attribution|paths < input")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
