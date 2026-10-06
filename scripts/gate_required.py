"""Whether a pull request needs scripts/quality_gate.py (AGENTS.md, Quality bar): it changes a module the model server
loads (d1a.serving.serve, d1a.serving.lib, d1a.serving.media and every d1a module they import, found by reading the
imports, lazy ones included) or the dependencies. CI's quality-gate job (.github/workflows/quality-gate.yml) asks this
for every pull request and leaves the required `quality-gate` status pending until the gate passes on the exact head.

    python scripts/gate_required.py <changed file>...     # the reasons, one per line; nothing when the gate is not required
    python base/scripts/gate_required.py --root <pr checkout> <changed file>...    # CI: the base's rules on the PR's tree
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ("d1a/serving/serve.py", "d1a/serving/lib.py", "d1a/serving/media.py")
DEPENDENCIES = ("pyproject.toml", "uv.lock")


def module_file(name, root=ROOT):
    """d1a.x.y -> its file relative to root (d1a/x/y.py or d1a/x/y/__init__.py), or None outside d1a."""
    if name != "d1a" and not name.startswith("d1a."):
        return None
    base = root / name.replace(".", "/")
    for f in (base.with_suffix(".py"), base / "__init__.py"):
        if f.exists():
            return f.relative_to(root).as_posix()
    return None


def imports(path, root=ROOT):
    """The d1a modules a file imports anywhere in it (top level or inside functions), as files, with their packages."""
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    package = Path(path).parent.as_posix().replace("/", ".")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:   # relative: from .x import y
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parts + ([base] if base else []))
            names.add(base)
            names |= {f"{base}.{a.name}" for a in node.names}   # from d1a.core import api: api may be a module
    files = set()
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):   # every package on the way is imported too (its __init__.py runs)
            f = module_file(".".join(parts[:i]), root)
            if f:
                files.add(f)
    return files


def server_modules(root=ROOT):
    """Every d1a file the model server loads: SERVER and what they import, transitively."""
    seen, todo = set(), [f for f in SERVER if (root / f).exists()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo += sorted(imports(f, root) - seen)
    return seen


def reasons(changed, root=ROOT):
    loaded = server_modules(root)
    return [f"{f}: the model server loads it" if f in loaded else f"{f}: a dependency change"
            for f in sorted(set(changed)) if f in loaded or f in DEPENDENCIES]


if __name__ == "__main__":
    args = sys.argv[1:]
    root = ROOT
    if args[:1] == ["--root"]:
        root, args = Path(args[1]).resolve(), args[2:]
    print("\n".join(reasons(args, root)))
