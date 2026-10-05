"""Shared pieces of the equivalence checks: load a module as it was at a git commit, and compare two results exactly.

A behaviour-identical rewrite of a d1a module is checked by running the old module (from git) and the new one (the working
tree) on the same generated inputs and requiring the same results: equal values with the same types, floats equal to the
bit, dicts in the same key order, numpy arrays with the same dtype, the same exception type and message.
"""
import argparse
import importlib
import importlib.util
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module_at(ref, path):
    """The module at `path` (e.g. "d1a/api.py") as it was at git commit `ref`, imported under a private name. It imports
    the rest of d1a from the working tree."""
    source = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    copy = Path(tempfile.mkdtemp()) / f"old_{Path(path).stem}.py"
    copy.write_text(source, encoding="utf-8")
    package = ".".join(Path(path).with_suffix("").parts[:-1])   # "d1a": the old module's relative imports (from .api) resolve there
    importlib.import_module(package)
    spec = importlib.util.spec_from_file_location(f"{package}._old_{Path(path).stem}", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(value):
    """A comparable form of a result that keeps what == would forgive: types, float bits, key order, array dtypes."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, exact(value.tolist()))
    if isinstance(value, dict):
        return ("dict", [(exact(k), exact(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [exact(v) for v in value])
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return ("float", "nan" if math.isnan(value) else value.hex())
    if isinstance(value, np.integer):
        return ("int", int(value))
    return (type(value).__name__, value)


def outcome(call):
    """("ok", exact result) or ("error", exception type, message)."""
    try:
        return ("ok", exact(call()))
    except Exception as error:
        return ("error", type(error).__name__, str(error))


class Checker:
    """Counts comparisons and stops at the first difference with both outcomes."""

    def __init__(self):
        self.count = 0

    def same(self, label, old_call, new_call):
        self.count += 1
        old, new = outcome(old_call), outcome(new_call)
        if old != new:
            raise AssertionError(f"{label}: the old and new modules differ\n  old: {str(old)[:600]}\n  new: {str(new)[:600]}")

    def equal(self, label, old, new):
        self.count += 1
        if exact(old) != exact(new):
            raise AssertionError(f"{label}: the old and new modules differ\n  old: {str(old)[:600]}\n  new: {str(new)[:600]}")


def arguments(description, ref, iterations):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--ref", default=ref, help=f"the commit holding the module to compare against (default {ref}, the last commit before the rewrite)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=iterations)
    return ap.parse_args()
