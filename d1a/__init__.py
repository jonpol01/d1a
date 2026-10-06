"""D1A: a small decision model. `from d1a import D1A` for in-process use (d1a.serving.lib); d1a.serving.serve for the HTTP server."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("d1a")   # pyproject.toml's, as installed; CHANGELOG.md has a section for it
except PackageNotFoundError:        # a source tree that was never installed
    __version__ = "0+unknown"


def __getattr__(name):   # lazy, so `import d1a.core.api` (pydantic only) does not load torch
    if name == "D1A":
        from d1a.serving.lib import D1A
        return D1A
    raise AttributeError(f"module 'd1a' has no attribute {name!r}")
