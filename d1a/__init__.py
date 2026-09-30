"""D1A: a small decision model. `from d1a import D1A` for in-process use (d1a.lib); d1a.serve for the HTTP server."""


def __getattr__(name):   # lazy, so `import d1a.api` (pydantic only) does not load torch
    if name == "D1A":
        from .lib import D1A
        return D1A
    raise AttributeError(f"module 'd1a' has no attribute {name!r}")
