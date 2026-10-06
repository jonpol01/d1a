"""Moved to d1a.backends.torch (#60). This path works for one more release and is removed in D1A 0.5."""
import sys
import warnings

warnings.warn("d1a.model moved to d1a.backends.torch; the old path is removed in D1A 0.5", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    import runpy
    runpy.run_module("d1a.backends.torch", run_name="__main__", alter_sys=True)
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("d1a.backends.torch")
