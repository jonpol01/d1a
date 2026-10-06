"""Moved to d1a.training.train (#60). This path works for one more release and is removed in D1A 0.5."""
import sys
import warnings

warnings.warn("d1a.train moved to d1a.training.train; the old path is removed in D1A 0.5", DeprecationWarning, stacklevel=2)
if __name__ == "__main__":
    import runpy
    runpy.run_module("d1a.training.train", run_name="__main__", alter_sys=True)
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("d1a.training.train")
