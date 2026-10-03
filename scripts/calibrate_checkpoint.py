"""Kept so older commands keep working: the same as `python -m d1a.calibrate` (its docstring has the details)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1a.calibrate import main  # noqa: E402

if __name__ == "__main__":
    main()
