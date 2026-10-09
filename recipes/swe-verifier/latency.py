"""Seconds per run from saved zero_shot.py outputs: the median and 95th percentile of each file's per-run timings, and
which model (the `run` field) scored it.

    uv run python recipes/swe-verifier/latency.py runs/swe-verifier/eval/<file>.jsonl ...

zero_shot.py records `seconds` around each D1A.decide call (the model already loaded), so this is the per-run decision
time on the machine that scored the file. Prints aggregates only.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        rows = [json.loads(l) for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
        s = [r["seconds"] for r in rows if "seconds" in r]
        if not s:
            print(f"{f}: {len(rows)} runs, no timings (scored with --no-d1a?)"); continue
        print(f"{f}: {len(rows)} runs, model {sorted({r.get('run') for r in rows if r.get('run')})}, "
              f"median {np.median(s):.2f} s, p95 {np.percentile(s, 95):.2f} s")


if __name__ == "__main__":
    main()
