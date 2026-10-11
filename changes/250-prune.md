### Removed

- What D1A no longer needed (#65, #250): three unused `os` imports, the remnants of the 0.3 import shims removed in 0.5
  (`kev_share.py`'s `SHIMS`, the `_layout.py` docstring, a test fixture), six unused test imports, and
  `experiments/sft-v1-lengths.json` (Kev's SFT token counts, read by nothing). The survey found no Kev-only research code
  left; `docs/UPSTREAM.md` now explains how `kev_share` counts and why from-scratch rewrites that match Kev by design stay
  counted. Kev share: model code 26%, total 32% (both unchanged at the percent).
