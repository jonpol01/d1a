### Added

- `python -m d1a.training.calibrate --run <checkpoint> --check` lists the use cases a checkpoint trained on (routing today)
  that have no temperature, and exits 1 (#382). `d1a.training.study publish` refuses such a checkpoint. A value kept on
  purpose is written with `--use-case NAME --temperature T --reason "..."`, which records it.

### Fixed

- The D1A-E4B-it QAT builds (MLX 4-bit, its mlx-community mirror, PyTorch) now carry a routing temperature, 1.41, from tag
  `routing-t1.41` (#382). Requests with `"use_case": "routing"` were read at the general 1.66 and came out under-confident
  out of sample. The fit set was uninformative (accuracy 0.99), so 1.41 is the routing method's point estimate, confirmed
  directionally out of sample (factory-development ECE 0.049 to 0.035); it is not claimed optimal. No answer's top choice changes.
