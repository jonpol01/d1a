### Fixed

- The D1A-E2B-it QAT v0.7 MLX build, its mlx-community mirror and its Core ML build now carry the routing temperature 1.59,
  from tag `routing-t1.59` (#385), as the PyTorch build has since v0.7. Requests with `"use_case": "routing"` were read at the
  general 1.91. On the 640 factory development questions, not used for the fit, calibration error goes from 0.027 to 0.018,
  with accuracy unchanged. No answer's top choice changes. Not checked on the Core ML build.
