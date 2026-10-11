### Changed

- Temperature fits are about 38x faster on thousands of rows (#253): `d1a.eval.metrics.nll_on_grid` computes the grid
  losses per block of rows instead of once per row and temperature, and `fit_temperature` (so `d1a.training.calibrate`)
  and `temperature_gate.interval` use it. The fitted temperatures and intervals are identical, bit for bit.
