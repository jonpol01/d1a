### Added

- A versioned format for training runs (#64): `d1a_config.json` (format `d1a-torch`, version 1: the base, the head's size,
  the temperature, `weights`, and the run's recorded arguments as JSON) with the pointer head in `head.safetensors`, the
  file names MLX exports already use. `d1a.train` and `d1a.calibrate` write it, and loading it unpickles nothing. A run
  that has both it and a `head.pt` must say the same in both, or loading stops and names the field. Runs saved before
  it (head.pt only) load as before, with no end date; published checkpoints are unchanged.
