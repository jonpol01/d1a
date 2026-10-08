### Fixed

- `d1a.training.calibrate --judge` judges a fine-tune's refit against the temperature its `--init_from` checkpoint serves
  (#207). Training writes every run at temperature 1.0, so a refit was compared against an uncalibrated model and passed
  the rule where it should not. A fine-tune not calibrated since training now takes its init's served temperature as the
  incumbent, and says so; `--incumbent <run|T>` names another (it needs `--judge`). The incumbent and where it came from
  are recorded in the checkpoint's `temperature_fit.rule.incumbent`. A run trained from a base model is judged as before.
