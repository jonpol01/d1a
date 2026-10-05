### Fixed

- `d1a.benchmark` no longer crashes after scoring a suite when an identical-option control (#38) is too long: the
  first option repeated K times can outgrow the row its question fit in (20 of decision-v2's first 100 on Gemma 4).
  Such controls are skipped and counted in `identical_options.skipped_overlong`, and the report is written.
