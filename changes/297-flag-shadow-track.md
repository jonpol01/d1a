### Added

- Learned flags can run in shadow mode (`flags.<name>.mode = "shadow"`): the flag is logged but not served, so the response
  is unchanged (#297). A tracked threshold (`flags.<name>.track`) is refit by the loop's tick from logged scores without
  labels, clamped, with a `flag_alert` notification when the recent flag rate runs above `alert_rate` or `t` sits at a clamp.
