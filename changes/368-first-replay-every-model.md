### Fixed

- The self-learning loop runs its gate on the first replay after every model change, not only after the first one it ever
  saw (#366). The flag that records that run is now reset with the other per-model state when the served model changes.
