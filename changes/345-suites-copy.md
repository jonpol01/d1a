### Changed

- The frozen suites' large partitions download from D1A's own copy of Kev's `jaredpalmer/kev-suites` (#345), so
  training mixes no longer depend on that dataset staying up. The copy is byte for byte and every manifest pin is
  unchanged; an account that cannot read the copy falls back to Kev's dataset and says so.
