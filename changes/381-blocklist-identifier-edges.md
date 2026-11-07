### Fixed

- `D1A_DATA_BLOCKLIST` now also refuses a listed name inside a code identifier (`<Name>Tests`, `<name>_ios`, `IOS<Name>`)
  and with a plural s/es after it (#378). Before, these passed `freeze`, suite loading, `d1a.training.train` and
  `feedback reword`. A short name inside a longer word (a lowercase or all-caps continuation) still does not match.
