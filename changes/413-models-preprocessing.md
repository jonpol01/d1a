### Added

- `GET /v1/models` shows what the server rewrites in a state before the model reads it (#413):
  `"preprocessing": {"ja_dates": ..., "date_facts": ...}` when `D1A_JA_DATES` or `D1A_DATE_FACTS` is on. Absent when both
  are off, so that card is unchanged.
