### Added

- An opt-in rewrite of Japanese dates as `Month D, YYYY` before the decision pass (#412): `D1A_JA_DATES=1` on the server,
  or `"ja_dates": true` on a request (D1A's own field; `false` keeps the dates as written either way). It covers
  `2026年8月8日`, `2026/8/8`, `令和`/`平成`/`昭和` era dates, full-width digits and ranges, and leaves year-less dates,
  dates that are not real days and relative terms as written. A state with no Japanese is never touched. Off by default,
  so answers are unchanged. On Japanese documents whose dates alone were rewritten this way, urgency accuracy rose from
  75.0% to 81.7% (83.0% in English). `D1A.decide(..., ja_dates=True)`, `d1a.eval.benchmark --ja_dates` and both clients
  pass it; a replay sends a request's own `ja_dates` again.
