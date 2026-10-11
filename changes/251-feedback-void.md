### Added

- `python -m d1a.learning.feedback void <log> --ids ... --src ... --reason ...` (and `FeedbackLog.void`): withdraws one
  source's outcomes on some decisions, with a reason, by appending a `void` event; the log is never rewritten. Every
  reader (resolved, pending, status, records, promote, the tick) drops the outcomes it covers, and an outcome that source
  sends later counts again. Written for the live PR labeler, where 40 "human" outcomes were the PR author's own labels set
  while opening the PR, not a person correcting D1A. Earlier d1a versions take every non-decision event for an outcome, so
  update the server and the tick before voiding.
