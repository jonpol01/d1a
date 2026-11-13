### Fixed

- `POST /v1/feedback` answers 422 for a replay's `decision_id` and names the decision it re-scores (#401). A replay reads
  that decision's outcomes and never its own, so an outcome sent for it was accepted and silently lost.
  `FeedbackLog.outcome()` refuses it the same way.
