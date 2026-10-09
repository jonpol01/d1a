### Added

- `d1a.serving.serve`: a request that sends the header `x-d1a-decision-log: off` is answered but kept out of the decision
  log (`D1A_FEEDBACK_LOG`), on `/v1/systemone`, `/v1/systemone/media`, `/permute` and `/separate`. The playground's demos
  and its smoke test send it, so the log's pending decisions are the ones an outcome can still follow.
