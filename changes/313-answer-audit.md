### Added

- The request header `x-d1a-audit: on` adds an `audit` block to a `/v1/systemone` or `/v1/systemone/media` answer (#32,
  #313). It records what produced the answer, for a port or a support ticket:
  - the d1a version, the run, and the weights' sha256;
  - the sha256 of the state text the model read;
  - the temperature and its source;
  - the questions the outcome calibrator recalibrated, and the decode rules and flags applied;
  - the prefix cache per model pass, and whether the decision was logged.

  It holds hashes and settings, never text. Without the header, responses are unchanged.
