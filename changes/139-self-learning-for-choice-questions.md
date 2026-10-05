### Added

- Self-learning for choice questions (#139). `d1a.feedback`'s `OutcomeCalibrator` fits one temperature per choice
  question on the outcomes, which changes how sure an answer is but never which option it picks; `gate_choice()` applies
  the promotion rule to the multi-class log loss. Outcomes name their source (`POST /v1/feedback`'s `src`, or
  `meta["src"]`): labels merge per question, and a person's (`human`) beats another model's (e.g. `reviewer`) whatever
  arrived last. `python -m d1a.feedback status|records|calibrate --src` keeps one source. First stream: the PR labeler's
  type and blast-radius questions, with the review bot's labels as outcomes.
