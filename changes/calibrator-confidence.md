### Fixed

- The outcome calibrator (`D1A_OUTCOME_CALIBRATOR`) no longer serves a choice answer whose `confidence` contradicts its
  recalibrated `probabilities`: the confidence is read off the same probabilities (#195). It also no longer fits or
  applies a choice calibration to score answers, which carry probabilities by level too.
  `promote` drops such a choice calibration of a score question from the file it rewrites, so `/v1/feedback`'s
  `calibrated_questions` lists only what is applied.
