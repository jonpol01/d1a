### Fixed

- The outcome calibrator (`D1A_OUTCOME_CALIBRATOR`) no longer serves a choice answer whose `confidence` contradicts its
  recalibrated `probabilities`: the confidence is read off the same probabilities (#195). It also no longer fits or
  applies a choice calibration to score answers, which carry probabilities by level too.
