### Added

- `d1a.learning.flags` (#237, part 1): learned decision flags. A flag is a group of a choice question's options plus a
  threshold, raised when the top answer is in the group or the group's probability reaches the threshold; it is read on
  the model's raw probabilities and never changes the answer. `python -m d1a.learning.flags fit` sets the threshold at a
  flag-rate budget on a labelled kit, `gate` serves it only if it catches more positives than the top answer on a later
  kit (bootstrap over groups) within the budget. Serving the flag and its settings come with part 2, after #233.
