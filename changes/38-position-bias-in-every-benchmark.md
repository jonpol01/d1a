### Added

- Position bias in every benchmark report (#38). `position_bias`: how often a clean choice question's answer is its first
  option, against how often its label is. `identical_options` (`d1a.benchmark --identical-options N`, default 100; 0
  for none): the first N clean choice questions asked again with every option the first option's text, as a score
  question whose levels may repeat; it reports how far the answers are from uniform and how often the first slot is
  strictly the top one, with each control's answer in identical_options.json. The suite's own predictions and rows are
  unchanged.
