### Fixed

- The self-learning gate no longer promotes noise as often (#364). Each gate run can promote, and `tick` re-ran the gate
  after every 10 new outcomes, so on a slowly growing log a calibrator fitted to pure noise got promoted in a third to a
  half of replays by 1,000 outcomes. The gate now needs at least `promotion.min_held_out` (30) held-out groups before it
  may promote, and either trigger re-runs it only once the outcomes have grown by `schedule.growth` (0.5) since its last
  run. With these defaults, the same replays promote noise in under 10%. `schedule.growth: 0` keeps the old schedule.
- A changed `promotion.serve_choice` or `promotion.hold` now reaches the server at the next `tick`. Before, it was
  written into the calibrator file only at the next promotion, which may never come.
