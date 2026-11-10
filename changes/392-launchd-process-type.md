### Changed

- docs/CONFIGURATION.md says to run `d1a.serving.serve` as a launchd job with `ProcessType` `Interactive` (#17). With no
  `ProcessType`, launchd throttles the job's CPU, and the model server answered decisions at half speed (D1A-E2B-it QAT
  v0.7 on an Apple M1 Max: 4.7 a second against 9.5).
