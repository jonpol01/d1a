### Changed

- docs/FEATURE_TESTS.md and the pages that made each claim (#434), measured on the served build: Japanese dates pass on the
  same 300 synthetic documents (75.0 → 81.7, era dates 76.3 → 81.7 with more false urgent answers, now stated); the routing
  tier pass sorts 269 of 270 general prompts small vs not-small; self-learning is reworded to what the one deployment shows
  (automatic promotion paused, no accuracy gain shown, the severity flag in shadow); Core ML is measured on Macs only, and the
  ONNX build's unreported phone speeds are removed; video stays a demo.
