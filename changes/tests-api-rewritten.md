### Changed

- The HTTP API tests (`tests/test_api.py`) are rewritten as D1A's own and run in the unit suite: they start
  `d1a.serving.serve`'s app in process on the committed tiny checkpoint (CPU, a few seconds), or check a running
  server when `D1A_BASE_URL` is set (`pytest tests/test_api.py -m server`). They keep every old check (answer shapes,
  null and structured criteria, the one-level score, 422s, branch isolation, the model names and cards, the request id,
  the TypeSafe SDK clients) and add more: probabilities in the request's order summing to 1 within TypeSafe's 0.02, the
  confidence and expected-score formulas, 422s at both option limits, each question alone, reversed and through
  `/v1/systemone/separate`, and a client's request id echoed back. Of 24 bugs planted in the API and server code, the
  old tests caught 17 and the new ones 23, including all 17; the one neither catches (a yes/no answer reporting P(no))
  is caught by `tests/test_conformance.py`.
