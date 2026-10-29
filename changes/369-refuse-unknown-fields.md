### Changed

- A /v1/systemone request (or `D1A.decide`) that carries a field D1A does not define, on the request or on a question, is
  refused with a 422 naming that field. It used to be dropped: a question written with `"instr"` instead of
  `"instructions"` was answered with no question text (#340). A question's `type` now selects its schema, so the error
  names the one bad field instead of listing every question type.

### Fixed

- The server's start-up self-check and `recipes/swe-verifier/zero_shot.py` wrote `"instr"`, so both asked their questions
  with no text. They now send `"instructions"` (#340).
