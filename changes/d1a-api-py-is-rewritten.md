### Changed

- `d1a/api.py` is rewritten in D1A's own code, no longer derived from Kev. The schema, the text the model reads, the
  answers and the validation errors are unchanged: they were checked identical to the previous version on 166,718
  generated requests, distributions and dates. Its tests are rewritten too, as `tests/test_system_one.py`.
