### Added

- Conditional questions (#36): a question may carry `ask_if` ({question id: condition}) and is answered only when the
  answers it names meet their conditions (a choice's option keys, a noul's true/false, a score's range), in later passes
  of the same request. Otherwise it is reported as skipped. It is never model input: a request without `ask_if` is
  answered exactly as before. The server, `D1A.decide` and the MCP/clients' pass-through take it; permute and separate
  refuse it.
