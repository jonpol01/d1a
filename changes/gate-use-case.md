### Added

- `scripts/quality_gate.py --use-case NAME` sends `"use_case": NAME` with the routing requests: the playground's
  routing demo and any request asking a `d1a.agents.presets` question set, as the presets' clients send them. Both
  servers get the same requests, and a server without the field ignores it. So a change to the use-case temperature
  path can be gated with the field it serves. Without the flag every request goes out as before.
