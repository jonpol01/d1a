### Added

- `GET /v1/models` shows each chat route's live state (#440): `state` (`ready`, `busy` with every slot in use, or
  `offline` when its target could not be reached lately), `in_flight`, `max_in_flight` and `waiting`. With no chat to the
  target for a minute, reading the card asks the target's `/health` (no key, at most every 15 s). Never a URL, a key or
  the target's error text. The card also has `route_rule`: the route preset's fail-up thresholds at the routing
  temperature, the rule the chat routes apply.
