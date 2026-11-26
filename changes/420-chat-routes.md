### Added

- Chat routes (#420, #419): with `D1A_ROUTES=<file>`, `POST /v1/chat/completions` sends a request to review a pull
  request or a piece of code to a review service and a chat that needs a larger model (the route preset's tier, from
  `min_tier`) to an OpenAI-compatible one, and D1A answers the rest. Answers carry `x-d1a-route`; a failed route says
  why in `x-d1a-route-error`, and D1A never writes review text itself. Open to every caller unless the file sets a
  `client_key_file` (or `D1A_API_KEY` is set). Unset, the endpoint is unchanged. See docs/USAGE.md, "Route chats to
  other models".
