### Added

- Chat routes (#420, #419): with `D1A_ROUTES=<file>`, `POST /v1/chat/completions` sends a request to review a pull
  request or a piece of code to a review service and a hard request to a larger OpenAI-compatible model, and D1A answers
  the rest. The route comes from a yes/no question kept out of the decision log and from the route preset. Answers carry
  `x-d1a-route`; a failed route says why in `x-d1a-route-error`, and D1A never writes review text itself. Requires
  `D1A_API_KEY`. Unset, the endpoint is unchanged. See docs/USAGE.md, "Route chats to other models".
