### Fixed

- OpenAI chat refuses a request whose prompt and `max_tokens` (1024 when not sent) exceed the model's context
  (`max_position_embeddings`; 131,072 for D1A-E4B-it) with an immediate 400, OpenAI's `context_length_exceeded`,
  before any prefill (#319). A 237,592-token prompt was accepted 5 times on 2026-10-21 and held the model thread 16
  minutes each, so other chat clients (mem0) timed out. Clients that compress on that error, such as Hermes, now
  shorten and resend.

### Added

- `D1A_CHAT_CONTEXT` caps a chat's prompt plus `max_tokens` below the model's context (#319), with the same 400.
  A prompt near the model's context still holds the model for minutes (~230 tokens/s on an M4). Unset, the model's
  context is the limit.
