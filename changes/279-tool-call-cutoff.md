### Fixed

- OpenAI chat with `tools`: an answer that ran out of `max_tokens` inside a tool call returned the half-written Gemma 4
  markup as `content` (#278). It now returns the text before it and no tool call, with `finish_reason` `"length"`.
