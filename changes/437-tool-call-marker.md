### Fixed

- OpenAI-format chat reads a tool call only where the chat template writes it, right after its `<|tool_call>` marker
  (#437). Text elsewhere that looks like a call, in the answer or inside the argument of a call cut off by `max_tokens`,
  is no longer returned as a tool call.
