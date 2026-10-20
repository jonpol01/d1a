### Fixed

- OpenAI chat: an agent's step after a tool call failed with a 500 (#312). Clients such as Hermes and n8n send the
  earlier tool call's `arguments` back as a JSON string, as the OpenAI API defines it, and Gemma 4's chat template
  accepts only an object. The request now decodes them, and refuses arguments that are not a JSON object with a 422.
  Measured with D1A-E4B-it, an agent loop of 3 tool calls and a summary took 4.1 s cold, then 0.6–1.8 s a step from
  the prompt cache.
