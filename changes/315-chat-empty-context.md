### Changed

- The chat log line marks an answer with no text and no tool call: `, empty answer after <roles>`, the roles of the
  last three messages (for example `user > assistant(tool_calls) > tool`), never their text (#315). The d1a sandbox
  bot got 3 such answers in 23 Hermes steps, and the counts alone could not say what came before them.
