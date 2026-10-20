### Added

- The model server prints a line per chat: prompt tokens, how many came from the prompt cache, completion tokens and
  time (#306). It holds counts only, never text. `D1A_CHAT_LOG=0` turns it off. docs/CONFIGURATION.md gains a Logs
  section: what the server writes, what it never holds, and where the Mac mini keeps it.
