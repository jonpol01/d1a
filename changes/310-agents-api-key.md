### Fixed

- The MCP tools, the Claude Code gate hook and `python -m d1a.agents.demo` send `D1A_API_KEY` to the server (#310).
  A server that requires a key refused every call before, and the gate hook then asked about every command.
- `python -m d1a.agents.demo` keeps its decisions out of the server's learning log (`x-d1a-decision-log: off`), so it
  is safe to run against a live server. Agents' own decisions are still logged.
