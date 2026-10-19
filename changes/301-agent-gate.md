### Added

- `d1a.agents.claude_code_gate`, a Claude Code `PreToolUse` hook: D1A's gate preset decides allow, ask or deny before an
  agent's tool call runs. It fails safe: "allow" is passed on only with `D1A_GATE_ALLOW=1`, and an unreachable D1A asks
  a person. New settings: `D1A_GATE_CARD`, `D1A_GATE_ALLOW`.
- Tests prove agents can use D1A: a real MCP client lists and calls the MCP server's tools over stdio on a tiny trained
  checkpoint, and CI installs the `mcp` extra so the MCP server is tested at all.
