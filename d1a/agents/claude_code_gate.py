"""D1A as a Claude Code PreToolUse hook: before an agent runs a tool call, D1A's gate preset decides allow, ask or deny.

    // .claude/settings.json
    {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}

Claude Code sends the call as JSON on stdin (tool_name, tool_input, cwd); the hook asks D1A (D1A_URL, or D1A_RUN in
process, as d1a.agents.mcp_server does) the gate question about "agent: claude-code / card: D1A_GATE_CARD / command: the
call", and prints Claude Code's decision JSON. It fails safe:
- D1A's "deny" denies, and "ask" asks a person; both carry D1A's probabilities as the reason.
- D1A's "allow" is passed on only when D1A_GATE_ALLOW=1. By default the hook prints nothing, so Claude Code's own
  permission rules decide, as without the hook: D1A adds caution, never permission.
- D1A unreachable, slow or failing: ask a person.
"""
import json, sys

from d1a.agents.presets import GATE, USE_CASES, advise, gate_state
from d1a.core.settings import get as setting


def command_text(call):
    """The text the gate reads for one call: a shell command as it is, any other tool as its name and input."""
    tool, inp = call.get("tool_name", ""), call.get("tool_input") or {}
    if tool == "Bash" and isinstance(inp.get("command"), str): return inp["command"]
    return f"{tool} {json.dumps(inp, ensure_ascii=False, sort_keys=True)}"


def decide(call, ask=None, temperatures=None):
    """-> (decision, reason) for one PreToolUse call: decision is "allow", "ask", "deny", or None (no opinion: Claude Code's
    own rules decide). `ask`/`temperatures`: d1a.agents.mcp_server's, or test doubles."""
    if ask is None:
        from d1a.agents import mcp_server
        ask, temperatures = mcp_server.ask, mcp_server.temperatures
    card = setting("D1A_GATE_CARD") or f"work in {call.get('cwd') or 'this project'}"
    try:
        answers = ask(gate_state("claude-code", card, command_text(call)), GATE, USE_CASES["gate"])["answers"]
        temps = temperatures() if callable(temperatures) else temperatures
        decision = advise("gate", answers, temperatures=temps, use_case=USE_CASES["gate"])["decision"]
    except Exception as e:   # D1A down, slow or broken: a person decides
        return "ask", f"D1A gate unavailable ({type(e).__name__}): asking a person"
    p = answers["decision"]["probabilities"]
    reason = f"D1A gate: {decision} (allow {p['allow']:.2f}, ask {p['ask']:.2f}, deny {p['deny']:.2f})"
    if decision == "allow" and setting("D1A_GATE_ALLOW") != "1": return None, reason
    return decision, reason


def output(decision, reason):
    """Claude Code's PreToolUse decision JSON, or None for no opinion."""
    if decision is None: return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision, "permissionDecisionReason": reason}}


def main(stdin=sys.stdin, stdout=sys.stdout):
    try: call = json.load(stdin)
    except ValueError: call = {}
    out = output(*decide(call))
    if out is not None: stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
