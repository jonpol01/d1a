"""D1A as MCP tools for agents (Hermes, Claude Code, any MCP client): the agent-factory decision points as one call each.

    python -m d1a.agents.mcp_server            # stdio; asks the D1A server at D1A_URL (default http://127.0.0.1:8009)
    D1A_RUN=JohnP1/d1a-e4b-mlx-q8 python -m d1a.agents.mcp_server   # or load a checkpoint in this process

Tools: d1a_intake (a new request), d1a_judge (a card and its latest report), d1a_tier (which model size implements a
card), d1a_gate (may this tool call run?), d1a_route (which model size answers a prompt), d1a_decide (your own
questions). Each returns D1A's answers with their probabilities and `advice`: the preset's fail-safe action
(d1a.agents.presets.advise). Requires the `mcp` extra.
"""
import json, os, time, urllib.request

from d1a.agents.presets import PRESETS, USE_CASES, advise, gate_state, judge_state

URL = os.environ.get("D1A_URL", "http://127.0.0.1:8009").rstrip("/")
RUN = os.environ.get("D1A_RUN")
_local = None


def ask(state, questions, use_case=None):
    """-> the /v1/systemone response for `state` and `questions` (and the request's `use_case`, when given): from D1A_RUN
    in-process, else the server at D1A_URL."""
    global _local
    if RUN:
        if _local is None:
            from d1a.serving.lib import D1A
            _local = D1A.load(RUN)
        t = time.perf_counter(); answers = _local.decide(state, questions, use_case=use_case)
        return {"answers": answers, "latency_ms": round((time.perf_counter() - t) * 1000, 1)}
    body = json.dumps({"model": "d1a-latest", "state": state, "questions": questions, **({"use_case": use_case} if use_case else {})}).encode()
    req = urllib.request.Request(f"{URL}/v1/systemone", data=body, headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def preset(kind, state):
    r = ask(state, PRESETS[kind], USE_CASES[kind])
    return {"answers": r["answers"], "advice": advise(kind, r["answers"]), "latency_ms": r.get("latency_ms")}


def d1a_intake(request: str) -> dict:
    """A new request, before it becomes a card: which worker should take it, how large a model it needs, whether it names
    its target and states what done looks like, and whether to ask the user first (advice.consult)."""
    return preset("intake", request)


def d1a_judge(card: str, report: str) -> dict:
    """A card (request and completion criteria) and the worker's latest report: done? blocked? needs a person? and what
    to do next (complete, rework, consult, repair). advice.next only says complete when D1A is sure it is done."""
    return preset("judge", judge_state(card, report))


def d1a_tier(card: str) -> dict:
    """A card with its design notes, before implementation starts: does it need a small, medium or large model.
    advice.tier routes up when D1A is unsure."""
    return preset("tier", card)


def d1a_gate(agent: str, card: str, command: str) -> dict:
    """A tool call an agent wants to run: allow, ask a person, or deny. advice.decision asks when D1A is unsure."""
    return preset("gate", gate_state(agent, card, command))


def d1a_route(prompt: str) -> dict:
    """Any prompt: does a small, medium or large model answer it well. advice.tier routes up when D1A is unsure."""
    return preset("route", prompt)


def d1a_decide(state: str, questions: dict) -> dict:
    """Your own questions about a text, in the System One shape: {"id": {"type": "noul"|"choice"|"score",
    "instructions": "...", "criteria": {...} or [...]}}. Returns the answers with probabilities."""
    return ask(state, questions)


TOOLS = [d1a_intake, d1a_judge, d1a_tier, d1a_gate, d1a_route, d1a_decide]


def server():
    from mcp.server.mcpserver import MCPServer
    s = MCPServer("d1a", instructions="Fast, calibrated decisions (~0.2 s) for an agent factory: intake, judging a card, model size, tool-call gating. "
                                      "Use the probabilities: act on confident answers, ask a person on unsure ones.")
    for f in TOOLS: s.tool()(f)
    return s


if __name__ == "__main__":
    server().run("stdio")
