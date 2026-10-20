"""A walk through D1A's agent decisions: one task through an agent's day, each step one D1A call.

    python -m d1a.agents.demo                       # asks the D1A server at D1A_URL (default http://127.0.0.1:8009)
    D1A_RUN=JohnP1/d1a-e2b-mlx-q8 python -m d1a.agents.demo     # or loads a checkpoint in this process

Each step prints D1A's top answer with its probability and the action the preset's fail-safe advice takes
(d1a.agents.presets.advise): unsure answers route up, ask a person, or keep a task open. The demo's decisions are kept out
of the server's learning log, and D1A_API_KEY is sent when set. The steps are the MCP tools'
(d1a.agents.mcp_server) and the Claude Code gate hook's (d1a.agents.claude_code_gate) decisions.
"""
import json, sys

TASK = "Fix the failing login test in ~/work/app (branch fix-login). Done when tests/test_login.py passes."
STEPS = [
    ("intake", "A new request: who takes it, how large a model, ask the user first?", TASK),
    ("intake", "A vague request", "Make the app better."),
    ("gate", "Run the tests", ("developer", TASK, "pytest tests/test_login.py")),
    ("gate", "Push to the main branch", ("developer", TASK, "git push origin main")),
    ("gate", "Delete the project", ("developer", TASK, "rm -rf ~/work/app")),
    ("judge", "The worker reports success", (TASK, "Fixed the password comparison in auth.py; tests/test_login.py: 12 passed.")),
    ("judge", "The worker reports a partial fix", (TASK, "Changed auth.py; tests/test_login.py still fails on test_expired_token.")),
    ("route", "An easy prompt", "What is the capital of France?"),
    ("route", "A hard prompt", "Find the race condition in this multi-threaded job scheduler and prove the fix is correct."),
]


def state_of(kind, arg):
    from d1a.agents.presets import gate_state, judge_state
    return gate_state(*arg) if kind == "gate" else judge_state(*arg) if kind == "judge" else arg


def top(answers):
    """One line per question: its top answer and probability (P(yes) for a yes/no question)."""
    out = []
    for qid, a in answers.items():
        if a["type"] == "noul": out.append(f"{qid}={'yes' if a['noul'] >= 0.5 else 'no'} ({a['noul']:.2f})")
        else: out.append(f"{qid}={a['choice']} ({a['probabilities'][a['choice']]:.2f})")
    return ", ".join(out)


def run(ask=None, temperatures=None, out=sys.stdout):
    """Run the steps; -> [{"kind", "title", "answers", "advice", "latency_ms"}]. `ask`/`temperatures`: d1a.agents.mcp_server's,
    or test doubles."""
    from d1a.agents.presets import PRESETS, USE_CASES, advise
    if ask is None:
        from d1a.agents import mcp_server
        ask, temperatures = (lambda s, q, u=None: mcp_server.ask(s, q, u, log=False)), mcp_server.temperatures
    temps = temperatures() if callable(temperatures) else temperatures
    results = []
    for kind, title, arg in STEPS:
        r = ask(state_of(kind, arg), PRESETS[kind], USE_CASES[kind])
        adv = advise(kind, r["answers"], temperatures=temps, use_case=USE_CASES[kind])
        results.append({"kind": kind, "title": title, "answers": r["answers"], "advice": adv, "latency_ms": r.get("latency_ms")})
        shown = arg[2] if kind == "gate" else arg[1] if kind == "judge" else arg
        out.write(f"\n[{kind}] {title}\n  input:   {shown}\n  D1A:     {top(r['answers'])}\n  advice:  {json.dumps(adv)}"
                  + (f"   ({r['latency_ms']:.0f} ms)" if r.get("latency_ms") is not None else "") + "\n")
    return results


if __name__ == "__main__":
    run()
