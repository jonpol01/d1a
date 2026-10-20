# D1A for agents

An agent makes many small decisions between its real work: who should take a request, how large a model it needs,
whether a tool call is safe to run, whether a task is done. D1A answers each in one forward pass (about 0.2–1 s), with
a calibrated probability for every option. The defaults turn unsure answers into the safe action: route up, ask a
person, or keep a task open.

## What D1A decides

| Decision | Preset | Answers | Fail-safe action |
|---|---|---|---|
| **Intake:** a new request | `intake` | which worker, model size, does it name its target and say what done looks like, ask the user first? | asks the user when the target or the finished state looks missing |
| **Tier:** a task, before work starts | `tier` | small, medium or large model | routes up unless D1A is sure a smaller model is enough |
| **Route:** any prompt | `route` | small, medium or large model | the same |
| **Gate:** a tool call | `gate` | allow, ask, deny | allows only at p ≥ 0.8; deny at p ≥ 0.5; otherwise asks |
| **Judge:** a task and the worker's report | `judge` | done? blocked? needs a person? next: complete, rework, consult, repair | closes a task only when D1A is sure it is done |

The questions are in `d1a/agents/presets.py`, worded as D1A's routing and agent training data asks them. The
fail-safe rules are `advise()` in the same file.

## See it: the demo

```bash
python -m d1a.agents.demo              # against the D1A server at D1A_URL (default http://127.0.0.1:8009)
```

Its decisions are kept out of the server's learning log, so it is safe to run against a live server.

One task goes through an agent's day. Here is real output from D1A-E4B-it v0.6, served on an Apple M4 (the latencies
include an SSH tunnel):

```
[intake] A new request: who takes it, how large a model, ask the user first?
  input:   Fix the failing login test in ~/work/app (branch fix-login). Done when tests/test_login.py passes.
  D1A:     worker=developer (1.00), tier=medium (0.97), has_target=yes (0.86), has_spec=yes (0.92), clear_done=yes (0.98), consult=no (0.06)
  advice:  {"consult": false, "worker": "developer", "tier": "medium"}

[intake] A vague request
  input:   Make the app better.
  D1A:     worker=developer (0.91), tier=large (0.53), has_target=no (0.05), has_spec=no (0.02), clear_done=no (0.04), consult=yes (0.97)
  advice:  {"consult": true, "worker": "developer", "tier": "large"}

[gate] Run the tests
  input:   pytest tests/test_login.py
  D1A:     decision=allow (0.94)
  advice:  {"decision": "allow"}   (313 ms)

[gate] Push to the main branch
  input:   git push origin main
  D1A:     decision=ask (0.67)
  advice:  {"decision": "ask"}   (314 ms)

[gate] Delete the project
  input:   rm -rf ~/work/app
  D1A:     decision=deny (0.99)
  advice:  {"decision": "deny"}   (322 ms)

[judge] The worker reports success
  input:   Fixed the password comparison in auth.py; tests/test_login.py: 12 passed.
  D1A:     done=yes (0.94), blocked=no (0.01), human=no (0.02), next=complete (0.97)
  advice:  {"next": "complete", "human": false, "blocked": false}

[judge] The worker reports a partial fix
  input:   Changed auth.py; tests/test_login.py still fails on test_expired_token.
  D1A:     done=no (0.02), blocked=no (0.17), human=no (0.04), next=rework (0.93)
  advice:  {"next": "rework", "human": false, "blocked": false}

[route] An easy prompt
  input:   What is the capital of France?
  D1A:     route=small (1.00)
  advice:  {"tier": "small"}

[route] A hard prompt
  input:   Find the race condition in this multi-threaded job scheduler and prove the fix is correct.
  D1A:     route=large (1.00)
  advice:  {"tier": "large"}
```

The playground ([jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground)) has a tool-gate demo in its
browser UI.

## Plug it in

### 1. As MCP tools (Claude Code, Hermes, any MCP client)

```json
{"mcpServers": {"d1a": {"command": "python", "args": ["-m", "d1a.agents.mcp_server"], "env": {"D1A_URL": "http://127.0.0.1:8009"}}}}
```

- **The tools:** `d1a_intake`, `d1a_judge`, `d1a_tier`, `d1a_gate`, `d1a_route`, and `d1a_decide` (your own questions).
  Each returns D1A's answers and the `advice`.
- **The MCP extra:** it needs `pip install "d1a[mcp]"`.
- **A server that requires a key:** add `"D1A_API_KEY": "<the server's key>"` to `env`. The gate hook and the demo
  read it too.
- **Tested end to end:** a real MCP client starts the server, lists the tools and calls them in CI
  (`tests/test_agents.py`).

### 2. As a Claude Code gate hook

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A's answers:** "deny" denies the call, and "ask" asks you.
- **"allow" adds nothing by default:** Claude Code's own rules decide. Set `D1A_GATE_ALLOW=1` to pass it on.
- **Fail-safe:** if D1A is unreachable, slow (`D1A_GATE_TIMEOUT`, 10 s) or failing, the hook asks you.
- **More:** details and settings are in [USAGE.md](USAGE.md#from-agents-mcp).

### 3. In your own code

```python
from d1a import D1A
from d1a.agents.presets import PRESETS, USE_CASES, advise, gate_state

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")
answers = m.decide(gate_state("developer", "Fix the login test", "git push origin main"), PRESETS["gate"], use_case=USE_CASES["gate"])
advise("gate", answers, temperatures=m.temperatures, use_case=USE_CASES["gate"])   # {"decision": "ask"}
```

The same works over HTTP with the [clients](USAGE.md#clients) and `"use_case": "routing"`.

## How well it decides, and its limits

Measured on D1A-E4B-it v0.6, as served:

| Measurement | Result |
|---|---|
| Held-out agent-factory items (in-distribution: the same generator as the training data), top answer | gate 98%, tier 96%, judge next step 92%, intake worker 91% |
| The same items, with the fail-safe advice | **0 of 33** calls that should be asked or denied were allowed; **0 of 33** unfinished tasks closed; every "deny" denied |
| 45 hand-labelled prompts, route | 45 of 45 |
| Public prompt-injection detection (devtools-v1) | catches **60%** (45 of 75), no false alarms |
| Public unsafe-content detection (aegis, devtools-v1) | catches 84% (63 of 75), 11 false alarms of 75 |

**The gate is a second opinion, not a security boundary.** It is strong on the kinds of tool calls it was trained on,
and it misses 40% of prompt injections in public data. It never replaces Claude Code's own permissions, a sandbox, or
a person for anything irreversible.
