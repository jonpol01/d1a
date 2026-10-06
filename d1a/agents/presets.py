"""Ready-made question sets for an agent factory (a kanban of agents: intake, an orchestrator, workers, tool calls).

Each preset is a `questions` dict for /v1/systemone or D1A.decide, worded exactly as D1A's routing and agent-kit training
data asks it (JohnP1/d1a-routing), so a trained checkpoint is asked what it learned. `advise` turns the answers into
an action with fail-safe defaults: an unsure model routes up, consults, or asks a person, never the reverse.

    from d1a.agents.presets import PRESETS, advise
    answers = D1A.load(run).decide(request_text, PRESETS["intake"])
    advise("intake", answers)   # {"consult": False, "worker": "developer", "tier": "medium"}
"""

TIER = {"type": "choice", "instructions": "How large a model does the implementation need?", "criteria": {
    "small": "Trivial: a lookup, rename, config value, or a one-line or copy change",
    "medium": "Routine: a feature, fix, test or document in one area",
    "large": "Hard: design, a cross-service change, a hard or intermittent bug, security, data migration or performance work"}}

ROUTE = {"type": "choice", "instructions": "How hard is this request for an AI assistant?", "criteria": {
    "small": "Easy: a simple fact, a greeting, a short rewrite or formatting",
    "medium": "Moderate: a summary, an ordinary email or piece of writing, a routine code change",
    "large": "Hard: expert reasoning, math proofs, debugging complex systems, multi-step analysis"}}

INTAKE = {
    "worker": {"type": "choice", "instructions": "Which agent should do this work?", "criteria": {
        "developer": "write or change code, tests or a pull request",
        "handler": "read or update an external service: GitHub issues or PR comments, Backlog, Notion, Slack",
        "broker": "hand the work to another AI agent", "avatar": "operate this computer's screen or take a screenshot",
        "recruiter": "create or change an agent itself", "mechanic": "fix the agent system: board, gateway, scheduled jobs, settings, shared memory"}},
    "tier": TIER,
    "has_target": {"type": "noul", "instructions": "Does the request name the repository, service, screen, document or system to work on?"},
    "has_spec": {"type": "noul", "instructions": "Does the request give enough current behaviour, inputs, examples or data to start?"},
    "clear_done": {"type": "noul", "instructions": "Is the expected result or finished state clear?"},
    "consult": {"type": "noul", "instructions": "Should the operator ask the user questions before any work starts?"},
}

JUDGE = {
    "done": {"type": "noul", "instructions": "Does the report show every completion criterion is met?"},
    "blocked": {"type": "noul", "instructions": "Is the work stuck waiting on something the worker cannot resolve itself?"},
    "human": {"type": "noul", "instructions": "Must a person decide, approve or provide something before the work continues?"},
    "next": {"type": "choice", "instructions": "What should the orchestrator do with this card next?", "criteria": {
        "complete": "close the card: the work is done", "rework": "send it back to the same worker: incomplete or wrong",
        "consult": "ask the user for information or a decision", "repair": "hand it to mechanic: the agent system itself is broken"}},
}

GATE = {"decision": {"type": "choice", "instructions": "Should this tool call run?", "criteria": {
    "allow": "yes: read-only, or changes only inside the agent's own workspace or branch, within the card's scope",
    "ask": "only with a person's OK: changes outside the workspace, posts to people, sends email, spends money or changes settings",
    "deny": "never: destructive or irreversible, leaks secrets, or outside the card's scope"}}}

PRESETS = {"intake": INTAKE, "judge": JUDGE, "tier": {"tier": TIER}, "gate": GATE, "route": {"route": ROUTE}}


def judge_state(card, report):
    """The text the judge preset reads: the card (request, completion criteria) and the worker's latest report."""
    return f"{card.strip()}\n\nReport:\n{report.strip()}"


def gate_state(agent, card, command):
    """The text the gate preset reads: who wants to run what, for which card."""
    return f"agent: {agent}\ncard: {card.strip()}\ncommand: {command.strip()}"


def fail_up(probs, small=0.7, medium=0.5):
    """The smallest tier the model is sure enough is enough: small only if p(small) >= `small`, medium only if
    p(small) + p(medium) >= `medium`, else large. On the calibrated routing checkpoint (JohnP1/d1a-e4b-routing, held-out
    factory cards) small >= 0.7 sends 2.8% too low against 4.6% at 0.6, for 9% sent too high; zero-shot it also beats
    argmax (6% vs 8% too low)."""
    if probs["small"] >= small: return "small"
    if probs["small"] + probs["medium"] >= medium: return "medium"
    return "large"


def advise(kind, answers, consult_at=0.5, allow_at=0.8, deny_at=0.5, done_at=0.8):
    """An action from a preset's answers (the /v1/systemone `answers` dict), each default failing safe."""
    p = lambda q: answers[q]["probabilities"]
    if kind in ("tier", "route"):
        return {"tier": fail_up(p(kind))}
    if kind == "intake":
        # consult when D1A says so, or when what the consult label is made of looks missing (target, finished state)
        missing = answers["has_target"]["noul"] < 0.5 or answers["clear_done"]["noul"] < 0.5
        return {"consult": answers["consult"]["noul"] >= consult_at or missing, "worker": answers["worker"]["choice"], "tier": fail_up(p("tier"))}
    if kind == "gate":
        g = p("decision")
        return {"decision": "deny" if g["deny"] >= deny_at else "allow" if g["allow"] >= allow_at else "ask"}
    if kind == "judge":
        nxt = answers["next"]["choice"]
        if nxt == "complete" and answers["done"]["noul"] < done_at: nxt = "consult"   # not sure it is done: ask, do not close
        return {"next": nxt, "human": answers["human"]["noul"] >= 0.5, "blocked": answers["blocked"]["noul"] >= 0.5}
    raise ValueError(f"unknown preset {kind!r}; one of {sorted(PRESETS)}")
