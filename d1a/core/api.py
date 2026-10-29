"""The System One request and answer, in TypeSafe's shapes (POST /v1/systemone), and their mapping onto D1A's one primitive:
pick one of K options for a question about a state.

A request becomes an internal record (`to_record`): the state and each question's instructions flattened to text
(`render`), and each question's options in a fixed order. The model returns one distribution per question over those
options, and `to_answers` reports it in the shape of the question's type:

    noul    options "no", "yes" (each with its criteria text, if any)   -> {"noul": P(yes)}
    choice  one option per criteria key, "key" or "key: description"    -> {"choice", "confidence", "probabilities" by key}
    score   one option per level, in order                             -> {"score": expected level, "legend", "probabilities" by
                                                                            level index, "confidence"}

Every string built here is model input, so the text formats (render, option_text, date_facts) are part of the trained
models' contract: a change to them changes every answer.
"""
import json
import re
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255   # TypeSafe's limit for a choice's criteria and a score's levels


class ScoreRange(BaseModel):
    """An ask_if condition on a score question: its answer's score (the expected level, counted from 0) within [min, max]."""
    model_config = {"extra": "forbid"}
    min: float | None = None
    max: float | None = None


# D1A's own optional field (#36), never model input: ask a question only when earlier answers allow it, {question id:
# condition}, every condition required. A condition is a list of option keys (a choice: its answer is one of them), true or
# false (a noul: its answer, P(yes) >= 0.5, is that), or a ScoreRange (a score). The question is answered after the ones it
# names, and reported as skipped when a condition fails or one of them was itself skipped.
AskIf = dict[str, Union[bool, list[str], ScoreRange]] | None


class Noul(BaseModel):
    model_config = {"extra": "forbid"}   # refused, not dropped: a misspelt "instr" was answered as an empty question (#340)
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None   # optional descriptions of "true" and "false"
    ask_if: AskIf = None

    @field_validator("criteria")
    @classmethod
    def _true_false_only(cls, criteria):   # any other key (e.g. "yes"/"no") was dropped unseen, and with it the criteria (#265)
        if criteria and (unknown := sorted(set(criteria) - {"true", "false"})):
            raise ValueError(f"a noul question's criteria take only the keys 'true' and 'false', not {unknown}")
        return criteria


class Choice(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]                  # option key -> description (None: the key alone)
    ask_if: AskIf = None

    @model_validator(mode="after")
    def _check(self):   # the name is in a 422's error location ("function-after[_check(), Choice]"): keep it
        if len(self.criteria) < 1 or len(self.criteria) > MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)   # the levels, lowest first
    ask_if: AskIf = None


Question = Annotated[Union[Noul, Choice, Score], Field(discriminator="type")]   # one error, naming the field, not one per type
USE_CASE_MAX_LENGTH = 64   # SystemOneRequest.use_case, and a name in a checkpoint's use_case_temperatures


class SystemOneRequest(BaseModel):
    model_config = {"extra": "forbid"}
    state: JSONContent
    model: str = "d1a-latest"
    questions: dict[str, Question] = Field(min_length=1)
    # D1A's own optional field (#209), never model input: the request's use case (e.g. "routing"). A checkpoint with a
    # temperature for it (d1a.training.calibrate --use-case) reads this request's probabilities at that temperature; absent,
    # or a name the checkpoint has no temperature for, the checkpoint's own temperature applies. Never guessed from the text.
    # Bounded: the decision log records it verbatim. Stripped, as the map's names are (" routing " is routing).
    use_case: str | None = Field(None, max_length=USE_CASE_MAX_LENGTH)

    @field_validator("use_case", mode="before")
    @classmethod
    def _strip_use_case(cls, v):
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_ask_if(self):
        ask_order(self)   # raises on an unknown or wrong-typed condition, or a cycle
        return self


def conditional(req) -> bool:
    """Whether any question carries an ask_if (#36): without one, a request is answered in a single pass, as before."""
    return any(q.ask_if for q in req.questions.values())


def ask_order(req) -> list[list[str]]:
    """The questions in passes: those with no ask_if first, then each question one pass after the last one it names.
    Raises ValueError on a condition that names an unknown question or itself, does not fit that question's type, or forms
    a cycle."""
    qs, level = req.questions, {}
    for qid, q in qs.items():
        for parent, cond in (q.ask_if or {}).items():
            p = qs.get(parent)
            if p is None or parent == qid:
                raise ValueError(f"question {qid!r}: ask_if names {parent!r}, " + ("itself" if parent == qid else "which is not in this request"))
            fits = (p.type == "choice" and isinstance(cond, list) and cond and set(cond) <= set(p.criteria)) or \
                   (p.type == "noul" and isinstance(cond, bool)) or (p.type == "score" and isinstance(cond, ScoreRange))
            if not fits:
                want = (f"a non-empty list of its option keys {sorted(p.criteria)}" if p.type == "choice" else
                        "true or false" if p.type == "noul" else 'a range {"min": .., "max": ..}')
                raise ValueError(f"question {qid!r}: ask_if on {parent!r} (a {p.type}) takes {want}")
    def depth(qid, seen=()):
        if qid in seen: raise ValueError(f"ask_if forms a cycle: {' -> '.join((*seen, qid))}")
        if qid not in level: level[qid] = 1 + max((depth(p, (*seen, qid)) for p in qs[qid].ask_if or {}), default=-1)
        return level[qid]
    for qid in qs: depth(qid)
    return [[q for q in qs if level[q] == d] for d in range(max(level.values()) + 1)]


def ask_holds(cond, ans: dict) -> bool:
    """Whether one ask_if condition holds for an answer (to_answers' shape)."""
    if ans["type"] == "choice": return ans["choice"] in cond
    if ans["type"] == "noul": return (ans["noul"] >= 0.5) == cond
    return (cond.min is None or ans["score"] >= cond.min) and (cond.max is None or ans["score"] <= cond.max)


def skipped(question) -> dict:
    """A question not asked because its ask_if did not hold: its type, "skipped": true, and the condition as sent."""
    return {"type": question.type, "skipped": True, "ask_if": {k: v.model_dump(exclude_none=True) if isinstance(v, ScoreRange) else v
                                                                 for k, v in question.ask_if.items()}}


def answer_in_passes(req, run):
    """Answer a request with ask_if conditions (#36): each pass of ask_order asks the questions whose conditions hold on
    the answers so far (the model's own answers, as the learned flags read them), through run(sub_request) -> (meta,
    probs, stats). -> (meta, probs, [stats]) of the questions asked, in pass order; the rest are skipped."""
    metas, probs, stats, got = [], [], [], {}
    for level in ask_order(req):
        ask = [q for q in level if all(p in got and ask_holds(c, got[p]) for p, c in (req.questions[q].ask_if or {}).items())]
        if not ask: continue
        meta, ps, m = run(req.model_copy(update={"questions": {q: req.questions[q] for q in ask}}))
        got.update(to_answers([list(p) for p in ps], meta)); metas += meta; probs += list(ps); stats.append(m)
    return metas, probs, stats


def in_request_order(req, answers: dict) -> dict:
    """The answers keyed in the request's question order, a skipped one reported as skipped()."""
    return {qid: answers[qid] if qid in answers else skipped(q) for qid, q in req.questions.items()}


# --- text the model reads -----------------------------------------------------------------------------------------------

SCALARS = (str, int, float, bool)


def render(value: JSONContent, indent: int = 0) -> str:
    """JSON content as text: a scalar as itself, None as nothing, a list as "- item" lines, an object as "key: value"
    lines with nested lists and objects on the lines below their key, two spaces deeper per level."""
    if value is None:
        return ""
    if isinstance(value, SCALARS):
        return str(value)
    pad = "  " * indent
    if isinstance(value, list):
        return "\n".join(pad + "- " + render(item, indent + 1).lstrip() for item in value)
    lines = []
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            lines.append(f"{pad}{key}:\n" + render(item, indent + 1))
        else:
            lines.append(f"{pad}{key}: " + render(item))
    return "\n".join(lines)


def option_text(name: str, description: JSONContent) -> str:
    """One option as the model sees it: its name, then ": description" when there is one."""
    if description is None or description == "":
        return name
    return name + ": " + render(description)


MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
DATE_PATTERN = re.compile(r"\b(?:" + "|".join(MONTH_NAMES) + r") \d{1,2}, \d{4}\b|\b\d{4}-\d{2}-\d{2}\b")


def parse_date(text: str):
    """A "July 22, 2026" or "2026-07-22" date, or None when it is not a real day."""
    try:
        return datetime.strptime(text, "%B %d, %Y" if "," in text else "%Y-%m-%d")
    except ValueError:
        return None


def day_gap(later: str, d_later, earlier: str, d_earlier) -> str:
    days = (d_later - d_earlier).days
    if days == 0:
        return f"{later} is the same day as {earlier}."
    unit = "day" if abs(days) == 1 else "days"
    return f"{later} is {abs(days)} {unit} {'after' if days > 0 else 'before'} {earlier}."


def date_facts(text: str) -> str:
    """One sentence per pair of distinct dates written in `text`, in order of first appearance ("August 3, 2026 is 12 days
    after July 22, 2026."). The model cannot subtract dates reliably (issue #8), but it can read a stated day count.
    "" when fewer than two dates appear."""
    dates = {}
    for match in DATE_PATTERN.finditer(text):
        written = match.group(0)
        if written not in dates and (day := parse_date(written)) is not None:
            dates[written] = day
    found = list(dates.items())
    return " ".join(day_gap(*found[j], *found[i]) for i in range(len(found)) for j in range(i + 1, len(found)))


def with_date_facts(state):
    """The state with its date_facts added when it holds two or more dates: a `date_facts` field (object), a
    {"date_facts": ...} item (list) or a final "date_facts: ..." paragraph (text). Opt-in (D1A_DATE_FACTS=1 in d1a.serving.serve,
    --date_facts in d1a.eval.benchmark)."""
    facts = date_facts(render(state))
    if not facts:
        return state
    if isinstance(state, dict):
        return {**state, "date_facts": facts}
    if isinstance(state, list):
        return [*state, {"date_facts": facts}]
    return f"{state}\n\ndate_facts: {facts}"


# --- request -> record -> answers ---------------------------------------------------------------------------------------

def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria keys (choice), "false" then
    "true" (noul), the level indices "0", "1", ... (score). Labels and targets use the same keys."""
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "choice":
        return list(criteria)
    return [str(level) for level, _ in enumerate(criteria)]


def options(question) -> list[str]:
    """A question's options as text, in the order of question_keys."""
    if question.type == "noul":
        described = question.criteria or {}
        return [option_text("no", described.get("false")), option_text("yes", described.get("true"))]
    if question.type == "choice":
        return [option_text(key, description) for key, description in question.criteria.items()]
    return [render(level) for level in question.criteria]


def to_record(req: SystemOneRequest):
    """-> (the internal record d1a.core.encoding.encode takes, one metadata dict per question: its "id", "type" and "keys", and for
    a score the "legend" of level index -> level text)."""
    questions, meta = [], []
    for qid, question in req.questions.items():
        texts = options(question)
        info = {"id": qid, "type": question.type, "keys": question_keys(question.type, question.criteria)}
        if question.type == "score":
            info["legend"] = dict(zip(info["keys"], texts))
        questions.append({"instr": render(question.instructions), "options": texts, "label": 0})
        meta.append(info)
    return {"state": render(req.state), "questions": questions}, meta


# The two confidence measures are TypeSafe's (its reference adapter, system-one-adapter 0.2.1): both read the distribution
# normalised to sum 1 (all zeros read as uniform), and a single option is certain.
def first_max(p) -> int:
    """The index of the first largest value."""
    return max(range(len(p)), key=p.__getitem__)


def normalised(p: list[float]) -> list[float]:
    total = sum(p)
    return [x / total for x in p] if total else [1 / len(p)] * len(p)


def choice_confidence(p: list[float]) -> float:
    """How far the top option is above uniform, (p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    k = len(p)
    if k == 1:
        return 1.0
    return (max(normalised(p)) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """1 - E|level - mode| / D, floored at 0, where mode is the first most likely level and D is that expected distance
    for a uniform distribution measured from its centre, (L-1)/2. 1 when one level holds all the mass, 0 at uniform or
    any spread as wide."""
    levels = len(p)
    if levels == 1:
        return 1.0
    p = normalised(p)
    mode = first_max(p)
    centre = (levels - 1) / 2
    uniform_spread = sum(abs(level - centre) for level in range(levels)) / levels
    spread = sum(mass * abs(level - mode) for level, mass in enumerate(p))
    return max(0.0, 1.0 - spread / uniform_spread)


def round_prob(x: float) -> float:
    """The precision answers are served at. 4 decimals keeps a rounded distribution's sum within TypeSafe's tolerance
    (|sum - 1| < 0.02) even at 255 options: 255 * 0.00005 < 0.02."""
    return round(float(x), 4)


def answer(p: list[float], info: dict) -> dict:
    if info["type"] == "noul":
        return {"type": "noul", "noul": round_prob(p[1])}
    if info["type"] == "choice":
        return {"type": "choice", "choice": info["keys"][first_max(p)], "confidence": round_prob(choice_confidence(p)),
                "probabilities": {key: round_prob(x) for key, x in zip(info["keys"], p)}}
    return {"type": "score", "score": round_prob(sum(level * x for level, x in enumerate(p))), "legend": info["legend"],
            "probabilities": {str(level): round_prob(x) for level, x in enumerate(p)}, "confidence": round_prob(score_confidence(p))}


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    """The /v1/systemone "answers" object: one answer per question, from its distribution in option order."""
    return {info["id"]: answer(p, info) for p, info in zip(probs, meta)}


def output_tokens(tok, answers: dict) -> int:
    """A billing-style count: the tokens of the serialised answers (D1A generates no text)."""
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
