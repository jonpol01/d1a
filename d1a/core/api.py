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
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255   # TypeSafe's limit for a choice's criteria and a score's levels


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None   # optional descriptions of "true" and "false"


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]                  # option key -> description (None: the key alone)

    @model_validator(mode="after")
    def _check(self):   # the name is in a 422's error location ("function-after[_check(), Choice]"): keep it
        if len(self.criteria) < 1 or len(self.criteria) > MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)   # the levels, lowest first


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "d1a-latest"
    questions: dict[str, Question] = Field(min_length=1)


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
    """-> (the internal record d1a.backends.torch.encode takes, one metadata dict per question: its "id", "type" and "keys", and for
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
