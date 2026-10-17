"""docs/MODELS.md and docs/USAGE.md must agree on which D1A models can chat: every model whose column in MODELS.md's
capabilities table says chat **yes** is named in USAGE.md's "Which models can chat" line as available today, not "when
trained" (#292: the E4B-it column was added while USAGE.md still called it future).
    uv run python -m pytest tests/test_models_doc.py -q
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chat_models(models=ROOT / "docs" / "MODELS.md"):
    """The models whose capabilities column says chat **yes**: (D1A-E4B-it, ...)."""
    rows = [line for line in models.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
    names = [re.sub(r"\s+v[\d.]+$", "", c.replace("*", "").strip()) for c in cells(rows[0])[1:]]
    chat = next(cells(r)[1:] for r in rows if "Chat / text generation" in r)
    return [n for n, c in zip(names, chat) if c.startswith("**yes**")]


def chat_today(usage=ROOT / "docs" / "USAGE.md"):
    """The models USAGE.md's "Which models can chat" line names as chatting "today" (the text before that word)."""
    line = next(l for l in usage.read_text(encoding="utf-8").splitlines() if "Which models can chat" in l)
    assert " today" in line, line
    return line.split(" today")[0]


def test_every_model_that_can_chat_is_named_as_chatting_today():
    models = chat_models()
    assert "D1A-E2B-it" in models   # the table's parse found the chat row
    assert [m for m in models if not re.search(rf"\b{re.escape(m)}\b", chat_today())] == []
