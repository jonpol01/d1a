"""Every D1A_* environment setting the package reads: its default, the module that reads it, and whether it is a secret.
Readers take their value through get(), so this list is the one place a setting and its default are named;
docs/CONFIGURATION.md documents each of them (tests/test_configuration_doc.py), and
`python -m d1a.serving.serve --show-config` prints them as they are in force, secrets masked.

    from d1a.core.settings import get
    size = int(get("D1A_PREFIX_CACHE"))     # the value set in the environment, else the default below

Defaults are the strings the environment would hold (None: unset); each reader parses its own, as it did before this
list existed.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    name: str
    default: str | None
    reader: str
    secret: bool = False


SETTINGS = {s.name: s for s in (
    # the server (d1a.serving.serve)
    Setting("D1A_PREFIX_CACHE", "4", "d1a.serving.serve"),
    Setting("D1A_PREFIX_MIN_TOKENS", None, "d1a.serving.serve"),
    Setting("D1A_PREFIX_MAX_TOKENS", "65536", "d1a.serving.serve"),
    Setting("D1A_DATE_FACTS", "0", "d1a.serving.serve"),
    Setting("D1A_FEEDBACK_LOG", None, "d1a.serving.serve"),
    Setting("D1A_OUTCOME_CALIBRATOR", None, "d1a.serving.serve"),
    Setting("D1A_API_KEY", None, "d1a.serving.serve, d1a.learning.feedback", secret=True),
    # loading a checkpoint (d1a.backends.checkpoint.LoadOptions.from_env)
    Setting("D1A_DTYPE", "", "d1a.backends.checkpoint"),
    Setting("D1A_MERGE", "1", "d1a.backends.checkpoint"),
    Setting("D1A_ATTN", None, "d1a.backends.checkpoint"),
    Setting("D1A_LORA_SCALE", "1", "d1a.backends.checkpoint"),
    Setting("D1A_TEMPERATURE", None, "d1a.backends.checkpoint"),
    Setting("D1A_BACKEND", None, "d1a.backends.checkpoint"),
    Setting("D1A_CUDA_GRAPHS", "", "d1a.backends.checkpoint"),
    Setting("D1A_FUSED", "", "d1a.backends.checkpoint"),
    Setting("D1A_PLE_FLASH", "", "d1a.backends.checkpoint"),
    Setting("D1A_SHAPE_BUCKET", "64", "d1a.backends.torch"),
    # self-learning, evaluation and agents
    Setting("D1A_LEARNING", None, "d1a.learning.settings"),
    Setting("D1A_REMOTE_API_KEY", "local", "d1a.eval.benchmark", secret=True),
    Setting("D1A_URL", "http://127.0.0.1:8009", "d1a.agents.mcp_server"),
    Setting("D1A_RUN", None, "d1a.agents.mcp_server"),
)}


def get(name, env=None):
    """The value of a registered setting: the environment's, else its default. An unregistered name is a bug."""
    s = SETTINGS[name]
    return (os.environ if env is None else env).get(name, s.default)


def effective(env=None):
    """[(name, value as shown, "set" or "default", default, reader)], secrets masked: what --show-config prints."""
    env = os.environ if env is None else env
    rows = []
    for s in SETTINGS.values():
        value = env.get(s.name)
        shown = ("(set, hidden)" if value else "(empty)") if s.secret and value is not None else value
        rows.append((s.name, shown if value is not None else s.default, "set" if value is not None else "default", s.default, s.reader))
    return rows


def show(env=None, extra=()):
    """The effective settings as text, one per line; `extra`: more (label, value) lines to add, e.g. the learning file."""
    lines = [f"{n:24} {str(v):40} {src:8} default {d!r:24} read by {r}" for n, v, src, d, r in effective(env)]
    return "\n".join(lines + [f"{label}: {value}" for label, value in extra])
