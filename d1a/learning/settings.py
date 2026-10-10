"""Self-learning settings: one JSON file, read by `d1a.learning.feedback` (promote, replay, tick) on every run, so a change
takes effect on the next run without a restart or a reinstall. `D1A_LEARNING=<path>` names the file; unset, every job
behaves as before this file existed (DEFAULTS are today's behaviour).

    python -m d1a.learning.feedback config init          # write DEFAULTS to $D1A_LEARNING (refuses to overwrite)
    python -m d1a.learning.feedback config show          # the effective settings, and where each value came from
    python -m d1a.learning.feedback config set promotion.min_outcomes=10 schedule.daily_at=03:30
    python -m d1a.learning.feedback config validate

A file that fails validation is never used: a job falls back to the last file that passed (kept beside it as
<file>.last-valid.json) and reports the error, and `config set` refuses the change. Every accepted change appends a line to
<file>.audit.jsonl (time, key, old, new, source). The server's decision log stays an environment setting
(D1A_FEEDBACK_LOG): /review and the outcome poster read that log, so it is not switched from here.
"""
import copy
import json
import os
import re
import time
from pathlib import Path

ENV = "D1A_LEARNING"
VERSION = 1
DEFAULTS = {
    "version": VERSION,
    "outcomes": {"enabled": True, "sources": None},           # sources: the outcome sources that count (None: all of them)
    "promotion": {"auto": True, "questions": None,             # questions: the ones that learn (None: all of them)
                  "min_outcomes": 20, "held_out": 0.3, "ci_level": 0.95, "tolerance": 0.01, "bootstrap": 2000},
    "schedule": {"daily_at": "04:00", "after_new_outcomes": 10},   # local time HH:MM, or None; >= N new outcomes, or None
    "replay": {"enabled": True, "on_model_change": True, "exclude_no_pr_id": True, "per_tick_cap": 50, "pause_s": 1.0},
    "reports": {"webhook_file": None, "notify": ["promotion", "min_reached"]},
}
CI_LEVELS = (0.9, 0.95, 0.99)
NOTIFY = ("promotion", "min_reached", "every_gate")
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class SettingsError(ValueError):
    pass


def _is_int(v): return isinstance(v, int) and not isinstance(v, bool)
def _is_num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def _str_list(v): return v is None or (isinstance(v, list) and v and all(isinstance(x, str) and x.strip() for x in v) and len(set(v)) == len(v))


RULES = {   # key -> (check, what it must be)
    "outcomes.enabled": (lambda v: isinstance(v, bool), "true or false"),
    "outcomes.sources": (_str_list, "null (every source) or a non-empty list of distinct source names, e.g. [\"reviewer\", \"human\"]"),
    "promotion.auto": (lambda v: isinstance(v, bool), "true or false (false: the gate runs and reports, and never writes the calibrator)"),
    "promotion.questions": (_str_list, "null (every question) or a non-empty list of distinct question ids"),
    "promotion.min_outcomes": (lambda v: _is_int(v) and v >= 10, "an integer >= 10"),
    "promotion.held_out": (lambda v: _is_num(v) and 0.1 <= v <= 0.5, "a number from 0.1 to 0.5"),
    "promotion.ci_level": (lambda v: _is_num(v) and any(abs(v - c) < 1e-9 for c in CI_LEVELS), "0.9, 0.95 or 0.99"),
    "promotion.tolerance": (lambda v: _is_num(v) and 0 <= v <= 0.1, "a number from 0 to 0.1"),
    "promotion.bootstrap": (lambda v: _is_int(v) and 200 <= v <= 20000, "an integer from 200 to 20000"),
    "schedule.daily_at": (lambda v: v is None or (isinstance(v, str) and bool(HHMM.match(v))), "null or HH:MM in this machine's local time"),
    "schedule.after_new_outcomes": (lambda v: v is None or (_is_int(v) and v >= 1), "null or an integer >= 1"),
    "replay.enabled": (lambda v: isinstance(v, bool), "true or false"),
    "replay.on_model_change": (lambda v: isinstance(v, bool), "true or false"),
    "replay.exclude_no_pr_id": (lambda v: isinstance(v, bool), "true or false (true: a decision with no pull-request id is never replayed)"),
    "replay.per_tick_cap": (lambda v: _is_int(v) and 1 <= v <= 500, "an integer from 1 to 500"),
    "replay.pause_s": (lambda v: _is_num(v) and 0 <= v <= 60, "a number of seconds from 0 to 60"),
    "reports.webhook_file": (lambda v: v is None or (isinstance(v, str) and v.strip()), "null or the path of a file holding a webhook URL (the URL itself never goes here)"),
    "reports.notify": (lambda v: isinstance(v, list) and all(x in NOTIFY for x in v) and len(set(v)) == len(v), f"a list of distinct values from {list(NOTIFY)}"),
}


def flat(d, prefix=""):
    """{"a": {"b": 1}} -> {"a.b": 1} (one level of sections)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update({f"{prefix}{k}.{kk}": vv for kk, vv in v.items()})
        else: out[f"{prefix}{k}"] = v
    return out


def validate(data):
    """-> the settings with defaults filled in; raises SettingsError listing every problem (unknown keys included)."""
    if not isinstance(data, dict): raise SettingsError("the settings file must hold a JSON object")
    problems = []
    if data.get("version", VERSION) != VERSION: problems.append(f"version: {data.get('version')!r}, this d1a reads version {VERSION}")
    for section, v in data.items():
        if section == "version": continue
        if section not in DEFAULTS: problems.append(f"{section}: unknown section (known: {sorted(k for k in DEFAULTS if k != 'version')})"); continue
        if not isinstance(v, dict): problems.append(f"{section}: must be an object"); continue
        for key in v:
            if key not in DEFAULTS[section]: problems.append(f"{section}.{key}: unknown setting (known: {sorted(DEFAULTS[section])})")
    out = copy.deepcopy(DEFAULTS)
    for section, v in data.items():
        if section in DEFAULTS and isinstance(v, dict) and section != "version":
            out[section].update({k: x for k, x in v.items() if k in DEFAULTS[section]})
    for key, value in flat(out).items():
        if key == "version": continue
        check, must = RULES[key]
        if not check(value): problems.append(f"{key}: {value!r} is not allowed; it must be {must}")
    if problems: raise SettingsError("; ".join(problems))
    return out


def path_from_env():
    p = os.environ.get(ENV)
    return Path(p) if p else None


def _last_valid(path): return path.with_name(path.name + ".last-valid.json")
def _audit(path): return path.with_name(path.name + ".audit.jsonl")


def _write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)   # a reader never sees half a file


def load(path=None):
    """The settings a job runs with -> (settings, error or None, source): DEFAULTS when no file is configured or it does not
    exist yet; the file when it validates (and it becomes the last valid one); otherwise the last valid file, or DEFAULTS
    when there has never been one, with the error, so a typo never stops learning or switches it to something unintended."""
    path = Path(path) if path else path_from_env()
    if path is None: return copy.deepcopy(DEFAULTS), None, "defaults (D1A_LEARNING is not set)"
    if not path.exists(): return copy.deepcopy(DEFAULTS), None, f"defaults ({path} does not exist)"
    try:
        settings = validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, SettingsError) as e:
        error = f"{path}: {e}"
        last = _last_valid(path)
        try: return validate(json.loads(last.read_text(encoding="utf-8"))), error, f"{last} (the file has an error)"
        except (OSError, json.JSONDecodeError, SettingsError): return copy.deepcopy(DEFAULTS), error, "defaults (the file has an error and no earlier file was valid)"
    if not _last_valid(path).exists() or json.loads(_last_valid(path).read_text(encoding="utf-8")) != settings:
        try: _write_json(_last_valid(path), settings)
        except OSError: pass
    return settings, None, str(path)


def parse_value(text):
    """A command-line value: JSON when it parses (10, 0.25, true, null, ["type","sev"]), else the plain string (03:30)."""
    try: return json.loads(text)
    except json.JSONDecodeError: return text


def init(path):
    path = Path(path)
    if path.exists(): raise SettingsError(f"{path} exists; edit it with `config set`, or remove it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, DEFAULTS); _write_json(_last_valid(path), DEFAULTS)
    _append_audit(path, [("(file)", None, "created with the defaults")], "cli")
    return copy.deepcopy(DEFAULTS)


def set_values(path, assignments, source="cli"):
    """Apply {"section.key": value, ...} to the file; validated as a whole before anything is written. -> [(key, old, new)]."""
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else copy.deepcopy(DEFAULTS)
    new = copy.deepcopy(current); changes = []
    for key, value in assignments.items():
        section, _, name = key.partition(".")
        if not name or section not in DEFAULTS or section == "version" or name not in DEFAULTS[section]:
            raise SettingsError(f"{key}: unknown setting (known: {sorted(k for k in RULES)})")
        old = new.get(section, {}).get(name, DEFAULTS[section][name])
        new.setdefault(section, {})[name] = value
        if old != value: changes.append((key, old, value))
    validate(new)   # raises with every problem; nothing is written
    if changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, new); _write_json(_last_valid(path), validate(new))
        _append_audit(path, changes, source)
    return changes


def _append_audit(path, changes, source):
    with open(_audit(Path(path)), "a", encoding="utf-8") as f:
        for key, old, new in changes:
            f.write(json.dumps({"ts": time.time(), "key": key, "old": old, "new": new, "source": source}, ensure_ascii=False) + "\n")


def audit(path, last=50):
    p = _audit(Path(path))
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()][-last:]
