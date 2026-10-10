"""The self-learning loop on a live server (d1a#233): the trained manifest, replay of outcomes across model versions, and
the tick that decides when the promotion gate runs and what it reports. All of it reads d1a.learning.settings each run.

    python -m d1a.learning.feedback trained --mix train.jsonl [--mix ...] --run JohnP1/d1a-e4b-mlx-q8@v0.6 --out trained.json
    python -m d1a.learning.feedback replay <log> --server http://127.0.0.1:8009 --trained trained.json
    python -m d1a.learning.feedback tick <log> --calibrator calibrator.json --server http://127.0.0.1:8009 --trained-dir <dir>

Replay. When the served model changes, the decisions that already have outcomes were made by the previous model, and a
calibrator fitted on them would correct the new model by the old one's errors (promote --run). Replay re-sends each such
decision's request to the live server, which logs it as a new decision of the served model with replay_of = the original;
FeedbackLog.resolved() gives it the original's outcomes. So the new model starts with every earlier outcome instead of none.

It fails closed. A decision is never replayed unless the served model's trained manifest (every PR and state it trained on,
from every mix in its lineage) exists and names that model, and the decision is excluded when its PR (its outcome's group)
or its state is in that manifest, or, by default, when it has no PR id at all. A model scored on its own training data
would look better calibrated than it is, and the gate would promote on that.
"""
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from d1a.learning import settings as S

REPLAY_HEADER = "x-d1a-replay-of"
PR_ID = re.compile(r"^(?:(?P<owner>[\w.-]+)/)?(?P<repo>[\w.-]+)#(?P<number>\d+)(?::[\w-]+)?$")


# --- the trained manifest -----------------------------------------------------------------------------------------------

def state_sha256(state):
    """The identity recipes/mix/build_mix.py's exact screens use (state_hash): sha256 of the normalised state text."""
    from d1a.eval.suite import normalise_text
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


def pr_key(pr_id):
    """"NousResearch/hermes-agent#123", "hermes-agent#123" or "hermes-agent#123:ja" -> "hermes-agent#123"; None when it
    is not a PR id. The owner is dropped on purpose: a training id may carry none, and two repositories of the same name
    can only make the exclusion stricter."""
    m = PR_ID.match(str(pr_id or "").strip())
    return f"{m['repo'].lower()}#{int(m['number'])}" if m else None


def build_manifest(mixes, run):
    """{run, pr_ids, state_sha256, mixes} from training mixes (JSONL records with a state and, for PRs, an id)."""
    prs, states, sources = set(), set(), []
    for mix in mixes:
        n = 0
        for line in Path(mix).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line); n += 1
            states.add(state_sha256(r["state"]))
            if (k := pr_key(r.get("id"))): prs.add(k)
        sources.append({"mix": Path(mix).name, "records": n, "sha256": hashlib.sha256(Path(mix).read_bytes()).hexdigest()})
    return {"run": run, "created": time.time(), "mixes": sources, "pr_ids": sorted(prs), "state_sha256": sorted(states)}


def load_manifest(path, run):
    """The manifest for `run`, or raise: replay never runs without one (fail closed)."""
    if path is None: raise ReplayRefused(f"no trained manifest for {run}: replay needs one (python -m d1a.learning.feedback trained ...)")
    p = Path(path)
    if not p.exists(): raise ReplayRefused(f"no trained manifest for {run} at {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("run") != run: raise ReplayRefused(f"{p} is the trained manifest of {m.get('run')!r}, not of the served {run!r}")
    if not m.get("mixes"): raise ReplayRefused(f"{p} names no training mixes")
    return {"run": run, "pr_ids": set(m["pr_ids"]), "state_sha256": set(m["state_sha256"])}


def manifest_path(directory, run):
    """Where tick looks for a run's manifest: <dir>/<repo with / as __>@<tag>.json."""
    return None if directory is None else Path(directory) / (run.replace("/", "__") + ".json")


class ReplayRefused(RuntimeError):
    pass


# --- choosing what to replay --------------------------------------------------------------------------------------------

def candidates(log, served_run, manifest, exclude_no_pr_id=True, src=None):
    """-> (decisions to replay, oldest first; {reason: count} of the ones excluded). A candidate is a decision with an
    outcome, made by another model, not itself a replay, and not yet replayed for `served_run`."""
    events = log.events()
    done = {e["replay_of"] for e in events if e["kind"] == "decision" and e.get("replay_of") and e.get("run") == served_run}
    out, excluded = [], {}
    for d in log.resolved(src=src, events=events):
        if d.get("replay_of") or d.get("run") == served_run or d["id"] in done: continue
        key = pr_key(d.get("group"))
        reason = ("its pull request is in the served model's training" if key and key in manifest["pr_ids"] else
                  "its state is in the served model's training" if state_sha256(d["state"]) in manifest["state_sha256"] else
                  "no pull-request id to check against training" if key is None and exclude_no_pr_id else None)
        if reason: excluded[reason] = excluded.get(reason, 0) + 1
        else: out.append(d)
    return out, excluded


# --- talking to the live server -----------------------------------------------------------------------------------------

def _http(url, body=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())


def server_card(server, api_key=None):
    """GET /v1/models' first card: its `run` and `batches.queued` (how many requests wait for the model)."""
    return _http(f"{server.rstrip('/')}/v1/models", headers=_auth(api_key))["models"][0]


def _auth(api_key): return {"authorization": f"Bearer {api_key}"} if api_key else {}


def replay(log, server, manifest_file, cap=50, pause_s=1.0, exclude_no_pr_id=True, src=None, api_key=None, now=time.time,
           sleep=time.sleep):
    """Re-send up to `cap` candidates to the live server, one at a time, `pause_s` apart, each only while no request waits
    for the model (batches.queued == 0), so the labeller never waits behind a replay. The server logs each as a decision
    of the model it serves with replay_of (the x-d1a-replay-of header). -> a report dict."""
    card = server_card(server, api_key); run = card["run"]
    manifest = load_manifest(manifest_file, run)
    todo, excluded = candidates(log, run, manifest, exclude_no_pr_id, src)
    sent, skipped_busy, errors = 0, 0, []
    for d in todo[:cap]:
        if (server_card(server, api_key).get("batches") or {}).get("queued", 0) > 0:
            skipped_busy += 1; break   # live traffic first: the rest waits for the next tick
        body = {"state": d["state"], "questions": d["questions"], "model": "d1a-latest", **({"use_case": d["use_case"]} if d.get("use_case") else {})}
        try:
            _http(f"{server.rstrip('/')}/v1/systemone", body, {REPLAY_HEADER: d["id"], **_auth(api_key)}); sent += 1
        except (urllib.error.URLError, OSError, ValueError) as e:
            errors.append(f"{d['id'][:8]}: {e}"); break
        sleep(pause_s)
    return {"run": run, "candidates": len(todo), "replayed": sent, "left": len(todo) - sent, "excluded": excluded,
            "paused_for_live_traffic": bool(skipped_busy), "errors": errors, "ts": now()}


# --- the tick -----------------------------------------------------------------------------------------------------------

def status_path(log_path): return Path(log_path).with_name("learning-status.json")


def load_status(log_path):
    p = status_path(log_path)
    try: return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {"reports": [], "reached_min": {}}


def save_status(log_path, status):
    p = status_path(log_path); tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); tmp.replace(p)


def due(settings, status, outcomes_now, now):
    """Why the gate should run now (a reason string), or None: after_new_outcomes new outcomes since the last gate run, or
    the daily time has passed since it (local time)."""
    sch = settings["schedule"]
    last = status.get("last_gate")
    if sch["after_new_outcomes"] and outcomes_now - status.get("outcomes_at_last_gate", 0) >= sch["after_new_outcomes"]:
        return f"{outcomes_now - status.get('outcomes_at_last_gate', 0)} new outcomes (trigger: {sch['after_new_outcomes']})"
    if sch["daily_at"]:
        h, m = map(int, sch["daily_at"].split(":"))
        today = datetime.fromtimestamp(now).replace(hour=h, minute=m, second=0, microsecond=0)
        slot = today if datetime.fromtimestamp(now) >= today else today - timedelta(days=1)
        if last is None or datetime.fromtimestamp(last) < slot: return f"daily at {sch['daily_at']}"
    return None


def power_line(report):
    """One line per question: n, delta with its interval, and the smallest delta this n can detect (mde)."""
    parts = []
    for q, r in sorted(report.items()):
        if "delta_log_loss" not in r: parts.append(f"{q}: {r.get('held_out', 0)} held out, {'; '.join(r.get('reasons', [])) or 'not gated'}"); continue
        lo, hi = r["ci"]
        parts.append(f"{q}: {'PROMOTED' if r['promote'] else 'kept'}, fit {r['fit']}, held out {r['held_out']} ({r.get('held_out_replayed', 0)} replayed), "
                     f"d log loss {r['delta_log_loss']:+.3f} [{lo:+.3f}, {hi:+.3f}], detectable {r['mde']:.3f}")
    return " | ".join(parts)


def notify(settings, text):
    """Post to the webhook named by reports.webhook_file (Discord-compatible {"content": ...}); the URL is read from that
    file and never logged. -> True when sent."""
    f = settings["reports"]["webhook_file"]
    if not f: return False
    try:
        url = Path(f).expanduser().read_text(encoding="utf-8").strip()
        if not url.startswith("https://"): return False
        urllib.request.urlopen(urllib.request.Request(url, data=json.dumps({"content": text[:1900]}).encode(),
                                                      headers={"content-type": "application/json"}), timeout=15)
        return True
    except (OSError, urllib.error.URLError):
        return False


def progress(resolved, settings):
    """{question: {outcomes, fit, needs}}: `fit` is what the gate's fit side would get (promote's split), `needs` how many
    more fit outcomes the question lacks before promote fits it at all."""
    from d1a.learning.feedback import split
    p = settings["promotion"]; keep = None if p["questions"] is None else set(p["questions"])
    fit, _ = split(resolved, p["held_out"])
    out = {}
    for rows, key in ((resolved, "outcomes"), (fit, "fit")):
        for d in rows:
            for q in d["labels"]:
                if keep is None or q in keep: out.setdefault(q, {"outcomes": 0, "fit": 0})[key] += 1
    for q, v in out.items(): v["needs"] = max(0, p["min_outcomes"] - v["fit"])
    return out


def tick(log_path, calibrator_path, server=None, run=None, trained_dir=None, settings_path=None, api_key=None, now=None):
    """One run of the loop, meant every few minutes (a LaunchAgent, cron): replay on a model change, then the promotion gate
    when it is due, a report line in <log dir>/learning-status.json, and notifications. -> that report line."""
    from d1a.learning.feedback import FeedbackLog, OutcomeCalibrator, promote
    now = time.time() if now is None else now
    settings, error, source = S.load(settings_path)
    status = load_status(log_path)
    line = {"ts": now, "settings": source, **({"settings_error": error} if error else {})}
    served = run
    if server:
        try: served = server_card(server, api_key)["run"]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e: line["server"] = f"unreachable: {e}"
    if served is None:
        line["skipped"] = "no served model known (server unreachable and no --run)"
        status["reports"] = (status.get("reports", []) + [line])[-100:]; save_status(log_path, status); return line
    if status.get("run") != served:   # a new model: its own counts, schedule and first-reach events start over
        line["model_change"] = {"from": status.get("run"), "to": served}
        status.update({"run": served, "outcomes_at_last_gate": 0, "last_gate": None, "reached_min": {}, "replay_complete": False})
    log = FeedbackLog(log_path)
    first_replay = False
    r = settings["replay"]
    if r["enabled"] and r["on_model_change"] and server and not status.get("replay_complete"):
        try:
            rep = replay(log, server, manifest_path(trained_dir, served), r["per_tick_cap"], r["pause_s"], r["exclude_no_pr_id"],
                         settings["outcomes"]["sources"], api_key)
            line["replay"] = rep
            first_replay = not status.get("replay_reported")
            if rep["left"] == 0 and not rep["errors"] and not rep["paused_for_live_traffic"]: status["replay_complete"] = True
        except ReplayRefused as e:
            line["replay"] = {"refused": str(e)}
        except (urllib.error.URLError, OSError, ValueError) as e:
            line["replay"] = {"error": str(e)}
    resolved = log.resolved(src=settings["outcomes"]["sources"], run=served)
    per_q = progress(resolved, settings)
    status["per_question"] = per_q
    p = settings["promotion"]
    reason = due(settings, status, len(resolved), now) or ("the first replay after a model change" if first_replay else None)
    if reason:
        path = Path(calibrator_path); current = OutcomeCalibrator.load(path) if path.exists() else None
        cal, report = promote(resolved, current, p["held_out"], p["min_outcomes"], p["questions"], tolerance=p["tolerance"],
                              n=p["bootstrap"], level=p["ci_level"])
        promoted = sorted(q for q, x in report.items() if x["promote"])
        wrote = False
        if p["auto"] and any(x["promote"] or x.get("dropped") for x in report.values()):   # whole file, renamed into place
            tmp = path.with_name(path.name + ".tmp"); cal.save(tmp); tmp.replace(path); wrote = True
        line["gate"] = {"reason": reason, "outcomes": len(resolved), "replayed": sum(1 for d in resolved if d.get("replay_of")),
                        "promoted": promoted, "calibrator_written": wrote, "auto": p["auto"], "report": report, "power": power_line(report)}
        status.update({"last_gate": now, "outcomes_at_last_gate": len(resolved)})
        if first_replay: status["replay_reported"] = True
        notes = settings["reports"]["notify"]
        if promoted and "promotion" in notes:
            line["notified"] = notify(settings, f"D1A {served}: promoted {', '.join(promoted)} " + ("(written)" if wrote else "(auto-promotion is off: not written)") + f" — {line['gate']['power']}")
        elif "every_gate" in notes:
            line["notified"] = notify(settings, f"D1A {served} gate ({reason}): {line['gate']['power']}")
    newly = [q for q, v in per_q.items() if v["needs"] == 0 and q not in status.get("reached_min", {})]
    if newly:
        status.setdefault("reached_min", {}).update({q: now for q in newly})
        line["reached_min"] = newly
        if "min_reached" in settings["reports"]["notify"]:
            line["notified_min"] = notify(settings, f"D1A {served}: {', '.join(newly)} reached {p['min_outcomes']} fit outcomes; the gate can now decide on them")
    status["reports"] = (status.get("reports", []) + [line])[-100:]
    status["settings"] = settings
    save_status(log_path, status)
    return line
