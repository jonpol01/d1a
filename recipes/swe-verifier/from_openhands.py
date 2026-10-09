"""OpenHands runs (nebius/SWE-rebench-openhands-trajectories) in the shape zero_shot.py scores, so the verifier trained on
public SWE-agent runs can be tested on another agent unchanged (E2).

    uv run --extra mlx python recipes/swe-verifier/zero_shot.py --source openhands --shards 12 --split test \
        --per-issue 2 --mixed-only --issues 250 --run runs/swe-verifier/train/r1 --out runs/swe-verifier/e2/<name>.jsonl

Data (CC-BY-4.0): 67,074 runs of OpenHands v0.54 with Qwen3-Coder-480B on 6,306 SWE-rebench issues, one 2.1 GB parquet
file at the pinned commit (REVISION). `resolved` is the hidden tests' verdict; gen_tests_correct and pred_passes_gen_tests
come from test runs too, so they are never read. The row becomes zero_shot.py's:
- target = resolved; model_name = the one agent model;
- exit_status: "submit" -> "submitted" (heuristics' clean submit), or "submitted_no_patch" when the patch is empty, as in
  SWE-agent's data; otherwise the error's class ("RuntimeError" for the iteration limit, "AgentStuckInLoopError", ...);
- generated_patch = model_patch without a root `issue.md`: in a fifth of the runs the task workspace held the issue as an
  untracked issue.md, and the diff picked it up. It is per issue (all of an issue's runs or none), is the issue text the
  state already shows, and comes with a 77% resolved rate against 40%, so leaving it would hand the scorer a shortcut.
  Each row says whether its patch carried it (issue_md), hydrate() prints the counts for the rows it reads, and
  keep_issue_md=True (zero_shot.py --keep-issue-md) leaves the diff in, to measure the shortcut;
- trajectory: the system prompt, the issue (as "ISSUE: ... INSTRUCTIONS:", which zero_shot.issue_of reads), each assistant
  turn ("ai": its text and tool calls, as name({arguments})), each tool result and later user nudge ("user").

Shards: the file has no shards, so shard_of() buckets issues into 12 by a hash of the instance id (an issue's runs stay
together), and --shards / --stream-shards / --eval-shards keep the meaning and roughly the size they have on SWE-agent's
12 files. index() reads only the id and outcome columns, a draw is made on those, and hydrate() reads the drawn rows.
"""
import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "nebius/SWE-rebench-openhands-trajectories"
REVISION = "35455389ab51bf5e2306bfd436ef72d0f98bf882"   # the dataset commit E2 was drawn from
FILE = "trajectories.parquet"
SHARDS = 12
MODEL = "Qwen3-Coder-480B-A35B-Instruct (OpenHands 0.54)"
COLS = ["trajectory_id", "instance_id", "trajectory", "model_patch", "exit_status", "resolved"]
ISSUE = re.compile(r"<issue_description>\s*(.*?)\s*</issue_description>(.*)", re.DOTALL)


def path(revision=REVISION):
    """Local path of the dataset's one parquet file at `revision`, downloaded once into the Hugging Face cache."""
    return hf_hub_download(DATASET, FILE, repo_type="dataset", revision=revision)


def shard_of(instance_id):
    return int(hashlib.sha1(instance_id.encode()).hexdigest(), 16) % SHARDS


def _call(c):
    f = c.get("function") or {}; args = f.get("arguments")
    try: args = json.loads(args) if isinstance(args, str) else (args or {})
    except ValueError: pass   # an unparsable argument string is shown as it was sent
    return f"{f.get('name', '')}({json.dumps(args, ensure_ascii=False)})"


def trajectory(msgs):
    traj, seen_issue = [], False
    for m in msgs:
        role, text = m["role"], m.get("content") or ""
        if role == "system":
            traj.append({"role": "system", "text": text})
        elif role == "user" and not seen_issue:
            seen_issue = True; i = ISSUE.search(text)
            traj.append({"role": "user", "text": f"ISSUE:\n{i.group(1)}\n\nINSTRUCTIONS:\n{i.group(2).strip()}" if i else text})
        elif role == "assistant":
            traj.append({"role": "ai", "text": "\n".join(x for x in [text] + [_call(c) for c in m.get("tool_calls") or []] if x)})
        else:   # tool results and the harness's "please continue" nudges: what the environment said
            traj.append({"role": "user", "text": text})
    return traj


def strip_issue_md(patch):
    """The patch without a diff of the repository-root issue.md (see the module docstring)."""
    parts = re.split(r"(?m)^(?=diff --git )", patch or "")
    return "".join(p for p in parts if not p.startswith("diff --git a/issue.md b/issue.md"))


def exit_status(status, patch):
    if status == "submit": return "submitted" if patch.strip() else "submitted_no_patch"
    return (status or "unknown").split(":")[0]


def to_row(r, keep_issue_md=False):
    stripped = strip_issue_md(r["model_patch"]); patch = (r["model_patch"] or "") if keep_issue_md else stripped
    return {"instance_id": r["instance_id"], "repo": r["instance_id"].rsplit("-", 1)[0], "trajectory_id": r["trajectory_id"],
            "model_name": MODEL, "target": bool(r["resolved"]), "exit_status": exit_status(r["exit_status"], patch),
            "generated_patch": patch, "issue_md": stripped != (r["model_patch"] or ""), "trajectory": trajectory(r["trajectory"])}


def shortcut_counts(rows):
    """How many of to_row's rows carried the issue.md diff, and the resolved rate with and without it."""
    w = [r["target"] for r in rows if r["issue_md"]]; wo = [r["target"] for r in rows if not r["issue_md"]]
    rate = lambda v: round(sum(v) / len(v), 3) if v else None
    return {"runs": len(rows), "issue_md": len(w), "issues_with_issue_md": len({r["instance_id"] for r in rows if r["issue_md"]}),
            "resolved_with": rate(w), "resolved_without": rate(wo)}


def index(revision=REVISION):
    """One light row per run, in file order: trajectory_id, instance_id, target, shard and the row's position (_row), for drawing."""
    t = pq.read_table(path(revision), columns=["trajectory_id", "instance_id", "resolved"])
    return [{"trajectory_id": d, "instance_id": i, "target": bool(y), "shard": shard_of(i), "_row": k}
            for k, (d, i, y) in enumerate(zip(t["trajectory_id"].to_pylist(), t["instance_id"].to_pylist(), t["resolved"].to_pylist()))]


def hydrate(light, revision=REVISION, keep_issue_md=False):
    """The full rows (to_row) of index() rows, in the order given, reading only the row groups that hold them; prints
    shortcut_counts() for them."""
    f = pq.ParquetFile(path(revision)); starts, n = [], 0
    for g in range(f.metadata.num_row_groups): starts.append(n); n += f.metadata.row_group(g).num_rows
    want = sorted({r["_row"] for r in light}); got = {}
    for g, s in enumerate(starts):
        local = [k - s for k in want if s <= k < s + f.metadata.row_group(g).num_rows]
        if not local: continue
        for k, r in zip(local, f.read_row_group(g, columns=COLS).take(pa.array(local)).to_pylist()):
            got[s + k] = to_row(r, keep_issue_md)
    out = []
    for r in light:
        row = got[r["_row"]]
        assert row["instance_id"] == r["instance_id"], (r, row["instance_id"])
        out.append(row)
    print(f"from_openhands: issue.md diff {'kept' if keep_issue_md else 'stripped'}; {json.dumps(shortcut_counts(out))}", flush=True)
    return out
