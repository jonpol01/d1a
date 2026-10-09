"""recipes/swe-verifier/from_openhands.py: an OpenHands row (nebius/SWE-rebench-openhands-trajectories) becomes the row
zero_shot.py scores, and a draw made on the light index reads back the same runs. Synthetic rows; no download, no model."""
import importlib.util
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"swe_verifier_{name}", ROOT / f"recipes/swe-verifier/{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


oh, zero_shot = _load("from_openhands"), _load("zero_shot")
ISSUE_MD = "diff --git a/issue.md b/issue.md\nnew file mode 100644\n--- /dev/null\n+++ b/issue.md\n@@ -0,0 +1 @@\n+the issue\n"
FIX = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"


def row(iid="org__repo-7", resolved=1, status="submit", patch=ISSUE_MD + FIX, tid="t0"):
    call = lambda name, args: [{"function": {"name": name, "arguments": json.dumps(args)}, "id": "c", "type": "function"}]
    msg = lambda role, content, tool_calls=None, name=None: {"role": role, "content": content, "tool_calls": tool_calls, "name": name, "tool_call_id": None}
    return {"trajectory_id": tid, "instance_id": iid, "repo": "org/repo", "model_patch": patch, "exit_status": status, "resolved": resolved,
            "trajectory": [msg("system", "You are OpenHands agent."),
                           msg("user", "<uploaded_files>\n/workspace/x\n</uploaded_files>\n<issue_description>\nx should be 2\n</issue_description>\n\nFix it."),
                           msg("assistant", "Let me look.", call("execute_bash", {"command": "pytest -q"})),
                           msg("tool", "Traceback (most recent call last):\nAssertionError", name="execute_bash"),
                           msg("user", "Please continue working on the task."),
                           msg("assistant", None, call("str_replace_editor", {"command": "str_replace", "path": "pkg/core.py"})),
                           msg("tool", "3 passed in 0.1s", name="str_replace_editor"),
                           msg("assistant", "Done.", call("finish", {"message": "fixed"}))]}


def test_row_maps_to_zero_shot_shape():
    r = oh.to_row(row())
    assert (r["target"], r["exit_status"], r["repo"], r["generated_patch"]) == (True, "submitted", "org__repo", FIX)   # issue.md dropped
    assert [m["role"] for m in r["trajectory"]] == ["system", "user", "ai", "user", "user", "ai", "user", "ai"]
    assert r["trajectory"][2]["text"] == 'Let me look.\nexecute_bash({"command": "pytest -q"})'
    assert zero_shot.issue_of(r["trajectory"]) == "x should be 2"
    h = zero_shot.heuristics(r)
    assert (h["patch_lines"], h["clean_submit"], h["steps"], h["error_obs"], h["success_printed"]) == (2, 1.0, 3, 1, 1.0)
    state = zero_shot.state_of(r, 3000, 6000, 3000)
    assert state.startswith("ISSUE:\nx should be 2\n\nPATCH:\n" + FIX) and state.endswith('[agent] Done.\nfinish({"message": "fixed"})')
    assert zero_shot.cut(r, 2)["trajectory"][-1]["text"] == "3 passed in 0.1s"
    assert oh.to_row(row(status="RuntimeError: Agent reached maximum iteration."))["exit_status"] == "RuntimeError"
    assert oh.to_row(row(patch=ISSUE_MD))["exit_status"] == "submitted_no_patch"   # only the harness's issue.md: no patch of its own


def test_draw_on_the_index_reads_back_the_drawn_runs(tmp_path, monkeypatch):
    rows = [row(f"org__repo{i % 3}-{i}", int(random.Random(j * 31 + i).random() < 0.5), tid=f"{i}.{j}") for i in range(9) for j in range(4)]
    random.Random(0).shuffle(rows)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "t.parquet", row_group_size=5)
    monkeypatch.setattr(oh, "path", lambda revision=oh.REVISION: str(tmp_path / "t.parquet"))
    light = oh.index()
    drawn = zero_shot.per_issue_draw(light, 2, True, random.Random(0))
    full = oh.hydrate(drawn)
    by_tid = {r["trajectory_id"]: r for r in rows}
    assert [r["trajectory_id"] for r in full] == [rows[r["_row"]]["trajectory_id"] for r in drawn]
    assert all(f["target"] == bool(by_tid[f["trajectory_id"]]["resolved"]) == d["target"] for f, d in zip(full, drawn))
    assert drawn and all({r["target"] for r in full if r["instance_id"] == i} == {True, False} for i in {r["instance_id"] for r in full})
