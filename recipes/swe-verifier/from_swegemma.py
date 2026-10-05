"""Gemma 4 agent runs (the competition harness, swegemma) in the shape zero_shot.py scores, so the verifier trained on
public SWE-agent runs can be tested on another agent unchanged.

    uv run --extra mlx python recipes/swe-verifier/zero_shot.py --swegemma <results dir> --tasks <tasks.jsonl> \
        --run runs/swe-verifier/train/r1 --out runs/swe-verifier/gemma/<name>.jsonl

A results directory holds task_results.jsonl (resolved by the harness's hidden tests), traces/trace_<id>.json (ATIF) and
patches/<id>.patch; the issue text comes from the tasks file. The trajectory becomes the system prompt, the issue (as
"ISSUE: ... INSTRUCTIONS:", which zero_shot.issue_of reads), then each agent turn ("ai": its text and tool calls) and each
tool result ("user"). Competition data stays local: outputs carry ids and scores, never the text.
"""
import json
from pathlib import Path


def _call(c):
    return f"{c.get('function_name', '')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})"


def trajectory(trace, issue):
    traj = [{"role": "system", "text": ""}, {"role": "user", "text": f"ISSUE:\n{issue}\n\nINSTRUCTIONS:"}]
    for s in trace.get("steps", []):
        if s.get("source") != "agent": continue
        text = "\n".join(x for x in [s.get("message") or ""] + [_call(c) for c in s.get("tool_calls") or []] if x)
        if text: traj.append({"role": "ai", "text": text})
        obs = s.get("observation")
        if obs:
            content = obs.get("content") if "content" in obs else [r.get("content") for r in obs.get("results", [])]
            for c in content if isinstance(content, list) else [content]:
                traj.append({"role": "user", "text": c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)})
    return traj


def load_runs(results_dir, tasks_path):
    d = Path(results_dir)
    issues = {t["instance_id"]: t["problem_statement"] for t in map(json.loads, Path(tasks_path).read_text(encoding="utf-8").splitlines()) if t}
    out = []
    for r in map(json.loads, (d / "task_results.jsonl").read_text(encoding="utf-8").splitlines()):
        tid = r["instance_id"]; safe = tid.replace("/", "__")
        tp, pp = d / "traces" / f"trace_{safe}.json", d / "patches" / f"{safe}.patch"
        trace = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {"steps": []}
        patch = pp.read_text(encoding="utf-8") if pp.exists() else ""
        exit_status = r["error"].split(":")[0] if r.get("error") else ("submitted" if patch else "submitted_no_patch")
        out.append({"instance_id": tid, "repo": str(r.get("repo") or tid), "target": bool(r.get("resolved")), "exit_status": exit_status,
                    "generated_patch": patch, "trajectory": trajectory(trace, issues.get(tid, "")), "model_name": "gemma-4-31b-agent"})
    return out
