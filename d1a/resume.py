"""Resume points for d1a.train (--save_every_steps / --save_every_minutes, then --resume 1).

A point is one file, `<out>/resume/step-N/state.pt`: the trainable tensors (LoRA and pointer head), the optimizer and
scheduler states and the RNG state. `latest.json` names the point to continue from and carries the position (epoch,
micro-batch, counters) and the arguments the run must be resumed with; it is written after the point, atomically, so a
run killed mid-write continues from the previous point. Earlier points are then removed, and train removes the folder
when the checkpoint is saved.
"""
import os
import shutil
import time
from pathlib import Path

import torch

from .suite import read_json, write_json

LATEST = "latest.json"
STATE = "state.pt"
WRITE_SHARE = 0.05   # at most this share of wall time may block on writing resume points


def due(step, every_steps, every_minutes, since, blocked):
    """Whether to write a point after this optimizer step: every `every_steps` steps, or once `every_minutes` have passed
    since `since`, stretched so the time spent writing (`blocked` seconds per point) stays under WRITE_SHARE of it."""
    interval = max(60 * every_minutes, blocked / WRITE_SHARE)
    return bool(every_steps and step % every_steps == 0) or bool(every_minutes and time.time() - since >= interval)


def save(resume_dir, step, params, opt, sched, position):
    """Write the point for `step` and make it the latest. Returns the seconds it took."""
    started, resume_dir = time.time(), Path(resume_dir)
    target = resume_dir / f"step-{step:07d}"
    target.mkdir(parents=True, exist_ok=True)
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}
    state = {"params": [p.detach().to("cpu", copy=True) for p in params], "optimizer": opt.state_dict(),
             "scheduler": sched.state_dict(), "rng": rng}
    torch.save(state, target / f".{STATE}.tmp")
    os.replace(target / f".{STATE}.tmp", target / STATE)
    write_json(resume_dir / LATEST, {"dir": target.name, "step": step, **position}, atomic=True)
    for old in resume_dir.glob("step-*"):
        if old != target: shutil.rmtree(old)
    return round(time.time() - started, 1)


def load(resume_dir, params, opt, sched, args):
    """-> the saved position, or None when there is no point, after restoring the trainable tensors, the optimizer, the
    scheduler and the RNG state. Refuses a point written with other arguments."""
    resume_dir = Path(resume_dir)
    if not (resume_dir / LATEST).exists(): return None
    position = read_json(resume_dir / LATEST)
    if position["args"] != args:
        changed = sorted(k for k in set(args) | set(position["args"]) if args.get(k) != position["args"].get(k))
        raise ValueError(f"resume point {resume_dir} was written with other arguments: {changed}")
    saved = torch.load(resume_dir / position["dir"] / STATE, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p, t in zip(params, saved["params"], strict=True): p.copy_(t)
    opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["rng"]["torch"])
    if saved["rng"]["cuda"] is not None: torch.cuda.set_rng_state(saved["rng"]["cuda"])
    return position
