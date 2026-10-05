"""Verifier-guided restarts under a step budget, replayed from real agent runs.

An agent has B actions for an issue. It starts a run; at action C a verifier scores the unfinished run (zero_shot.py
--cut-step C). Below a threshold the run is stopped and a fresh run of the same issue starts; a run that finishes is
submitted, and its recorded outcome is the result. Spending more than B actions is a failure. Each issue's pool is its
recorded runs (drawn without replacement), so every outcome is a real one.

    uv run python recipes/swe-verifier/budget_sim.py --full <test, --per-issue> --cut <test, --cut-step C> \
        --dev-full <dev ...> --dev-cut <dev ...> --step C --budgets 30,45,60

Policies: never stop (one run, as most agents do); stop at random (same stop rate as the verifier on dev, the control);
stop every run still going at C (the cheap "long runs fail" rule); the verifier, its threshold chosen on dev to maximise
the resolve rate at each budget; and an oracle that stops exactly the runs that would fail (the ceiling).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pools(full, cut, key):
    """{issue: [(actions, resolved, p_at_C or None)]}: p is None when the run had finished before action C."""
    p = {r["run_key"]: r.get(key) for r in cut}
    out = {}
    for r in full: out.setdefault(r["instance_id"], []).append((r["steps"], bool(r["y"]), p.get(r["run_key"])))
    return out


def simulate(pool, budget, step, stop, rng, draws):
    """Mean resolve rate over random run orders; stop(p, resolved) decides at action `step`."""
    wins = 0
    for _ in range(draws):
        order = rng.permutation(len(pool)); spent = 0; won = False
        for j in order:
            actions, resolved, p = pool[j]
            if p is not None and stop(p, resolved):
                spent += step
                if spent >= budget: break
                continue
            spent += actions
            won = resolved and spent <= budget
            break
        wins += won
    return wins / draws


def evaluate(pools_, budget, step, stop, rng, draws):
    return float(np.mean([simulate(p, budget, step, stop, rng, draws) for p in pools_.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in ("full", "cut", "dev-full", "dev-cut"): ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--step", type=int, required=True); ap.add_argument("--budgets", default="30,45,60")
    ap.add_argument("--key", default="p_d1a"); ap.add_argument("--draws", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sets = {}
    for name, fp, cp in (("test", a.full, a.cut), ("dev", a.dev_full, a.dev_cut)):
        sets[name] = pools(load(fp), load(cp), a.key)
    thresholds = np.linspace(0.02, 0.98, 49)
    for budget in (int(b) for b in a.budgets.split(",")):
        dev = sets["dev"]
        best = max(thresholds, key=lambda t: evaluate(dev, budget, a.step, lambda p, y, t=t: p < t, rng, 50))
        stop_rate = np.mean([p < best for runs in dev.values() for _, _, p in runs if p is not None])
        policies = {"never stop": lambda p, y: False, f"stop at random ({stop_rate:.0%})": lambda p, y: rng.random() < stop_rate,
                    f"stop all at action {a.step}": lambda p, y: True, f"verifier (t={best:.2f} from dev)": lambda p, y: p < best,
                    "oracle": lambda p, y: not y}
        print(f"\nbudget {budget} actions, check at action {a.step}: {len(sets['test'])} test issues")
        for name, stop in policies.items():
            print(f"  {name:32s} resolve rate {evaluate(sets['test'], budget, a.step, stop, rng, a.draws):.3f}")


if __name__ == "__main__":
    main()
