"""nll_on_grid and the fits built on it (fit_temperature, temperature_gate.interval) give exactly what the per-row loop
they replaced gave: bit for bit, on rows with and without logits, several option counts, tied logits, and a grid holding 1."""
from collections import defaultdict

import numpy as np
import pytest

from d1a.eval import metrics
from d1a.eval.metrics import TEMPERATURE_FIT, TEMPERATURE_GRID, cluster_resamples, nll_at_temperature, nll_on_grid
from d1a.training import temperature_gate


def rows_mixed(n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = int(rng.choice([2, 3, 5, 9, 40]))
        z = rng.normal(0, 3, k)
        if i % 17 == 0: z[:2] = z.max()   # a tie at the top
        p = np.exp(z - z.max()); p /= p.sum()
        row = {"id": f"r{i}", "question": "q", "label": int(rng.integers(k)), "variant": "clean", "task": f"t{i % 4}",
               "source": f"s{i % 3}", "group": f"g{i % 29}", "p": p.tolist()}
        if i % 5:
            row["logits"] = z.tolist()
        else:   # no logits: the floored-probability path, with an exact zero that hits the floor
            p[int(rng.integers(k))] = 0.0
            row["p"] = (p / p.sum()).tolist()
        out.append(row)
    return out


def old_fit(rows, aggregation="macro", points=81):
    """fit_temperature as it was: one nll_at_temperature call per row and grid point."""
    rows = metrics.scored_rows(rows)
    grid = np.exp(np.linspace(*np.log(TEMPERATURE_GRID), points))
    if aggregation == "micro":
        weight = np.ones(len(rows))
    else:
        per_task = defaultdict(int)
        for row in rows: per_task[row["task"]] += 1
        weight = np.asarray([1.0 / per_task[row["task"]] for row in rows])
    loss = [np.average([nll_at_temperature(row, float(t)) for row in rows], weights=weight) for t in grid]
    return float(grid[int(np.argmin(loss))]), loss


def old_interval_fits(rows, samples):
    nll = np.asarray([[nll_at_temperature(row, float(t)) for t in temperature_gate.GRID] for row in rows])
    return [temperature_gate.GRID[int(np.argmin(nll[idx].mean(axis=0)))] for idx in cluster_resamples(rows, samples, temperature_gate.SEED)]


def test_the_grid_losses_equal_the_per_row_ones_bit_for_bit():
    rows = rows_mixed()
    grid = [0.25, 0.5, 1.0, 1.7, 4.0, *np.exp(np.linspace(*np.log(TEMPERATURE_GRID), TEMPERATURE_FIT["points"]))]
    got = nll_on_grid(rows, grid)
    want = np.asarray([[nll_at_temperature(row, float(t)) for row in rows] for t in grid])
    assert got.shape == (len(grid), len(rows)) and got.flags["C_CONTIGUOUS"]
    assert np.array_equal(got, want)   # exact, not close


@pytest.mark.parametrize("aggregation,points", [("micro", 121), ("macro", 81)])
def test_fit_temperature_picks_the_same_grid_point_from_the_same_losses(aggregation, points):
    rows = rows_mixed()
    want, loss = old_fit(rows, aggregation, points)
    grid = np.exp(np.linspace(*np.log(TEMPERATURE_GRID), points))
    assert metrics.fit_temperature(rows, aggregation, points) == want
    weight = np.ones(len(rows)) if aggregation == "micro" else None
    if weight is not None:
        assert [np.average(per_t, weights=weight) for per_t in nll_on_grid(rows, grid)] == loss


def test_the_temperature_interval_equals_the_per_row_loop():
    rows = rows_mixed(200, seed=1)
    want = old_interval_fits(rows, 300)
    got = temperature_gate.interval(rows, samples=300)
    lower, upper = np.quantile(want, [0.05, 0.95])
    assert (got["lower"], got["upper"]) == (float(lower), float(upper))


def test_bad_rows_fail_as_the_per_row_path_does():
    good = rows_mixed(3)
    for bad in ({**good[1], "logits": good[1]["logits"][:-1]}, {**good[1], "logits": [float("nan")] * len(good[1]["p"])}):
        with pytest.raises(ValueError, match="logits must be finite and match the option count"):
            nll_at_temperature(bad, 2.0)
        with pytest.raises(ValueError, match="logits must be finite and match the option count"):
            nll_on_grid([good[0], bad], [2.0])
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        nll_on_grid(good, [1.0, 0.0])
