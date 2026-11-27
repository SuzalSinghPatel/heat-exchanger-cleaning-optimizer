"""Sanity tests:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.backtest import NeverClean, simulate  # noqa: E402
from src.features import FEATURES, add_targets, current_state, preprocess  # noqa: E402
from src.forecast import split  # noqa: E402
from src.generate_data import simulate_history  # noqa: E402
from src.optimize import HORIZON, CostParams, build_plan  # noqa: E402
from src.plant import fouling_step, heat_loss_frac  # noqa: E402


@pytest.fixture(scope="module")
def history():
    df, log = simulate_history()
    return df, log


def test_fouling_grows_and_stays_below_plateau():
    rf = np.full(C.N_EX, 0.1)
    for _ in range(2000):
        new = fouling_step(rf, 32.0, 1.6, C.FLOW_NOM, np.ones(C.N_EX))
        assert (new >= rf).all()
        rf = new
    assert (rf < C.RF_MAX).all() and (rf > 0.9 * C.RF_MAX).all()


def test_heat_loss_monotone_and_bounded():
    x = np.linspace(0, 8, 50)[:, None]
    loss = heat_loss_frac(x)
    assert (np.diff(loss, axis=0) > 0).all() and (loss >= 0).all() and (loss < 1).all()


def test_dataset_shape_seed_and_gaps(history):
    df, log = history
    assert len(df) == C.N_DAYS * C.N_EX and len(log) > 20
    assert df.loc[df["status"] == "cleaning", "fouling_resistance"].isna().all()
    df2, _ = simulate_history()
    assert df.equals(df2)                                       # fixed seed -> identical data
    assert 0.005 < df["fouling_resistance"].isna().mean() < 0.08


def test_no_leakage_between_train_and_test(history):
    df = add_targets(preprocess(history[0]))
    for h in (7, 14, 30):
        tr, te = split(df, h)
        assert (tr["day"] + h < C.TRAIN_END_DAY).all()          # training targets end before the test year
        assert (te["day"] >= C.TRAIN_END_DAY).all()
        assert tr[FEATURES].notna().all().all()


def test_current_state(history):
    s = current_state(preprocess(history[0]))
    assert s["rf_state"].shape == (C.N_EX,) and (s["rf_state"] > 0).all()


class LinearForecaster:
    """Stand-in forecaster: Rf grows linearly, so the optimiser can be tested without trained models."""

    def rollout(self, idx, rf0, dsc0, flow, api_path, sul_path):
        t = np.arange(HORIZON + 1)
        return np.minimum(np.asarray(rf0)[:, None] + 0.02 * t, C.RF_MAX[np.asarray(idx)][:, None])


def _state(rf):
    n = C.N_EX
    return {"rf_state": np.full(n, rf), "dsc": np.full(n, 150.0), "flow_mean": C.FLOW_NOM.copy(),
            "api_mean": np.full(n, 32.0), "sulfur_mean": np.full(n, 1.6), "api_long": np.full(n, 32.0),
            "sulfur_long": np.full(n, 1.6), "busy": np.zeros(n, int)}


@pytest.mark.parametrize("max_offline", [1, 2])
def test_schedule_respects_crew_limit(max_offline):
    plan = build_plan(LinearForecaster(), _state(3.5), CostParams(max_offline=max_offline))
    assert plan["schedule"], "heavily fouled exchangers should be scheduled"
    active = np.zeros(200, int)
    for i, d in plan["schedule"]:
        active[d:d + C.DOWNTIME[i]] += 1
    assert active.max() <= max_offline
    assert len({i for i, _ in plan["schedule"]}) == len(plan["schedule"])      # at most one cleaning each
    assert plan["cost_plan"] <= plan["cost_none"]


def test_clean_exchangers_are_not_scheduled_soon():
    plan = build_plan(LinearForecaster(), _state(0.12), CostParams())
    assert all(d > 20 for _, d in plan["schedule"])


def test_backtest_engine_accounting():
    r = simulate(NeverClean(), 0, current_state(preprocess(simulate_history()[0])), CostParams())
    assert r["cleaning"] == 0 and r["downtime"] == 0 and r["n_cleanings"] == 0
    assert np.isclose(r["total"], r["fouling"]) and r["total"] > 0
