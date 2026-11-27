"""Closed-loop backtest of cleaning strategies on simulated future years.

A fixed historical dataset cannot say what would have happened under a different
cleaning schedule, so each strategy is run day by day against the plant model in
plant.py. Every strategy faces exactly the same crude slate, throughput and random
noise in a given scenario. The ML strategy only sees what a real planner would see:
a noisy fouling reading, crude and flow history, and its own trained forecasting model.
Baselines are tuned in hindsight on the same scenarios, which favours them.
"""
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import LONG_WIN, build_dataset, current_state
from .optimize import CostParams, Forecaster, build_plan
from .plant import (daily_energy_cost, exchanger_flow, fouling_step, make_environment,
                    offline_cost_per_day)

REPLAN_DAYS = 14


class NeverClean:
    def decide(self, t, s):
        return []


class FixedInterval:
    def __init__(self, days):
        self.days = days

    def decide(self, t, s):
        return [i for i in range(C.N_EX) if s["busy"][i] == 0 and s["dsc"][i] >= self.days]


class Threshold:
    def __init__(self, theta):
        self.theta = theta

    def decide(self, t, s):
        return [i for i in range(C.N_EX) if s["busy"][i] == 0 and s["rf_state"][i] >= self.theta]


class Operator:
    """Status quo: the historical practice. Clean some days after Rf crosses a trigger level that
    differs from cycle to cycle (same rule that generated the historical data)."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.thr = self.rng.uniform(1.4, 0.8 * C.RF_MAX)
        self.planned = np.full(C.N_EX, -1)

    def decide(self, t, s):
        req = []
        for i in range(C.N_EX):
            if s["busy"][i] > 0:                              # cleaning started: next cycle gets a new trigger
                if self.planned[i] >= 0:
                    self.planned[i] = -1
                    self.thr[i] = self.rng.uniform(1.4, 0.8 * C.RF_MAX[i])
                continue
            if self.planned[i] < 0 and s["rf_state"][i] >= self.thr[i]:
                self.planned[i] = t + 1 + int(self.rng.integers(0, 40))
            if self.planned[i] >= 0 and t >= self.planned[i]:
                req.append(i)
        return req


class MLOptimizer:
    """Replans every REPLAN_DAYS days and executes the cleanings that fall before the next replan."""

    def __init__(self, forecaster, cp, replan=REPLAN_DAYS):
        self.fc, self.cp, self.replan, self.queue = forecaster, cp, replan, []

    def decide(self, t, s):
        if t % self.replan == 0:
            plan = build_plan(self.fc, s, self.cp)
            self.queue = [(i, t + d) for i, d in plan["schedule"] if d < self.replan]
        due = [i for i, start in self.queue if start <= t]
        self.queue = [(i, start) for i, start in self.queue if start > t]
        return due


def simulate(policy, scenario, init, cp):
    rng = np.random.default_rng(1000 + scenario)
    n, m = C.N_BT_DAYS, C.N_EX
    env = make_environment(n, rng)
    rf = init["rf_state"].copy()
    dsc = init["dsc"].copy()
    off = init["busy"].astype(int).copy()
    ret = np.zeros(m, bool)
    h_rf = np.tile(rf, (5, 1))
    h_flow = np.tile(init["flow_mean"], (14, 1))
    h_api = np.full(LONG_WIN, init["api_long"][0])
    h_sul = np.full(LONG_WIN, init["sulfur_long"][0])
    h_api[-5:], h_sul[-5:] = init["api_mean"][0], init["sulfur_mean"][0]
    full = offline_cost_per_day(cp.price)
    foul = down = clean = 0.0
    events, trace = [], np.full((n, m), np.nan)

    for t in range(n):
        for i in np.nonzero(ret)[0]:                         # back from cleaning
            rf[i], dsc[i] = env["after"][t, i], 0
            h_rf[:, i] = rf[i]
        ret[:] = False
        state = {"rf_state": h_rf.mean(0), "dsc": dsc.copy(), "flow_mean": h_flow.mean(0),
                 "api_mean": np.full(m, h_api[-5:].mean()), "sulfur_mean": np.full(m, h_sul[-5:].mean()),
                 "api_long": np.full(m, h_api.mean()), "sulfur_long": np.full(m, h_sul.mean()),
                 "busy": off.copy()}
        n_off = int((off > 0).sum())
        for i in sorted(set(policy.decide(t, state)), key=lambda j: -state["rf_state"][j]):
            if off[i] == 0 and n_off < cp.max_offline:       # crew limit applies to every strategy
                off[i] = C.DOWNTIME[i]
                n_off += 1
                clean += C.CLEAN_COST[i] * cp.clean_mult
                events.append((t, int(i)))

        flow_t = exchanger_flow(env["thr"][t], env["flow_noise"][t])
        new_rf = fouling_step(rf, env["api"][t], env["sulfur"][t], flow_t, env["proc"][t])
        online = off == 0
        rf = np.where(online, new_rf, rf)
        dsc = np.where(online, dsc + 1, dsc)
        foul += daily_energy_cost(rf, cp.price)[online].sum()
        down += full[~online].sum()
        trace[t] = np.where(online, rf, np.nan)

        obs = rf + env["meas"][t]
        h_rf[:, online] = np.vstack([h_rf[1:, online], obs[online]])
        h_flow[:, online] = np.vstack([h_flow[1:, online], flow_t[online]])
        h_api = np.r_[h_api[1:], env["api"][t]]
        h_sul = np.r_[h_sul[1:], env["sulfur"][t]]
        idx_off = np.nonzero(~online)[0]
        off[idx_off] -= 1
        ret[idx_off[off[idx_off] == 0]] = True

    return {"fouling": foul, "downtime": down, "cleaning": clean, "total": foul + down + clean,
            "n_cleanings": len(events), "events": events, "rf": trace}


FIXED_GRID = (120, 150, 180, 210, 240, 300)
THRESHOLD_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
VAL_SCENARIOS = (100, 101, 102)          # used only to tune the baselines
YEARS = C.N_BT_DAYS / 365.0


def baseline_factories():
    """(name, family, parameter, factory(scenario) -> policy) for every non-ML strategy."""
    out = [("Never clean", "never", None, lambda s: NeverClean()),
           ("Operator (status quo)", "operator", None, lambda s: Operator(5000 + s))]
    out += [(f"Fixed {d} d", "fixed", d, lambda s, d=d: FixedInterval(d)) for d in FIXED_GRID]
    out += [(f"Threshold {th:g}", "threshold", th, lambda s, th=th: Threshold(th)) for th in THRESHOLD_GRID]
    return out


def _row(name, family, param, split, scenario, r):
    return {"policy": name, "family": family, "param": param, "split": split, "scenario": scenario,
            **{k: r[k] / YEARS for k in ("total", "fouling", "downtime", "cleaning")},   # USD per year
            "cleanings_per_year": r["n_cleanings"] / YEARS}


def run_backtest(bundle=None, cp=None, n_scen=C.N_SCENARIOS, verbose=True):
    df = build_dataset()
    init = current_state(df)
    cp = cp or CostParams()
    fc = Forecaster(bundle)
    rows, traces = [], {}
    # 1. tune the baselines on validation scenarios the ML optimiser never sees
    for name, fam, par, make in baseline_factories():
        for s in VAL_SCENARIOS:
            rows.append(_row(name, fam, par, "validation", s, simulate(make(s), s, init, cp)))
    # 2. every strategy on the held-out test scenarios
    pols = baseline_factories() + [("ML optimiser", "ml", None, lambda s: MLOptimizer(fc, cp))]
    for name, fam, par, make in pols:
        for s in range(n_scen):
            r = simulate(make(s), s, init, cp)
            rows.append(_row(name, fam, par, "test", s, r))
            if s == 0:
                traces[name] = {"rf": r["rf"], "events": r["events"]}
        if verbose:
            sub = pd.DataFrame(rows).query("policy == @name and split == 'test'")
            print(f"{name:<22} ${sub['total'].mean() / 1e6:6.3f} M/yr   cleanings/yr {sub['cleanings_per_year'].mean():5.1f}", flush=True)
    return pd.DataFrame(rows), traces


def main(n_scen=C.N_SCENARIOS):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res, traces = run_backtest(n_scen=n_scen)
    res.to_csv(C.BACKTEST_FILE, index=False)
    joblib.dump(traces, C.TRACES_FILE)


if __name__ == "__main__":
    main()
