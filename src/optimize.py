"""Cleaning scheduler.

1. Roll the fouling model forward week by week to get two daily trajectories per exchanger:
   "no cleaning" (from today's state) and "fresh" (starting from a just-cleaned state).
2. Turn them into the total cost of cleaning on each candidate day d (or not at all):
      energy lost to fouling before the cleaning
    + the whole duty lost while the exchanger is bypassed for cleaning
    + energy lost after it returns (fresh trajectory)
    + cleaning cost
   evaluated over a 301-day window so that benefits after the decision window still count.
3. Solve a mixed-integer program: at most one cleaning per exchanger, and at most
   `max_offline` exchangers out of service on any day.
"""
from dataclasses import dataclass

import joblib
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config as C
from .forecast import predict_quantiles
from .plant import daily_energy_cost, offline_cost_per_day

STEP = 7
N_STEPS = 43
HORIZON = STEP * N_STEPS        # 301-day evaluation window
DECISION_DAYS = 90              # cleanings may be scheduled in the next 90 days
TAU = 15.0                      # days the current crude campaign is assumed to persist


@dataclass
class CostParams:
    price: float = C.ENERGY_PRICE
    clean_mult: float = 1.0
    max_offline: int = C.MAX_OFFLINE
    rate_scale: float = 1.0     # planner calibration: scales the long-run cost rate c*


class Forecaster:
    """Rolls the 7-day fouling model forward to build long trajectories."""

    def __init__(self, bundle=None):
        self.bundle = bundle or joblib.load(C.MODEL_FILE)

    def rollout(self, idx, rf0, dsc0, flow, api_path, sul_path):
        """Daily Rf trajectories, shape (batch, HORIZON + 1); column 0 is today."""
        idx = np.asarray(idx, int)
        rf = np.asarray(rf0, float).copy()
        dsc = np.asarray(dsc0, float).copy()
        flow = np.asarray(flow, float)
        nodes = [rf.copy()]
        for k in range(N_STEPS):
            X = np.column_stack([idx, rf, dsc, flow, api_path[k], sul_path[k]])
            rf = rf + np.maximum(predict_quantiles(self.bundle, X, STEP)[1], 0.0)
            dsc = dsc + STEP
            nodes.append(rf.copy())
        nodes = np.array(nodes)
        t_nodes, t = np.arange(N_STEPS + 1) * STEP, np.arange(HORIZON + 1)
        return np.array([np.interp(t, t_nodes, nodes[:, b]) for b in range(nodes.shape[1])])


def crude_paths(now, long_run):
    """Current crude campaign persists briefly, then reverts to the long-run slate."""
    w = np.exp(-STEP * np.arange(N_STEPS)[:, None] / TAU)
    return np.asarray(long_run) + (np.asarray(now) - np.asarray(long_run)) * w


def solve_schedule(costs, none, busy, max_offline, time_limit=20):
    """Mixed-integer assignment of cleaning start days. Returns [(exchanger, day), ...]."""
    m, W = costs.shape
    nv = m * W
    c = (costs - none[:, None]).ravel()
    rows, rhs = [], []
    for i in range(m):                                   # at most one cleaning per exchanger
        r = np.zeros(nv)
        r[i * W:(i + 1) * W] = 1
        rows.append(r)
        rhs.append(1)
    for t in range(W + int(C.DOWNTIME.max())):           # capacity on every day
        r = np.zeros(nv)
        for i in range(m):
            lo, hi = max(0, t - int(C.DOWNTIME[i]) + 1), min(W - 1, t)
            if lo <= hi:
                r[i * W + lo:i * W + hi + 1] = 1
        rows.append(r)
        rhs.append(max(max_offline - int((busy > t).sum()), 0))
    ub = np.ones(nv)
    for i in range(m):
        if busy[i] > 0:
            ub[i * W:(i + 1) * W] = 0                    # already being cleaned
    res = milp(c, constraints=LinearConstraint(np.array(rows), -np.inf, np.array(rhs)),
               integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub), options={"time_limit": time_limit})
    if res.x is None:
        return []
    x = np.round(res.x).astype(int).reshape(m, W)
    return sorted([(int(i), int(d)) for i, d in zip(*np.nonzero(x))], key=lambda p: p[1])


def build_plan(fc, state, cp=None, decision_days=DECISION_DAYS):
    """Forecast, cost out every option and solve for the cheapest feasible schedule."""
    cp = cp or CostParams()
    m, T = C.N_EX, HORIZON
    idx = np.arange(m)
    api_p = np.tile(crude_paths(state["api_mean"], state["api_long"]), (1, 2))
    sul_p = np.tile(crude_paths(state["sulfur_mean"], state["sulfur_long"]), (1, 2))
    traj = fc.rollout(np.r_[idx, idx], np.r_[state["rf_state"], np.full(m, C.RF_AFTER_CLEAN)],
                      np.r_[state["dsc"], np.zeros(m)], np.tile(state["flow_mean"], 2), api_p, sul_p)
    nc, fresh = traj[:m], traj[m:]
    cost_nc = daily_energy_cost(nc[:, 1:], cp.price, C.DUTY[:, None], C.U0[:, None])
    cost_fc = daily_energy_cost(fresh[:, :T], cp.price, C.DUTY[:, None], C.U0[:, None])
    full = offline_cost_per_day(cp.price)
    clean = C.CLEAN_COST * cp.clean_mult
    busy = np.asarray(state["busy"], int)

    costs = np.zeros((m, decision_days))
    none = np.zeros(m)               # value of "wait": clean at the best day after the decision window
    never = np.zeros(m)              # cost if the exchanger is never cleaned during the whole window
    t_star = np.zeros(m, int)
    ages = np.arange(T + 1)
    for i in range(m):
        cum_nc = np.r_[0.0, np.cumsum(cost_nc[i])]
        cum_fc = np.r_[0.0, np.cumsum(cost_fc[i])]
        D = int(C.DOWNTIME[i])
        # Renewal argument. Cleaning a fresh exchanger every `a` days costs
        # (clean + downtime + fouling energy) / (a + D) per day. Its minimum c* (at the best cycle
        # length T*) is the long-run cost rate of a well-run exchanger. After a cleaning the rest of
        # the window is therefore valued at c* per day, and a run that is not cleaned carries the
        # relative value V(a) = c* min(a,T*) - cum(min(a,T*)) at the end of the window.
        a_grid = np.arange(20, T + 1)
        cycle = (clean[i] + D * full[i] + cum_fc[a_grid]) / (a_grid + D)
        t_star[i] = int(a_grid[np.argmin(cycle)])
        c_star = float(cycle.min()) * cp.rate_scale

        def V(age):
            a = int(min(max(round(float(age)), 0), t_star[i]))
            return c_star * a - cum_fc[a]

        if busy[i] > 0:                                   # offline now, back fresh after `busy` days
            never[i] = none[i] = busy[i] * full[i] + c_star * (T - busy[i])
            costs[i, :] = none[i]
            continue
        never[i] = cum_nc[T] + V(np.interp(nc[i, T], fresh[i], ages))  # age with the same Rf on the fresh curve
        d_all = np.arange(0, T - D)                       # every possible cleaning day in the window
        all_costs = cum_nc[d_all] + D * full[i] + clean[i] + c_star * (T - d_all - D)
        costs[i] = all_costs[:decision_days]
        none[i] = min(never[i], all_costs[decision_days:].min())
    schedule = solve_schedule(costs, none, busy, cp.max_offline)
    plan_cost = none.sum() + sum(costs[i, d] - none[i] for i, d in schedule)
    return {"schedule": schedule, "nc": nc, "fresh": fresh, "costs": costs, "none": none, "never": never,
            "t_star": t_star, "cost_none": float(never.sum()), "cost_plan": float(plan_cost), "busy": busy}


def planned_trajectory(plan, i):
    """Expected daily Rf of exchanger i under the plan (NaN while it is out of service)."""
    traj = plan["nc"][i].copy()
    for j, d in plan["schedule"]:
        if j == i:
            D = int(C.DOWNTIME[i])
            out = np.full(HORIZON + 1, np.nan)
            out[:d + 1] = plan["nc"][i, :d + 1]
            n_after = HORIZON + 1 - (d + D)
            if n_after > 0:
                out[d + D:] = plan["fresh"][i, :n_after]
            return out
    return traj


def threshold_rule_cost(plan, theta):
    """Forecast cost of 'clean when Rf reaches theta' (ignores crew limits, so flatters the rule)."""
    W = plan["costs"].shape[1]
    total = 0.0
    for i in range(C.N_EX):
        hit = np.nonzero(plan["nc"][i, :W] >= theta)[0]
        total += plan["costs"][i, hit[0]] if len(hit) and plan["busy"][i] == 0 else plan["none"][i]
    return float(total)
