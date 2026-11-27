"""Plant model shared by the data generator and the closed-loop backtest.

Fouling resistance Rf (units of 1e-4 m2K/W) grows towards an asymptote. The growth
rate depends on the crude slate (lower API and higher sulfur foul faster) and on
flow velocity (higher velocity fouls slower). A cleaning resets Rf to a small
residual value. Fouling lowers heat recovery, which the furnace has to make up with
extra fuel; that extra fuel is the energy cost the optimiser tries to reduce.
"""
import numpy as np
import pandas as pd

from . import config as C


def make_environment(n_days, rng):
    """Crude slate, plant throughput and all random draws for one simulated period."""
    lengths, api_v, sul_v = [], [], []
    total = 0
    while total < n_days:
        length = int(rng.integers(10, 26))                      # crude campaign length
        api = float(np.clip(rng.normal(32, 4.5), 22, 42))
        sul = float(np.clip(1.3 + (34 - api) * 0.13 + rng.normal(0, 0.25), 0.3, 3.5))
        lengths.append(length)
        api_v.append(api)
        sul_v.append(sul)
        total += length
    api = pd.Series(np.repeat(api_v, lengths)[:n_days])
    sul = pd.Series(np.repeat(sul_v, lengths)[:n_days])
    api = api.rolling(3, center=True, min_periods=1).mean().to_numpy() + rng.normal(0, 0.15, n_days)
    sul = sul.rolling(3, center=True, min_periods=1).mean().to_numpy() + rng.normal(0, 0.03, n_days)

    x = np.zeros(n_days)
    for t in range(1, n_days):
        x[t] = 0.97 * x[t - 1] + rng.normal(0, 0.012)
    thr = np.clip(1.0 + 0.04 * np.sin(2 * np.pi * (np.arange(n_days) / 365 - 0.2)) + x, 0.8, 1.08)

    sigma = 0.3
    shape = (n_days, C.N_EX)
    return {
        "api": api,
        "sulfur": np.clip(sul, 0.2, None),
        "thr": thr,
        "proc": np.exp(rng.normal(-sigma ** 2 / 2, sigma, shape)),       # process noise on fouling rate
        "meas": rng.normal(0, 0.10, shape),                              # Rf soft-sensor noise
        "flow_noise": rng.normal(0, 0.015, shape),
        "after": np.clip(rng.normal(C.RF_AFTER_CLEAN, 0.04, shape), 0.03, None),
    }


def crude_factor(api, sulfur):
    """Relative fouling propensity of the crude slate (1.0 for a 32 API, 1.6 % S crude)."""
    return np.clip(1.0 + 0.55 * (sulfur - 1.6) + 0.035 * (32.0 - api), 0.35, 2.5)


def exchanger_flow(thr_t, noise_t):
    return C.FLOW_NOM * thr_t * (1.0 + noise_t)


def fouling_step(rf, api, sulfur, flow, proc):
    """Advance Rf by one day for all exchangers (vectorised)."""
    velocity = np.clip(flow / C.FLOW_NOM, 0.5, 1.5) ** -1.0
    rate = C.K * crude_factor(api, sulfur) * velocity * (1.0 - rf / C.RF_MAX) * proc
    return rf + np.maximum(rate, 0.0)


def heat_loss_frac(rf, u0=None):
    """Fraction of clean heat duty lost to fouling."""
    u0 = C.U0 if u0 is None else u0
    return 1.0 - 1.0 / (1.0 + u0 * rf * 1e-4)


def daily_energy_cost(rf, price, duty=None, u0=None):
    """Extra furnace fuel cost per day [USD] caused by fouling."""
    duty = C.DUTY if duty is None else duty
    return price * duty * 24.0 * heat_loss_frac(rf, u0)


def offline_cost_per_day(price, duty=None):
    """Cost per day while an exchanger is bypassed for cleaning: its whole duty is lost."""
    duty = C.DUTY if duty is None else duty
    return price * duty * 24.0


def observables(i, rf, flow, rng):
    """Synthetic sensor readings for exchanger i derived from its fouling state."""
    n = len(rf)
    t = np.arange(n)
    eps = C.EPS_CLEAN * (1.0 - heat_loss_frac(rf, C.U0[i]))
    tc_in = C.TC_IN0[i] + 3 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 0.8, n)
    th_in = C.TC_IN0[i] + 160 + 4 * np.sin(2 * np.pi * t / 365 + 1) + rng.normal(0, 1.0, n)
    span = th_in - tc_in
    tc_out = tc_in + eps * span + rng.normal(0, 0.4, n)
    th_out = th_in - 0.85 * eps * span + rng.normal(0, 0.4, n)
    dp = C.DP0[i] * (flow / C.FLOW_NOM[i]) ** 1.8 * (1 + 0.12 * rf) * (1 + rng.normal(0, 0.02, n))
    return tc_in, tc_out, th_in, th_out, dp
