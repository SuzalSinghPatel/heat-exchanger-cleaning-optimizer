"""Generate three years of synthetic daily preheat-train data.

The history is produced under an "operator style" cleaning policy: an exchanger is
cleaned some days after its fouling resistance crosses a threshold that differs from
one cleaning to the next. This variety is what lets the forecasting models see
fouling behaviour over a wide range of Rf values.

The data is SYNTHETIC. It is produced by formulas in plant.py, not by a refinery.
"""
import numpy as np
import pandas as pd

from . import config as C
from .plant import exchanger_flow, fouling_step, make_environment, observables


def simulate_history(seed=C.SEED, n_days=C.N_DAYS, max_offline_hist=1):
    rng = np.random.default_rng(seed)
    env = make_environment(n_days, rng)
    m = C.N_EX
    rf_true = np.full((n_days, m), np.nan)
    flow = np.full((n_days, m), np.nan)
    dsc_arr = np.full((n_days, m), np.nan)
    cleaned = np.zeros((n_days, m), int)
    online = np.zeros((n_days, m), bool)

    rf = rng.uniform(0.3, 2.2, m)
    dsc = rng.integers(20, 150, m).astype(float)
    thr = rng.uniform(1.4, 0.8 * C.RF_MAX)      # operator trigger, always below the fouling plateau
    planned = np.full(m, -1)
    clean_end = np.full(m, -1)
    events = []

    for t in range(n_days):
        for i in np.argsort(-rf):               # most fouled exchanger gets the crew first
            if (t >= clean_end[i] and planned[i] >= 0 and t >= planned[i]
                    and (clean_end > t).sum() < max_offline_hist):
                clean_end[i] = t + C.DOWNTIME[i]
                planned[i] = -1
                events.append((i, t, t + C.DOWNTIME[i]))
        f_t = exchanger_flow(env["thr"][t], env["flow_noise"][t])
        new_rf = fouling_step(rf, env["api"][t], env["sulfur"][t], f_t, env["proc"][t])
        for i in range(m):
            if t < clean_end[i]:
                continue                        # exchanger is out for cleaning
            if t == clean_end[i]:               # back in service, freshly cleaned
                rf[i] = env["after"][t, i]
                dsc[i] = 0
                thr[i] = rng.uniform(1.4, 0.8 * C.RF_MAX[i])
                cleaned[t, i] = 1
            else:
                rf[i] = new_rf[i]
                dsc[i] += 1
            online[t, i] = True
            rf_true[t, i] = rf[i]
            flow[t, i] = f_t[i]
            dsc_arr[t, i] = dsc[i]
            if planned[i] < 0 and rf[i] >= thr[i]:
                planned[i] = t + 1 + int(rng.integers(0, 40))   # operator reacts with a delay

    df = _assemble(env, rf_true, flow, dsc_arr, cleaned, online, rng)
    dates = pd.date_range(C.START_DATE, periods=n_days)
    log = pd.DataFrame([
        {"exchanger_id": C.EX_ID[i], "start_date": dates[s].date(),
         "return_date": dates[e].date() if e < n_days else None,
         "cleaning_cost_usd": C.CLEAN_COST[i]}
        for i, s, e in events
    ])
    return df, log


def _assemble(env, rf_true, flow, dsc_arr, cleaned, online, rng):
    n, m = rf_true.shape
    dates = pd.date_range(C.START_DATE, periods=n)
    frames = []
    for i in range(m):
        tc_in, tc_out, th_in, th_out, dp = observables(i, rf_true[:, i], flow[:, i], rng)
        drift = np.cumsum(rng.normal(0, 0.004, n))            # slow soft-sensor drift
        frames.append(pd.DataFrame({
            "date": dates,
            "exchanger_id": C.EX_ID[i],
            "status": np.where(online[:, i], "online", "cleaning"),
            "flow_m3h": flow[:, i],
            "t_cold_in_C": tc_in, "t_cold_out_C": tc_out,
            "t_hot_in_C": th_in, "t_hot_out_C": th_out,
            "dp_kpa": dp,
            "crude_api": env["api"], "crude_sulfur_pct": env["sulfur"],
            "fouling_resistance": rf_true[:, i] + env["meas"][:, i] + drift,
            "days_since_clean": dsc_arr[:, i],
            "cleaned": cleaned[:, i],
        }))
    df = pd.concat(frames, ignore_index=True)

    on = df["status"].eq("online").to_numpy()
    missing = {"flow_m3h": .015, "t_cold_in_C": .015, "t_cold_out_C": .015, "t_hot_in_C": .015,
               "t_hot_out_C": .015, "dp_kpa": .015, "fouling_resistance": .015,
               "crude_api": .01, "crude_sulfur_pct": .01}
    for col, p in missing.items():
        df.loc[on & (rng.random(len(df)) < p), col] = np.nan
    spike = on & (rng.random(len(df)) < 0.004)                # Rf outliers
    df.loc[spike, "fouling_resistance"] = df.loc[spike, "fouling_resistance"] * rng.uniform(1.6, 2.5, spike.sum())
    spike = on & (rng.random(len(df)) < 0.003)                # flow-meter glitches
    df.loc[spike, "flow_m3h"] = df.loc[spike, "flow_m3h"] * rng.choice([0.5, 1.6], spike.sum())
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    return df


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df, log = simulate_history()
    df.to_csv(C.DATA_FILE, index=False)
    log.to_csv(C.LOG_FILE, index=False)
    print(f"wrote {C.DATA_FILE.name}: {len(df):,} rows, {df['exchanger_id'].nunique()} exchangers, "
          f"{len(log)} cleaning events")


if __name__ == "__main__":
    main()
