"""Fouling forecasting: a physics-informed hybrid model.

Physics part (Kern-Seaton asymptotic fouling, fitted per exchanger by least squares):
    d Rf / dt = beta * (R* - Rf)   =>   change over h days  p_h = (R* - Rf) * (1 - exp(-beta h))
ML part (gradient boosting, quantiles 0.1 / 0.5 / 0.9): learns the multiplier
    r = actual change / p_h
from the crude slate (API, sulfur), flow, run age and the exchanger identity. Because the
multiplier is applied on top of the physical law, long roll-outs approach the plateau
instead of running off the range of fouling levels seen in the history (a pure tree model
cannot extrapolate beyond it).

Intervals are widened with split-conformal calibration. Time split: fit on years 1-2,
score on year 3.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .features import FEATURES, HORIZONS, build_dataset

QUANTILES = (0.1, 0.5, 0.9)
CAL_DAYS = 120          # last days of the training period reserved to calibrate the intervals
MIN_P = 0.03            # ignore rows where the physics predicts (almost) no growth


def split(df, h):
    ok = df["online"] & df[FEATURES].notna().all(axis=1) & df[f"y{h}"].notna()
    train = df[ok & (df["day"] + h < C.TRAIN_END_DAY)]        # target window ends before the test year
    test = df[ok & (df["day"] >= C.TRAIN_END_DAY)]
    return train, test


def _kern_seaton(tr):
    """Per-exchanger fit of  rate = alpha - beta * Rf  ->  plateau R* = alpha / beta."""
    rstar, beta = np.zeros(C.N_EX), np.zeros(C.N_EX)
    for i in range(C.N_EX):
        g = tr[tr["exch_idx"] == i]
        slope, icpt = np.polyfit(g["rf_state"].to_numpy(), g["y7"].to_numpy() / 7.0, 1)
        alpha, b = max(icpt, 1e-3), max(-slope, 1e-4)
        rstar[i] = np.clip(alpha / b, g["rf_state"].max() * 1.05, 15.0)
        beta[i] = alpha / rstar[i]
    return rstar, beta


def _physics(bundle, X, h):
    """Physical h-day change p_h and the feature matrix for the ML multiplier."""
    idx = X[:, 0].astype(int)
    rf = X[:, 1]
    p = np.maximum(bundle["rstar"][idx] - rf, 0.0) * (1.0 - np.exp(-bundle["beta"][idx] * h))
    Z = X.copy()
    Z[:, 1] = np.clip(rf / bundle["rstar"][idx], 0.0, 1.3)       # fraction of the plateau
    return p, Z


def _model(q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=200, learning_rate=0.06, max_leaf_nodes=12,
        min_samples_leaf=60, l2_regularization=1.0, random_state=0,
        categorical_features=[True] + [False] * (len(FEATURES) - 1))


def _fit_ratio(bundle, tr, h, q):
    p, Z = _physics(bundle, tr[FEATURES].to_numpy(), h)
    keep = p > MIN_P
    return _model(q).fit(Z[keep], tr[f"y{h}"].to_numpy()[keep] / p[keep], sample_weight=p[keep])


def _predict_ratio(model, bundle, X, h):
    p, Z = _physics(bundle, X, h)
    return model.predict(Z) * p


def _conformal_margin(bundle, tr, h):
    """Split-conformal widening of [q10, q90] so that it covers ~80 % of outcomes.

    Core models are fitted on the older part of the training period and scored on the
    last CAL_DAYS days, which none of them has seen."""
    cut = C.TRAIN_END_DAY - CAL_DAYS
    core, cal = tr[tr["day"] + h < cut], tr[tr["day"] >= cut]
    lo, hi = _fit_ratio(bundle, core, h, 0.1), _fit_ratio(bundle, core, h, 0.9)
    Xc, yc = cal[FEATURES].to_numpy(), cal[f"y{h}"].to_numpy()
    scores = np.maximum(_predict_ratio(lo, bundle, Xc, h) - yc, yc - _predict_ratio(hi, bundle, Xc, h))
    level = min(1.0, np.ceil((len(scores) + 1) * 0.8) / len(scores))
    return float(np.quantile(scores, level))


def fit(df):
    tr7, _ = split(df, 7)
    rstar, beta = _kern_seaton(tr7)
    bundle = {"rstar": rstar, "beta": beta, "models": {}, "mean_delta": {}, "margin": {}, "features": FEATURES}
    for h in HORIZONS:
        tr, _ = split(df, h)
        for q in QUANTILES:
            bundle["models"][(h, q)] = _fit_ratio(bundle, tr, h, q)
        bundle["mean_delta"][h] = tr.groupby("exch_idx")[f"y{h}"].mean().to_dict()
        bundle["margin"][h] = _conformal_margin(bundle, tr, h)
    return bundle


def predict_quantiles(bundle, X, h):
    """(low, median, high) predictions of the h-day change in fouling state."""
    X = np.asarray(X, float)
    lo, med, hi = (_predict_ratio(bundle["models"][(h, q)], bundle, X, h) for q in QUANTILES)
    m = bundle["margin"][h]
    return np.minimum(lo, med) - m, med, np.maximum(hi, med) + m


def predict_frame(bundle, df, h):
    """Test-year rows with predicted quantiles of the h-day change."""
    _, te = split(df, h)
    te = te.copy()
    te["q10"], te["q50"], te["q90"] = predict_quantiles(bundle, te[FEATURES].to_numpy(), h)
    return te


def evaluate(bundle, df):
    rows = []
    for h in HORIZONS:
        te = predict_frame(bundle, df, h)
        y, pred = te[f"y{h}"].to_numpy(), te["q50"].to_numpy()
        mean_rate = te["exch_idx"].map(bundle["mean_delta"][h]).to_numpy()
        mae = lambda e: float(np.mean(np.abs(e)))
        rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
        row = {
            "horizon_days": h, "n_test": int(len(te)),
            "mae_model": mae(y - pred), "mae_persistence": mae(y), "mae_mean_rate": mae(y - mean_rate),
            "rmse_model": rmse(y - pred), "rmse_persistence": rmse(y), "rmse_mean_rate": rmse(y - mean_rate),
            "coverage_80": float(np.mean((y >= te["q10"]) & (y <= te["q90"]))),
            "mean_interval_width": float(np.mean(te["q90"] - te["q10"])),
        }
        row["skill_vs_persistence"] = 1 - row["mae_model"] / row["mae_persistence"]
        row["skill_vs_mean_rate"] = 1 - row["mae_model"] / row["mae_mean_rate"]
        rows.append(row)
    return rows


def importance(bundle, df, repeats=5, seed=0):
    """Permutation importance (increase of MAE when one feature is shuffled), 7-day horizon."""
    _, te = split(df, 7)
    X, y = te[FEATURES].to_numpy(), te["y7"].to_numpy()
    base = np.mean(np.abs(y - predict_quantiles(bundle, X, 7)[1]))
    rng = np.random.default_rng(seed)
    out = {}
    for j, name in enumerate(FEATURES):
        incr = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            incr.append(np.mean(np.abs(y - predict_quantiles(bundle, Xp, 7)[1])) - base)
        out[name] = float(np.mean(incr))
    return out


def main():
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    bundle = fit(df)
    joblib.dump(bundle, C.MODEL_FILE)
    metrics = {"forecast": evaluate(bundle, df), "importance": importance(bundle, df),
               "plateau_fitted": bundle["rstar"].round(2).tolist(), "plateau_true": C.RF_MAX.tolist()}
    (C.REPORT_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(pd.DataFrame(metrics["forecast"]).round(3).to_string(index=False))
    print("importance:", {k: round(v, 3) for k, v in metrics["importance"].items()})
    print("fitted plateau R*:", metrics["plateau_fitted"], "\ntrue plateau      :", metrics["plateau_true"])


if __name__ == "__main__":
    main()
