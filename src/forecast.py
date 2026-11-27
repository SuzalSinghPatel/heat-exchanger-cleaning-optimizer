"""Physics-informed fouling forecasts using Random Forest Regression.

The physical component is the Kern-Seaton asymptotic law, fitted per exchanger:
    dRf / dt = beta * (R* - Rf)
so the expected h-day change is:
    p_h = (R* - Rf) * (1 - exp(-beta h))

A RandomForestRegressor learns a multiplier on this physical change from crude
slate, flow, run age and exchanger identity. The prediction is therefore
p_h * predicted_multiplier. This retains the physical plateau behaviour while
using a standard regression algorithm.

Uncertainty is estimated with split-conformal calibration: a held-out slice of
the training period supplies an absolute-residual margin around the RF point
prediction. The resulting interval is symmetric; it is not quantile regression.
The chronological test split remains years 1-2 for training and year 3 for test.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config as C
from .features import FEATURES, HORIZONS, build_dataset

CAL_DAYS = 120
MIN_P = 0.03
MODEL_TYPE = "random_forest_regression"
MODEL_VERSION = 2
COVERAGE = 0.80


def split(df, h):
    ok = df["online"] & df[FEATURES].notna().all(axis=1) & df[f"y{h}"].notna()
    train = df[ok & (df["day"] + h < C.TRAIN_END_DAY)]
    test = df[ok & (df["day"] >= C.TRAIN_END_DAY)]
    return train, test


def _kern_seaton(tr):
    """Per-exchanger fit of rate = alpha - beta * Rf; plateau R* = alpha / beta."""
    rstar, beta = np.zeros(C.N_EX), np.zeros(C.N_EX)
    for i in range(C.N_EX):
        g = tr[tr["exch_idx"] == i]
        slope, icpt = np.polyfit(g["rf_state"].to_numpy(), g["y7"].to_numpy() / 7.0, 1)
        alpha, b = max(icpt, 1e-3), max(-slope, 1e-4)
        rstar[i] = np.clip(alpha / b, g["rf_state"].max() * 1.05, 15.0)
        beta[i] = alpha / rstar[i]
    return rstar, beta


def _physics(bundle, X, h):
    """Physical h-day change and transformed feature matrix for the RF."""
    idx = X[:, 0].astype(int)
    rf = X[:, 1]
    p = np.maximum(bundle["rstar"][idx] - rf, 0.0) * (
        1.0 - np.exp(-bundle["beta"][idx] * h)
    )
    Z = X.copy()
    Z[:, 1] = np.clip(rf / bundle["rstar"][idx], 0.0, 1.3)
    return p, Z


def _model():
    return RandomForestRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=30,
        bootstrap=True,
        n_jobs=-1,
        random_state=0,
    )


def _fit_ratio(bundle, tr, h):
    p, Z = _physics(bundle, tr[FEATURES].to_numpy(), h)
    keep = p > MIN_P
    if not np.any(keep):
        raise ValueError(f"No usable training rows for the {h}-day horizon.")
    target = tr[f"y{h}"].to_numpy()[keep] / p[keep]
    return _model().fit(Z[keep], target, sample_weight=p[keep])


def _predict_ratio(model, bundle, X, h):
    p, Z = _physics(bundle, np.asarray(X, float), h)
    return model.predict(Z) * p


def _conformal_margin(bundle, tr, h):
    """Calibrate a symmetric 80% interval using held-out absolute residuals."""
    cut = C.TRAIN_END_DAY - CAL_DAYS
    core = tr[tr["day"] + h < cut]
    cal = tr[tr["day"] >= cut]
    if core.empty or cal.empty:
        raise ValueError(f"Insufficient rows for conformal calibration at {h} days.")
    model = _fit_ratio(bundle, core, h)
    Xc = cal[FEATURES].to_numpy()
    yc = cal[f"y{h}"].to_numpy()
    scores = np.abs(yc - _predict_ratio(model, bundle, Xc, h))
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * COVERAGE)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def fit(df):
    tr7, _ = split(df, 7)
    rstar, beta = _kern_seaton(tr7)
    bundle = {
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "rstar": rstar,
        "beta": beta,
        "models": {},
        "mean_delta": {},
        "margin": {},
        "features": FEATURES,
        "interval_method": "split_conformal_absolute_residual",
        "nominal_coverage": COVERAGE,
    }
    for h in HORIZONS:
        tr, _ = split(df, h)
        bundle["models"][h] = _fit_ratio(bundle, tr, h)
        bundle["mean_delta"][h] = tr.groupby("exch_idx")[f"y{h}"].mean().to_dict()
        bundle["margin"][h] = _conformal_margin(bundle, tr, h)
    return bundle


def predict_quantiles(bundle, X, h):
    """Return (lower, point, upper) predictions for the h-day fouling change.

    The legacy function name is retained for compatibility with the scheduler.
    These are conformal prediction bounds, not learned conditional quantiles.
    """
    X = np.asarray(X, float)
    point = _predict_ratio(bundle["models"][h], bundle, X, h)
    margin = float(bundle["margin"][h])
    return point - margin, point, point + margin


def predict_frame(bundle, df, h):
    """Test-year rows with predicted bounds and point forecasts."""
    _, te = split(df, h)
    te = te.copy()
    te["q10"], te["q50"], te["q90"] = predict_quantiles(
        bundle, te[FEATURES].to_numpy(), h
    )
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
            "horizon_days": h,
            "n_test": int(len(te)),
            "mae_model": mae(y - pred),
            "mae_persistence": mae(y),
            "mae_mean_rate": mae(y - mean_rate),
            "rmse_model": rmse(y - pred),
            "rmse_persistence": rmse(y),
            "rmse_mean_rate": rmse(y - mean_rate),
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
            incr.append(
                np.mean(np.abs(y - predict_quantiles(bundle, Xp, 7)[1])) - base
            )
        out[name] = float(np.mean(incr))
    return out


def main():
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    bundle = fit(df)
    joblib.dump(bundle, C.MODEL_FILE)
    metrics = {
        "forecast": evaluate(bundle, df),
        "importance": importance(bundle, df),
        "plateau_fitted": bundle["rstar"].round(2).tolist(),
        "plateau_true": C.RF_MAX.tolist(),
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "interval_method": bundle["interval_method"],
    }
    (C.REPORT_DIR / "forecast_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(pd.DataFrame(metrics["forecast"]).round(3).to_string(index=False))
    print("importance:", {k: round(v, 3) for k, v in metrics["importance"].items()})
    print(
        "fitted plateau R*:",
        metrics["plateau_fitted"],
        "\ntrue plateau      :",
        metrics["plateau_true"],
    )


if __name__ == "__main__":
    main()
