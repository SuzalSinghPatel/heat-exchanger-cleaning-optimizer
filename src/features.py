"""Cleaning, feature engineering and target construction.

State of an exchanger at day t (all computed from past data only):
  rf_state          trailing 5-day mean of the cleaned fouling resistance, within the current run
  days_since_clean  days since the exchanger returned from its last cleaning
  flow_mean         trailing 14-day mean flow
  api_mean          trailing 14-day mean crude API gravity (plant level)
  sulfur_mean       trailing 14-day mean crude sulfur (plant level)
Targets: change of rf_state over the next h days (h = 7, 14, 30), only for windows
that contain no cleaning, so the models learn pure fouling behaviour.
"""
import numpy as np
import pandas as pd

from . import config as C

FEATURES = ["exch_idx", "rf_state", "days_since_clean", "flow_mean", "api_mean", "sulfur_mean"]
HORIZONS = (7, 14, 30)
STATE_WIN, ENV_WIN, CRUDE_WIN, LONG_WIN = 5, 14, 5, 180
REQUIRED_COLUMNS = ["date", "exchanger_id", "status", "flow_m3h", "crude_api", "crude_sulfur_pct",
                    "fouling_resistance", "days_since_clean", "cleaned"]


def load_raw(path=None):
    return pd.read_csv(path or C.DATA_FILE, parse_dates=["date"])


def _hampel(s, k=4.0, win=7):
    """Replace outliers (robust z-score above k) by NaN."""
    med = s.rolling(win, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=3).median()
    scale = np.maximum(1.4826 * mad, 0.01 * med.abs() + 1e-6)
    return s.mask((s - med).abs() > k * scale)


def _clean_col(s):
    return _hampel(s).interpolate(limit=3, limit_area="inside")


def preprocess(raw):
    """Outlier removal, gap filling and run segmentation. Returns a tidy frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = raw.sort_values(["exchanger_id", "date"]).reset_index(drop=True).copy()
    df["exch_idx"] = df["exchanger_id"].map({e: i for i, e in enumerate(C.EX_ID)})
    df = df[df["exch_idx"].notna()].reset_index(drop=True)
    df["exch_idx"] = df["exch_idx"].astype(int)
    df["day"] = (df["date"] - df["date"].min()).dt.days
    df["online"] = df["status"].eq("online")
    df["segment"] = df.groupby("exch_idx")["cleaned"].cumsum()       # a new run starts at each cleaning
    df["rf"] = df["fouling_resistance"]
    df["flow"] = df["flow_m3h"]
    grp = df.groupby(["exch_idx", "segment"])
    for col in ("rf", "flow"):
        df[col] = grp[col].transform(_clean_col)

    crude = raw.groupby("date")[["crude_api", "crude_sulfur_pct"]].median().sort_index()
    crude = crude.interpolate(limit_direction="both")
    roll = crude.rolling(CRUDE_WIN, min_periods=2).mean()
    df["api"] = df["date"].map(crude["crude_api"])
    df["sulfur"] = df["date"].map(crude["crude_sulfur_pct"])
    df["api_mean"] = df["date"].map(roll["crude_api"])
    df["sulfur_mean"] = df["date"].map(roll["crude_sulfur_pct"])

    df["rf_state"] = grp["rf"].transform(lambda s: s.rolling(STATE_WIN, min_periods=1).mean())
    df["flow_mean"] = df.groupby("exch_idx")["flow"].transform(lambda s: s.rolling(ENV_WIN, min_periods=3).mean())
    return df


def add_targets(df, horizons=HORIZONS):
    g = df.groupby("exch_idx")
    for h in horizons:
        fut = g["rf_state"].shift(-h)
        same_run = g["segment"].shift(-h) == df["segment"]
        df[f"y{h}"] = np.where(same_run, fut - df["rf_state"], np.nan)
    return df


def build_dataset(raw=None):
    df = preprocess(load_raw() if raw is None else raw)
    return add_targets(df)


def current_state(df, asof=None):
    """Per-exchanger state vector at the end of the data (or at date `asof`)."""
    d = df if asof is None else df[df["date"] <= pd.Timestamp(asof)]
    keys = ("rf_state", "dsc", "flow_mean", "api_mean", "sulfur_mean", "busy")
    out = {k: np.zeros(C.N_EX) for k in keys}
    last_day = d["day"].max()
    for i in range(C.N_EX):
        e = d[d["exch_idx"] == i].sort_values("day")
        flags = e["online"].to_numpy()[::-1]
        n_off = 0 if flags[0] else int(np.argmax(flags)) if flags.any() else len(flags)
        ok = e[e["online"] & e["rf_state"].notna() & e["flow_mean"].notna() & e["api_mean"].notna()]
        r = ok.iloc[-1]
        out["rf_state"][i] = r["rf_state"]
        out["dsc"][i] = (r["days_since_clean"] if pd.notna(r["days_since_clean"]) else 0) + (last_day - r["day"])
        out["flow_mean"][i] = r["flow_mean"]
        out["api_mean"][i] = r["api_mean"]
        out["sulfur_mean"][i] = r["sulfur_mean"]
        out["busy"][i] = max(C.DOWNTIME[i] - n_off, 0) if n_off > 0 else 0
    out["busy"] = out["busy"].astype(int)
    daily = d.drop_duplicates("date").sort_values("date").tail(LONG_WIN)     # long-run crude slate
    out["api_long"] = np.full(C.N_EX, daily["api"].mean())
    out["sulfur_long"] = np.full(C.N_EX, daily["sulfur"].mean())
    out["asof"] = d["date"].max()
    return out
