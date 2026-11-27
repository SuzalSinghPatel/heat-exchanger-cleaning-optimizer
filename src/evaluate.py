"""Summarise all results into reports/results.json and draw the report figures."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, build_dataset
from .forecast import predict_frame

INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"        # categorical slots 1-3, fixed order
LABELS = {"exch_idx": "Exchanger identity", "rf_state": "Current fouling level",
          "days_since_clean": "Days since cleaning", "flow_mean": "Flow (14-day mean)",
          "api_mean": "Crude API gravity", "sulfur_mean": "Crude sulfur"}
ML = "ML optimiser"
OPERATOR = "Operator (status quo)"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def summarise(forecast_metrics, bt):
    test, val = bt[bt["split"] == "test"], bt[bt["split"] == "validation"]
    per = {}
    for name, g in test.groupby("policy"):
        per[name] = {k: float(g[k].mean()) for k in ("total", "fouling", "downtime", "cleaning", "cleanings_per_year")}
        per[name]["total_std"] = float(g["total"].std(ddof=1))
    tuned = {fam: val[val["family"] == fam].groupby("policy")["total"].mean().idxmin() for fam in ("fixed", "threshold")}
    hindsight = {fam: test[test["family"] == fam].groupby("policy")["total"].mean().idxmin() for fam in ("fixed", "threshold")}
    ml = test[test["policy"] == ML].set_index("scenario")["total"]

    def compare(other):
        o = test[test["policy"] == other].set_index("scenario")["total"]
        saving = 1 - ml / o
        return {"other": other, "mean_saving_pct": float(100 * saving.mean()), "min_saving_pct": float(100 * saving.min()),
                "max_saving_pct": float(100 * saving.max()), "scenarios_won": int((saving > 0).sum()),
                "n_scenarios": int(len(saving)), "mean_saving_usd_per_year": float((o - ml).mean())}
    return {"per_year_usd": per, "tuned_on_validation": tuned, "best_in_hindsight": hindsight,
            "comparisons": {"vs_operator": compare(OPERATOR), "vs_never_clean": compare("Never clean"),
                            "vs_tuned_fixed": compare(tuned["fixed"]), "vs_tuned_threshold": compare(tuned["threshold"]),
                            "vs_hindsight_fixed": compare(hindsight["fixed"]),
                            "vs_hindsight_threshold": compare(hindsight["threshold"])},
            "n_test_scenarios": int(ml.shape[0]), "n_validation_scenarios": int(val["scenario"].nunique()),
            "years_per_scenario": C.N_BT_DAYS / 365.0}


# ----------------------------------------------------------------------------- figures
def fig_data(df):
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        e = df[df["exch_idx"] == i]
        ax.plot(e["date"], e["rf"], color=BLUE, lw=1.2)
        for d in e.loc[e["cleaned"] == 1, "date"]:
            ax.axvline(d, color=INK2, lw=0.8, alpha=0.6)
        ax.axvline(df["date"].min() + pd.Timedelta(days=C.TRAIN_END_DAY), color=INK, lw=1, ls=":")
        ax.set_title(C.EX_ID[i], loc="left", fontsize=9)
        ax.tick_params(axis="x", labelrotation=0, labelsize=7)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    fig.supylabel("Fouling resistance Rf (x1e-4 m2K/W)", fontsize=9, color=INK2)
    fig.suptitle("Synthetic fouling history: grey lines = cleanings, dotted line = start of the test year",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_fouling_history.png")


def fig_forecast(bundle, df):
    e_idx = 2                                                    # E-103: fast fouler, several cleanings
    te = predict_frame(bundle, df, 14)
    s = te[te["exch_idx"] == e_idx].set_index("date")
    full = pd.date_range(s.index.min(), s.index.max())
    s = s.reindex(full)                                          # NaN gaps at cleanings -> broken line
    actual, pred = s["rf_state"] + s["y14"], s["rf_state"] + s["q50"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = s.index + pd.Timedelta(days=14)
    a1.fill_between(x, s["rf_state"] + s["q10"], s["rf_state"] + s["q90"], color=BLUE, alpha=0.18, lw=0, label="80 % interval")
    a1.plot(x, actual, color=INK2, lw=1.4, label="Actual")
    a1.plot(x, pred, color=BLUE, lw=1.6, label="Forecast (median)")
    a1.set_title(f"{C.EX_ID[e_idx]}: 14-day-ahead fouling forecast, held-out year")
    a1.set_ylabel("Rf (x1e-4 m2K/W)")
    a1.legend(loc="upper left", ncol=3, fontsize=8)
    t30 = predict_frame(bundle, df, 30)
    a2.scatter(t30["y30"], t30["q50"], s=6, color=BLUE, alpha=0.35, lw=0)
    lim = [min(t30["y30"].min(), t30["q50"].min()), max(t30["y30"].max(), t30["q50"].max())]
    a2.plot(lim, lim, color=INK2, lw=1, ls="--")
    a2.set_title("30-day change: forecast vs actual (all exchangers)")
    a2.set_xlabel("Actual change in Rf")
    a2.set_ylabel("Forecast change in Rf")
    fig.tight_layout()
    _save(fig, "02_forecast_quality.png")


def fig_importance(metrics):
    imp = pd.Series(metrics["importance"]).rename(LABELS).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.5)
    ax.set_xlabel("Increase of 7-day forecast error when the feature is shuffled (MAE)")
    ax.set_title("What drives the fouling forecast")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "03_feature_importance.png")


def fig_costs(summary):
    per, tuned = summary["per_year_usd"], summary["tuned_on_validation"]
    rows = [(OPERATOR, "Operator\n(status quo)"),
            (tuned["threshold"], f"{tuned['threshold']}\n(tuned on validation)"),
            (tuned["fixed"], f"{tuned['fixed']}\n(tuned on validation)"),
            (ML, "ML optimiser\n(no tuning)")]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    comps = [("fouling", "Extra fuel from fouling", BLUE), ("downtime", "Fuel lost while exchanger is out", ORANGE),
             ("cleaning", "Cleaning cost", AQUA)]
    for y, (key, label) in enumerate(rows[::-1]):
        left = 0
        for comp, _, colour in comps:
            v = per[key][comp] / 1e6
            ax.barh(y, v, left=left, color=colour, height=0.5, edgecolor="white", linewidth=1.5)
            left += v
        ax.text(left + 0.06, y, f"${left:.2f} M", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows[::-1]])
    ax.set_xlim(0, 6.9)
    ax.set_xlabel("Total cost, USD million per year (mean of test scenarios)")
    ax.set_title("Annual cost of each cleaning strategy")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in comps]
    ax.legend(handles, [l for _, l, _ in comps], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, "04_cost_comparison.png")


def fig_gantt(traces):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    for ax, name in zip(axes, (OPERATOR, ML)):
        ev = traces[name]["events"]
        for t, i in ev:
            ax.barh(i, C.DOWNTIME[i], left=t, color=BLUE, height=0.55, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(C.N_EX))
        ax.set_yticklabels(C.EX_ID, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{name}: {len(ev) / (C.N_BT_DAYS / 365):.1f} cleanings per year")
        ax.grid(axis="y", visible=False)
    axes[1].set_xticks([0, 365, 730, 1095])
    axes[1].set_xticklabels(["Start", "Year 1", "Year 2", "Year 3"])
    fig.suptitle("Cleaning calendars in one simulated test scenario (bars = exchanger out of service)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_cleaning_calendars.png")


def fig_sensitivity(bt):
    test = bt[bt["split"] == "test"]
    ml = test[test["policy"] == ML]["total"].mean() / 1e6
    op = test[test["policy"] == OPERATOR]["total"].mean() / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, fam, xl in ((axes[0], "fixed", "Fixed interval between cleanings (days)"),
                        (axes[1], "threshold", "Clean when Rf reaches (x1e-4 m2K/W)")):
        g = test[test["family"] == fam].groupby("param")["total"].mean() / 1e6
        ax.plot(g.index.astype(float), g.values, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5,
                label="Simple rule (by its setting)")
        ax.axhline(ml, color=INK, lw=1.4, ls="--", label="ML optimiser (no setting)")
        ax.axhline(op, color=INK2, lw=1.4, ls=":", label="Operator (status quo)")
        ax.set_xlabel(xl)
    for y, text, colour in ((ml, "ML optimiser", INK), (op, "Operator\n(status quo)", INK2)):   # direct labels, right margin
        axes[1].annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=8, color=colour, annotation_clip=False)
    axes[0].set_ylabel("Cost, USD million per year")
    axes[0].set_title("Fixed-interval rule")
    axes[1].set_title("Threshold rule")
    axes[0].set_ylim(4.75, 6.7)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("A simple rule works only if its setting is right; the ML optimiser needs no tuning",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_rule_sensitivity.png")


def main():
    _style()
    fm = json.loads((C.REPORT_DIR / "forecast_metrics.json").read_text())
    bt = pd.read_csv(C.BACKTEST_FILE)
    traces = joblib.load(C.TRACES_FILE)
    bundle = joblib.load(C.MODEL_FILE)
    df = build_dataset()
    raw_log = pd.read_csv(C.LOG_FILE)
    summary = summarise(fm, bt)
    results = {"dataset": {"rows": int(len(df)), "exchangers": C.N_EX, "days": C.N_DAYS,
                           "cleaning_events_history": int(len(raw_log)), "train_days": C.TRAIN_END_DAY},
               "forecast": fm["forecast"], "importance": fm["importance"],
               "plateau_fitted": fm["plateau_fitted"], "plateau_true": fm["plateau_true"],
               "backtest": summary}
    C.RESULTS_FILE.write_text(json.dumps(results, indent=2))
    fig_data(df)
    fig_forecast(bundle, df)
    fig_importance(fm)
    fig_costs(summary)
    fig_gantt(traces)
    fig_sensitivity(bt)
    s = summary
    print("tuned on validation:", s["tuned_on_validation"], "| best in hindsight:", s["best_in_hindsight"])
    for k, v in s["comparisons"].items():
        print(f"ML vs {v['other']:<24} saving {v['mean_saving_pct']:+5.1f} % (range {v['min_saving_pct']:+.1f} .. {v['max_saving_pct']:+.1f}), "
              f"wins {v['scenarios_won']}/{v['n_scenarios']}, ${v['mean_saving_usd_per_year'] / 1e3:+.0f}k/yr")


if __name__ == "__main__":
    main()
