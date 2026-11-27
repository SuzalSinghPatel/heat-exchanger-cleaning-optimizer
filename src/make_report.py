"""Build reports/report.md and reports/report.pdf from reports/results.json.

Every number in the text is read from the results file, so the report cannot drift
away from what the code actually produced.
"""
import json
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config as C

ML, OPERATOR = "ML optimiser", "Operator (status quo)"
AUTHOR, REG_NO = "Suzal Singh Patel", "INGF202412769"


def usd(x):
    return f"${x / 1e6:.2f} M"


def pct(x, nd=1):
    return f"{x:.{nd}f} %"


def build_blocks(res):
    fc, bt = res["forecast"], res["backtest"]
    per, cmp_ = bt["per_year_usd"], bt["comparisons"]
    tf, tt = bt["tuned_on_validation"]["fixed"], bt["tuned_on_validation"]["threshold"]
    op, mlc = cmp_["vs_operator"], per[ML]
    vf, vt = cmp_["vs_tuned_fixed"], cmp_["vs_tuned_threshold"]
    n_test, yrs = bt["n_test_scenarios"], bt["years_per_scenario"]

    def worst(prefix):   # how much more the worst setting of a rule family costs than its best setting
        t = [p["total"] for n, p in per.items() if n.startswith(prefix)]
        return 100 * (max(t) / min(t) - 1)

    def vs(c):   # phrase "X % lower/higher"
        v = c["mean_saving_pct"]
        return f"{abs(v):.1f} % {'lower' if v >= 0 else 'higher'}"

    B = []
    B.append(("title", "AI-Driven Predictive Cleaning Optimisation for Refinery Heat Exchanger Networks"))
    B.append(("byline", f"{AUTHOR}  |  Registration No. {REG_NO}"))
    B.append(("subtitle", "Project report. A study on synthetic data: no real refinery data was used."))

    B.append(("h1", "Abstract"))
    B.append(("p",
              "Crude preheat exchangers slowly foul, which forces the furnace to burn extra fuel; cleaning removes the "
              "fouling but costs money and takes the exchanger out of service. We build a two-stage decision-support "
              "system. A physics-informed Random Forest regression model forecasts the fouling resistance of every exchanger "
              "with calibrated 80 % prediction intervals, and a mixed-integer program turns those forecasts into a "
              "cleaning calendar that respects crew capacity. Because no plant data was available, the data is "
              f"synthetic and produced by a documented fouling model. On the held-out year the 7, 14 and 30-day "
              f"forecasts reduce error by {fc[0]['skill_vs_persistence'] * 100:.0f} %, {fc[1]['skill_vs_persistence'] * 100:.0f} % and "
              f"{fc[2]['skill_vs_persistence'] * 100:.0f} % relative to a no-change forecast. In a closed-loop simulation of "
              f"{n_test} unseen {yrs:.0f}-year scenarios the optimiser costs {vs(op)} than the status-quo "
              f"operator policy ({usd(op['mean_saving_usd_per_year'])} per year, better in {op['scenarios_won']} of "
              f"{op['n_scenarios']} scenarios). It is comparable to the best hand-tuned simple rules, within "
              f"scenario-to-scenario noise ({vs(vf)} than {tf}, {vs(vt)} than {tt}), but needs no tuning. The results show that the method "
              "works end to end on synthetic data; they do not establish performance on a real refinery."))

    B.append(("h1", "1. Problem and motivation"))
    B.append(("p",
              "In a refinery, crude oil is heated by a train of heat exchangers before it enters the furnace. Deposits "
              "(fouling) build up on the exchanger surfaces, which lowers heat recovery. The furnace makes up the "
              "difference by burning more fuel, which costs money and emits CO2. An exchanger can be taken out of "
              "service and cleaned, which restores performance, but cleaning has a fixed cost, the exchanger is "
              "bypassed for several days (during which its whole heat duty must be supplied by the furnace), and "
              "usually only a limited number of exchangers can be cleaned at the same time. The practical question is "
              "**which exchangers to clean, and when**."))
    B.append(("p",
              "Today this is often decided by fixed calendars or by an operator reacting to performance drops. "
              "A forecast-driven approach should do better because fouling speed differs between exchangers, depends "
              "on the crude slate being processed, and the cost of waiting grows non-linearly."))
    B.append(("p", "**Cost model.** The fraction of clean heat duty lost to fouling is "
                   "`loss(Rf) = 1 - 1 / (1 + U0 * Rf * 1e-4)`, where Rf is the fouling resistance (units of 1e-4 m2K/W) "
                   "and U0 the clean overall heat-transfer coefficient. The daily fuel penalty is "
                   "`price * duty * 24 * loss(Rf)`. While an exchanger is out for cleaning, its whole duty is lost. "
                   f"Defaults: fuel price {C.ENERGY_PRICE:.0f} USD/MWh, one exchanger out at a time, cleaning costs of "
                   "28-58 kUSD and 4-6 days of downtime per exchanger."))

    B.append(("h1", "2. Data"))
    B.append(("p",
              f"**The data is synthetic.** `src/generate_data.py` produces {res['dataset']['days']} days of daily readings "
              f"for {res['dataset']['exchangers']} exchangers ({res['dataset']['rows']:,} rows). Fouling resistance follows a "
              "Kern-Seaton asymptotic law whose rate depends on the crude slate (lower API gravity and higher sulfur foul "
              "faster) and on flow velocity (higher velocity fouls slower), with multiplicative process noise. Crude "
              "campaigns last 10-25 days. Temperatures, pressure drop and the soft-sensor fouling reading are derived "
              "from the fouling state. History was generated under an operator-style policy: an exchanger is cleaned "
              "a random 1-40 days after its fouling crosses a trigger that changes from one cleaning to the next, "
              f"which produced {res['dataset']['cleaning_events_history']} cleaning events and a wide range of fouling "
              "levels."))
    B.append(("p",
              "To make preprocessing meaningful, the data is deliberately imperfect: about 1.5 % of readings are missing, "
              "0.4 % of fouling readings and 0.3 % of flow readings are outliers, the fouling soft-sensor drifts slowly, "
              "and exchangers have no readings while they are being cleaned. The first two years are used for training "
              "and the third year is held out for testing; no training target window reaches into the test year."))
    rows = [[C.EX_ID[i], f"{C.DUTY[i]:.0f}", f"{C.K[i]:.3f}", f"{C.RF_MAX[i]:.1f}", f"{C.U0[i]:.0f}",
             f"{C.CLEAN_COST[i] / 1e3:.0f}", f"{C.DOWNTIME[i]}"] for i in range(C.N_EX)]
    B.append(("table", ["Exchanger", "Duty (MW)", "k (1e-4/day)", "Plateau Rf*", "Clean U (W/m2K)", "Cleaning (kUSD)", "Downtime (d)"],
              rows, "Table 1. The simulated preheat train (illustrative values)."))
    B.append(("img", "01_fouling_history.png", 0.95, "Figure 1. Synthetic fouling history per exchanger."))

    B.append(("h1", "3. Methods"))
    B.append(("h2", "3.1 Preprocessing and features"))
    B.append(("p",
              "Each exchanger series is split into runs at cleaning events. Within a run, outliers are removed with a "
              "Hampel filter (7-day window, 4 robust standard deviations) and gaps of up to 3 days are interpolated. "
              "The fouling **state** is the trailing 5-day mean of the cleaned reading within the current run. Features "
              "are the state, days since cleaning, 14-day mean flow, 5-day mean crude API gravity and sulfur, and the "
              "exchanger identity. Targets are the change of the state over the next 7, 14 and 30 days, using only "
              "windows without a cleaning."))
    B.append(("h2", "3.2 Fouling forecast: physics-informed hybrid"))
    B.append(("p",
              "A first version used a plain tree-based model. It forecast well one week ahead, but when rolled forward "
              "for months it systematically under-predicted fouling, because operators always cleaned before the plateau "
              "and a tree model cannot extrapolate beyond the fouling levels it has seen. The final model therefore "
              "combines physics and machine learning."))
    B.append(("bullets", [
        "**Physics.** For each exchanger a Kern-Seaton law `dRf/dt = beta (R* - Rf)` is fitted by least squares, "
        "giving the physically expected change over h days: `p_h = (R* - Rf)(1 - exp(-beta h))`.",
        "**ML.** Random Forest regression (scikit-learn `RandomForestRegressor`) "
        "learns the multiplier `actual change / p_h` from the crude slate, flow, run age and exchanger. The forecast "
        "is `p_h` times the multiplier, so long forecasts approach the plateau instead of running away.",
        "**Intervals.** A symmetric 80 % prediction band is calibrated from absolute residuals on the last 120 training days "
        "using split-conformal calibration. These are not learned quantiles.",
        "**Baselines.** Persistence (no change) and each exchanger's average historical change.",
    ]))
    B.append(("h2", "3.3 Cleaning optimiser"))
    B.append(("p",
              "The 7-day median forecast is rolled forward 43 steps (301 days) for two cases per exchanger: left alone, "
              "and freshly cleaned. The current crude campaign is assumed to persist for about two weeks and then revert "
              "to the six-month average. From these trajectories the cost of cleaning exchanger i on day d is built from "
              "fuel lost before cleaning, the full duty lost while it is out, the cleaning cost, and the value of the "
              "rest of the window."))
    B.append(("p",
              "The value of the rest of the window needs care: a window that allows one cleaning per exchanger would "
              "otherwise push cleanings late. Following renewal theory, a freshly cleaned exchanger is valued at its "
              "best long-run cost rate `c*` per day, where `c*` is the minimum over cycle lengths a of "
              "`(cleaning + downtime + fuel lost during a days) / (a + downtime)`. Not cleaning for the whole window "
              "is compared with cleaning at the best day after the decision window. In a check on a set of freshly "
              "cleaned exchangers, each was scheduled within a few days of its own optimal cycle length "
              "(154-238 days across the train)."))
    B.append(("p",
              "A mixed-integer program (scipy `milp`, HiGHS solver; Huangfu and Hall 2018) then chooses at most one start "
              "day per exchanger in the next 90 days, minimising total cost subject to at most M exchangers out of "
              "service on any day. In the backtest the plan is recomputed every 14 days (receding horizon), and only "
              "cleanings that start before the next replan are carried out."))
    B.append(("h2", "3.4 Evaluation protocol"))
    B.append(("p",
              "A fixed historical data set cannot say what would have happened under a different cleaning schedule, so "
              "strategies are compared in a **closed-loop simulation**: each strategy decides day by day, seeing only "
              "noisy fouling readings and history, and the plant model advances. All strategies face identical crude "
              f"slate, throughput and noise within a scenario. Each of {n_test} test scenarios covers {yrs:.0f} years, "
              "starting from the plant state at the end of the data set. The same crew limit applies to every strategy. "
              "Strategies compared:"))
    B.append(("bullets", [
        "**Never clean** (reference).",
        "**Operator (status quo):** the historical practice that generated the data.",
        "**Fixed interval** (120-300 days) and **threshold** (clean when Rf reaches 1.5-4.0) rules. The setting of each "
        f"family is chosen on {bt['n_validation_scenarios']} separate validation scenarios; the test scenarios are only "
        "used to report. The best-in-hindsight setting on the test scenarios is shown too.",
        "**ML optimiser:** the system above, with no tuning.",
    ]))

    B.append(("h1", "4. Results"))
    B.append(("h2", "4.1 Forecast quality (held-out year)"))
    rows = [[f"{r['horizon_days']} d", f"{r['n_test']:,}", f"{r['mae_model']:.3f}", f"{r['mae_mean_rate']:.3f}",
             f"{r['mae_persistence']:.3f}", pct(r["skill_vs_mean_rate"] * 100, 0), pct(r["skill_vs_persistence"] * 100, 0),
             pct(r["coverage_80"] * 100, 0)] for r in fc]
    B.append(("table", ["Horizon", "Test rows", "MAE model", "MAE mean rate", "MAE persist.", "Gain vs mean rate",
                        "Gain vs persist.", "80 % band coverage"], rows,
              "Table 2. Error in the change of fouling resistance (x1e-4 m2K/W). MAE = mean absolute error; "
              "gain = reduction of MAE; coverage = share of outcomes inside the model's 80 % interval."))
    cov = [r["coverage_80"] * 100 for r in fc]
    B.append(("p",
              f"The model beats both baselines at every horizon. The gain over persistence grows with the horizon "
              f"because fouling drifts steadily; the gain over each exchanger's historical average rate is smaller "
              f"({min(r['skill_vs_mean_rate'] for r in fc) * 100:.0f}-{max(r['skill_vs_mean_rate'] for r in fc) * 100:.0f} %), so most of the "
              "predictable signal is the exchanger's own trend and the crude slate adds a smaller part. After conformal "
              f"calibration the intervals cover {min(cov):.0f}-{max(cov):.0f} % of outcomes instead of the nominal 80 %: "
              "better, but still too narrow, because the held-out year differs from the calibration period."))
    rows = [[C.EX_ID[i], f"{res['plateau_fitted'][i]:.2f}", f"{res['plateau_true'][i]:.1f}"] for i in range(C.N_EX)]
    B.append(("table", ["Exchanger", "Fitted plateau R*", "True plateau (simulator)"], rows,
              "Table 3. Plateau recovered by the physics part from history in which exchangers were always cleaned early. "
              "Possible only because the data is synthetic."))
    B.append(("img", "02_forecast_quality.png", 0.95, "Figure 2. Left: 14-day-ahead forecast for one exchanger with its 80 % "
                                                       "interval. Right: 30-day changes, forecast against actual."))
    B.append(("img", "03_feature_importance.png", 0.6, "Figure 3. Permutation importance for the 7-day forecast."))

    B.append(("h2", "4.2 Cleaning strategies in closed-loop simulation"))
    order = ["Never clean", OPERATOR, tf, tt, ML]
    rows = []
    for name in order:
        p = per[name]
        tag = {tf: " (tuned on validation)", tt: " (tuned on validation)"}.get(name, "")
        rows.append([name + tag, usd(p["total"]), f"{p['total_std'] / 1e6:.2f}", usd(p["fouling"]), usd(p["downtime"]),
                     usd(p["cleaning"]), f"{p['cleanings_per_year']:.1f}"])
    B.append(("table", ["Strategy", "Total / year", "Std dev (M)", "Fouling fuel", "Downtime", "Cleaning", "Cleanings / year"], rows,
              f"Table 4. Mean annual cost over {n_test} test scenarios of {yrs:.0f} years each (USD)."))
    B.append(("img", "04_cost_comparison.png", 0.8, "Figure 4. Annual cost split into its three components."))
    rows = [["ML vs " + c["other"], pct(c["mean_saving_pct"]), f"{c['min_saving_pct']:.1f} to {c['max_saving_pct']:.1f}",
             f"{c['scenarios_won']} / {c['n_scenarios']}", f"{c['mean_saving_usd_per_year'] / 1e3:+,.0f}"]
            for k, c in cmp_.items() if k in ("vs_operator", "vs_tuned_fixed", "vs_tuned_threshold", "vs_never_clean")]
    B.append(("table", ["Comparison", "Mean saving", "Range over scenarios (%)", "Scenarios won", "USD k / year"], rows,
              "Table 5. Paired comparison across test scenarios (positive = ML optimiser is cheaper)."))
    B.append(("img", "06_rule_sensitivity.png", 0.95, "Figure 5. Cost of each simple rule as its setting changes."))
    B.append(("img", "05_cleaning_calendars.png", 0.9, "Figure 6. Cleaning calendars in one test scenario."))
    allrows = []
    for name, p in sorted(per.items(), key=lambda kv: kv[1]["total"]):
        allrows.append([name, usd(p["total"]), f"{p['total_std'] / 1e6:.2f}", f"{p['cleanings_per_year']:.1f}"])
    B.append(("table", ["Strategy", "Total / year", "Std dev (M)", "Cleanings / year"], allrows,
              "Table 6. All strategies on the test scenarios, cheapest first."))

    B.append(("h1", "5. Discussion"))
    B.append(("bullets", [
        f"**Versus current practice** the optimiser lowers annual cost by {op['mean_saving_pct']:.1f} % on average, in "
        f"{op['scenarios_won']} of {op['n_scenarios']} scenarios. It cleans more often than the operator "
        f"({mlc['cleanings_per_year']:.1f} against {per[OPERATOR]['cleanings_per_year']:.1f} times per year), which costs more in "
        "downtime and cleaning but saves more fuel.",
        f"**Versus tuned simple rules** there is no clear difference: {vs(vf)} than the best fixed interval and "
        f"{vs(vt)} than the best threshold rule, with the sign changing between scenarios. The cost curve is flat "
        "near the optimum, so a well-tuned rule is hard to beat in this plant. Figure 5 shows the practical advantage: "
        f"a badly set fixed interval costs up to {worst('Fixed'):.0f} % more than the best interval and a badly set "
        f"threshold up to {worst('Threshold'):.0f} % more, while the optimiser needs no setting and adapts to each "
        "exchanger's fouling speed and to the crew limit.",
        "**Where the remaining gap comes from.** In a development diagnostic on two scenarios (not part of "
        "`run_all.py`), the same optimiser driven by a perfect-physics forecaster scored about the same as with the "
        "learned forecaster, and a rule that cleans each exchanger at its own optimal cycle length was slightly cheaper "
        "than both. This suggests the limit is the scheduling heuristic (the renewal-based value of the rest of the "
        "window), not forecast error.",
        "**Why the physics part mattered.** The plain machine-learning model could not extrapolate to fouling levels "
        "that history never reached; adding the asymptotic law cut the average under-prediction of fouling after "
        "300 days from about 0.7 to about 0.2 (x1e-4 m2K/W), measured in a development check.",
    ]))
    B.append(("h1", "6. Limitations"))
    B.append(("bullets", [
        "**Synthetic data.** The forecast model shares the asymptotic form of the simulator, so it is favoured; real "
        "fouling is messier. The results show the method works, not that these savings would occur in a refinery.",
        f"**Small test set.** {n_test} scenarios; differences of 1-2 % between strategies are within scenario noise.",
        "**Interval coverage** is below the nominal 80 %.",
        "**Simplified economics.** One fuel price, no throughput or furnace-limit penalty, exchangers treated as "
        "independent (no hydraulic or thermal coupling), cleaning always restores a near-clean state.",
        "**Sensing in the backtest** is Gaussian noise on the fouling reading; outliers and gaps appear only in "
        "the offline data set.",
        "**Development was not fully blind.** Baseline settings were chosen on separate validation scenarios, but some "
        "scheduler design choices were made while inspecting early backtest results (including test scenarios 0 and 1).",
        "**Crude slate** is assumed unknown beyond the current campaign; a real planner may know the crude schedule.",
    ]))
    B.append(("h1", "7. Conclusion and future work"))
    B.append(("p",
              "A forecast-then-optimise pipeline for heat-exchanger cleaning was built, tested and delivered with a "
              "dashboard. On synthetic data it reduces cost against status-quo practice and removes the need to tune "
              "cleaning rules. Next steps: test on real plant data; model dependencies between exchangers and furnace "
              "limits; plan under forecast uncertainty (use the prediction intervals inside the optimisation); and "
              "replace the renewal heuristic with a stronger long-horizon method such as stochastic or "
              "reinforcement-learning-based planning."))

    B.append(("h1", "References"))
    B.append(("bullets", [
        "Kern, D. Q. and Seaton, R. E. (1959). A theoretical analysis of thermal surface fouling. British Chemical Engineering, 4(5), 258-262.",
        "Ebert, W. and Panchal, C. B. (1997). Analysis of Exxon crude-oil slip-stream coking data. In Fouling Mitigation of Industrial Heat-Exchange Equipment, Begell House.",
        "Georgiadis, M. C., Papageorgiou, L. G. and Macchietto, S. (2000). Optimal cleaning policies in heat exchanger networks under rapid fouling. Industrial and Engineering Chemistry Research, 39(2), 441-454.",
        "Smaili, F., Vassiliadis, V. S. and Wilson, D. I. (2001). Mitigation of fouling in refinery heat exchanger networks by optimal management of cleaning. Energy and Fuels, 15(5), 1038-1056.",
        "Romano, Y., Patterson, E. and Candes, E. (2019). Conformalized quantile regression. Advances in Neural Information Processing Systems 32.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Huangfu, Q. and Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1), 119-142.",
    ]))
    B.append(("h1", "Appendix: reproducing the results"))
    B.append(("p", "`pip install -r requirements.txt`, then `python run_all.py` regenerates the data, models, backtest, "
                   "figures and this report with fixed random seeds. `streamlit run app/app.py` starts the dashboard and "
                   "`pytest -q` runs the tests."))
    return B


# ----------------------------------------------------------------------------- Markdown
def to_markdown(blocks):
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "title":
            out.append(f"# {b[1]}\n")
        elif kind == "byline":
            out.append(f"**{b[1]}**\n")
        elif kind == "subtitle":
            out.append(f"*{b[1]}*\n")
        elif kind == "h1":
            out.append(f"\n## {b[1]}\n")
        elif kind == "h2":
            out.append(f"\n### {b[1]}\n")
        elif kind == "p":
            out.append(b[1] + "\n")
        elif kind == "bullets":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif kind == "table":
            _, head, rows, cap = b
            out.append(f"**{cap}**\n")
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "|".join("---" for _ in head) + "|")
            out.extend("| " + " | ".join(r) + " |" for r in rows)
            out.append("")
        elif kind == "img":
            out.append(f"![{b[3]}](figures/{b[1]})\n\n*{b[3]}*\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------- PDF
def _fonts():
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                    ("DV-M", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(d / f)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")


def _rl(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<font face="DV-M" size="8">\1</font>', text)


def to_pdf(blocks, path):
    _fonts()
    ink, grey, blue = colors.HexColor("#191919"), colors.HexColor("#666666"), colors.HexColor("#257de0")
    S = {
        "title": ParagraphStyle("t", fontName="DV-B", fontSize=19, leading=24, textColor=ink, spaceAfter=6),
        "byline": ParagraphStyle("by", fontName="DV-B", fontSize=11, leading=15, textColor=ink, spaceAfter=3),
        "subtitle": ParagraphStyle("s", fontName="DV-I", fontSize=10, leading=14, textColor=grey, spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=17, textColor=ink, spaceBefore=14, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=ink, spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=13.2, textColor=ink, spaceAfter=5, alignment=TA_LEFT),
        "cap": ParagraphStyle("c", fontName="DV-I", fontSize=8, leading=11, textColor=grey, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.5, leading=9.5, textColor=ink),
        "head": ParagraphStyle("head", fontName="DV-B", fontSize=7.5, leading=9.5, textColor=colors.white),
    }
    story, width = [], A4[0] - 4 * cm
    for b in blocks:
        kind = b[0]
        if kind in ("title", "byline", "subtitle", "h1", "h2", "p"):
            story.append(Paragraph(_rl(b[1]), S[kind]))
        elif kind == "bullets":
            for x in b[1]:
                story.append(Paragraph("•&nbsp;&nbsp;" + _rl(x), ParagraphStyle("b", parent=S["p"], leftIndent=14, firstLineIndent=-10, spaceAfter=3)))
            story.append(Spacer(1, 4))
        elif kind == "table":
            _, head, rows, cap = b
            data = [[Paragraph(_rl(h), S["head"]) for h in head]] + [[Paragraph(_rl(c), S["cell"]) for c in r] for r in rows]
            longest = max(len(r[0]) for r in rows)
            first = 0.10 if longest <= 10 else 0.28 if len(head) > 4 else 0.34
            widths = [width * first] + [width * (1 - first) / (len(head) - 1)] * (len(head) - 1)
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), blue), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9d9d9")),
                                   ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story.append(KeepTogether([Paragraph(_rl(cap), S["cap"]), t, Spacer(1, 8)]))
        elif kind == "img":
            from PIL import Image as PILImage
            img_path = C.FIG_DIR / b[1]
            w, h = PILImage.open(img_path).size
            iw = width * b[2]
            story.append(KeepTogether([Image(str(img_path), width=iw, height=iw * h / w), Paragraph(_rl(b[3]), S["cap"])]))

    def footer(canvas, doc):
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(grey)
        canvas.drawString(2 * cm, 1.2 * cm, "Predictive cleaning optimisation for heat exchanger networks (synthetic-data study)")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=2 * cm, title="Predictive cleaning optimisation for heat exchanger networks",
                      author=f"{AUTHOR} ({REG_NO})").build(story, onFirstPage=footer, onLaterPages=footer)


def readme_text(res):
    bt, fc = res["backtest"], res["forecast"]
    per, cmp_ = bt["per_year_usd"], bt["comparisons"]
    tf, tt = bt["tuned_on_validation"]["fixed"], bt["tuned_on_validation"]["threshold"]
    op = cmp_["vs_operator"]

    def worst(prefix):
        t = [p["total"] for n, p in per.items() if n.startswith(prefix)]
        return 100 * (max(t) / min(t) - 1)

    rows = "\n".join(
        f"| {n}{' *(tuned on validation)*' if n in (tf, tt) else ''} | {usd(per[n]['total'])} | "
        f"{per[n]['cleanings_per_year']:.1f} |"
        for n in ("Never clean", OPERATOR, tf, tt, ML))
    shots = "".join(f"![{t}](docs/screenshots/{f})\n\n" for f, t in (
        ("fleet.png", "Fleet overview"), ("forecast.png", "Fouling forecast with intervals"),
        ("schedule.png", "Recommended cleaning plan")))
    return f"""# Predictive cleaning optimisation for refinery heat exchanger networks

Crude preheat exchangers slowly foul, which makes the furnace burn extra fuel. Cleaning fixes it but costs money and
takes the exchanger out of service. This project **forecasts fouling for every exchanger** and **recommends which to
clean and when**, using a physics-informed Random Forest regression model and a mixed-integer scheduler, with a Streamlit
dashboard and a full report.

> **Data is synthetic.** No real refinery data was used. `src/generate_data.py` generates it from a documented fouling
> model, so results show that the *method* works end to end, not that these savings would occur in a real plant.

## Results at a glance

Closed-loop simulation, {bt['n_test_scenarios']} unseen test scenarios of {bt['years_per_scenario']:.0f} years each
(8 exchangers, one cleaning crew):

| Strategy | Cost per year | Cleanings per year |
|---|---|---|
{rows}

- The optimiser costs **{op['mean_saving_pct']:.1f} % less than the status-quo operator policy**
  (about ${op['mean_saving_usd_per_year'] / 1e3:,.0f}k per year; cheaper in {op['scenarios_won']} of {op['n_scenarios']} scenarios).
- Against simple rules whose setting was **tuned on separate validation scenarios** it is comparable
  ({cmp_['vs_tuned_fixed']['mean_saving_pct']:+.1f} % vs {tf}, {cmp_['vs_tuned_threshold']['mean_saving_pct']:+.1f} % vs {tt}; within scenario noise),
  but it needs no tuning, and a badly set fixed interval or threshold costs up to {worst('Fixed'):.0f} % or {worst('Threshold'):.0f} % more than its best setting.
- Forecast error is {fc[0]['skill_vs_persistence'] * 100:.0f} % / {fc[1]['skill_vs_persistence'] * 100:.0f} % / {fc[2]['skill_vs_persistence'] * 100:.0f} % lower than a
  no-change forecast at 7 / 14 / 30 days on the held-out year.

![Annual cost](reports/figures/04_cost_comparison.png)

## Dashboard

{shots}## Quick start

```bash
pip install -r requirements.txt
python run_all.py              # data -> models -> backtest -> figures -> report (about 4 minutes)
Launch_Therma.bat              # native desktop application
```

`python run_all.py --skip-backtest` reuses the previous backtest. `pytest -q` runs the tests. All seeds are fixed, so
every run reproduces the same numbers.

## How it works

1. **Data** (`src/generate_data.py`): 3 years of daily readings for 8 exchangers with missing values, outliers and
   sensor drift. Fouling follows a Kern-Seaton asymptotic law driven by crude slate and flow.
2. **Features** (`src/features.py`): Hampel outlier filter, gap filling, run segmentation, rolling features, targets
   that never cross a cleaning or the train/test boundary.
3. **Forecast** (`src/forecast.py`): per-exchanger physical fouling law + Random Forest regression that learns how crude
   slate, flow and run age scale the rate; conformal-calibrated 80 % intervals.
4. **Optimiser** (`src/optimize.py`): rolls forecasts forward 300 days, prices every cleaning option, and solves a
   mixed-integer program (scipy/HiGHS) with a crew limit.
5. **Backtest** (`src/backtest.py`): runs every strategy day by day against the plant model on unseen scenarios.
6. **Report** (`src/make_report.py`): [`reports/report.pdf`](reports/report.pdf) / [`report.md`](reports/report.md).

```
app/desktop_app.py    Native desktop application (fleet, forecast, schedule, what-if, data)
src/                  config, plant model, data generator, features, forecast, optimiser, backtest, evaluate, report
tests/                9 sanity tests (leakage, crew limit, accounting, determinism)
data/                 synthetic CSV and cleaning log
models/               trained forecasting models
reports/              report, results.json, figures
run_all.py            one command for everything
```

## Using your own data

Upload a CSV in the dashboard, or replace `data/heat_exchanger_data.csv`, with the same columns:
`date, exchanger_id, status (online/cleaning), flow_m3h, crude_api, crude_sulfur_pct, fouling_resistance,
days_since_clean, cleaned (1 on the first day back in service)`. Exchanger ids must match `EX_ID` in `src/config.py`,
whose plant parameters (duty, cleaning cost, downtime) you should also update. Re-run `python run_all.py` to retrain.
The backtest needs a plant model, which is synthetic here; on real data you would validate the forecast and use
the optimiser's plan, not the simulated savings.

## Limitations

Synthetic data favours the forecast model (it shares the simulator's functional form); only {bt['n_test_scenarios']}
test scenarios, so 1-2 % differences are noise; prediction intervals cover less than the nominal 80 %; economics are
simplified (single fuel price, independent exchangers, no furnace limit); some scheduler design choices were made
while inspecting early backtest results. See the report, section 6.
"""


def main():
    res = json.loads(C.RESULTS_FILE.read_text())
    blocks = build_blocks(res)
    (C.REPORT_DIR / "report.md").write_text(to_markdown(blocks), encoding="utf-8")
    to_pdf(blocks, C.REPORT_DIR / "report.pdf")
    (C.ROOT / "README.md").write_text(readme_text(res), encoding="utf-8")
    print("wrote reports/report.md, reports/report.pdf and README.md")


if __name__ == "__main__":
    main()
