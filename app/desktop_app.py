"""Native desktop application for the heat-exchanger cleaning optimiser.

Launch with launch.bat on Windows. Uses Tkinter (included with standard Python)
rather than a browser dashboard or Streamlit.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src import config as C
from src.features import current_state, load_raw, preprocess, REQUIRED_COLUMNS
from src.forecast import predict_quantiles
from src.optimize import CostParams, Forecaster, build_plan, planned_trajectory, threshold_rule_cost
from src.plant import daily_energy_cost, heat_loss_frac

# Palette
BG = "#F4F7FB"
SURFACE = "#FFFFFF"
INK = "#17243A"
MUTED = "#718096"
BLUE = "#356AE6"
BLUE_DARK = "#244FC2"
TEAL = "#0F9D8A"
AMBER = "#D99022"
RED = "#D9534F"
BORDER = "#E5EAF2"
DARK = "#101B2D"


def money(v):
    return f"${v:,.0f}"


class HeatExchangerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Therma | Predictive Cleaning Optimiser")
        self.geometry("1440x920")
        self.minsize(1120, 760)
        self.configure(bg=BG)
        self.raw = None
        self.df = None
        self.bundle = None
        self.forecaster = None
        self.state = None
        self.plan = None
        self.source_name = "Sample plant dataset"
        self.selected_exchanger = tk.StringVar(value=C.EX_ID[0])
        self.fuel_price = tk.DoubleVar(value=C.ENERGY_PRICE)
        self.clean_mult = tk.DoubleVar(value=1.0)
        self.max_offline = tk.IntVar(value=C.MAX_OFFLINE)
        self.threshold = tk.DoubleVar(value=2.5)
        self.asof_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self._style()
        self._load_models()
        self._load_default_data()
        self._build_shell()
        self.recalculate()

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", font=("Segoe UI", 10), background=BG, foreground=INK)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=SURFACE, relief="flat")
        s.configure("TLabel", background=BG, foreground=INK)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("Title.TLabel", font=("Segoe UI Semibold", 22), background=BG, foreground=INK)
        s.configure("Section.TLabel", font=("Segoe UI Semibold", 13), background=BG, foreground=INK)
        s.configure("Metric.TLabel", font=("Segoe UI Semibold", 23), background=SURFACE, foreground=INK)
        s.configure("MetricName.TLabel", font=("Segoe UI", 9), background=SURFACE, foreground=MUTED)
        s.configure("CardValue.TLabel", font=("Segoe UI Semibold", 12), background=SURFACE, foreground=INK)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(18, 11), font=("Segoe UI Semibold", 10), background="#E9EEF6")
        s.map("TNotebook.Tab", background=[("selected", SURFACE)], foreground=[("selected", BLUE)])
        s.configure("Treeview", rowheight=31, background=SURFACE, fieldbackground=SURFACE,
                    foreground=INK, bordercolor=BORDER, font=("Segoe UI", 9))
        s.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), background="#EEF2F8",
                    foreground=INK, relief="flat", padding=8)
        s.map("Treeview", background=[("selected", "#E6EEFF")], foreground=[("selected", INK)])
        s.configure("TButton", padding=(12, 8), font=("Segoe UI Semibold", 9))
        s.configure("Primary.TButton", background=BLUE, foreground="white")
        s.map("Primary.TButton", background=[("active", BLUE_DARK)])
        s.configure("Secondary.TButton", background="#EAF0FF", foreground=BLUE)
        s.configure("TEntry", padding=6)
        s.configure("TCombobox", padding=6)
        s.configure("Horizontal.TScale", background=BG)

    def _load_models(self):
        # Refresh legacy bundles trained with Gradient Boosting (or older schemas).
        # The desktop app can therefore be opened directly after updating the project.
        valid = False
        if C.MODEL_FILE.exists():
            try:
                saved = joblib.load(C.MODEL_FILE)
                valid = (
                    saved.get("model_type") == "random_forest_regression"
                    and saved.get("model_version") == 2
                )
            except Exception:
                valid = False
        if not valid:
            from src import forecast
            forecast.main()
        self.bundle = joblib.load(C.MODEL_FILE)
        if self.bundle.get("model_type") != "random_forest_regression":
            raise RuntimeError("The forecasting model could not be updated. Run python run_all.py.")
        self.forecaster = Forecaster(self.bundle)

    def _load_default_data(self):
        self.raw = load_raw()
        self.df = preprocess(self.raw)

    def _build_shell(self):
        header = tk.Frame(self, bg=DARK, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = tk.Frame(header, bg=DARK)
        brand.pack(side="left", padx=28, pady=15)
        tk.Label(brand, text="THERMA", bg=DARK, fg="white",
                 font=("Segoe UI Semibold", 17)).pack(anchor="w")
        tk.Label(brand, text="PREDICTIVE MAINTENANCE  /  HEAT EXCHANGER NETWORK",
                 bg=DARK, fg="#9EADC4", font=("Segoe UI", 8)).pack(anchor="w")
        right = tk.Frame(header, bg=DARK)
        right.pack(side="right", padx=24)
        tk.Label(right, text="●  SYNTHETIC DEMONSTRATION DATA", bg=DARK, fg="#8AD8C9",
                 font=("Segoe UI Semibold", 8)).pack(side="left", padx=12)
        ttk.Button(right, text="Import CSV", command=self.import_csv, style="Secondary.TButton").pack(side="left", padx=5)
        ttk.Button(right, text="Export schedule", command=self.export_schedule).pack(side="left", padx=5)

        main = ttk.Frame(self, padding=(26, 20, 26, 8))
        main.pack(fill="both", expand=True)
        title_row = ttk.Frame(main)
        title_row.pack(fill="x", pady=(0, 14))
        ttk.Label(title_row, text="Cleaning intelligence", style="Title.TLabel").pack(side="left")
        ttk.Label(title_row, text="Forecast fouling. Prioritise interventions. Reduce avoidable energy loss.",
                  style="Muted.TLabel").pack(side="left", padx=18, pady=(8, 0))
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)
        self._build_overview()
        self._build_forecast()
        self._build_schedule()
        self._build_scenarios()
        self._build_data()
        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=28, pady=(2, 10))
        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Label(footer, text=f"Model assets loaded  •  {len(C.EX_ID)} exchangers",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="right")

    def _tab(self, name):
        frame = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(frame, text=name)
        return frame

    def _card(self, parent, name, value="—", note=""):
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(card, text=name.upper(), bg=SURFACE, fg=MUTED,
                 font=("Segoe UI Semibold", 8)).pack(anchor="w", padx=16, pady=(13, 3))
        val = tk.Label(card, text=value, bg=SURFACE, fg=INK,
                       font=("Segoe UI Semibold", 22))
        val.pack(anchor="w", padx=16)
        sub = tk.Label(card, text=note, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8))
        sub.pack(anchor="w", padx=16, pady=(1, 13))
        return card, val, sub

    def _build_overview(self):
        tab = self._tab("Overview")
        controls = tk.Frame(tab, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        controls.pack(fill="x", pady=(0, 14))
        left = tk.Frame(controls, bg=SURFACE)
        left.pack(side="left", padx=16, pady=12)
        tk.Label(left, text="AS-OF DATE", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI Semibold", 8)).pack(anchor="w")
        dates = sorted(self.df.date.dt.strftime("%Y-%m-%d").unique())
        self.asof_var.set(dates[-1])
        self.date_combo = ttk.Combobox(left, textvariable=self.asof_var, values=dates[40:], width=14, state="readonly")
        self.date_combo.pack(anchor="w", pady=(4, 0))
        ttk.Button(controls, text="Recalculate recommendations", style="Primary.TButton",
                   command=self.recalculate).pack(side="right", padx=16, pady=16)
        self.metric_frame = tk.Frame(tab, bg=BG)
        self.metric_frame.pack(fill="x", pady=(0, 14))
        self.metric_cards = []
        for _ in range(4):
            card, val, sub = self._card(self.metric_frame, "Metric")
            card.pack(side="left", fill="x", expand=True, padx=(0, 10))
            self.metric_cards.append((val, sub))
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        self.fleet_panel = ttk.Frame(body)
        self.fleet_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.alert_panel = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, width=330)
        self.alert_panel.pack(side="right", fill="y")
        self.alert_panel.pack_propagate(False)
        ttk.Label(self.fleet_panel, text="Asset health", style="Section.TLabel").pack(anchor="w", pady=(0, 9))
        self.fleet_tree = self._tree(self.fleet_panel,
            ("asset", "rf", "loss", "daily", "age", "status"),
            ("Exchanger", "Fouling Rf", "Heat loss", "Fuel loss / day", "Days in run", "Recommendation"),
            (90, 90, 90, 130, 90, 220))
        self._place_tree(self.fleet_tree, fill="both", expand=True)
        tk.Label(self.alert_panel, text="ACTION CENTRE", bg=SURFACE, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w", padx=16, pady=(16, 4))
        self.alert_text = tk.Text(self.alert_panel, bg=SURFACE, fg=INK, relief="flat",
                                  font=("Segoe UI", 9), wrap="word", padx=16, pady=6,
                                  height=20, state="disabled")
        self.alert_text.pack(fill="both", expand=True)

    def _build_forecast(self):
        tab = self._tab("Forecast")
        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Exchanger", style="Section.TLabel").pack(side="left")
        self.ex_combo = ttk.Combobox(top, textvariable=self.selected_exchanger,
                                     values=C.EX_ID, state="readonly", width=12)
        self.ex_combo.pack(side="left", padx=10)
        self.ex_combo.bind("<<ComboboxSelected>>", lambda e: self.draw_forecast())
        self.forecast_summary = ttk.Label(top, text="", style="Muted.TLabel")
        self.forecast_summary.pack(side="left", padx=14)
        self.forecast_plot_frame = ttk.Frame(tab)
        self.forecast_plot_frame.pack(fill="both", expand=True)
        self.forecast_canvas = None
        self.forecast_explain = ttk.Label(tab, text="The shaded band is an illustrative ±12% range, not a calibrated prediction interval.",
                                           style="Muted.TLabel")
        self.forecast_explain.pack(anchor="w", pady=(8, 0))

    def _build_schedule(self):
        tab = self._tab("Schedule")
        heading = ttk.Frame(tab)
        heading.pack(fill="x", pady=(0, 12))
        ttk.Label(heading, text="Optimised 90-day intervention plan", style="Section.TLabel").pack(side="left")
        ttk.Button(heading, text="Save schedule as CSV", command=self.export_schedule,
                   style="Secondary.TButton").pack(side="right")
        self.schedule_summary = ttk.Label(tab, text="", style="Muted.TLabel")
        self.schedule_summary.pack(anchor="w", pady=(0, 10))
        self.schedule_tree = self._tree(tab, ("asset", "start", "return", "rf", "cycle", "downtime"),
            ("Exchanger", "Cleaning begins", "Back in service", "Rf at start", "Optimal cycle", "Offline days"),
            (110, 150, 150, 100, 120, 100))
        self._place_tree(self.schedule_tree, fill="x", pady=(0, 14))
        self.cost_plot_frame = ttk.Frame(tab)
        self.cost_plot_frame.pack(fill="both", expand=True)

    def _build_scenarios(self):
        tab = self._tab("What-if studio")
        ttk.Label(tab, text="Test operating and economic assumptions", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="Adjust the inputs, then apply them to rebuild the cleaning plan. Your source data and trained model remain unchanged.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 16))
        grid = tk.Frame(tab, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        grid.pack(fill="x", pady=(0, 14))
        self._slider(grid, "Furnace fuel price", self.fuel_price, 20, 120, "$/MWh", 0)
        self._slider(grid, "Cleaning cost multiplier", self.clean_mult, .5, 3, "× baseline", 1)
        self._slider(grid, "Maximum simultaneous outages", self.max_offline, 1, 3, "exchangers", 2, integer=True)
        self._slider(grid, "Rule-based threshold", self.threshold, 1.5, 4.5, "Rf threshold", 3)
        ttk.Button(tab, text="Apply scenario & recalculate", style="Primary.TButton",
                   command=self.recalculate).pack(anchor="w")
        self.scenario_result = tk.Text(tab, height=13, bg=SURFACE, fg=INK, relief="flat",
                                       font=("Segoe UI", 10), padx=16, pady=14, wrap="word")
        self.scenario_result.pack(fill="both", expand=True, pady=(16, 0))

    def _slider(self, parent, label, var, lo, hi, suffix, row, integer=False):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.grid(row=row, column=0, sticky="ew", padx=18, pady=11)
        parent.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text=label, bg=SURFACE, fg=INK, font=("Segoe UI Semibold", 9), width=29, anchor="w").pack(side="left")
        scale = tk.Scale(frame, variable=var, from_=lo, to=hi, resolution=1 if integer else .1,
                         orient="horizontal", showvalue=False, length=330, bg=SURFACE,
                         highlightthickness=0, troughcolor="#E8EDF5", activebackground=BLUE)
        scale.pack(side="left", padx=10)
        value = tk.Label(frame, text="", bg=SURFACE, fg=BLUE, font=("Segoe UI Semibold", 10), width=13, anchor="e")
        value.pack(side="left")
        def update(*_):
            v = var.get()
            value.config(text=f"{int(v)} {suffix}" if integer else f"{v:.1f} {suffix}")
        var.trace_add("write", update)
        update()

    def _build_data(self):
        tab = self._tab("Data explorer")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(0, 10))
        ttk.Label(bar, text="Dataset inspection", style="Section.TLabel").pack(side="left")
        self.data_info = ttk.Label(bar, text="", style="Muted.TLabel")
        self.data_info.pack(side="left", padx=14)
        ttk.Button(bar, text="Import another CSV", command=self.import_csv).pack(side="right")
        ttk.Button(bar, text="Export cleaned view", command=self.export_data).pack(side="right", padx=8)
        filters = ttk.Frame(tab)
        filters.pack(fill="x", pady=(0, 10))
        ttk.Label(filters, text="Exchanger").pack(side="left")
        self.data_filter = tk.StringVar(value="All")
        self.data_combo = ttk.Combobox(filters, textvariable=self.data_filter,
                                       values=["All"] + C.EX_ID, width=12, state="readonly")
        self.data_combo.pack(side="left", padx=8)
        self.data_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_data_table())
        ttk.Label(filters, text="Showing most recent 500 records").pack(side="left", padx=12)
        self.data_tree = self._tree(tab, ("date", "asset", "status", "flow", "api", "sulfur", "rf", "age"),
            ("Date", "Exchanger", "Status", "Flow m³/h", "Crude API", "Sulfur %", "Fouling Rf", "Days since clean"),
            (110, 100, 90, 110, 90, 90, 100, 130))
        self._place_tree(self.data_tree, fill="both", expand=True)

    def _tree(self, parent, cols, headings, widths):
        wrap = ttk.Frame(parent)
        tree = ttk.Treeview(wrap, columns=cols, show="headings", height=12)
        for col, heading, width in zip(cols, headings, widths):
            tree.heading(col, text=heading)
            tree.column(col, width=width, minwidth=65, anchor="w")
        sy = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        sx = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1); wrap.columnconfigure(0, weight=1)
        # Pack the inner tree wrapper's parent through this frame proxy.
        tree._wrapper = wrap
        return tree

    def _place_tree(self, tree, **kwargs):
        tree._wrapper.pack(**kwargs)

    def _replace_tree(self, tree, rows):
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", "end", values=row)

    def recalculate(self):
        try:
            self.status_var.set("Calculating forecasts and optimised schedule…")
            self.update_idletasks()
            dates = pd.to_datetime(self.df["date"])
            asof = pd.Timestamp(self.asof_var.get()) if self.asof_var.get() else dates.max()
            self.state = current_state(self.df, asof)
            cp = CostParams(price=float(self.fuel_price.get()), clean_mult=float(self.clean_mult.get()),
                            max_offline=int(self.max_offline.get()))
            self.plan = build_plan(self.forecaster, self.state, cp)
            self._refresh_overview(cp)
            self._refresh_schedule(cp)
            self._refresh_scenario(cp)
            self.draw_forecast()
            self.refresh_data_table()
            self.status_var.set(f"Updated for {asof.date()}  •  {self.source_name}")
        except Exception as exc:
            self.status_var.set("Calculation failed")
            messagebox.showerror("Unable to calculate", f"{exc}\n\n{traceback.format_exc()}")

    def _status(self, i):
        if self.state["busy"][i] > 0:
            return f"In cleaning · {self.state['busy'][i]}d left"
        starts = {j: d for j, d in self.plan["schedule"]}
        if i in starts:
            d = starts[i]
            return "Urgent · within 7d" if d <= 7 else "Due · within 30d" if d <= 30 else f"Planned · day {d}"
        return "Stable · no action in 90d"

    def _refresh_overview(self, cp):
        loss = heat_loss_frac(self.state["rf_state"]) * 100
        daily = daily_energy_cost(self.state["rf_state"], cp.price)
        starts = [d for _, d in self.plan["schedule"]]
        vals = [f"${daily.sum()*365/1e6:.2f}M", str(sum(d <= 30 for d in starts)),
                str(int((self.state["busy"] > 0).sum())), f"{len(self.plan['schedule'])}"]
        notes = ["Annualised fuel penalty at current fouling", "Recommended within next 30 days",
                 "Currently unavailable for service", "Total interventions in next 90 days"]
        for (v, n), value, note in zip(self.metric_cards, vals, notes):
            v.config(text=value); n.config(text=note)
        rows = []
        for i, ex in enumerate(C.EX_ID):
            rows.append((ex, f"{self.state['rf_state'][i]:.2f}", f"{loss[i]:.1f}%",
                         money(daily[i]), f"{self.state['dsc'][i]:.0f}", self._status(i)))
        self._replace_tree(self.fleet_tree, rows)
        alerts = []
        for i, ex in enumerate(C.EX_ID):
            st = self._status(i)
            if "Urgent" in st or "Due" in st or "In cleaning" in st:
                alerts.append(f"• {ex} — {st}. Current Rf {self.state['rf_state'][i]:.2f}.")
        if not alerts:
            alerts = ["No immediate interventions are recommended under the current assumptions."]
        alerts.append("\nOperating note\nRecommendations are forecast-based estimates. Confirm plant constraints and maintenance readiness before execution.")
        self.alert_text.config(state="normal"); self.alert_text.delete("1.0", "end")
        self.alert_text.insert("1.0", "\n\n".join(alerts)); self.alert_text.config(state="disabled")

    def _refresh_schedule(self, cp):
        self.schedule_tree._wrapper.pack_forget()
        rows = []
        today = pd.Timestamp(self.state["asof"])
        for i, d in self.plan["schedule"]:
            start = today + pd.Timedelta(days=int(d))
            end = start + pd.Timedelta(days=int(C.DOWNTIME[i]))
            rows.append((C.EX_ID[i], start.strftime("%d %b %Y"), end.strftime("%d %b %Y"),
                         f"{self.plan['nc'][i,d]:.2f}", f"{self.plan['t_star'][i]} days", str(C.DOWNTIME[i])))
        self._replace_tree(self.schedule_tree, rows)
        self._place_tree(self.schedule_tree, fill="x", pady=(0, 14))
        self.schedule_summary.config(text=f"{len(rows)} recommended cleaning events  •  Crew capacity: {cp.max_offline} concurrent outage(s)  •  90-day decision window")
        for child in self.cost_plot_frame.winfo_children(): child.destroy()
        fig = Figure(figsize=(8, 2.5), dpi=100, facecolor=BG)
        ax = fig.add_subplot(111); ax.set_facecolor(SURFACE)
        names = ["Never clean", "Threshold rule", "Optimised plan"]
        costs = [self.plan["cost_none"]/1e6,
                 threshold_rule_cost(self.plan, float(self.threshold.get()))/1e6,
                 self.plan["cost_plan"]/1e6]
        bars = ax.bar(names, costs, color=["#A8B3C5", AMBER, BLUE], width=.55)
        ax.set_ylabel("Forecast cost · USD million")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelcolor=MUTED)
        for b, v in zip(bars, costs): ax.text(b.get_x()+b.get_width()/2, v, f"${v:.2f}M", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.cost_plot_frame).get_tk_widget().pack(fill="both", expand=True)

    def _refresh_scenario(self, cp):
        rule = threshold_rule_cost(self.plan, float(self.threshold.get()))
        saving = 100*(1-self.plan["cost_plan"]/self.plan["cost_none"]) if self.plan["cost_none"] else 0
        text = (f"SCENARIO SUMMARY\n\n"
                f"Fuel price: ${cp.price:.1f}/MWh\nCleaning cost multiplier: {cp.clean_mult:.1f}×\n"
                f"Maximum simultaneous outages: {cp.max_offline}\nRule threshold: Rf ≥ {self.threshold.get():.1f}\n\n"
                f"Optimised forecast cost: {money(self.plan['cost_plan'])}\n"
                f"Never-clean forecast cost: {money(self.plan['cost_none'])}\n"
                f"Threshold-rule forecast cost: {money(rule)}\n\n"
                f"Optimised plan vs never clean: {saving:.1f}% lower forecast cost.\n"
                f"Recommended cleaning events: {len(self.plan['schedule'])}\n\n"
                "These are model-based projections over the scheduler's evaluation window, not guaranteed savings. "
                "The threshold comparison does not enforce crew limits.")
        self.scenario_result.config(state="normal"); self.scenario_result.delete("1.0", "end")
        self.scenario_result.insert("1.0", text); self.scenario_result.config(state="disabled")

    def draw_forecast(self):
        if self.plan is None: return
        ex = self.selected_exchanger.get()
        if ex not in C.EX_ID: return
        i = C.EX_ID.index(ex)
        today = pd.Timestamp(self.state["asof"])
        hist = self.df[(self.df.exch_idx == i) & (self.df.date <= today)].tail(240)
        future = today + pd.to_timedelta(np.arange(121), unit="D")
        forecast = self.plan["nc"][i, :121]
        traj = planned_trajectory(self.plan, i)[:121]
        fig = Figure(figsize=(10, 4.8), dpi=100, facecolor=BG)
        ax = fig.add_subplot(111); ax.set_facecolor(SURFACE)
        ax.scatter(hist.date, hist.rf, s=9, color="#AAB4C4", alpha=.55, label="Observed readings")
        ax.plot(hist.date, hist.rf_state, color=BLUE, lw=2, label="Smoothed fouling state")
        ax.plot(future, forecast, color=BLUE, lw=2, ls="--", label="Forecast · no cleaning")
        ax.fill_between(future, forecast*.88, forecast*1.12, color=BLUE, alpha=.09, label="Indicative range (±12%; not a calibrated interval)")
        if np.isfinite(traj).any() and not np.allclose(traj, forecast, equal_nan=True):
            ax.plot(future, traj, color=TEAL, lw=2.4, label="With recommended cleaning")
        ax.axvline(today, color=MUTED, lw=1, ls=":")
        ax.set_ylabel("Fouling resistance Rf")
        ax.grid(alpha=.17); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, labelcolor=MUTED)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout()
        if self.forecast_canvas:
            self.forecast_canvas.get_tk_widget().destroy()
        self.forecast_canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
        self.forecast_canvas.draw(); self.forecast_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.forecast_summary.config(text=f"{ex}  •  Current Rf {self.state['rf_state'][i]:.2f}  •  {self._status(i)}")

    def import_csv(self):
        path = filedialog.askopenfilename(title="Select plant data CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        try:
            candidate = pd.read_csv(path, parse_dates=["date"])
            missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
            if missing:
                raise ValueError("Missing required columns: " + ", ".join(missing))
            unknown = sorted(set(candidate.exchanger_id.dropna()) - set(C.EX_ID))
            if unknown:
                raise ValueError("Unknown exchanger IDs: " + ", ".join(map(str, unknown)) +
                                 ". Update src/config.py to define the plant's exchanger IDs and parameters.")
            if candidate.empty: raise ValueError("The selected CSV contains no records.")
            tidy = preprocess(candidate)
            if tidy.empty: raise ValueError("No usable rows remain after preprocessing.")
            self.raw, self.df = candidate, tidy
            self.source_name = Path(path).name
            dates = sorted(self.df.date.dt.strftime("%Y-%m-%d").unique())
            self.date_combo.configure(values=dates[40:])
            self.asof_var.set(dates[-1])
            self.refresh_data_table()
            self.recalculate()
            messagebox.showinfo("Data imported", f"Loaded {len(candidate):,} rows from {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("CSV import failed", str(exc))

    def export_schedule(self):
        if self.plan is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="recommended_cleaning_schedule.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path: return
        today = pd.Timestamp(self.state["asof"])
        rows = []
        for i, d in self.plan["schedule"]:
            start = today + pd.Timedelta(days=int(d))
            rows.append({"exchanger_id": C.EX_ID[i], "cleaning_start": start.date(),
                         "back_in_service": (start + pd.Timedelta(days=int(C.DOWNTIME[i]))).date(),
                         "rf_at_start": float(self.plan["nc"][i,d]), "downtime_days": int(C.DOWNTIME[i]),
                         "optimal_cycle_days": int(self.plan["t_star"][i])})
        pd.DataFrame(rows).to_csv(path, index=False)
        messagebox.showinfo("Export complete", f"Schedule saved to:\n{path}")

    def export_data(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="plant_data_export.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path: return
        self.raw.to_csv(path, index=False)
        messagebox.showinfo("Export complete", f"Plant data saved to:\n{path}")

    def refresh_data_table(self):
        if self.df is None: return
        d = self.raw.copy()
        sel = self.data_filter.get() if hasattr(self, "data_filter") else "All"
        if sel != "All": d = d[d.exchanger_id == sel]
        d = d.tail(500)
        rows = []
        for r in d.itertuples():
            rows.append((pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.exchanger_id, r.status,
                         f"{r.flow_m3h:.1f}", f"{r.crude_api:.2f}", f"{r.crude_sulfur_pct:.3f}",
                         f"{r.fouling_resistance:.3f}", f"{r.days_since_clean:.0f}"))
        if hasattr(self, "data_tree"):
            self._replace_tree(self.data_tree, rows)
            self.data_info.config(text=f"{len(self.raw):,} records  •  {self.raw.exchanger_id.nunique()} exchangers  •  {self.source_name}")

if __name__ == "__main__":
    app = HeatExchangerApp()
    app.mainloop()
