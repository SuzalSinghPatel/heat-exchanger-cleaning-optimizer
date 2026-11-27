"""Project-wide configuration: plant description, economics and paths.

All plant numbers are illustrative values for a generic crude preheat train.
They do not describe any real refinery.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DATA_FILE = DATA_DIR / "heat_exchanger_data.csv"
LOG_FILE = DATA_DIR / "cleaning_log.csv"
MODEL_FILE = MODEL_DIR / "forecast_models.joblib"
BACKTEST_FILE = REPORT_DIR / "backtest_results.csv"
TRACES_FILE = REPORT_DIR / "backtest_traces.joblib"
RESULTS_FILE = REPORT_DIR / "results.json"

SEED = 42
START_DATE = "2023-01-01"
N_DAYS = 1095            # three years of daily history
TRAIN_END_DAY = 730      # first two years train, third year is the held-out test year
N_BT_DAYS = 1095         # length of each closed-loop backtest scenario (3 years)
N_SCENARIOS = 5          # independent future years used in the backtest

# id, duty [MW], fouling rate k [1e-4 m2K/W per day], asymptotic Rf [1e-4 m2K/W],
# clean U [W/m2K], nominal flow [m3/h], cleaning cost [USD], downtime [days]
_EXCHANGERS = [
    ("E-101", 18, 0.024, 5.0, 400, 320, 38000, 5),
    ("E-102", 15, 0.016, 4.6, 380, 260, 32000, 4),
    ("E-103", 22, 0.030, 6.0, 450, 410, 52000, 6),
    ("E-104", 12, 0.012, 4.2, 350, 210, 28000, 4),
    ("E-105", 17, 0.020, 5.4, 420, 300, 36000, 5),
    ("E-106", 25, 0.027, 6.4, 460, 450, 58000, 6),
    ("E-107", 14, 0.014, 4.4, 360, 240, 30000, 4),
    ("E-108", 20, 0.018, 5.0, 410, 360, 44000, 5),
]
EX_ID = [e[0] for e in _EXCHANGERS]
N_EX = len(_EXCHANGERS)
DUTY = np.array([e[1] for e in _EXCHANGERS], float)
K = np.array([e[2] for e in _EXCHANGERS], float)
RF_MAX = np.array([e[3] for e in _EXCHANGERS], float)
U0 = np.array([e[4] for e in _EXCHANGERS], float)
FLOW_NOM = np.array([e[5] for e in _EXCHANGERS], float)
CLEAN_COST = np.array([e[6] for e in _EXCHANGERS], float)
DOWNTIME = np.array([e[7] for e in _EXCHANGERS], int)

# temperature / pressure-drop baselines used for the synthetic sensor columns
TC_IN0 = 110.0 + 14.0 * np.arange(N_EX)     # cold-side inlet [C], rises along the train
DP0 = 55.0 + 5.0 * np.arange(N_EX)          # clean pressure drop [kPa]
EPS_CLEAN = 0.62                            # clean effectiveness

# economics (defaults, adjustable in the dashboard)
ENERGY_PRICE = 45.0      # USD per MWh of furnace fuel
MAX_OFFLINE = 1          # exchangers that may be out for cleaning at the same time
RF_AFTER_CLEAN = 0.12    # residual fouling resistance after a cleaning [1e-4 m2K/W]
