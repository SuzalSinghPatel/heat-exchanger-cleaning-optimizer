"""Regenerate everything with one command:  python run_all.py

Steps: synthetic data -> forecasting models -> closed-loop backtest -> figures/results -> report.
All random seeds are fixed, so every run reproduces the same numbers.
Options:  --scenarios N     number of test scenarios in the backtest (default 5)
          --skip-backtest   reuse the previous backtest results (the slowest step, ~3 min)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import backtest, config as C, evaluate, forecast, generate_data, make_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=C.N_SCENARIOS)
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()

    steps = [("1/5 Generate synthetic data", generate_data.main),
             ("2/5 Train forecasting models", forecast.main)]
    if not args.skip_backtest:
        steps.append(("3/5 Closed-loop backtest", lambda: backtest.main(args.scenarios)))
    steps += [("4/5 Results and figures", evaluate.main), ("5/5 Report", make_report.main)]
    for name, fn in steps:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        fn()
        print(f"({time.time() - t0:.0f} s)")
    print("\nDone. Launch the desktop application with:  Launch_Therma.bat")


if __name__ == "__main__":
    main()
