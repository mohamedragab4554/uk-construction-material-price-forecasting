"""``matprice backtest|escalate|coursework``"""
from __future__ import annotations

import argparse
import sys

from . import backtest, data, escalation
from .coursework import reproduce


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="matprice")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="rolling-origin backtest of every model on one series")
    b.add_argument("--series", default="ew7c_metal_doors_windows", choices=data.all_names())
    b.add_argument("--min-train", type=int, default=36)
    e = sub.add_parser("escalate", help="P10/P50/P90 cost escalation for a package")
    e.add_argument("--series", default="ew7c_metal_doors_windows", choices=data.all_names())
    e.add_argument("--model", default="ets_damped")
    e.add_argument("--horizon", type=int, default=12)
    e.add_argument("--base-cost", type=float, default=100_000)
    sub.add_parser("coursework", help="reproduce the original coursework metrics")
    a = ap.parse_args(argv)
    if a.cmd == "coursework":
        print(reproduce().to_string(index=False))
        return 0
    y = data.get(a.series)
    if a.cmd == "backtest":
        _, s = backtest.run(y, min_train=a.min_train)
        print(s.pivot(index="model", columns="h", values="MAE_vs_naive").round(2).to_string())
        return 0
    errs = backtest.rolling_origin(y, a.model, horizons=(a.horizon,))
    print(escalation.escalate(y, errs, a.model, a.horizon, a.base_cost).summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
