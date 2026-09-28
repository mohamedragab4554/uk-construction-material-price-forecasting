"""Banner for the README (1280 x 400).  python scripts/make_banner.py"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matprice import data  # noqa: E402
from matprice.models import MODELS  # noqa: E402

BG, INK, MUTED, GRID = "#0b1220", "#e6edf5", "#8aa0b8", "#22324a"
y = data.get("ew7c_metal_doors_windows")
train = y[:"2020-12-01"]
fig = plt.figure(figsize=(12.8, 4.0), dpi=100, facecolor=BG)
ax = fig.add_axes([0.535, 0.14, 0.44, 0.72], facecolor=BG)
ax.plot(y.index, y.values, color=INK, lw=2)
idx = y["2021-01-01":"2023-12-01"].index
for name, col in (("rf_time_index", "#c98500"), ("ets_damped", "#d95926"), ("rf_lags", "#199e70")):
    ls = "--" if name == "rf_time_index" else "-"
    ax.plot(idx, MODELS[name]().fit(train).predict(len(idx)), color=col, lw=2.2, ls=ls)
ax.axvline(train.index[-1], color=MUTED, lw=1, ls=":")
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=10)
ax.grid(color=GRID, lw=0.6)
ax.set_title("ONS EW7C metal doors & windows: forecasts from Dec 2020", color=MUTED, fontsize=10, loc="left")
fig.text(0.04, 0.70, "UK construction-material", color=INK, fontsize=25, weight="bold")
fig.text(0.04, 0.58, "price forecasting", color=INK, fontsize=25, weight="bold")
fig.text(0.04, 0.44, "Rolling-origin backtests of 7 models on ONS and DBT", color=MUTED, fontsize=13)
fig.text(0.04, 0.37, "indices, turned into P10 / P50 / P90 cost escalation.", color=MUTED, fontsize=13)
fig.text(0.04, 0.20, "Which models beat \"no change\"?  How wide should", color="#2dd4bf", fontsize=12.5)
fig.text(0.04, 0.13, "a 12-month contingency be?  Measured, not assumed.", color="#2dd4bf", fontsize=12.5)
fig.savefig(ROOT / "docs" / "images" / "banner.png", facecolor=BG)
