"""Run the full benchmark and write results/ and docs/images/.  python scripts/run_benchmark.py"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matprice import backtest, data, escalation  # noqa: E402
from matprice.coursework import original_parse, reproduce  # noqa: E402
from matprice.models import MODELS  # noqa: E402

RES, IMG = ROOT / "results", ROOT / "docs" / "images"
BG, PANEL, GRID, INK, MUTED = "#0b1220", "#101a2b", "#22324a", "#e6edf5", "#8aa0b8"
COL = {"naive": "#8aa0b8", "drift": "#3987e5", "ets_damped": "#d95926", "rf_lags": "#199e70",
       "rf_time_index": "#c98500", "linear_trend": "#9085e9", "seasonal_naive": "#e87ba4"}
NICE = {"naive": "Naive (last value)", "drift": "Drift", "ets_damped": "Damped ETS",
        "rf_lags": "RF on lagged changes", "rf_time_index": "RF on time index (coursework)",
        "linear_trend": "Linear trend", "seasonal_naive": "Seasonal naive"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True, "grid.linewidth": 0.6,
                     "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "lines.linewidth": 2,
                     "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    RES.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    reproduce().to_csv(RES / "coursework_reproduction.csv", index=False)
    all_scores, all_errs = [], []
    for name in data.all_names():
        y = data.get(name)
        errs, sc = backtest.run(y, min_train=36)
        errs["series"], sc["series"] = name, name
        all_errs.append(errs)
        all_scores.append(sc)
        print(name, len(y), "months")
    scores = pd.concat(all_scores, ignore_index=True)
    errors = pd.concat(all_errs, ignore_index=True)
    scores.round(4).to_csv(RES / "backtest_scores.csv", index=False)
    errors.to_csv(RES / "backtest_errors.csv.gz", index=False)
    # headline: mean relative MAE across the three long series (>=132 months)
    long = [n for n in data.all_names() if len(data.get(n)) >= 132]
    head = (scores[scores.series.isin(long)].groupby(["model", "h"]).MAE_vs_naive.mean().unstack().round(3))
    head.to_csv(RES / "headline_relative_mae.csv")
    print(head)
    esc = []
    for name in long:
        y = data.get(name)
        for model in ("ets_damped", "rf_lags", "drift"):
            e = errors[(errors.series == name)]
            x = escalation.escalate(y, e, model, 12, 100_000)
            esc.append({"series": name, "model": model, "last_month": y.index[-1].strftime("%Y-%m"),
                        "P10_%": round(x.pct("p10"), 2), "P50_%": round(x.pct("p50"), 2),
                        "P90_%": round(x.pct("p90"), 2),
                        "coverage_static": round(escalation.coverage(e, model, 12), 3),
                        "coverage_rolling36": round(escalation.rolling_coverage(e, model, 12, 36), 3),
                        "coverage_rolling36_P5_P95": round(
                            escalation.rolling_coverage(e, model, 12, 36, (0.05, 0.95)), 3),
                        "coverage_rolling36_h3": round(escalation.rolling_coverage(e, model, 3, 36), 3)})
    pd.DataFrame(esc).to_csv(RES / "escalation_12m.csv", index=False)
    figures(scores, errors)


def figures(scores: pd.DataFrame, errors: pd.DataFrame) -> None:
    y = data.get("ew7c_metal_doors_windows")
    # 1. held-out demo: train to Dec 2020, forecast 2021-2024 (the post-pandemic surge)
    cut = y.index.get_loc(pd.Timestamp("2020-12-01")) + 1
    train, test = y.iloc[:cut], y.iloc[cut:cut + 36]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(y.index[:cut + 36], y.iloc[:cut + 36], color=INK, lw=1.6, label="Actual (ONS EW7C)")
    for m in ("naive", "rf_time_index", "ets_damped", "rf_lags"):
        f = MODELS[m]().fit(train).predict(len(test))
        ax.plot(test.index, f, color=COL[m], ls="--" if m in ("naive", "rf_time_index") else "-", label=NICE[m])
    ax.axvline(train.index[-1], color=MUTED, lw=1, ls=":")
    ax.text(train.index[-1], ax.get_ylim()[1] * 0.97, "  forecast origin: Dec 2020", color=MUTED, fontsize=9, va="top")
    ax.set_ylabel("index (2015 = 100)")
    ax.set_title("Metal doors & windows PPI: 36-month forecasts from Dec 2020 vs what happened", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.015, "No model foresaw the 2021-23 surge; the time-index Random Forest cannot extrapolate at all "
             "(flat line). Source: ONS, OGL v3.0.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "holdout_2021_2023.png", dpi=150)
    plt.close(fig)

    # 2. relative MAE by horizon (mean over the three long series)
    long = [n for n in data.all_names() if len(data.get(n)) >= 132]
    rel = scores[scores.series.isin(long)].groupby(["model", "h"]).MAE_vs_naive.mean().unstack()
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    shown = ("naive", "drift", "ets_damped", "rf_lags", "rf_time_index")
    ypos, last = {}, -np.inf
    for m in sorted(shown, key=lambda k: rel.loc[k, 12]):          # spread end labels so they never overlap
        last = ypos[m] = max(rel.loc[m, 12], last + 0.045)
    for m in shown:
        ax.plot(rel.columns, rel.loc[m], marker="o", ms=7, color=COL[m], label=NICE[m])
        ax.text(12.4, ypos[m], NICE[m], color=COL[m] if m != "naive" else INK, fontsize=8.5, va="center")
    ax.set_xticks([1, 3, 6, 12])
    ax.set_xlim(0.5, 17.5)
    ax.set_xlabel("forecast horizon (months)")
    ax.set_ylabel("MAE / MAE of naive")
    ax.set_title("Rolling-origin backtest: error relative to the naive forecast (lower is better)", loc="left")
    fig.text(0.01, 0.015, "Mean over the three long ONS series, every origin from month 36. "
             "Linear trend and seasonal naive are off the scale (results/).", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "relative_mae.png", dpi=150)
    plt.close(fig)

    # 3. escalation fan: last 5 years + 12-month P10-P90 for the best long-horizon model
    e = errors[errors.series == "ew7c_metal_doors_windows"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    hist = y.iloc[-60:]
    ax.plot(hist.index, hist, color=INK, lw=1.6, label="Actual")
    horizons = range(1, 13)
    ratios = {h: (lambda d: (d.actual / d.forecast).to_numpy())(e[(e.model == "rf_lags") & (e.h == h)])
              for h in (1, 3, 6, 12)}
    point = MODELS["rf_lags"]().fit(y).predict(12)
    fidx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    hs = np.array([1, 3, 6, 12])
    lo = [np.interp(h, hs, [np.quantile(ratios[k], 0.1) for k in hs]) for h in horizons]
    hi = [np.interp(h, hs, [np.quantile(ratios[k], 0.9) for k in hs]) for h in horizons]
    ax.fill_between(fidx, point * np.array(lo), point * np.array(hi), color=COL["rf_lags"], alpha=0.25,
                    label="P10-P90 (empirical backtest errors)")
    ax.plot(fidx, point, color=COL["rf_lags"], label="RF on lagged changes (P50 path)")
    ax.set_ylabel("index (2015 = 100)")
    ax.set_title("12-month escalation range for a metal doors & windows package", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cov = escalation.rolling_coverage(e, "rf_lags", 12, 36)
    fig.text(0.01, 0.015, f"Caution: in the rolling backtest a band built this way contained the outcome {cov:.0%} of "
             "the time, not 80%. Size contingency above P90.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "escalation_fan.png", dpi=150)
    plt.close(fig)

    # 4. the parsing bug: what the original regression actually saw
    raw = data.get("materials_index")
    kept = original_parse(data.DATA / "materials_index_2014_2024.csv")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(raw.index, raw, color=MUTED, lw=1.4, label="All 132 months in the file")
    ax.scatter(kept["Time period"], kept.iloc[:, 1], s=46, color=COL["ets_damped"], zorder=3,
               label="The 11 rows the coursework parser kept (January only)")
    ax.set_ylabel("index (2015 = 100)")
    ax.set_title("Why the coursework R² of 0.89 was fitted on 11 points", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.02, "errors='coerce' turned the bare 'Feb', 'Mar', … rows into NaT and dropna() removed them "
             "silently. load_ons_csv carries the year forward and fails loudly.",
             color=MUTED, fontsize=8.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IMG / "parsing_bug.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
