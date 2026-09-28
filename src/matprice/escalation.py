"""Turn forecasts into cost-escalation ranges for construction budgets.

Intervals are empirical (split-conformal style): the distribution of past backtest ratios
actual / forecast at horizon h is applied to today's forecast. No distributional assumption, and the
interval width reflects how wrong the model has actually been on this series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import MODELS


@dataclass(frozen=True)
class Escalation:
    series: str
    model: str
    horizon: int
    index_now: float
    index_p10: float
    index_p50: float
    index_p90: float
    base_cost: float

    def pct(self, q: str) -> float:
        return (getattr(self, f"index_{q}") / self.index_now - 1) * 100

    def cost(self, q: str) -> float:
        return self.base_cost * getattr(self, f"index_{q}") / self.index_now

    def summary(self) -> str:
        return (f"{self.series}: {self.horizon}-month escalation with {self.model} — P10 {self.pct('p10'):+.1f}%, "
                f"P50 {self.pct('p50'):+.1f}%, P90 {self.pct('p90'):+.1f}%. On a £{self.base_cost:,.0f} package: "
                f"P50 £{self.cost('p50'):,.0f}, P90 £{self.cost('p90'):,.0f} "
                f"(contingency P90-P50 £{self.cost('p90') - self.cost('p50'):,.0f}). Backtests show these bands "
                "under-cover in regime shifts (see README): treat P90 as a floor for contingency, not a ceiling.")


def escalate(y: pd.Series, errors: pd.DataFrame, model: str, horizon: int, base_cost: float) -> Escalation:
    e = errors[(errors.model == model) & (errors.h == horizon)]
    if len(e) < 10:
        raise ValueError("need at least 10 backtest errors at this horizon")
    ratio = (e.actual / e.forecast).to_numpy()
    point = float(MODELS[model]().fit(y).predict(horizon)[-1])
    p10, p50, p90 = (point * np.quantile(ratio, q) for q in (0.1, 0.5, 0.9))
    return Escalation(str(y.name), model, horizon, float(y.iloc[-1]), p10, p50, p90, base_cost)


def coverage(errors: pd.DataFrame, model: str, horizon: int, calib_frac: float = 0.5) -> float:
    """Honest check of the interval: calibrate quantiles on the first half of origins, measure how often
    the second half falls inside P10-P90 (target 80%)."""
    e = errors[(errors.model == model) & (errors.h == horizon)].sort_values("origin")
    k = int(len(e) * calib_frac)
    cal, test = e.iloc[:k], e.iloc[k:]
    lo, hi = np.quantile(cal.actual / cal.forecast, [0.1, 0.9])
    inside = (test.actual >= test.forecast * lo) & (test.actual <= test.forecast * hi)
    return float(inside.mean())


def rolling_coverage(errors: pd.DataFrame, model: str, horizon: int, window: int = 36,
                     q: tuple[float, float] = (0.1, 0.9)) -> float:
    """Adaptive interval check without look-ahead: at each origin t, use only the last ``window`` errors whose
    targets were already observed (origin <= t - horizon months), and test whether the realised value falls
    inside [q_lo, q_hi]."""
    e = errors[(errors.model == model) & (errors.h == horizon)].sort_values("origin").reset_index(drop=True)
    e["ratio"] = e.actual / e.forecast
    hits = []
    for _, row in e.iterrows():
        known = e[e.origin <= row.origin - pd.DateOffset(months=horizon)].tail(window)
        if len(known) < max(12, window // 2):
            continue
        lo, hi = np.quantile(known.ratio, q)
        hits.append(row.forecast * lo <= row.actual <= row.forecast * hi)
    return float(np.mean(hits)) if hits else float("nan")
