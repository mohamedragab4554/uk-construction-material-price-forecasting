"""Rolling-origin (time-series cross-validation) backtest.

At every origin t the model is fitted on y[:t] only and asked for h-step-ahead forecasts; nothing after
t is ever visible. This is the evaluation the coursework lacked (its scores were in-sample).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import MODELS

HORIZONS = (1, 3, 6, 12)


def rolling_origin(y: pd.Series, model: str, horizons=HORIZONS, min_train: int = 36, step: int = 1) -> pd.DataFrame:
    H = max(horizons)
    rows = []
    for t in range(min_train, len(y) - min(horizons) + 1, step):
        train = y.iloc[:t]
        f = MODELS[model]().fit(train).predict(H)
        for h in horizons:
            if t + h - 1 < len(y):
                actual = float(y.iloc[t + h - 1])
                rows.append({"model": model, "origin": y.index[t - 1], "h": h, "forecast": float(f[h - 1]),
                             "actual": actual, "last": float(train.iloc[-1])})
    return pd.DataFrame(rows)


def score(errors: pd.DataFrame) -> pd.DataFrame:
    e = errors.assign(err=errors.forecast - errors.actual)
    g = e.groupby(["model", "h"])
    out = pd.DataFrame({
        "MAE": g.err.apply(lambda s: s.abs().mean()),
        "RMSE": g.err.apply(lambda s: np.sqrt((s ** 2).mean())),
        "MAPE_%": g.apply(lambda d: (d.err.abs() / d.actual.abs()).mean() * 100, include_groups=False),
        "n": g.size(),
    }).reset_index()
    naive = out[out.model == "naive"].set_index("h").MAE
    out["MAE_vs_naive"] = out.apply(lambda r: r.MAE / naive.get(r.h, np.nan), axis=1)
    return out


def run(y: pd.Series, models=None, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = models or list(MODELS)
    errs = pd.concat([rolling_origin(y, m, **kw) for m in models], ignore_index=True)
    return errs, score(errs)
