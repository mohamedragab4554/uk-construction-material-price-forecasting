"""Forecasters with a common interface: ``fit(y) -> self`` and ``predict(h) -> np.ndarray``."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Naive:
    """Last observed value (random walk). The baseline every model must beat."""
    name = "naive"

    def fit(self, y: pd.Series):
        self.last = float(y.iloc[-1])
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.full(h, self.last)


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, m: int = 12):
        self.m = m

    def fit(self, y: pd.Series):
        self.tail = y.iloc[-self.m:].to_numpy(float)
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.array([self.tail[i % self.m] for i in range(h)])


class Drift:
    """Random walk with drift (average historical monthly change)."""
    name = "drift"

    def fit(self, y: pd.Series):
        v = y.to_numpy(float)
        self.last, self.slope = v[-1], (v[-1] - v[0]) / max(1, len(v) - 1)
        return self

    def predict(self, h: int) -> np.ndarray:
        return self.last + self.slope * np.arange(1, h + 1)


class LinearTrend:
    """Least-squares straight line through time (the coursework's linear-regression idea)."""
    name = "linear_trend"

    def fit(self, y: pd.Series):
        self.n = len(y)
        self.m = LinearRegression().fit(np.arange(self.n)[:, None], y.to_numpy(float))
        return self

    def predict(self, h: int) -> np.ndarray:
        return self.m.predict(np.arange(self.n, self.n + h)[:, None])


class RandomForestTimeIndex:
    """The coursework's Random Forest on (time index, sin/cos month). Trees cannot extrapolate beyond the
    training range, so every forecast is roughly the last fitted level: kept to demonstrate that."""
    name = "rf_time_index"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, y: pd.Series):
        self.n, self.start_month = len(y), y.index[0].month
        self.rf = RandomForestRegressor(n_estimators=100, random_state=self.seed).fit(self._X(0, self.n),
                                                                                     y.to_numpy(float))
        return self

    def _X(self, a: int, b: int) -> np.ndarray:
        t = np.arange(a, b)
        month = (self.start_month - 1 + t) % 12 + 1
        return np.c_[t, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]

    def predict(self, h: int) -> np.ndarray:
        return self.rf.predict(self._X(self.n, self.n + h))


class RandomForestLags:
    """Random Forest on the last 12 month-on-month changes, forecasting recursively. Modelling changes
    rather than levels lets a tree model follow a trend it has not seen before."""
    name = "rf_lags"

    def __init__(self, lags: int = 12, seed: int = 42):
        self.lags, self.seed = lags, seed

    def fit(self, y: pd.Series):
        d = np.diff(y.to_numpy(float))
        X = np.array([d[i - self.lags:i] for i in range(self.lags, len(d))])
        self.rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=self.seed)
        self.rf.fit(X, d[self.lags:])
        self.hist, self.last = list(d[-self.lags:]), float(y.iloc[-1])
        return self

    def predict(self, h: int) -> np.ndarray:
        hist, level, out = list(self.hist), self.last, []
        for _ in range(h):
            step = float(self.rf.predict(np.array(hist[-self.lags:])[None])[0])
            level += step
            hist.append(step)
            out.append(level)
        return np.array(out)


class DampedETS:
    """Exponential smoothing with an additive damped trend (statsmodels)."""
    name = "ets_damped"

    def fit(self, y: pd.Series):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ExponentialSmoothing(y.to_numpy(float), trend="add", damped_trend=True,
                                            initialization_method="estimated").fit()
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.asarray(self.res.forecast(h))


MODELS = {c.name: c for c in (Naive, SeasonalNaive, Drift, LinearTrend, RandomForestTimeIndex, RandomForestLags,
                              DampedETS)}
