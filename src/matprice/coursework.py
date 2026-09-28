"""Reproduce the original coursework metrics, and show why they were optimistic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .data import DATA


def original_parse(path) -> pd.DataFrame:
    """The coursework parser: ``pd.to_datetime(format='%Y %b', errors='coerce')`` then ``dropna``. Rows written
    as a bare month ('Feb') become NaT and are silently dropped."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Time period"] = pd.to_datetime(df["Time period"], format="%Y %b", errors="coerce")
    return df.dropna(subset=["Time period"]).sort_values("Time period").reset_index(drop=True)


def in_sample_linear(df: pd.DataFrame) -> tuple[float, float]:
    X = np.c_[df["Time period"].dt.year, df["Time period"].dt.month]
    y = df.iloc[:, 1].to_numpy(float)
    p = LinearRegression().fit(X, y).predict(X)
    return mean_squared_error(y, p), r2_score(y, p)


def in_sample_rf(df: pd.DataFrame, seed: int = 42) -> tuple[float, float]:
    m = df["Time period"].dt.month
    X = np.c_[np.arange(len(df)), np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]
    y = df.iloc[:, 1].to_numpy(float)
    p = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X, y).predict(X)
    return mean_squared_error(y, p), r2_score(y, p)


def reproduce() -> pd.DataFrame:
    rows = []
    b = original_parse(DATA / "materials_index_2014_2024.csv")
    rows.append(("materials_index", "LinearRegression(year, month)", len(b), 132, *in_sample_linear(b), 0.8946))
    for name, f, rep_rf in (("ceramic_tiles", "ceramic_tiles_ppi_2010_2025.csv", 0.9965),
                            ("ew7c_metal_doors_windows", "metal_doors_windows_ppi_EW7C_2010_2025.csv", 0.9996)):
        d = original_parse(DATA / f)
        if name == "ceramic_tiles":
            rows.append((name, "LinearRegression(year, month)", len(d), 180, *in_sample_linear(d), 0.4197))
        rows.append((name, "RandomForest(time index, sin/cos month)", len(d), 180, *in_sample_rf(d), rep_rf))
    return pd.DataFrame(rows, columns=["series", "model", "rows_used", "rows_in_file", "in_sample_MSE",
                                       "in_sample_R2", "reported_R2"])
