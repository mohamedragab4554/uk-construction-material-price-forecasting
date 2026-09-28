"""Loaders for ONS time-series CSVs and the DBT Building Materials workbook table 1a."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"
_YEAR_MONTH = re.compile(r"^\s*(\d{4})\s+([A-Za-z]{3,9})\s*$")
_MONTH_ONLY = re.compile(r"^\s*([A-Za-z]{3,9})\s*$")


def load_ons_csv(path: str | Path) -> pd.Series:
    """Parse ONS-style monthly CSVs. Handles both '2014 Jan' on every row and the layout where only
    January carries the year ('2014 Jan', 'Feb', 'Mar', ...), carrying the year forward.

    Raises if any row cannot be parsed, instead of silently dropping it.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    period, value = df.columns[0], df.columns[-1]
    year, dates = None, []
    for raw in df[period].astype(str):
        m = _YEAR_MONTH.match(raw)
        if m:
            year, mon = int(m.group(1)), m.group(2)
        else:
            m2 = _MONTH_ONLY.match(raw)
            if not m2 or year is None:
                raise ValueError(f"cannot parse period {raw!r} in {path}")
            mon = m2.group(1)
        dates.append(pd.Timestamp(year=year, month=pd.to_datetime(mon[:3].title(), format="%b").month, day=1))
    s = pd.Series(pd.to_numeric(df[value]).values, index=pd.DatetimeIndex(dates, freq=None), name=Path(path).stem)
    s = s.sort_index()
    # a month-only row after December starts a new year in the source layout
    if not s.index.is_monotonic_increasing or s.index.duplicated().any():
        raise ValueError(f"non-monotonic or duplicate months in {path}")
    return s.asfreq("MS")


def load_dbt_table1a(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA / "dbt_table1a_material_price_indices_2019_2023.csv"
    df = pd.read_csv(p)
    df.index = pd.to_datetime(df.pop("date"), format="%Y-%m")
    return df.asfreq("MS")


SERIES = {
    "ew7c_metal_doors_windows": ("metal_doors_windows_ppi_EW7C_2010_2025.csv",
                                 "PPI: doors and windows of metal (ONS EW7C)", True),
    "ceramic_tiles": ("ceramic_tiles_ppi_2010_2025.csv", "PPI: ceramic tiles and flags (code not recorded)", False),
    "materials_index": ("materials_index_2014_2024.csv", "Construction material price index (code not recorded)",
                        False),
}
DBT_COLUMNS = {"dbt_all_work": "all_work", "dbt_new_housing": "new_housing",
               "dbt_other_new_work": "other_new_work", "dbt_repair_maintenance": "repair_maintenance"}


def get(name: str) -> pd.Series:
    """Load a series by name (see SERIES and DBT_COLUMNS)."""
    if name in SERIES:
        return load_ons_csv(DATA / SERIES[name][0]).rename(name)
    if name in DBT_COLUMNS:
        return load_dbt_table1a()[DBT_COLUMNS[name]].astype(float).rename(name)
    raise KeyError(f"unknown series {name!r}; choose from {sorted(SERIES) + sorted(DBT_COLUMNS)}")


def all_names() -> list[str]:
    return list(SERIES) + list(DBT_COLUMNS)
