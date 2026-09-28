import pandas as pd
import pytest

from matprice import data


def test_carry_forward_parser_keeps_every_month():
    s = data.get("materials_index")
    assert len(s) == 132 and s.index[0] == pd.Timestamp("2014-01-01") and s.index[-1] == pd.Timestamp("2024-12-01")
    assert s.index.freqstr == "MS" and not s.isna().any()


def test_all_series_are_complete_monthly():
    for name in data.all_names():
        s = data.get(name)
        assert s.index.freqstr == "MS" and not s.isna().any(), name


def test_ew7c_matches_published_ons_values():
    s = data.get("ew7c_metal_doors_windows")
    assert s["2010-02-01"] == 87.0 and s["2010-03-01"] == 87.5
    assert s["2024-12-01"] == 177.4 and s["2025-01-01"] == 177.7


def test_parser_fails_loudly(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Time period, value\nFeb,1\n2014 Jan,2\n")
    with pytest.raises(ValueError):
        data.load_ons_csv(p)


def test_dbt_table1a():
    t = data.load_dbt_table1a()
    assert len(t) == 60 and t.index[0] == pd.Timestamp("2019-01-01")
    assert t.loc["2019-01-01", "all_work"] == 111.8 and t.provisional.iloc[-1]
