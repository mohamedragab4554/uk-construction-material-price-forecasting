import numpy as np
import pandas as pd
import pytest

from matprice import backtest, escalation
from matprice.coursework import reproduce
from matprice.models import MODELS


def series(values):
    return pd.Series(np.asarray(values, float), index=pd.date_range("2015-01-01", periods=len(values), freq="MS"))


def test_baselines_exact():
    y = series(np.arange(24) + 100.0)
    assert MODELS["naive"]().fit(y).predict(3).tolist() == [123.0] * 3
    assert np.allclose(MODELS["drift"]().fit(y).predict(3), [124, 125, 126])
    assert np.allclose(MODELS["linear_trend"]().fit(y).predict(2), [124, 125])
    assert MODELS["seasonal_naive"]().fit(y).predict(13).tolist() == list(np.arange(12, 24) + 100.0) + [112.0]


def test_rf_time_index_cannot_extrapolate_but_rf_lags_can():
    y = series(np.linspace(100, 200, 120))
    flat = MODELS["rf_time_index"]().fit(y).predict(24)
    assert flat.max() <= y.max() + 1e-9                      # trees never exceed the training range
    trend = MODELS["rf_lags"]().fit(y).predict(24)
    assert trend[-1] > y.iloc[-1] + 10                        # modelling changes follows the trend


def test_backtest_has_no_lookahead():
    y = series(np.r_[np.full(40, 100.0), np.full(20, 200.0)])   # step change at month 40
    e = backtest.rolling_origin(y, "naive", horizons=(1,), min_train=36)
    first_after = e[e.origin == y.index[39]].iloc[0]           # origin = last pre-step month
    assert first_after.forecast == 100.0 and first_after.actual == 200.0
    assert (e.origin < e.origin.max() + pd.DateOffset(months=1)).all()


def test_score_relative_to_naive():
    y = series(np.arange(60) + 100.0)
    _, s = backtest.run(y, models=["naive", "drift"], horizons=(1, 3), min_train=24)
    d = s.set_index(["model", "h"])
    assert d.loc[("naive", 1), "MAE_vs_naive"] == 1.0
    assert d.loc[("drift", 3), "MAE"] < 1e-9                   # drift is exact on a straight line


def test_escalation_and_coverage():
    rng = np.random.default_rng(0)
    y = series(100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, 150))))
    e = backtest.rolling_origin(y, "drift", horizons=(12,), min_train=36)
    x = escalation.escalate(y, e, "drift", 12, 100_000)
    assert x.index_p10 < x.index_p50 < x.index_p90 and x.cost("p90") > x.cost("p50")
    assert 0.5 <= escalation.coverage(e, "drift", 12) <= 1.0  # stationary noise: roughly nominal coverage
    assert 0.0 <= escalation.rolling_coverage(e, "drift", 12, 36) <= 1.0


def test_coursework_reproduction():
    r = reproduce()
    lr = r[(r.series == "materials_index")].iloc[0]
    assert lr.rows_used == 11 and lr.rows_in_file == 132       # the parsing bug
    assert lr.in_sample_R2 == pytest.approx(0.8946, abs=1e-4)
    ce = r[(r.series == "ceramic_tiles") & r.model.str.startswith("Linear")].iloc[0]
    assert ce.in_sample_R2 == pytest.approx(0.4197, abs=1e-4)
    rf = r[r.model.str.startswith("Random")]
    assert (rf.in_sample_R2 > 0.99).all()                      # in-sample RF looks near-perfect
