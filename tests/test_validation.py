"""Tests for the validation-notebook helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.validation import (
    assign_deciles,
    decile_flows,
    decile_returns,
    lookahead_baseline,
    momentum_12_1,
    neutralize,
    transition_matrix,
    value_asof_before,
)


def test_lookahead_baseline_reproduces_adjusted_price_caps():
    df = pd.DataFrame({"price_formation": [5.0, np.nan], "shares_outstanding": [2e9, 1e9],
                       "book_equity": [1e9, 1e9], "net_income_annual": [5e8, np.nan]})
    out = lookahead_baseline(df)
    assert np.isclose(out["log_market_cap_lookahead"].iloc[0], np.log(1e10))
    assert np.isclose(out["book_to_market_lookahead"].iloc[0], 0.1)
    assert np.isclose(out["earnings_yield_lookahead"].iloc[0], 0.05)
    assert out.iloc[1][["log_market_cap_lookahead", "book_to_market_lookahead"]].isna().all()


def test_deciles_top_score_is_d1_and_equal_sized():
    df = pd.DataFrame({"period": 1, "score": np.arange(100.0)})
    dec = assign_deciles(df, "score")
    assert dec[df["score"] == 99].iloc[0] == 1      # highest score -> D1
    assert dec[df["score"] == 0].iloc[0] == 10      # lowest score  -> D10
    assert (dec.value_counts() == 10).all()


def test_value_weighted_returns():
    df = pd.DataFrame({"period": [1, 1], "decile": [1, 1],
                       "ret": [0.10, 0.00], "mcap": [3.0, 1.0]})
    ew = decile_returns(df, "ret")
    vw = decile_returns(df, "ret", weight_col="mcap")
    assert np.isclose(ew.loc[1, 1], 0.05)
    assert np.isclose(vw.loc[1, 1], 0.075)          # (0.3 + 0) / 4


def test_asof_before_never_looks_ahead():
    dates = pd.to_datetime(["2020-01-01", "2020-01-10"]).to_numpy("datetime64[ns]")
    vals = np.array([1.0, 2.0])
    assert value_asof_before(dates, vals, pd.Timestamp("2020-01-09")) == 1.0
    assert value_asof_before(dates, vals, pd.Timestamp("2020-01-10")) == 2.0
    assert np.isnan(value_asof_before(dates, vals, pd.Timestamp("2019-12-31")))


def test_momentum_skips_last_month():
    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    px = np.ones(len(dates))
    px[dates >= "2020-06-01"] = 2.0                  # jump inside the skipped month
    d = dates.to_numpy("datetime64[ns]")
    assert np.isclose(momentum_12_1(d, px, pd.Timestamp("2020-06-15")), 0.0)


def test_neutralized_score_is_uncorrelated_with_control():
    rng = np.random.default_rng(0)
    mom = rng.normal(size=500)
    df = pd.DataFrame({"period": 1, "mom": mom, "score": mom + rng.normal(size=500)})
    resid = neutralize(df, "score", ["mom"])
    assert abs(np.corrcoef(resid, df["mom"].rank(pct=True))[0, 1]) < 1e-8


def test_flows_and_transitions():
    df = pd.DataFrame({
        "period": [1, 1, 2, 2],
        "composite_figi": ["A", "B", "A", "C"],
        "decile": [1, 2, 1, 2],
        "cv_fold": [0, 0, 1, 1],
    })
    flows = decile_flows(df, n=2).set_index("decile")
    assert flows.loc[1, "retained"] == 1 and flows.loc[1, "entered"] == 0
    assert flows.loc[2, "left"] == 1 and flows.loc[2, "entered"] == 1
    assert flows["cross_fold"].all()
    tm = transition_matrix(df, n=2)
    assert tm.loc[1, 1] == 1.0
    assert tm.loc[2, "Left universe"] == 1.0
