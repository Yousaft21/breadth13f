"""Tests for the pure forward-return math (as-of lookup, dropout flags)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prices.returns import forward_total_return, price_asof


def _series(start="2020-01-01", periods=400):
    dates = pd.bdate_range(start, periods=periods)  # business days only
    prices = np.linspace(100.0, 200.0, periods)
    return dates.to_numpy(dtype="datetime64[ns]"), prices


def test_price_asof_takes_first_trading_day_on_or_after():
    dates, prices = _series()
    # 2020-01-04 is a Saturday; first trading day on/after is Monday 2020-01-06.
    val = price_asof(dates, prices, pd.Timestamp("2020-01-04"), max_gap_days=7)
    monday_idx = np.searchsorted(dates, np.datetime64("2020-01-06"))
    assert np.isclose(val, prices[monday_idx])


def test_price_asof_returns_nan_past_series_end():
    dates, prices = _series(periods=50)
    # Target well beyond the last available date -> treated as delisted/missing.
    assert np.isnan(price_asof(dates, prices, pd.Timestamp("2025-01-01")))


def test_price_asof_gap_enforced():
    dates, prices = _series(periods=10)
    last = pd.Timestamp(dates[-1])
    # A target 30 days before series would be fine, but a big forward gap is nan.
    assert np.isnan(price_asof(dates, prices, last + pd.Timedelta(days=30), max_gap_days=7))


def test_forward_return_normal():
    dates, prices = _series()
    formation = pd.Timestamp("2020-02-03")
    ret, had0, had1 = forward_total_return(dates, prices, formation, horizon_months=6)
    assert had0 and had1
    p0 = price_asof(dates, prices, formation)
    p1 = price_asof(dates, prices, formation + pd.DateOffset(months=6))
    assert np.isclose(ret, p1 / p0 - 1)


def test_forward_return_delisting_dropout():
    # Series ends 3 months after formation -> formation price exists, forward doesn't.
    dates, prices = _series(start="2020-01-01", periods=65)  # ~3 months of bdays
    formation = pd.Timestamp("2020-01-06")
    ret, had0, had1 = forward_total_return(dates, prices, formation, horizon_months=12)
    assert had0 is True and had1 is False
    assert np.isnan(ret)


def test_forward_return_never_present():
    dates, prices = _series(start="2021-01-01", periods=50)
    # Formation before the series starts, beyond gap -> no formation price.
    ret, had0, had1 = forward_total_return(dates, prices, pd.Timestamp("2020-01-06"), 12)
    assert had0 is False and had1 is False
    assert np.isnan(ret)
