"""Tests for point-in-time XBRL fact selection (no look-ahead)."""

from __future__ import annotations

import pandas as pd

from src.prices.fundamentals import _norm_ticker, latest_fact_asof

ENTRIES = [
    {"end": "2019-12-31", "filed": "2020-02-20", "val": 100, "form": "10-K"},
    {"end": "2020-03-31", "filed": "2020-05-05", "val": 110, "form": "10-Q"},
    {"end": "2020-06-30", "filed": "2020-08-05", "val": 120, "form": "10-Q"},
    {"end": "2020-12-31", "filed": "2021-02-20", "val": 130, "form": "10-K"},
]


def test_picks_latest_period_known_by_asof():
    # As of 2020-07-01 only the first two are filed & ended by then.
    val = latest_fact_asof(ENTRIES, pd.Timestamp("2020-07-01"))
    assert val == 110.0  # 2020-03-31 period, filed 2020-05-05


def test_excludes_future_filings():
    # As of 2020-08-01 the 2020-06-30 fact ended by then but was filed 08-05.
    val = latest_fact_asof(ENTRIES, pd.Timestamp("2020-08-01"))
    assert val == 110.0  # still the Q1 fact; Q2 not yet filed


def test_form_filter_selects_annual_only():
    val = latest_fact_asof(ENTRIES, pd.Timestamp("2021-06-01"), forms={"10-K"})
    assert val == 130.0  # latest annual (FY2020)


def test_returns_none_when_nothing_eligible():
    assert latest_fact_asof(ENTRIES, pd.Timestamp("2019-01-01")) is None
    assert latest_fact_asof([], pd.Timestamp("2020-01-01")) is None


def test_norm_ticker_share_classes():
    assert _norm_ticker("BRK.B") == "BRK-B"
    assert _norm_ticker("brk/b") == "BRK-B"
