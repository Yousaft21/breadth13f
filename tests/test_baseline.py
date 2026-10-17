"""Tests for the point-in-time size/value baseline (as-traded prices, cap checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prices.as_traded import as_traded_price
from src.prices.baseline import build_baseline_features, market_cap_ok


class _Fundamentals:
    """Stand-in for FundamentalsProvider: constant as-of fundamentals."""

    def get_asof(self, cik, as_of):
        return {"shares_outstanding": 1e8, "book_equity": 2e9, "net_income_annual": 5e8}


def _raw(closes, splits):
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes, "split": splits})


def test_as_traded_price_undoes_later_splits_only():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-06-01", "2021-01-04"]),
        "close": [10.0, 12.0, 15.0],          # back-adjusted for the 4:1 split below
        "split": [0.0, 4.0, 0.0],             # 4-for-1 split on 2020-06-01
    })
    # Before the split, the stock actually traded at 4x the back-adjusted close.
    assert as_traded_price(raw, pd.Timestamp("2020-01-06")) == 40.0
    # After the split, no later splits -> price is as reported.
    assert as_traded_price(raw, pd.Timestamp("2021-01-06")) == 15.0
    # Never uses a close dated after the target.
    assert np.isnan(as_traded_price(raw, pd.Timestamp("2019-12-31")))


def test_baseline_cap_is_point_in_time():
    """A split after the formation date must not change the cap at formation.

    yfinance back-adjusts every earlier close when a stock splits, so the same
    stock downloaded before and after a later 10-for-1 split gives different
    closes for the formation date. The as-traded cap must be identical either way.
    """
    n = 60
    closes = np.full(n, 50.0)
    before = _raw(closes, np.zeros(n))                    # downloaded before the split
    split = np.zeros(n)
    split[40] = 10.0
    after = _raw(closes / np.where(np.arange(n) < 40, 10.0, 1.0), split)  # back-adjusted

    formation = before["date"].iloc[20]                   # well before the split
    panel = pd.DataFrame({"period": [pd.Timestamp("2019-12-31")], "composite_figi": ["F1"],
                          "formation_date": [formation], "ticker": ["XYZ"],
                          "log_total_inst_shares": [np.log(6e7)]})   # 60% institutional
    caps = [build_baseline_features(panel, {"XYZ": raw}, _Fundamentals(), [1])
            for raw in (before, after)]
    for b in caps:
        assert np.isclose(b["market_cap"].iloc[0], 50.0 * 1e8)
        assert np.isclose(b["log_market_cap"].iloc[0], np.log(5e9))
        assert np.isclose(b["book_to_market"].iloc[0], 0.4)
        assert np.isclose(b["earnings_yield"].iloc[0], 0.1)
    # The adjusted close the old baseline used would have shown a 10x smaller cap.
    assert np.isclose(after["close"].iloc[20] * 1e8, caps[0]["market_cap"].iloc[0] / 10)


def test_failed_share_count_check_blanks_features():
    raw = _raw(np.full(30, 50.0), np.zeros(30))
    panel = pd.DataFrame({"period": [pd.Timestamp("2019-12-31")], "composite_figi": ["F1"],
                          "formation_date": [raw["date"].iloc[20]], "ticker": ["XYZ"],
                          # 13F holders report 50x the XBRL share count -> bad count
                          "log_total_inst_shares": [np.log(5e9)]})
    b = build_baseline_features(panel, {"XYZ": raw}, _Fundamentals(), [1])
    assert not b["cap_ok"].iloc[0]
    assert np.isclose(b["io_ratio"].iloc[0], 50.0)
    assert b[["log_market_cap", "book_to_market", "earnings_yield"]].iloc[0].isna().all()


def test_rows_without_a_cik_get_no_features():
    raw = _raw(np.full(30, 50.0), np.zeros(30))
    panel = pd.DataFrame({"period": [pd.Timestamp("2019-12-31")] * 2, "composite_figi": ["F1", "F2"],
                          "formation_date": [raw["date"].iloc[20]] * 2, "ticker": ["XYZ", "XYZ"],
                          "log_total_inst_shares": [np.log(6e7)] * 2})
    b = build_baseline_features(panel, {"XYZ": raw}, _Fundamentals(), pd.array([1, pd.NA], dtype="Int64"))
    assert np.isclose(b["market_cap"].iloc[0], 50.0 * 1e8)
    assert np.isnan(b["shares_outstanding"].iloc[1]) and not b["cap_ok"].iloc[1]


def test_market_cap_ok_bounds():
    # sane $10bn / IO 0.6 | IO below 1% | IO above 300% | cap above $6T | no cap
    ok = market_cap_ok([1e10, 1e10, 1e10, 7e12, np.nan], [0.6, 0.005, 3.5, 0.6, 0.6])
    assert ok.tolist() == [True, False, False, False, False]
