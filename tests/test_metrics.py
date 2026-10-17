"""Tests for cross-sectional IC / decile-spread / turnover metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    decile_spread,
    ic_summary,
    newey_west_t,
    nw_lags_for,
    portfolio_turnover,
    rank_ic,
)


def _perfect_panel(n_periods=6, n_names=40):
    """Prediction perfectly rank-orders realized return within each period."""
    rows = []
    for p in range(n_periods):
        for j in range(n_names):
            rows.append({
                "period": pd.Timestamp("2020-03-31") + pd.DateOffset(months=3 * p),
                "composite_figi": f"F{j:04d}",
                "pred": float(j),
                "label": float(j) + np.random.default_rng(p * 100 + j).normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_rank_ic_perfect_is_one():
    df = _perfect_panel()
    ic = rank_ic(df, "pred", "label")
    assert len(ic) == df["period"].nunique()
    assert np.allclose(ic.to_numpy(), 1.0, atol=1e-6)


def test_rank_ic_reversed_is_minus_one():
    df = _perfect_panel()
    df["pred"] = -df["pred"]
    ic = rank_ic(df, "pred", "label")
    assert np.allclose(ic.to_numpy(), -1.0, atol=1e-6)


def test_ic_summary_stats():
    ic = pd.Series([0.1, 0.2, 0.0, 0.3, 0.1])
    s = ic_summary(ic)
    assert np.isclose(s["mean_ic"], 0.14)
    assert s["n_periods"] == 5
    # ICIR = mean/std ; t = ICIR*sqrt(n)
    assert np.isclose(s["t_stat"], s["icir"] * np.sqrt(5))
    assert 0 <= s["hit_rate"] <= 1


def test_newey_west_t_known_value_and_overlap_penalty():
    x = np.array([0.1, 0.2, 0.0, 0.3, 0.1])
    # lags=0: mean / sqrt(population variance / n)
    assert np.isclose(newey_west_t(x, 0), x.mean() / np.sqrt(x.var() / len(x)))
    # Overlapping labels -> positively autocorrelated ICs: a smoothed series
    # (moving average of 4, like 12m returns sampled quarterly) must get a
    # smaller NW t than its naive t.
    rng = np.random.default_rng(0)
    smooth = pd.Series(0.02 + rng.normal(0, 0.05, 400)).rolling(4).mean().dropna()
    s = ic_summary(smooth, nw_lags=4)
    assert s["t_stat_nw"] < 0.7 * s["t_stat"]
    assert nw_lags_for(12) == 4 and nw_lags_for(6) == 2


def test_decile_spread_positive_when_pred_orders_labels():
    df = _perfect_panel(n_periods=4, n_names=50)
    spread, summary = decile_spread(df, "pred", "label", n_bins=10)
    assert (spread > 0).all()
    assert summary["mean_ls_spread"] > 0


def test_turnover_bounds_and_known_value():
    # Two periods; top-decile long book fully rotates -> turnover 1.0.
    p1 = pd.Timestamp("2020-03-31")
    p2 = pd.Timestamp("2020-06-30")
    rows = []
    for j in range(10):
        rows.append({"period": p1, "composite_figi": f"A{j}", "pred": float(j)})
    for j in range(10):
        rows.append({"period": p2, "composite_figi": f"B{j}", "pred": float(j)})
    df = pd.DataFrame(rows)
    out = portfolio_turnover(df, "pred", top_quantile=0.1)
    # top 10% of 10 names = 1 name each period; different ids -> full turnover.
    assert out["n_transitions"] == 1
    assert np.isclose(out["mean_top_turnover"], 1.0)
