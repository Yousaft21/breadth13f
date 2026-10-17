"""Point-in-time size/value baseline features.

The breadth model's headline claim is *incremental* rank IC over a size/value
baseline, so the baseline must be a legitimate control built from the same
point-in-time information set:

* ``log_market_cap`` (size)  = log( formation-date price x shares outstanding )
* ``book_to_market`` (value) = book equity / market cap
* ``earnings_yield`` (value) = trailing annual net income / market cap

The price is the **as-traded** close on or before the formation date (see
:mod:`as_traded`) — not the adjusted close used for returns, which is rescaled
for later splits and dividends and would leak the future into the cap.
Fundamentals come from SEC XBRL selected as-of the formation date (see
:mod:`fundamentals`).

SEC share counts occasionally carry unit/scale errors, and for ADRs they count
ordinary shares while the price is per ADR. A cap is therefore trusted only if
(a) the implied institutional-ownership ratio — 13F shares held / XBRL shares
outstanding, a price-free cross-check of the share count — is in
``IO_RATIO_BOUNDS`` and (b) the cap is below ``MCAP_MAX``. Rows that fail, or
lack a CIK match or inputs, get ``NaN`` features (LightGBM treats them as
missing); coverage is logged so the data loss is explicit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .as_traded import as_traded_prices
from .fundamentals import FundamentalsProvider

logger = logging.getLogger("breadth13f.baseline")

BASELINE_FEATURES = ["log_market_cap", "book_to_market", "earnings_yield"]

MCAP_MAX = 6e12
IO_RATIO_BOUNDS = (0.01, 3.0)


def market_cap_ok(market_cap, io_ratio) -> np.ndarray:
    """True where the cap and the share count behind it pass the sanity checks."""
    lo, hi = IO_RATIO_BOUNDS
    io = np.asarray(io_ratio, dtype=float)
    cap = np.asarray(market_cap, dtype=float)
    return (io >= lo) & (io <= hi) & (cap > 0) & (cap <= MCAP_MAX)


def build_baseline_features(
    panel: pd.DataFrame,
    raw_closes: dict[str, pd.DataFrame],
    provider: FundamentalsProvider,
    ciks,
) -> pd.DataFrame:
    """Size/value baseline features and their audit inputs per (period, security).

    ``raw_closes`` maps ticker -> ``(date, close, split)`` from
    :func:`as_traded.load_raw_closes`. ``ciks`` holds each panel row's issuer
    CIK (missing where unmapped), from the point-in-time map in
    :mod:`src.mapping.issuer_cik`. Besides the three features, the output
    keeps the inputs (``price_traded``, ``shares_outstanding``, ``book_equity``,
    ``net_income_annual``), the unfiltered ``market_cap``, the ``io_ratio`` and
    the ``cap_ok`` flag, so every feature value can be audited.
    """
    n = len(panel)
    formations = pd.to_datetime(panel["formation_date"]).to_numpy()
    ciks = pd.Series(ciks, dtype="Int64").to_numpy(dtype=object, na_value=None)

    price = np.full(n, np.nan)
    for ticker, idx in panel.groupby("ticker").indices.items():
        raw = raw_closes.get(ticker)
        if raw is not None:
            price[idx] = as_traded_prices(raw, formations[idx])

    # Cache fundamentals per (cik, formation_date) — many securities repeat.
    fund_cache: dict[tuple[int, pd.Timestamp], dict] = {}
    shares = np.full(n, np.nan)
    equity = np.full(n, np.nan)
    income = np.full(n, np.nan)
    for i in range(n):
        cik = ciks[i]
        if cik is None:
            continue
        cik = int(cik)
        key = (cik, pd.Timestamp(formations[i]))
        if key not in fund_cache:
            fund_cache[key] = provider.get_asof(cik, key[1])
        f = fund_cache[key]
        if f.get("shares_outstanding") and f["shares_outstanding"] > 0:
            shares[i] = f["shares_outstanding"]
        for arr, name in ((equity, "book_equity"), (income, "net_income_annual")):
            if f.get(name) is not None:
                arr[i] = f[name]

    mcap = price * shares
    io_ratio = np.exp(panel["log_total_inst_shares"].to_numpy(dtype=float)) / shares
    ok = market_cap_ok(mcap, io_ratio)
    trusted = np.where(ok, mcap, np.nan)

    baseline = pd.DataFrame({
        "period": panel["period"].to_numpy(),
        "composite_figi": panel["composite_figi"].to_numpy(),
        "log_market_cap": np.log(trusted),
        "book_to_market": equity / trusted,
        "earnings_yield": income / trusted,
        "price_traded": price,
        "shares_outstanding": shares,
        "book_equity": equity,
        "net_income_annual": income,
        "market_cap": mcap,
        "io_ratio": io_ratio,
        "cap_ok": ok,
    })
    has_cap = np.isfinite(mcap)
    cov = {
        "rows": n,
        "with_shares": int(np.isfinite(shares).sum()),
        "with_price_traded": int(np.isfinite(price).sum()),
        "with_cap": int(has_cap.sum()),
        "cap_failed_checks": int((has_cap & ~ok).sum()),
        "with_size": int(baseline["log_market_cap"].notna().sum()),
        "with_book_to_market": int(baseline["book_to_market"].notna().sum()),
        "with_earnings_yield": int(baseline["earnings_yield"].notna().sum()),
    }
    logger.info("Baseline coverage: %s", cov)
    return baseline
