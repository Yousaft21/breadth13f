"""Point-in-time size/value baseline features.

The breadth model's headline claim is *incremental* rank IC over a size/value
baseline, so the baseline must be a legitimate control built from the same
point-in-time information set:

* ``log_market_cap`` (size)  = log( formation-date price x shares outstanding )
* ``book_to_market`` (value) = book equity / market cap
* ``earnings_yield`` (value) = trailing annual net income / market cap

Prices are formation-date adjusted closes; fundamentals come from SEC XBRL
selected as-of the formation date (see :mod:`fundamentals`). Rows without a CIK
match or missing inputs get ``NaN`` and are simply absent from the baseline —
coverage is reported so the data loss is explicit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .fundamentals import FundamentalsProvider, _norm_ticker
from .returns import price_asof

logger = logging.getLogger("breadth13f.baseline")


def build_baseline_features(
    panel: pd.DataFrame,
    price_lookup: dict[str, pd.DataFrame],
    provider: FundamentalsProvider,
    ticker_cik: dict[str, int],
    max_gap_days: int = 7,
) -> pd.DataFrame:
    """Compute size/value baseline features for each (period, security) row."""
    arrays = {
        t: (df["date"].to_numpy(dtype="datetime64[ns]"), df["adj_close"].to_numpy(dtype=float))
        for t, df in price_lookup.items()
        if df is not None and not df.empty
    }

    # Cache fundamentals per (cik, formation_date) — many securities repeat.
    fund_cache: dict[tuple[int, pd.Timestamp], dict] = {}

    out_rows: list[dict] = []
    for row in panel[["period", "formation_date", "composite_figi", "ticker"]].itertuples(index=False):
        period, formation, figi, ticker = row
        rec = {"period": period, "composite_figi": figi,
               "log_market_cap": np.nan, "book_to_market": np.nan, "earnings_yield": np.nan}
        cik = ticker_cik.get(_norm_ticker(str(ticker))) if ticker else None
        arr = arrays.get(ticker)
        if cik is None or arr is None:
            out_rows.append(rec)
            continue

        dates, prices = arr
        price = price_asof(dates, prices, pd.Timestamp(formation), max_gap_days)
        if not np.isfinite(price) or price <= 0:
            out_rows.append(rec)
            continue

        key = (cik, pd.Timestamp(formation))
        if key not in fund_cache:
            fund_cache[key] = provider.get_asof(cik, pd.Timestamp(formation))
        f = fund_cache[key]

        shares = f.get("shares_outstanding")
        if shares and shares > 0:
            mktcap = price * shares
            rec["log_market_cap"] = float(np.log(mktcap))
            eq = f.get("book_equity")
            if eq is not None:
                rec["book_to_market"] = float(eq / mktcap)
            ni = f.get("net_income_annual")
            if ni is not None:
                rec["earnings_yield"] = float(ni / mktcap)
        out_rows.append(rec)

    baseline = pd.DataFrame(out_rows)
    cov = {
        "rows": len(baseline),
        "with_size": int(baseline["log_market_cap"].notna().sum()),
        "with_book_to_market": int(baseline["book_to_market"].notna().sum()),
        "with_earnings_yield": int(baseline["earnings_yield"].notna().sum()),
    }
    logger.info("Baseline coverage: %s", cov)
    return baseline
