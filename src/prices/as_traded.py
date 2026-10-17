"""As-traded closing prices, for point-in-time price *levels* (market cap).

Forward returns use yfinance's auto-adjusted close (see :mod:`returns`), which is
right for returns but wrong for levels: an adjusted series is rescaled backwards
for every *later* split and dividend, so the price it shows for a past date
depends on events after that date. A market cap built from it understates the
size of any stock that later split (AMZN by 20x, NVDA by 39x at end-2019) — a
look-ahead, since stocks tend to split after rising.

Here we load yfinance's ``Close`` with ``auto_adjust=False`` (not adjusted for
dividends, but still back-adjusted for splits) together with the split history,
and undo the splits that happened after each date. The result is the price the
stock actually traded at, using only information available on that date.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .returns import _cache_path, value_asof_before

logger = logging.getLogger("breadth13f.as_traded")


def load_raw_closes(tickers, cache_dir: Path, start: str = "2013-01-01",
                    end: str | None = None, batch: int = 50) -> dict[str, pd.DataFrame]:
    """Closes *not* adjusted for dividends, plus split events, cached per ticker.

    Returns ``ticker -> DataFrame(date, close, split)``; ``split`` is the split
    ratio on its ex-date (4.0 for a 4-for-1) and 0 otherwise. Tickers are fetched
    in batches; a ticker yfinance has no data for is cached as an empty frame so
    it is not re-requested, while a failed batch is left uncached and retried on
    the next run.

    ``end`` defaults to today: yfinance back-adjusts ``Close`` for every split up
    to the download date, so the split history must run to the same date for
    :func:`as_traded_prices` to undo all of them.
    """
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = lambda t: _cache_path(cache_dir, str(t))  # noqa: E731
    todo = [t for t in tickers if not path(t).exists()]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            raw = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001 - leave uncached, retried next run
            logger.warning("raw-close batch failed (%s): %s", chunk[:3], exc)
            continue
        for t in chunk:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                sub = None
            df = pd.DataFrame(columns=["date", "close", "split"])
            if sub is not None and "Close" in sub and sub["Close"].notna().any():
                df = pd.DataFrame({
                    "date": pd.to_datetime(sub.index),
                    "close": sub["Close"].to_numpy(dtype=float),
                    "split": (sub["Stock Splits"] if "Stock Splits" in sub else 0.0),
                }).dropna(subset=["close"])
                df["split"] = df["split"].fillna(0.0).astype(float)
            df.to_parquet(path(t), index=False)
        logger.info("raw closes: %d / %d tickers fetched", min(i + batch, len(todo)), len(todo))
        time.sleep(1.0)
    out = {}
    for t in tickers:
        if path(t).exists():
            df = pd.read_parquet(path(t))
            if not df.empty:
                out[t] = df.sort_values("date")
    return out


def as_traded_prices(raw: pd.DataFrame, whens, max_staleness_days: int = 10) -> np.ndarray:
    """As-traded price on/before each date in ``whens``: undo splits made later.

    ``raw`` is one ticker's ``(date, close, split)`` frame from
    :func:`load_raw_closes`, sorted by date. The close is the last one on or
    before the date (never after it); it is multiplied by every split dated
    strictly after the date. ``NaN`` where no close falls within
    ``max_staleness_days`` before the date.
    """
    dates = raw["date"].to_numpy(dtype="datetime64[ns]")
    close = raw["close"].to_numpy(dtype=float)
    split = raw["split"].to_numpy(dtype=float)
    # later[k] = product of the split ratios on rows k..end; later[n] = 1.
    later = np.append(np.cumprod(np.where(split > 0, split, 1.0)[::-1])[::-1], 1.0)
    out = np.full(len(whens), np.nan)
    for j, when in enumerate(whens):
        p = value_asof_before(dates, close, pd.Timestamp(when), max_staleness_days)
        if np.isfinite(p):
            k = np.searchsorted(dates, np.datetime64(pd.Timestamp(when), "ns"), side="right")
            out[j] = p * later[k]
    return out


def as_traded_price(raw: pd.DataFrame, when: pd.Timestamp) -> float:
    """Scalar version of :func:`as_traded_prices`."""
    return float(as_traded_prices(raw, [when])[0])
