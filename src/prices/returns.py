"""Forward total returns aligned to formation dates (free price sources).

Prices come from **yfinance** (primary) with a **Stooq** fallback, both free.
Every ticker's adjusted-close series is cached to disk so a resumed run never
re-downloads. Returns use dividend-adjusted close (yfinance ``auto_adjust``), so
they are *total* returns.

Survivorship / delisting bias
-----------------------------
Free feeds carry no delisting returns: a delisted name's series simply ends, so
its forward return becomes ``NaN`` and it drops out. We do **not** paper over
this — :func:`compute_forward_returns` records, per formation date, how many
securities had a valid formation-date price but no price at the horizon
(``dropped_no_forward_price``). Because delistings are disproportionately poor
performers, silently dropping them biases measured returns **optimistically**;
the counts let the write-up quantify the exposure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("breadth13f.prices")


# --- pure return math (unit-tested) ----------------------------------------
def price_asof(
    dates: np.ndarray,
    prices: np.ndarray,
    target: pd.Timestamp,
    max_gap_days: int = 7,
) -> float:
    """First available price on/after ``target`` within ``max_gap_days``.

    Markets are closed on weekends/holidays, so we take the first trading day at
    or after the target. If the nearest available day is more than
    ``max_gap_days`` past the target (e.g. the series ended — a likely
    delisting), return ``NaN``.
    """
    if len(dates) == 0:
        return np.nan
    target = np.datetime64(pd.Timestamp(target), "ns")
    idx = np.searchsorted(dates, target, side="left")
    if idx >= len(dates):
        return np.nan
    gap_days = (dates[idx] - target) / np.timedelta64(1, "D")
    if gap_days > max_gap_days:
        return np.nan
    return float(prices[idx])


def value_asof_before(
    dates: np.ndarray,
    values: np.ndarray,
    target: pd.Timestamp,
    max_staleness_days: int = 10,
) -> float:
    """Last observation on or before ``target`` (never after it — no look-ahead).

    Returns ``NaN`` if there is no observation within ``max_staleness_days``
    before the target (e.g. the series had not started trading yet).
    """
    if len(dates) == 0:
        return np.nan
    t = np.datetime64(pd.Timestamp(target), "ns")
    idx = np.searchsorted(dates, t, side="right") - 1
    if idx < 0 or (t - dates[idx]) / np.timedelta64(1, "D") > max_staleness_days:
        return np.nan
    return float(values[idx])


def forward_total_return(
    dates: np.ndarray,
    prices: np.ndarray,
    formation: pd.Timestamp,
    horizon_months: int,
    max_gap_days: int = 7,
) -> tuple[float, bool, bool]:
    """Return ``(ret, has_formation_price, has_forward_price)``.

    ``ret`` is ``NaN`` unless both endpoint prices exist. The two booleans let the
    caller distinguish "never in the universe here" (no formation price) from
    "dropped out before the horizon" (delisting-type dropout).
    """
    p0 = price_asof(dates, prices, formation, max_gap_days)
    if not np.isfinite(p0) or p0 <= 0:
        return np.nan, False, False
    target = pd.Timestamp(formation) + pd.DateOffset(months=horizon_months)
    p1 = price_asof(dates, prices, target, max_gap_days)
    if not np.isfinite(p1) or p1 <= 0:
        return np.nan, True, False
    return p1 / p0 - 1.0, True, True


# --- price fetching / caching ----------------------------------------------
def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace("\\", "-").upper()
    return cache_dir / f"{safe}.parquet"


def load_price_series(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame | None:
    """Load one ticker's adjusted-close series (date, adj_close), cached.

    Tries yfinance first, then Stooq. Returns None if neither yields data. An
    empty-but-attempted result is cached as an empty frame so we don't re-hit the
    network for a permanently missing ticker.

    The cache is **range-aware**: a companion ``.range.json`` records the window
    that was requested when the series was fetched. If a later call needs a wider
    window than the cache was built for, the series is re-fetched. This prevents
    reusing a narrow series (e.g. one built during a shorter pilot run) and
    silently losing forward prices outside that window.
    """
    path = _cache_path(cache_dir, ticker)
    range_path = path.with_name(path.stem + ".range.json")
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)

    if path.exists() and not force and _range_covers(range_path, req_start, req_end):
        df = pd.read_parquet(path)
        return df if not df.empty else None

    df = _fetch_yfinance(ticker, start, end)
    if df is None or df.empty:
        df = _fetch_stooq(ticker, start, end)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame(columns=["date", "adj_close"])).to_parquet(path, index=False)
    range_path.write_text(
        json.dumps({"start": str(req_start.date()), "end": str(req_end.date())}),
        encoding="utf-8",
    )
    return df if (df is not None and not df.empty) else None


def _range_covers(range_path: Path, req_start: pd.Timestamp, req_end: pd.Timestamp) -> bool:
    """True if a prior fetch already requested a window covering [req_start, req_end]."""
    if not range_path.exists():
        return False
    try:
        r = json.loads(range_path.read_text(encoding="utf-8"))
        return pd.Timestamp(r["start"]) <= req_start and pd.Timestamp(r["end"]) >= req_end
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # multi-column when ticker is odd
            close = close.iloc[:, 0]
        out = close.reset_index()
        out.columns = ["date", "adj_close"]
        out["date"] = pd.to_datetime(out["date"])
        return out.dropna()
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance failed for %s: %s", ticker, exc)
        return None


def _fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        from pandas_datareader import data as pdr

        raw = pdr.DataReader(ticker, "stooq", start=start, end=end)
        if raw is None or raw.empty:
            return None
        out = raw.sort_index()["Close"].reset_index()
        out.columns = ["date", "adj_close"]
        out["date"] = pd.to_datetime(out["date"])
        return out.dropna()
    except Exception as exc:  # noqa: BLE001
        logger.debug("stooq failed for %s: %s", ticker, exc)
        return None


# --- panel-level attachment -------------------------------------------------
def compute_forward_returns(
    panel: pd.DataFrame,
    price_lookup: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = (6, 12),
    max_gap_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach forward returns to the panel and return dropout diagnostics.

    ``price_lookup`` maps ticker -> DataFrame(date, adj_close). Returns
    ``(panel_with_returns, dropout_by_period)`` where the diagnostics count, per
    formation date and horizon, securities with a formation price but no forward
    price (delisting-type dropout).
    """
    panel = panel.copy()
    # Pre-extract numpy arrays per ticker for fast as-of lookups.
    arrays = {
        t: (df["date"].to_numpy(dtype="datetime64[ns]"), df["adj_close"].to_numpy(dtype=float))
        for t, df in price_lookup.items()
        if df is not None and not df.empty
    }
    # Latest date for which any price exists: periods whose horizon extends past
    # this are simply not yet labelable (future), which must not be confused with
    # delisting dropout.
    max_price_date = max((d[-1] for d, _ in arrays.values()), default=np.datetime64("NaT"))

    diag_rows: list[dict] = []
    for h in horizons:
        col = f"fwd_ret_{h}m"
        rets = np.full(len(panel), np.nan)
        had_formation = np.zeros(len(panel), dtype=bool)
        had_forward = np.zeros(len(panel), dtype=bool)

        tickers = panel["ticker"].to_numpy()
        formations = pd.to_datetime(panel["formation_date"]).to_numpy()
        for i in range(len(panel)):
            t = tickers[i]
            if t not in arrays:
                continue
            dates, prices = arrays[t]
            r, hf, hfwd = forward_total_return(dates, prices, pd.Timestamp(formations[i]), h, max_gap_days)
            rets[i] = r
            had_formation[i] = hf
            had_forward[i] = hfwd
        panel[col] = rets
        panel[f"_had_formation_{h}m"] = had_formation

        # Dropout diagnostics per period.
        tmp = pd.DataFrame({
            "period": panel["period"],
            "formation_date": pd.to_datetime(panel["formation_date"]),
            "had_formation": had_formation,
            "had_forward": had_forward,
        })
        grp = tmp.groupby("period").agg(
            formation_date=("formation_date", "first"),
            n_with_formation_price=("had_formation", "sum"),
            n_with_forward_price=("had_forward", "sum"),
        )
        grp["dropped_no_forward_price"] = grp["n_with_formation_price"] - grp["n_with_forward_price"]
        grp["dropout_rate"] = grp["dropped_no_forward_price"] / grp["n_with_formation_price"].replace(0, np.nan)
        grp["horizon_months"] = h
        # A period is labelable only if its horizon end is within available data;
        # future periods (horizon end beyond the last price) are not delisting drop.
        horizon_end = grp["formation_date"] + pd.DateOffset(months=h)
        grp["labelable"] = horizon_end.to_numpy() <= max_price_date
        diag_rows.append(grp.reset_index())

    dropout = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    return panel, dropout
