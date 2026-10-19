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
def yahoo_symbol(ticker: str) -> str:
    """The symbol to request from yfinance/Stooq for a panel ticker.

    OpenFIGI writes share classes with a slash (``BRK/B``, ``MOG/A``); Yahoo and
    Stooq use a dash (``BRK-B``). Requested verbatim, these names come back empty
    and silently drop out of the universe. The panel keeps OpenFIGI's form; only
    the request is translated.
    """
    return ticker.strip().upper().replace("/", "-")


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "-").replace("\\", "-").upper()
    return cache_dir / f"{safe}.parquet"


def _read_sidecar(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def fetched_with_current_symbol(sidecar: dict, ticker: str) -> bool:
    """True if a cached series was requested under today's :func:`yahoo_symbol`.

    Sidecars written before symbols were normalised record no symbol: those
    requests used the panel ticker verbatim. For a share-class ticker that was
    the wrong symbol, so its (empty) cache must be re-fetched.
    """
    return sidecar.get("symbol", ticker) == yahoo_symbol(ticker)


def load_price_series(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    force: bool = False,
    data_end: str | None = None,
) -> pd.DataFrame | None:
    """Load one ticker's adjusted-close series (date, adj_close), cached.

    Tries yfinance first, then Stooq. Returns None if neither yields data. An
    empty-but-attempted result is cached as an empty frame so we don't re-hit the
    network for a permanently missing ticker.

    The cache is **range-aware**: a companion ``.range.json`` records the window
    that was requested and the day it was fetched. A cache is reused only if the
    window it actually holds — the requested window, cut at the fetch date —
    covers the new request; otherwise the series is re-fetched. This prevents
    reusing a narrow series (e.g. one built during a shorter pilot run, or one
    fetched months ago) and silently losing forward prices. A re-fetch that comes
    back empty keeps the previous series instead of overwriting it.

    ``data_end`` caps the request and cuts the returned series, so labels do not
    depend on when each cache was filled.

    The cache is keyed by the panel ticker, but the request uses
    :func:`yahoo_symbol` (``BRK/B`` -> ``BRK-B``), which the sidecar also
    records; a cache requested under a different symbol is re-fetched.
    """
    path = _cache_path(cache_dir, ticker)
    range_path = path.with_name(path.stem + ".range.json")
    end = _cap_end(end, data_end)
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)
    symbol = yahoo_symbol(ticker)

    sidecar = _read_sidecar(range_path)
    if (path.exists() and not force and _range_covers(sidecar, req_start, req_end, path)
            and fetched_with_current_symbol(sidecar, ticker)):
        return _cut(pd.read_parquet(path), data_end)

    df = _fetch_yfinance(symbol, start, end)
    if df is None or df.empty:
        df = _fetch_stooq(symbol, start, end)
    if (df is None or df.empty) and path.exists():
        old = pd.read_parquet(path)
        if not old.empty:
            logger.warning("%s: re-fetch returned nothing; keeping cached series", ticker)
            return _cut(old, data_end)

    _write_price_cache(path, range_path, df, req_start, req_end, symbol)
    return _cut(df, data_end)


def prefetch_price_series(tickers, start: str, end: str, cache_dir: Path,
                          data_end: str | None = None, batch: int = 50) -> None:
    """Batch-download every ticker whose cache does not cover ``[start, end]``.

    One yfinance request per ``batch`` tickers instead of one per ticker; the
    cache layout is the one :func:`load_price_series` reads. Tickers a batch
    returns nothing for are left alone, so :func:`load_price_series` still tries
    them singly with its Stooq fallback (and keeps any older series).
    """
    import time

    import yfinance as yf

    end = _cap_end(end, data_end)
    req_start, req_end = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in tickers:
        if not isinstance(t, str) or not t:
            continue
        path = _cache_path(cache_dir, t)
        sidecar = _read_sidecar(path.with_name(path.stem + ".range.json"))
        if not (path.exists() and _range_covers(sidecar, req_start, req_end, path)
                and fetched_with_current_symbol(sidecar, t)):
            todo.append(t)
    logger.info("prices: %d / %d tickers need (re-)fetching", len(todo), len(tickers))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        symbols = {t: yahoo_symbol(t) for t in chunk}
        try:
            raw = yf.download(sorted(set(symbols.values())), start=start, end=end, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001 - left stale, retried singly
            logger.warning("price batch failed (%s): %s", chunk[:3], exc)
            continue
        for t in chunk:
            try:
                sub = raw[symbols[t]] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            if sub is None or "Close" not in sub or not sub["Close"].notna().any():
                continue
            df = pd.DataFrame({"date": pd.to_datetime(sub.index),
                               "adj_close": sub["Close"].to_numpy(dtype=float)}).dropna()
            path = _cache_path(cache_dir, t)
            _write_price_cache(path, path.with_name(path.stem + ".range.json"),
                               df, req_start, req_end, symbols[t])
        logger.info("prices: %d / %d tickers fetched", min(i + batch, len(todo)), len(todo))
        time.sleep(1.0)


def _write_price_cache(path: Path, range_path: Path, df, req_start, req_end, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else pd.DataFrame(columns=["date", "adj_close"])).to_parquet(path, index=False)
    range_path.write_text(
        json.dumps({"start": str(req_start.date()), "end": str(req_end.date()), "symbol": symbol,
                    "fetched": str(pd.Timestamp.now().date())}),
        encoding="utf-8",
    )


def _cap_end(end: str, data_end: str | None) -> str:
    """Request end (exclusive, as yfinance takes it) capped at the day after ``data_end``."""
    if data_end is None:
        return end
    cap = pd.Timestamp(data_end) + pd.Timedelta(days=1)
    return min(pd.Timestamp(end), cap).strftime("%Y-%m-%d")


def _cut(df: pd.DataFrame | None, data_end: str | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if data_end is not None:
        df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(data_end)]
    return df if not df.empty else None


def _range_covers(sidecar: dict, req_start: pd.Timestamp, req_end: pd.Timestamp,
                  path: Path | None = None) -> bool:
    """True if a prior fetch holds data covering [req_start, req_end].

    A fetch cannot hold prices after the day it ran, so the covered end is the
    requested end cut at the fetch date (the sidecar's ``fetched``, or the cache
    file's modification date for sidecars written before it was recorded).
    """
    try:
        covered_end = pd.Timestamp(sidecar["end"])
        fetched = sidecar.get("fetched")
        if fetched is not None:
            covered_end = min(covered_end, pd.Timestamp(fetched))
        elif path is not None and path.exists():
            covered_end = min(covered_end, pd.Timestamp(path.stat().st_mtime, unit="s").normalize())
        return pd.Timestamp(sidecar["start"]) <= req_start and covered_end >= req_end
    except (KeyError, ValueError, TypeError):
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
def price_data_cutoff(series_ends) -> np.datetime64:
    """Last date the price data as a whole covers: the most common series end.

    Caches are fetched at different times, so series end on different dates:
    delisted names earlier, and any ticker fetched after the main run later. The
    latest end would let those few late-fetched names carry labels for a period
    no one else can be scored on; the most common end is the date the bulk of
    the data was fetched up to (ties go to the later date).
    """
    ends = pd.Series(pd.to_datetime(list(series_ends)), dtype="datetime64[ns]")
    if ends.empty:
        return np.datetime64("NaT")
    return ends.mode().max().to_datetime64()


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
    # Periods whose horizon extends past the price data are simply not yet
    # labelable (future), which must not be confused with delisting dropout.
    cutoff = price_data_cutoff([d[-1] for d, _ in arrays.values()])

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
        # Rows whose horizon ends past the cutoff get no label, even if their own
        # series runs further (a cache fetched later than the rest).
        beyond = (pd.DatetimeIndex(formations) + pd.DateOffset(months=h)).to_numpy() > cutoff
        rets[beyond] = np.nan
        had_forward[beyond] = False
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
        # future periods (horizon end beyond the cutoff) are not delisting drop.
        horizon_end = grp["formation_date"] + pd.DateOffset(months=h)
        grp["labelable"] = horizon_end.to_numpy() <= cutoff
        diag_rows.append(grp.reset_index())

    dropout = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    return panel, dropout
