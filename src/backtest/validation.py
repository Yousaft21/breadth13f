"""Validation analytics for the breadth factor.

Supports ``notebooks/validation_questions.ipynb``: decile portfolios, momentum
neutralisation, regime splits, decile stability and cap- vs equal-weighting.

Everything is computed from **out-of-sample** model scores (purged & embargoed
K-fold, see :mod:`src.backtest.cv`) and from the on-disk price cache — no network
access for equities and no refitting beyond the cross-validation itself.

Conventions
-----------
* Deciles are formed per formation date on a model score; **D1 = highest score**,
  D10 = lowest. Constituents are re-formed every quarter.
* Portfolio paths use the 3-month return from one formation date to the next,
  so successive quarterly portfolios never overlap in time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..features.breadth import FEATURE_COLUMNS as BREADTH_FEATURES
from ..prices.baseline import BASELINE_FEATURES, IO_RATIO_BOUNDS, MCAP_MAX  # noqa: F401 (re-exported)
from ..prices.returns import (
    _cache_path,
    forward_total_return,
    load_price_series,
    price_asof,
    price_data_cutoff,
    value_asof_before,
)
from .cv import PurgedKFold
from .model import run_cv_predictions

logger = logging.getLogger("breadth13f.validation")

FULL_FEATURES = BREADTH_FEATURES + BASELINE_FEATURES
N_DECILES = 10


# --- price helpers -----------------------------------------------------------
def momentum_12_1(dates: np.ndarray, prices: np.ndarray, formation: pd.Timestamp) -> float:
    """Skip-month momentum: return from 12 months to 1 month before formation."""
    f = pd.Timestamp(formation)
    p_start = value_asof_before(dates, prices, f - pd.DateOffset(months=12))
    p_end = value_asof_before(dates, prices, f - pd.DateOffset(months=1))
    if not (np.isfinite(p_start) and np.isfinite(p_end)) or p_start <= 0:
        return np.nan
    return p_end / p_start - 1.0


def _load_cached_prices(tickers, price_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in tickers:
        path = _cache_path(price_dir, str(t))
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.sort_values("date")
        out[t] = (df["date"].to_numpy(dtype="datetime64[ns]"), df["adj_close"].to_numpy(dtype=float))
    return out


def attach_price_features(df: pd.DataFrame, price_dir: Path) -> pd.DataFrame:
    """Add ``fwd_ret_3m`` (next-quarter holding return), ``mom_12_1`` and the
    formation-date *adjusted* close ``price_formation`` (right for returns; for
    price levels such as market cap Phase 4 uses as-traded prices instead)."""
    df = df.copy()
    prices = _load_cached_prices(df["ticker"].dropna().unique(), price_dir)
    formations = pd.to_datetime(df["formation_date"]).to_numpy()
    fwd3 = np.full(len(df), np.nan)
    mom = np.full(len(df), np.nan)
    p0 = np.full(len(df), np.nan)
    for ticker, idx in df.groupby("ticker").indices.items():
        if ticker not in prices:
            continue
        dates, px = prices[ticker]
        for i in idx:
            f = pd.Timestamp(formations[i])
            fwd3[i] = forward_total_return(dates, px, f, 3)[0]
            mom[i] = momentum_12_1(dates, px, f)
            p0[i] = price_asof(dates, px, f)
    # Same cutoff as Phase 4: no label from a series fetched later than the rest.
    cutoff = price_data_cutoff([d[-1] for d, _ in prices.values()])
    fwd3[(pd.DatetimeIndex(formations) + pd.DateOffset(months=3)).to_numpy() > cutoff] = np.nan
    df["fwd_ret_3m"] = fwd3
    df["mom_12_1"] = mom
    df["price_formation"] = p0
    return df


# --- the pre-fix look-ahead baseline, for reference ----------------------------
def lookahead_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the size/value features the way Phase 4 did before the fix.

    Market cap = formation-date *adjusted* close (``price_formation``) x the same
    XBRL shares outstanding, with no share-count check. Adjusted closes are
    rescaled for later splits and dividends, so these features leak the future;
    they are kept only to measure what the leak was worth (validation Q7).
    """
    df = df.copy()
    cap = df["price_formation"] * df["shares_outstanding"]
    cap = cap.where(cap > 0)
    df["market_cap_lookahead"] = cap
    df["log_market_cap_lookahead"] = np.log(cap)
    df["book_to_market_lookahead"] = df["book_equity"] / cap
    df["earnings_yield_lookahead"] = df["net_income_annual"] / cap
    return df


LOOKAHEAD_BASELINE_FEATURES = [f"{c}_lookahead" for c in BASELINE_FEATURES]


def build_validation_panel(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Out-of-sample scores + returns + controls, cached to ``data/processed``.

    Scores are produced on identical purged folds (12-month label):

    * ``score_full`` — 13 breadth features + the point-in-time size/value
      features from Phase 4. This is the factor the validation notebook studies;
      it matches the README's full model.
    * ``score_breadth`` — the 13 breadth features only (no price-derived inputs).
    * ``score_baseline`` — size/value only, for incremental-IC comparisons.
    * ``score_full_lookahead`` / ``score_baseline_lookahead`` — the same models
      on the pre-fix, look-ahead size/value features (:func:`lookahead_baseline`),
      kept only to measure the effect of the fix.

    The cache is rebuilt when ``force`` is set or the model panel is newer.
    """
    path = cfg.path("processed") / "validation_scores_12m.parquet"
    panel_path = cfg.path("processed") / "model_panel.parquet"
    if path.exists() and not force and path.stat().st_mtime > panel_path.stat().st_mtime:
        return pd.read_parquet(path)

    panel = pd.read_parquet(panel_path)
    min_holders = int(cfg.raw["universe"]["min_holders"])
    panel = panel[(panel["n_holders_p"] >= min_holders) & panel["fwd_ret_12m"].notna()]
    panel = panel.reset_index(drop=True)

    logger.info("Attaching 3m holding returns and 12-1 momentum from price cache")
    panel = lookahead_baseline(attach_price_features(panel, cfg.path("raw") / "prices"))

    cvcfg = cfg.raw["cv"]
    cv = PurgedKFold(n_splits=int(cvcfg["n_splits"]), horizon_months=12,
                     embargo_days=int(cvcfg["embargo_days"]))
    keys = ["period", "composite_figi"]
    scores = {
        "score_full": FULL_FEATURES,
        "score_breadth": BREADTH_FEATURES,
        "score_baseline": BASELINE_FEATURES,
        "score_full_lookahead": BREADTH_FEATURES + LOOKAHEAD_BASELINE_FEATURES,
        "score_baseline_lookahead": LOOKAHEAD_BASELINE_FEATURES,
    }
    out = panel[keys + ["formation_date", "ticker", "fwd_ret_12m"]].copy()
    for name, feats in scores.items():
        logger.info("Scoring %s OOS (%d features)", name, len(feats))
        pred = run_cv_predictions(panel, feats, "fwd_ret_12m", cv)
        out = out.merge(pred[keys + ["pred"]].rename(columns={"pred": name}), on=keys, how="left")

    keep = ["n_holders_p", "log_total_inst_shares", "log_market_cap", "price_formation",
            "price_traded", "shares_outstanding", "market_cap", "market_cap_lookahead",
            "io_ratio", "cap_ok", "fwd_ret_3m", "mom_12_1"]
    out = out.merge(panel[keys + keep], on=keys, how="left")

    # Which purged test fold scored each quarter (adjacent quarters in different
    # folds were scored by different models).
    fold_of = {pd.Timestamp(d): i for i, dates in enumerate(cv.fold_dates(panel["formation_date"]))
               for d in dates}
    out["cv_fold"] = pd.to_datetime(out["formation_date"]).map(fold_of)
    out.to_parquet(path, index=False)
    return out


# --- portfolio construction --------------------------------------------------
def assign_deciles(df: pd.DataFrame, score_col: str, by: str = "period",
                   n: int = N_DECILES) -> pd.Series:
    """Decile of ``score_col`` within each period; 1 = highest score."""
    def _q(s: pd.Series) -> pd.Series:
        return pd.qcut(s.rank(method="first", ascending=False), n, labels=False) + 1
    return df.groupby(by)[score_col].transform(_q).astype("Int64")


def decile_returns(df: pd.DataFrame, ret_col: str, decile_col: str = "decile",
                   by: str = "period", weight_col: str | None = None) -> pd.DataFrame:
    """Period x decile portfolio returns, equal- or ``weight_col``-weighted."""
    need = [ret_col, decile_col] + ([weight_col] if weight_col else [])
    d = df.dropna(subset=need)
    if weight_col is None:
        out = d.groupby([by, decile_col])[ret_col].mean()
    else:
        w = d[weight_col]
        grp = [d[by], d[decile_col]]
        out = (d[ret_col] * w).groupby(grp).sum() / w.groupby(grp).sum()
    return out.unstack(decile_col).sort_index()


def long_short_stats(r: pd.Series, periods_per_year: int = 4) -> dict:
    """Summary of a dollar-neutral long-short return series (arithmetic, no
    compounding — a short leg can lose more than 100% in a quarter)."""
    r = r.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    cum = r.cumsum()
    return {
        "Ann. return": r.mean() * periods_per_year,
        "Ann. vol": sd * np.sqrt(periods_per_year),
        "Sharpe": r.mean() / sd * np.sqrt(periods_per_year),
        "t-stat": r.mean() / (sd / np.sqrt(n)),
        "Worst quarter": r.min(),
        "Max drawdown": (cum - cum.cummax()).min(),
        "% quarters > 0": (r > 0).mean(),
        "Quarters": n,
    }


def perf_stats(r: pd.Series, periods_per_year: int = 4) -> dict:
    """Annualised summary of a long-only periodic return series (compounded)."""
    r = r.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    wealth = (1 + r).cumprod()
    return {
        "CAGR": wealth.iloc[-1] ** (periods_per_year / n) - 1,
        "Ann. vol": sd * np.sqrt(periods_per_year),
        "Sharpe": r.mean() / sd * np.sqrt(periods_per_year),
        "t-stat": r.mean() / (sd / np.sqrt(n)),
        "Max drawdown": (wealth / wealth.cummax() - 1).min(),
        "% quarters > 0": (r > 0).mean(),
        "Quarters": n,
    }


# --- factor diagnostics ------------------------------------------------------
def neutralize(df: pd.DataFrame, col: str, controls: list[str],
               by: str = "period") -> pd.Series:
    """Per-period residual of rank(``col``) regressed on rank(``controls``).

    The residual is the part of the score that the controls cannot explain; its
    rank IC measures what is left after stripping them out.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(by):
        g = g.dropna(subset=[col, *controls])
        if len(g) < len(controls) + 3:
            continue
        y = g[col].rank(pct=True).to_numpy()
        X = np.column_stack([np.ones(len(g))] + [g[c].rank(pct=True).to_numpy() for c in controls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[g.index] = y - X @ beta
    return out


def decile_flows(df: pd.DataFrame, decile_col: str = "decile", id_col: str = "composite_figi",
                 by: str = "period", n: int = N_DECILES) -> pd.DataFrame:
    """Per consecutive quarter and decile: size, names entering, leaving, retained."""
    periods = sorted(df[by].unique())
    fold = df.groupby(by)["cv_fold"].first() if "cv_fold" in df.columns else None
    snap = {p: df.loc[df[by] == p].set_index(id_col)[decile_col] for p in periods}
    rows = []
    for p0, p1 in zip(periods[:-1], periods[1:]):
        a, b = snap[p0], snap[p1]
        for d in range(1, n + 1):
            A, B = set(a.index[a == d]), set(b.index[b == d])
            rows.append({
                "period": p1, "decile": d, "size": len(B), "prior_size": len(A),
                "entered": len(B - A), "left": len(A - B), "retained": len(A & B),
                "cross_fold": bool(fold is not None and fold[p0] != fold[p1]),
            })
    return pd.DataFrame(rows)


def transition_matrix(df: pd.DataFrame, decile_col: str = "decile", id_col: str = "composite_figi",
                      by: str = "period", n: int = N_DECILES) -> pd.DataFrame:
    """Row-normalised P(decile at t+1 | decile at t), incl. leaving the universe."""
    periods = sorted(df[by].unique())
    cols = list(range(1, n + 1)) + ["Left universe"]
    counts = pd.DataFrame(0.0, index=range(1, n + 1), columns=cols)
    snap = {p: df.loc[df[by] == p].set_index(id_col)[decile_col] for p in periods}
    for p0, p1 in zip(periods[:-1], periods[1:]):
        nxt = snap[p1].reindex(snap[p0].index)
        for d0, d1 in zip(snap[p0].to_numpy(), nxt.to_numpy()):
            counts.loc[int(d0), "Left universe" if pd.isna(d1) else int(d1)] += 1
    return counts.div(counts.sum(axis=1), axis=0)


# --- macro regimes -------------------------------------------------------------
def load_macro_series(cfg: Config, ticker: str, start: str = "2012-01-01",
                      end: str = "2026-12-31") -> pd.DataFrame:
    """Daily macro series (e.g. ``^TNX`` 10y yield) via the cached price loader."""
    df = load_price_series(ticker, start, end, cfg.path("raw") / "macro")
    if df is None:
        raise RuntimeError(f"No data for {ticker}")
    return df.sort_values("date")


def rate_regimes(formations: pd.Series, rates: pd.DataFrame) -> pd.DataFrame:
    """10y-yield regime at each formation date, using data on/before that date.

    * ``trend``: yield higher vs lower than 12 months earlier.
    * ``level``: yield above vs below its sample median.
    """
    dates = rates["date"].to_numpy(dtype="datetime64[ns]")
    vals = rates["adj_close"].to_numpy(dtype=float)
    f = pd.Series(sorted(pd.to_datetime(formations).unique()))
    lvl = f.apply(lambda d: value_asof_before(dates, vals, d))
    prev = f.apply(lambda d: value_asof_before(dates, vals, d - pd.DateOffset(months=12)))
    out = pd.DataFrame({"formation_date": f, "yield_10y": lvl, "change_12m": lvl - prev})
    out["trend"] = np.where(out["change_12m"] > 0, "Rising rates", "Falling rates")
    out["level"] = np.where(out["yield_10y"] > out["yield_10y"].median(), "High rates", "Low rates")
    return out
