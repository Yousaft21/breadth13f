"""Point-in-time breadth-feature panel from consolidated 13F holdings.

For each formation date (a quarter-end plus the filing lag) we build a
security-level feature vector from two consecutive quarterly *snapshots* of
institutional holdings. All features derive from quarter-over-quarter changes in
the set of institutional holders and their positions, following the breadth
intuition of Chen, Hong & Stein (2002).

Point-in-time discipline
------------------------
A snapshot for period ``p`` uses only filings whose ``period_of_report == p`` and
whose ``acceptance_datetime <= formation_date(p)``, where
``formation_date(p) = p + filing_lag_days``. Each quarter's holdings are thus
frozen at a fixed lag after its own quarter-end — the information a real-time
investor would have just after the filing deadline. Features at formation date
``F_p`` compare snapshot(p) against snapshot(p-1); because ``F_{p-1} < F_p`` the
prior snapshot was already known, so no future information leaks in.

Amendment handling
------------------
Within a snapshot, a manager's holdings for a period come from its
**latest-accepted filing** for that period (a 13F-HR/A restatement supersedes the
original). This is the standard convention and handles the dominant restatement
case; additive "NEW HOLDINGS" amendments are rare and treated the same way (a
documented simplification).

Units note
----------
Concentration features use **share counts**, not the ``value`` field, so they are
unaffected by the 2023 change in 13F value units (thousands -> whole USD).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("breadth13f.features")

FEATURE_COLUMNS = [
    "breadth_level",
    "delta_breadth",
    "delta_breadth_pct",
    "net_buyer_breadth",
    "new_position_rate",
    "exit_rate",
    "hhi",
    "delta_hhi",
    "top5_share",
    "top10_share",
    "log_total_inst_shares",
    "delta_total_inst_shares_pct",
    "position_size_dispersion",
]


def formation_date(period: pd.Timestamp, filing_lag_days: int) -> pd.Timestamp:
    """Formation (as-of) date for a period-of-report quarter-end."""
    return pd.Timestamp(period) + pd.Timedelta(days=filing_lag_days)


def consolidate_snapshot(
    holdings: pd.DataFrame,
    period: pd.Timestamp,
    as_of: pd.Timestamp,
    secmaster: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Consolidate one period's holdings as known by ``as_of``.

    Returns ``(positions, n_active_managers)`` where ``positions`` has one row per
    ``(manager_cik, composite_figi)`` with summed shares/value, restricted to US
    common equity, and ``n_active_managers`` is the number of distinct managers
    that filed for the period by ``as_of`` (the breadth denominator).
    """
    df = holdings[
        (holdings["period_of_report"] == period)
        & (holdings["acceptance_datetime"] <= as_of)
        & (holdings["put_call"].isna())        # long equity only, exclude options
        & (holdings["shares"] > 0)
    ].copy()
    if df.empty:
        return df.assign(composite_figi=pd.Series(dtype=str)), 0

    # Distinct managers that filed for this period by as_of (all filers).
    n_active_managers = int(df["manager_cik"].nunique())

    # Latest-accepted filing per manager supersedes earlier ones (amendments).
    latest_acc = (
        df.groupby("manager_cik")["acceptance_datetime"].transform("max")
    )
    df = df[df["acceptance_datetime"] == latest_acc]

    # Map CUSIP -> tradable security identity, keep only US common equity.
    sm = secmaster[secmaster["is_us_common_equity"]][["cusip", "composite_figi", "ticker"]]
    df = df.merge(sm, on="cusip", how="inner")

    positions = (
        df.groupby(["manager_cik", "composite_figi", "ticker"], as_index=False)
        .agg(shares=("shares", "sum"), value=("value", "sum"))
    )
    return positions, n_active_managers


def _herfindahl(shares: np.ndarray) -> float:
    total = shares.sum()
    if total <= 0:
        return np.nan
    w = shares / total
    return float(np.sum(w * w))


def _top_k_share(shares: np.ndarray, k: int) -> float:
    total = shares.sum()
    if total <= 0:
        return np.nan
    top = np.sort(shares)[::-1][:k].sum()
    return float(top / total)


def _per_security_concentration(pos_p: pd.DataFrame) -> pd.DataFrame:
    """HHI, top-5/10 share, dispersion and total shares per security (period p)."""
    rows = []
    for figi, grp in pos_p.groupby("composite_figi"):
        shares = grp["shares"].to_numpy(dtype=float)
        total = shares.sum()
        # Dispersion: std of log position sizes across holders (0 if single holder).
        disp = float(np.std(np.log(shares[shares > 0]))) if (shares > 0).sum() > 1 else 0.0
        rows.append(
            {
                "composite_figi": figi,
                "hhi": _herfindahl(shares),
                "top5_share": _top_k_share(shares, 5),
                "top10_share": _top_k_share(shares, 10),
                "total_inst_shares": total,
                "position_size_dispersion": disp,
            }
        )
    return pd.DataFrame(rows)


def compute_pair_features(
    pos_p: pd.DataFrame,
    pos_pm1: pd.DataFrame,
    n_active_p: int,
) -> pd.DataFrame:
    """Per-security breadth features comparing snapshot(p) to snapshot(p-1)."""
    if pos_p.empty:
        return pd.DataFrame(columns=["composite_figi", "ticker", *FEATURE_COLUMNS])

    # Outer-join holders of each security across the two snapshots.
    merged = pos_p.merge(
        pos_pm1[["manager_cik", "composite_figi", "shares"]],
        on=["manager_cik", "composite_figi"],
        how="outer",
        suffixes=("_p", "_pm1"),
    )
    merged["in_p"] = merged["shares_p"].notna()
    merged["in_pm1"] = merged["shares_pm1"].notna()
    cont = merged["in_p"] & merged["in_pm1"]
    merged["is_new"] = merged["in_p"] & ~merged["in_pm1"]
    merged["is_exit"] = ~merged["in_p"] & merged["in_pm1"]
    merged["is_incr"] = cont & (merged["shares_p"] > merged["shares_pm1"])
    merged["is_decr"] = cont & (merged["shares_p"] < merged["shares_pm1"])

    agg = merged.groupby("composite_figi").agg(
        n_holders_p=("in_p", "sum"),
        n_holders_pm1=("in_pm1", "sum"),
        n_new=("is_new", "sum"),
        n_exit=("is_exit", "sum"),
        n_incr=("is_incr", "sum"),
        n_decr=("is_decr", "sum"),
    )

    conc_p = _per_security_concentration(pos_p).set_index("composite_figi")
    conc_pm1 = (
        _per_security_concentration(pos_pm1).set_index("composite_figi")
        if not pos_pm1.empty else pd.DataFrame(columns=["hhi", "total_inst_shares"])
    )

    out = agg.join(conc_p, how="left")
    out["hhi_pm1"] = conc_pm1["hhi"] if "hhi" in conc_pm1 else np.nan
    out["total_inst_shares_pm1"] = (
        conc_pm1["total_inst_shares"] if "total_inst_shares" in conc_pm1 else np.nan
    )

    n_holders_p = out["n_holders_p"].astype(float)
    n_holders_pm1 = out["n_holders_pm1"].astype(float)

    out["breadth_level"] = n_holders_p / n_active_p if n_active_p else np.nan
    out["delta_breadth"] = n_holders_p - n_holders_pm1
    out["delta_breadth_pct"] = out["delta_breadth"] / n_holders_pm1.replace(0, np.nan)
    out["net_buyer_breadth"] = (out["n_incr"] - out["n_decr"]) / n_holders_p.replace(0, np.nan)
    out["new_position_rate"] = out["n_new"] / n_holders_p.replace(0, np.nan)
    out["exit_rate"] = out["n_exit"] / n_holders_pm1.replace(0, np.nan)
    out["delta_hhi"] = out["hhi"] - out["hhi_pm1"]
    out["log_total_inst_shares"] = np.log(out["total_inst_shares"].replace(0, np.nan))
    out["delta_total_inst_shares_pct"] = (
        (out["total_inst_shares"] - out["total_inst_shares_pm1"])
        / out["total_inst_shares_pm1"].replace(0, np.nan)
    )

    # Attach a representative ticker per security (from snapshot p).
    tickers = pos_p.groupby("composite_figi")["ticker"].first()
    out = out.join(tickers.rename("ticker"), how="left")

    out = out.reset_index()
    keep = ["composite_figi", "ticker", "n_holders_p", "n_holders_pm1", *FEATURE_COLUMNS]
    return out[keep]


def build_feature_panel(
    holdings: pd.DataFrame | None,
    secmaster: pd.DataFrame,
    filing_lag_days: int,
    min_period: pd.Timestamp | None = None,
    min_prev_coverage_ratio: float = 0.25,
    holdings_loader=None,
    periods=None,
) -> pd.DataFrame:
    """Build the full security x formation-date breadth panel.

    Iterates over consecutive period-of-report quarters, builds each snapshot at
    its own formation date, and emits per-security features tagged with
    ``period``, ``formation_date`` and holder counts for universe filtering.

    Two input modes:

    * **In-memory** — pass the full ``holdings`` DataFrame (used in tests).
    * **Lazy/streaming** — pass ``holdings_loader`` (a callable ``period ->
      DataFrame``) and the explicit ``periods`` to iterate. Because a period's
      point-in-time filings all live in a single filing-quarter file, this keeps
      only one quarter in memory at a time (essential for the full 12-year run).

    A period is only emitted when its prior quarter is genuinely populated
    (prior securities >= ``min_prev_coverage_ratio`` x current). This drops the
    burn-in period at the start of any crawl window, whose prior quarter's
    filings were never crawled and would otherwise make every holder look newly
    initiated.
    """
    if holdings_loader is not None:
        if periods is None:
            raise ValueError("periods must be given when using holdings_loader")
        period_list = sorted(pd.Timestamp(p) for p in periods)
        get_holdings = holdings_loader
    else:
        period_list = sorted(pd.to_datetime(holdings["period_of_report"].dropna().unique()))
        get_holdings = lambda _p: holdings  # noqa: E731 - same frame for every period
    # Restrict to standard calendar quarter-ends (13F periods always are).
    period_list = [p for p in period_list if p.is_quarter_end]
    if min_period is not None:
        period_list = [p for p in period_list if p >= min_period]

    panels: list[pd.DataFrame] = []
    prev_positions: pd.DataFrame | None = None
    prev_period: pd.Timestamp | None = None

    for period in period_list:
        as_of = formation_date(period, filing_lag_days)
        pos_p, n_active = consolidate_snapshot(get_holdings(period), period, as_of, secmaster)
        if prev_positions is not None and prev_period is not None:
            cur_n = pos_p["composite_figi"].nunique() if not pos_p.empty else 0
            prev_n = prev_positions["composite_figi"].nunique() if not prev_positions.empty else 0
            adequately_covered = prev_n >= min_prev_coverage_ratio * max(cur_n, 1)
            # Only pair truly consecutive, adequately-covered quarters.
            if (period - prev_period).days <= 100 and adequately_covered:
                feats = compute_pair_features(pos_p, prev_positions, n_active)
                feats.insert(0, "period", period)
                feats.insert(1, "formation_date", as_of)
                feats.insert(2, "n_active_managers", n_active)
                panels.append(feats)
                logger.info(
                    "%s: %d securities, %d active managers",
                    period.date(), len(feats), n_active,
                )
        prev_positions, prev_period = pos_p, period

    if not panels:
        return pd.DataFrame(columns=["period", "formation_date", "n_active_managers",
                                     "composite_figi", "ticker", *FEATURE_COLUMNS])
    return pd.concat(panels, ignore_index=True)
