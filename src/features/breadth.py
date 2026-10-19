"""Point-in-time breadth-feature panel from consolidated 13F holdings.

For each formation date (the first business day after a quarter's 13F filing
deadline) we build a security-level feature vector from two consecutive
quarterly *snapshots* of institutional holdings. All features derive from
quarter-over-quarter changes in the set of institutional holders and their
positions, following the breadth intuition of Chen, Hong & Stein (2002).

Point-in-time discipline
------------------------
A snapshot for period ``p`` uses only filings whose ``period_of_report == p`` and
that were accepted by the end of the period's filing deadline
(:func:`snapshot_cutoff`). The deadline is 45 days after quarter-end, moved to
the next business day when that falls on a weekend or federal holiday (Exchange
Act Rule 0-3). About a quarter of managers file on the deadline day itself, so
a fixed calendar lag drops them whenever the deadline moves. The signal is
formed and traded on the next business day (:func:`formation_date`). Features
at formation date ``F_p`` compare snapshot(p) against snapshot(p-1); because
``F_{p-1} < F_p`` the prior snapshot was already known, so no future
information leaks in.

Continuing managers
-------------------
Features that compare the two snapshots (Δbreadth, new/exit rates, net buyers,
ΔHHI, Δ institutional shares) use only managers that filed in **both**
snapshots, as in Chen, Hong & Stein (2002). A manager missing from one snapshot
(it filed late, or started or stopped filing) would otherwise turn all its
positions into spurious initiations or exits. Level features (breadth,
concentration, total institutional shares) use every period-``p`` filer.

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
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

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


# SEC business days: weekdays that are not US federal holidays.
SEC_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def filing_deadline(period: pd.Timestamp, deadline_days: int = 45) -> pd.Timestamp:
    """13F due date: ``deadline_days`` after quarter-end, rolled to a business day."""
    due = pd.Timestamp(period).normalize() + pd.Timedelta(days=deadline_days)
    return SEC_BUSINESS_DAY.rollforward(due)


def snapshot_cutoff(period: pd.Timestamp, deadline_days: int = 45) -> pd.Timestamp:
    """Latest acceptance time in a period's snapshot: the end of its deadline day."""
    return filing_deadline(period, deadline_days) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def formation_date(period: pd.Timestamp, deadline_days: int = 45) -> pd.Timestamp:
    """Formation (trade) date: the first business day after the filing deadline."""
    return filing_deadline(period, deadline_days) + SEC_BUSINESS_DAY


def consolidate_snapshot(
    holdings: pd.DataFrame,
    period: pd.Timestamp,
    as_of: pd.Timestamp,
    secmaster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index]:
    """Consolidate one period's holdings as known by ``as_of``.

    Returns ``(positions, managers)`` where ``positions`` has one row per
    ``(manager_cik, composite_figi)`` with summed shares/value, restricted to US
    common equity, and ``managers`` holds the distinct managers that filed long
    positions for the period by ``as_of`` (its length is the breadth
    denominator).
    """
    df = holdings[
        (holdings["period_of_report"] == period)
        & (holdings["acceptance_datetime"] <= as_of)
        & (holdings["put_call"].isna())        # long equity only, exclude options
        & (holdings["shares"] > 0)
    ].copy()
    if df.empty:
        return df.assign(composite_figi=pd.Series(dtype=str)), pd.Index([])

    # Distinct managers that filed for this period by as_of (all filers).
    managers = pd.Index(df["manager_cik"].unique())

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
    return positions, managers


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
    managers_p: pd.Index,
    managers_pm1: pd.Index,
) -> pd.DataFrame:
    """Per-security breadth features comparing snapshot(p) to snapshot(p-1).

    Level features use every manager that filed for ``p``; change features use
    only the managers in both ``managers_p`` and ``managers_pm1`` (see the
    module docstring). ``n_holders_p`` / ``n_holders_pm1`` count all holders.
    """
    keep = ["composite_figi", "ticker", "n_holders_p", "n_holders_pm1", *FEATURE_COLUMNS]
    if pos_p.empty:
        return pd.DataFrame(columns=keep)
    continuing = pd.Index(managers_p).intersection(pd.Index(managers_pm1))
    n_active_p = len(managers_p)
    n_cont = len(continuing)

    # Outer-join holders of each security across the two snapshots.
    merged = pos_p.merge(
        pos_pm1[["manager_cik", "composite_figi", "shares"]],
        on=["manager_cik", "composite_figi"],
        how="outer",
        suffixes=("_p", "_pm1"),
    )
    merged["in_p"] = merged["shares_p"].notna()
    merged["in_pm1"] = merged["shares_pm1"].notna()
    is_cont = merged["manager_cik"].isin(continuing)
    merged["in_p_c"] = merged["in_p"] & is_cont
    merged["in_pm1_c"] = merged["in_pm1"] & is_cont
    held_both = merged["in_p"] & merged["in_pm1"]   # implies a continuing manager
    merged["is_new"] = merged["in_p_c"] & ~merged["in_pm1"]
    merged["is_exit"] = merged["in_pm1_c"] & ~merged["in_p"]
    merged["is_incr"] = held_both & (merged["shares_p"] > merged["shares_pm1"])
    merged["is_decr"] = held_both & (merged["shares_p"] < merged["shares_pm1"])
    merged["shares_p_c"] = merged["shares_p"].where(merged["in_p_c"], 0.0)
    merged["shares_pm1_c"] = merged["shares_pm1"].where(merged["in_pm1_c"], 0.0)

    agg = merged.groupby("composite_figi").agg(
        n_holders_p=("in_p", "sum"),
        n_holders_pm1=("in_pm1", "sum"),
        n_p_c=("in_p_c", "sum"),
        n_pm1_c=("in_pm1_c", "sum"),
        n_new=("is_new", "sum"),
        n_exit=("is_exit", "sum"),
        n_incr=("is_incr", "sum"),
        n_decr=("is_decr", "sum"),
        shares_p_c=("shares_p_c", "sum"),
        shares_pm1_c=("shares_pm1_c", "sum"),
    )
    # HHI among continuing managers in each snapshot.
    agg["hhi_p_c"] = _grouped_hhi(merged, "shares_p_c")
    agg["hhi_pm1_c"] = _grouped_hhi(merged, "shares_pm1_c")

    conc_p = _per_security_concentration(pos_p).set_index("composite_figi")
    out = agg.join(conc_p, how="left")

    n_holders_p = out["n_holders_p"].astype(float)
    n_p_c = out["n_p_c"].astype(float)
    n_pm1_c = out["n_pm1_c"].astype(float)

    out["breadth_level"] = n_holders_p / n_active_p if n_active_p else np.nan
    # Chen-Hong-Stein: change in holders among continuing managers, per continuing manager.
    out["delta_breadth"] = (n_p_c - n_pm1_c) / n_cont if n_cont else np.nan
    out["delta_breadth_pct"] = (n_p_c - n_pm1_c) / n_pm1_c.replace(0, np.nan)
    out["net_buyer_breadth"] = (out["n_incr"] - out["n_decr"]) / n_p_c.replace(0, np.nan)
    out["new_position_rate"] = out["n_new"] / n_p_c.replace(0, np.nan)
    out["exit_rate"] = out["n_exit"] / n_pm1_c.replace(0, np.nan)
    out["delta_hhi"] = out["hhi_p_c"] - out["hhi_pm1_c"]
    out["log_total_inst_shares"] = np.log(out["total_inst_shares"].replace(0, np.nan))
    out["delta_total_inst_shares_pct"] = (
        (out["shares_p_c"] - out["shares_pm1_c"]) / out["shares_pm1_c"].replace(0, np.nan)
    )

    # Attach a representative ticker per security (from snapshot p).
    tickers = pos_p.groupby("composite_figi")["ticker"].first()
    out = out.join(tickers.rename("ticker"), how="left")

    out = out.reset_index()
    return out[keep]


def _grouped_hhi(merged: pd.DataFrame, shares_col: str) -> pd.Series:
    """Per-security Herfindahl of ``shares_col`` (NaN where the total is zero)."""
    total = merged.groupby("composite_figi")[shares_col].transform("sum")
    w2 = (merged[shares_col] / total.where(total > 0)) ** 2
    hhi = w2.groupby(merged["composite_figi"]).sum(min_count=1)
    return hhi


def build_feature_panel(
    holdings: pd.DataFrame | None,
    secmaster: pd.DataFrame,
    deadline_days: int = 45,
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
    prev_managers: pd.Index = pd.Index([])
    prev_period: pd.Timestamp | None = None

    for period in period_list:
        as_of = snapshot_cutoff(period, deadline_days)
        pos_p, managers_p = consolidate_snapshot(get_holdings(period), period, as_of, secmaster)
        if prev_positions is not None and prev_period is not None:
            cur_n = pos_p["composite_figi"].nunique() if not pos_p.empty else 0
            prev_n = prev_positions["composite_figi"].nunique() if not prev_positions.empty else 0
            adequately_covered = prev_n >= min_prev_coverage_ratio * max(cur_n, 1)
            # Only pair truly consecutive, adequately-covered quarters.
            if (period - prev_period).days <= 100 and adequately_covered:
                feats = compute_pair_features(pos_p, prev_positions, managers_p, prev_managers)
                n_cont = len(managers_p.intersection(prev_managers))
                feats.insert(0, "period", period)
                feats.insert(1, "formation_date", formation_date(period, deadline_days))
                feats.insert(2, "n_active_managers", len(managers_p))
                feats.insert(3, "n_continuing_managers", n_cont)
                panels.append(feats)
                logger.info(
                    "%s: %d securities, %d active managers (%d continuing)",
                    period.date(), len(feats), len(managers_p), n_cont,
                )
        prev_positions, prev_managers, prev_period = pos_p, managers_p, period

    if not panels:
        return pd.DataFrame(columns=["period", "formation_date", "n_active_managers",
                                     "n_continuing_managers", "composite_figi", "ticker",
                                     "n_holders_p", "n_holders_pm1", *FEATURE_COLUMNS])
    return pd.concat(panels, ignore_index=True)
