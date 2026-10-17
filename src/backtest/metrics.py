"""Cross-sectional evaluation metrics for a return-prediction signal.

All metrics are computed **per formation date** and then aggregated across dates,
matching how a cross-sectional signal is actually traded:

* :func:`rank_ic` — Spearman rank correlation between prediction and realized
  forward return, one value per formation date.
* :func:`ic_summary` — mean IC, IC volatility, ICIR (mean/vol), the naive
  t-statistic, a Newey-West t-statistic, hit rate and number of periods.
* :func:`newey_west_t` — t-statistic of a mean, robust to autocorrelation.
* :func:`decile_spread` — top-minus-bottom decile mean-return spread per period.
* :func:`portfolio_turnover` — average one-sided turnover of the top-decile
  long book, security-identity aware.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rank_ic(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    by: str = "period",
) -> pd.Series:
    """Per-period Spearman rank IC between prediction and realized return."""
    out: dict = {}
    for key, grp in df.groupby(by):
        sub = grp[[pred_col, label_col]].dropna()
        if len(sub) < 3 or sub[pred_col].nunique() < 2 or sub[label_col].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(sub[pred_col], sub[label_col])
        out[key] = rho
    return pd.Series(out, name="rank_ic").sort_index()


def newey_west_t(x: pd.Series | np.ndarray, lags: int) -> float:
    """t-statistic of the mean of ``x`` with a Newey-West (Bartlett) variance.

    Forward returns longer than the spacing between formation dates overlap
    (a 12-month return measured every quarter shares 9 months with the next
    one), so consecutive ICs are positively autocorrelated and the naive
    t-statistic overstates the evidence. ``lags=0`` gives the naive t with a
    population (``ddof=0``) variance.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def nw_lags_for(horizon_months: int, spacing_months: int = 3) -> int:
    """Newey-West lags for a label of ``horizon_months`` sampled every
    ``spacing_months``: one per spacing the label spans (12m quarterly -> 4)."""
    return int(np.ceil(horizon_months / spacing_months))


def ic_summary(ic: pd.Series, nw_lags: int = 0) -> dict:
    """Aggregate a per-period IC series into headline statistics.

    ``t_stat`` is the naive t (iid periods); ``t_stat_nw`` uses ``nw_lags``
    Newey-West lags and is the one to trust for overlapping labels.
    """
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"mean_ic": np.nan, "ic_std": np.nan, "icir": np.nan,
                "t_stat": np.nan, "t_stat_nw": np.nan, "hit_rate": np.nan, "n_periods": 0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else np.nan
    icir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
    t_stat = icir * np.sqrt(n) if np.isfinite(icir) else np.nan
    return {
        "mean_ic": mean,
        "ic_std": std,
        "icir": icir,          # per-period information ratio of the IC series
        "t_stat": t_stat,      # H0: mean IC = 0, periods independent
        "t_stat_nw": newey_west_t(ic, nw_lags),   # H0: mean IC = 0, overlap-robust
        "hit_rate": float((ic > 0).mean()),
        "n_periods": n,
    }


def decile_spread(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    by: str = "period",
    n_bins: int = 10,
) -> tuple[pd.Series, dict]:
    """Top-minus-bottom decile realized-return spread per period.

    Returns ``(per_period_spread, summary)``. Bins are formed on the prediction
    within each period; the spread is the mean realized label of the top bin
    minus that of the bottom bin.
    """
    spreads: dict = {}
    for key, grp in df.groupby(by):
        sub = grp[[pred_col, label_col]].dropna()
        if len(sub) < n_bins * 2:
            continue
        try:
            bins = pd.qcut(sub[pred_col].rank(method="first"), n_bins, labels=False)
        except ValueError:
            continue
        top = sub.loc[bins == n_bins - 1, label_col].mean()
        bot = sub.loc[bins == 0, label_col].mean()
        spreads[key] = float(top - bot)
    s = pd.Series(spreads, name="ls_spread").sort_index()
    n = len(s)
    summary = {
        "mean_ls_spread": float(s.mean()) if n else np.nan,
        "ls_spread_std": float(s.std(ddof=1)) if n > 1 else np.nan,
        "ls_t_stat": (float(s.mean() / (s.std(ddof=1) / np.sqrt(n)))
                      if n > 1 and s.std(ddof=1) > 0 else np.nan),
        "n_periods": n,
    }
    return s, summary


def portfolio_turnover(
    df: pd.DataFrame,
    pred_col: str,
    id_col: str = "composite_figi",
    by: str = "period",
    top_quantile: float = 0.1,
) -> dict:
    """Average one-sided turnover of the top-quantile long book across periods."""
    periods = sorted(df[by].unique())
    prev_top: set | None = None
    turns: list[float] = []
    for p in periods:
        grp = df[df[by] == p][[id_col, pred_col]].dropna()
        if grp.empty:
            continue
        k = max(1, int(np.ceil(len(grp) * top_quantile)))
        top = set(grp.nlargest(k, pred_col)[id_col])
        if prev_top is not None and top:
            entering = len(top - prev_top)
            turns.append(entering / len(top))
        prev_top = top
    return {
        "mean_top_turnover": float(np.mean(turns)) if turns else np.nan,
        "n_transitions": len(turns),
    }
