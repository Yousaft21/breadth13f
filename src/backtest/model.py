"""Gradient-boosted return-rank model under purged/embargoed CV.

The model predicts the **cross-sectional rank** of forward returns: within each
formation date the label is rank-standardised, so the learner focuses on the
ordering that rank IC actually measures rather than on absolute return levels.

Out-of-sample predictions are assembled fold-by-fold using :class:`PurgedKFold`
so no test observation shares future information with its training set. The same
folds are used for the full breadth model and the size/value baseline, making the
**incremental IC** comparison apples-to-apples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cv import PurgedKFold
from .metrics import decile_spread, ic_summary, portfolio_turnover, rank_ic

logger = logging.getLogger("breadth13f.model")

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 13,
    "n_jobs": -1,
    "verbose": -1,
}


def cross_sectional_rank_target(df: pd.DataFrame, label_col: str, by: str = "period") -> pd.Series:
    """Within-period rank of the label, standardised to ~N(0,1)."""
    def _z(x: pd.Series) -> pd.Series:
        r = x.rank()
        return (r - r.mean()) / (r.std(ddof=0) or 1.0)
    return df.groupby(by)[label_col].transform(_z)


@dataclass
class BacktestResult:
    oos: pd.DataFrame                       # per-obs OOS predictions + realized label
    ic: pd.Series                           # per-period rank IC
    summary: dict = field(default_factory=dict)


def run_cv_predictions(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: PurgedKFold,
    params: dict | None = None,
    by: str = "period",
) -> pd.DataFrame:
    """Return an OOS prediction frame (period, id, pred, realized label)."""
    from lightgbm import LGBMRegressor

    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    work = panel.dropna(subset=[label_col]).reset_index(drop=True)
    work["_target"] = cross_sectional_rank_target(work, label_col, by=by)

    X = work[feature_cols]
    y = work["_target"].to_numpy()
    formation = pd.to_datetime(work["formation_date"])

    preds = np.full(len(work), np.nan)
    for train_idx, test_idx in cv.split(formation):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])

    work["pred"] = preds
    keep = [by, "formation_date", "composite_figi", "ticker", "pred", label_col]
    keep = [c for c in keep if c in work.columns]
    return work.loc[work["pred"].notna(), keep].reset_index(drop=True)


def evaluate(oos: pd.DataFrame, label_col: str, by: str = "period") -> BacktestResult:
    """Compute IC/ICIR, decile spread and turnover from OOS predictions."""
    ic = rank_ic(oos, "pred", label_col, by=by)
    summary = ic_summary(ic)
    spread, spread_summary = decile_spread(oos, "pred", label_col, by=by)
    turn = portfolio_turnover(oos, "pred", by=by)
    summary.update({f"ls_{k}": v for k, v in spread_summary.items()})
    summary.update(turn)
    return BacktestResult(oos=oos, ic=ic, summary=summary)


def run_full_vs_baseline(
    panel: pd.DataFrame,
    full_features: list[str],
    baseline_features: list[str],
    label_col: str,
    cv: PurgedKFold,
    params: dict | None = None,
    by: str = "period",
) -> dict:
    """Run both models on identical folds and report incremental IC."""
    full_oos = run_cv_predictions(panel, full_features, label_col, cv, params, by)
    base_oos = run_cv_predictions(panel, baseline_features, label_col, cv, params, by)
    full = evaluate(full_oos, label_col, by)
    base = evaluate(base_oos, label_col, by)

    incremental = full.summary["mean_ic"] - base.summary["mean_ic"]
    logger.info(
        "mean rank IC — full: %.4f | baseline: %.4f | incremental: %.4f",
        full.summary["mean_ic"], base.summary["mean_ic"], incremental,
    )
    return {
        "full": full,
        "baseline": base,
        "incremental_mean_ic": incremental,
    }
