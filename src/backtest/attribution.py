"""SHAP feature attribution and out-of-sample feature pruning.

We aggregate SHAP values from the **out-of-sample** test folds (via
``TreeExplainer``) into a mean|SHAP| ranking, then prune features whose
contribution is negligible and re-run the model. Pruning is validated by the
IC before vs after: a feature is only worth keeping if dropping it does not
improve (or at least does not materially hurt) out-of-sample IC.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .cv import PurgedKFold
from .model import (
    DEFAULT_LGBM_PARAMS,
    cross_sectional_rank_target,
    evaluate,
    run_cv_predictions,
)

logger = logging.getLogger("breadth13f.attribution")


def shap_importance(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: PurgedKFold,
    params: dict | None = None,
    by: str = "period",
) -> pd.Series:
    """Mean |SHAP| per feature, aggregated over out-of-sample test folds."""
    import shap
    from lightgbm import LGBMRegressor

    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    work = panel.dropna(subset=[label_col]).reset_index(drop=True)
    work["_target"] = cross_sectional_rank_target(work, label_col, by=by)
    X = work[feature_cols]
    y = work["_target"].to_numpy()
    formation = pd.to_datetime(work["formation_date"])

    abs_sum = np.zeros(len(feature_cols))
    n_rows = 0
    for train_idx, test_idx in cv.split(formation):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.iloc[test_idx])
        abs_sum += np.abs(sv).sum(axis=0)
        n_rows += len(test_idx)

    mean_abs = abs_sum / max(n_rows, 1)
    return pd.Series(mean_abs, index=feature_cols, name="mean_abs_shap").sort_values(ascending=False)


def prune_features(importance: pd.Series, min_ratio: float = 0.02) -> tuple[list[str], list[str]]:
    """Split features into kept/dropped by mean|SHAP| relative to the top feature.

    Features contributing less than ``min_ratio`` of the most important feature's
    mean|SHAP| are dropped as negligible.
    """
    if importance.empty:
        return [], []
    threshold = importance.max() * min_ratio
    kept = importance[importance >= threshold].index.tolist()
    dropped = importance[importance < threshold].index.tolist()
    return kept, dropped


def backward_eliminate(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: PurgedKFold,
    params: dict | None = None,
    by: str = "period",
    tol: float = 0.0,
) -> dict:
    """Greedy out-of-sample backward elimination of features.

    Features are considered weakest-first (by OOS mean|SHAP|). A feature is
    **dropped** if removing it does not reduce out-of-sample mean IC by more than
    ``tol`` — i.e. it contributes nothing (or is noise) out-of-sample. This makes
    concrete the claim of "discarding features that fail out-of-sample": the
    dropped set is exactly those whose removal the CV could not distinguish from
    keeping them.
    """
    importance = shap_importance(panel, feature_cols, label_col, cv, params, by)
    order = list(importance.sort_values().index)  # weakest contribution first

    def mean_ic(feats: list[str]) -> float:
        oos = run_cv_predictions(panel, feats, label_col, cv, params, by)
        return evaluate(oos, label_col, by).summary["mean_ic"]

    kept = list(feature_cols)
    best_ic = mean_ic(kept)
    ic_all = best_ic
    dropped: list[str] = []
    for feat in order:
        if len(kept) <= 1:
            break
        trial = [f for f in kept if f != feat]
        ic = mean_ic(trial)
        if ic >= best_ic - tol:      # removal doesn't hurt -> feature failed OOS
            kept, best_ic = trial, max(best_ic, ic)
            dropped.append(feat)
            logger.info("  drop %-28s IC %.4f -> %.4f (kept %d)", feat, ic_all, ic, len(kept))
    return {
        "importance": importance,
        "kept": kept,
        "dropped": dropped,
        "ic_all_features": ic_all,
        "ic_pruned": best_ic,
    }


def prune_and_rerun(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: PurgedKFold,
    params: dict | None = None,
    by: str = "period",
    min_ratio: float = 0.02,
) -> dict:
    """Rank features by OOS SHAP, prune the negligible ones, and compare IC."""
    importance = shap_importance(panel, feature_cols, label_col, cv, params, by)
    kept, dropped = prune_features(importance, min_ratio=min_ratio)
    if not kept:  # degenerate guard
        kept, dropped = feature_cols, []

    before = evaluate(run_cv_predictions(panel, feature_cols, label_col, cv, params, by), label_col, by)
    after = evaluate(run_cv_predictions(panel, kept, label_col, cv, params, by), label_col, by)
    logger.info(
        "Pruning: kept %d / dropped %d | mean IC %.4f -> %.4f",
        len(kept), len(dropped), before.summary["mean_ic"], after.summary["mean_ic"],
    )
    return {
        "importance": importance,
        "kept": kept,
        "dropped": dropped,
        "ic_before": before.summary,
        "ic_after": after.summary,
    }
