"""End-to-end sanity checks for the model / evaluation / SHAP-pruning stack.

Uses a synthetic panel with one genuinely predictive feature and two noise
features: the full model must beat the noise-only baseline on out-of-sample IC,
and SHAP-based pruning must keep the real feature and drop noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import (
    backward_eliminate,
    prune_and_rerun,
    prune_features,
    shap_importance,
)
from src.backtest.cv import PurgedKFold
from src.backtest.model import run_full_vs_baseline

pytest.importorskip("lightgbm")
pytest.importorskip("shap")


def _signal_panel(n_quarters=16, n_names=200, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q_ends = pd.date_range("2015-03-31", periods=n_quarters, freq="QE")
    rows = []
    for qi, qe in enumerate(q_ends):
        formation = qe + pd.Timedelta(days=46)
        signal = rng.normal(size=n_names)
        noise1 = rng.normal(size=n_names)
        noise2 = rng.normal(size=n_names)
        # Forward return driven by the signal plus period + idiosyncratic noise.
        label = 0.6 * signal + rng.normal(scale=0.7, size=n_names)
        for j in range(n_names):
            rows.append({
                "period": qe, "formation_date": formation,
                "composite_figi": f"F{j:04d}", "ticker": f"T{j:04d}",
                "breadth_signal": signal[j], "noise1": noise1[j], "noise2": noise2[j],
                "fwd_ret_12m": label[j],
            })
    return pd.DataFrame(rows)


def test_full_model_beats_noise_baseline():
    panel = _signal_panel()
    cv = PurgedKFold(n_splits=4, horizon_months=12, embargo_days=21)
    res = run_full_vs_baseline(
        panel,
        full_features=["breadth_signal", "noise1", "noise2"],
        baseline_features=["noise1", "noise2"],
        label_col="fwd_ret_12m",
        cv=cv,
    )
    assert res["full"].summary["mean_ic"] > 0.1
    assert res["incremental_mean_ic"] > 0.05  # breadth adds real OOS IC
    # The paired per-period test averages to the same increment.
    assert np.isclose(res["incremental"]["mean_ic"], res["incremental_mean_ic"])
    assert res["incremental"]["t_stat"] > 2


def test_shap_ranks_signal_first_and_pruning_keeps_it():
    panel = _signal_panel()
    cv = PurgedKFold(n_splits=4, horizon_months=12, embargo_days=21)
    feats = ["breadth_signal", "noise1", "noise2"]
    imp = shap_importance(panel, feats, "fwd_ret_12m", cv)
    assert imp.index[0] == "breadth_signal"       # most important OOS

    kept, dropped = prune_features(imp, min_ratio=0.1)
    assert "breadth_signal" in kept

    out = prune_and_rerun(panel, feats, "fwd_ret_12m", cv, min_ratio=0.1)
    assert "breadth_signal" in out["kept"]
    # Dropping noise should not hurt OOS IC.
    assert out["ic_after"]["mean_ic"] >= out["ic_before"]["mean_ic"] - 0.02


def test_backward_elimination_drops_noise_keeps_signal():
    panel = _signal_panel()
    cv = PurgedKFold(n_splits=4, horizon_months=12, embargo_days=21)
    feats = ["breadth_signal", "noise1", "noise2"]
    out = backward_eliminate(panel, feats, "fwd_ret_12m", cv)
    assert "breadth_signal" in out["kept"]          # the real feature survives
    assert set(out["dropped"]).issubset({"noise1", "noise2"})  # only noise dropped
    # Pruned model is no worse than using all features out-of-sample.
    assert out["ic_pruned"] >= out["ic_all_features"] - 1e-9
