"""Tests for the purged & embargoed CV splitter — the leakage guarantees.

These are the load-bearing assertions of the whole backtest: for every fold no
training observation may share future information with the test labels. We verify
purge (no label-interval overlap), embargo (a gap after the test span), fold
coverage, and that whole formation quarters are never split across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.cv import PurgedKFold, label_end_dates


def _make_panel(n_quarters: int = 16, n_names: int = 20) -> pd.DataFrame:
    """Quarterly formation dates, several names per quarter."""
    q_ends = pd.date_range("2015-03-31", periods=n_quarters, freq="QE")
    rows = []
    for qe in q_ends:
        formation = qe + pd.Timedelta(days=46)
        for j in range(n_names):
            rows.append({"formation_date": formation, "name": j})
    return pd.DataFrame(rows)


def test_folds_cover_all_samples_exactly_once():
    panel = _make_panel()
    cv = PurgedKFold(n_splits=5, horizon_months=12, embargo_days=21)
    test_union = []
    for _, test_idx in cv.split(panel["formation_date"]):
        test_union.extend(test_idx.tolist())
    assert sorted(test_union) == list(range(len(panel)))  # partition, no overlap


def test_train_test_disjoint_and_no_label_overlap():
    panel = _make_panel()
    formation = pd.to_datetime(panel["formation_date"])
    label_end = label_end_dates(formation, horizon_months=12)
    embargo = pd.Timedelta(days=21)
    cv = PurgedKFold(n_splits=5, horizon_months=12, embargo_days=21)

    for train_idx, test_idx in cv.split(formation, label_end):
        assert set(train_idx).isdisjoint(set(test_idx))

        test_t0 = formation.iloc[test_idx].min()
        test_t1 = label_end.iloc[test_idx].max()
        forbidden_end = test_t1 + embargo

        # No training observation's [t0, t1] may intersect the forbidden window.
        tr_t0 = formation.iloc[train_idx]
        tr_t1 = label_end.iloc[train_idx]
        overlap = (tr_t0 <= forbidden_end) & (tr_t1 >= test_t0)
        assert not overlap.any(), "purge/embargo failed: training label overlaps test"


def test_whole_quarters_not_split_across_folds():
    panel = _make_panel()
    formation = pd.to_datetime(panel["formation_date"])
    cv = PurgedKFold(n_splits=4, horizon_months=12, embargo_days=0)
    seen_dates: set = set()
    for _, test_idx in cv.split(formation):
        fold_dates = set(formation.iloc[test_idx].unique())
        # A formation date appears in exactly one test fold.
        assert fold_dates.isdisjoint(seen_dates)
        seen_dates |= fold_dates


def test_larger_embargo_removes_more_training():
    panel = _make_panel()
    formation = pd.to_datetime(panel["formation_date"])

    def train_count(embargo_days: int) -> int:
        cv = PurgedKFold(n_splits=5, horizon_months=12, embargo_days=embargo_days)
        return sum(len(tr) for tr, _ in cv.split(formation))

    assert train_count(180) <= train_count(0)


def test_too_many_splits_raises():
    formation = pd.to_datetime(_make_panel(n_quarters=3)["formation_date"])
    cv = PurgedKFold(n_splits=5)
    try:
        list(cv.split(formation))
    except ValueError:
        return
    raise AssertionError("expected ValueError when n_splits exceeds formation dates")
