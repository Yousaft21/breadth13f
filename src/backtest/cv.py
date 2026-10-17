"""Purged & embargoed time-series cross-validation (López de Prado, AFML Ch. 7).

The panel is formed quarterly but labels are forward 6–12 month returns, so a
label window spans the next several formation quarters. Naive K-fold would leak:
a training observation whose label window overlaps the test period shares
realized future information with the test labels. Two guards fix this:

* **Purge** — drop from the training set any observation whose label interval
  ``[formation, formation + horizon]`` overlaps the test fold's label span.
* **Embargo** — additionally drop training observations that begin shortly after
  the test span, since serial correlation can leak information across the
  boundary even without direct overlap.

Folds are built over **whole formation dates** (a quarter is never split across
folds), which is the natural grouping for a quarterly-formed panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def label_end_dates(formation: pd.Series, horizon_months: int) -> pd.Series:
    """Label interval end = formation date + horizon (in months)."""
    formation = pd.to_datetime(formation)
    return formation + pd.DateOffset(months=horizon_months)


@dataclass
class PurgedKFold:
    """K-fold splitter with purging and an embargo, grouped by formation date.

    Parameters
    ----------
    n_splits:
        Number of chronological folds.
    horizon_months:
        Forward-return label horizon; sets each observation's label interval.
    embargo_days:
        Extra gap (in days) dropped from training after the test label span.
    """

    n_splits: int = 5
    horizon_months: int = 12
    embargo_days: int = 21

    def split(
        self,
        formation: pd.Series | np.ndarray,
        label_end: pd.Series | np.ndarray | None = None,
    ):
        """Yield ``(train_idx, test_idx)`` positional-index arrays.

        ``formation`` is each observation's formation date. ``label_end`` may be
        supplied directly; otherwise it is derived from ``horizon_months``.
        """
        formation = pd.to_datetime(pd.Series(np.asarray(formation)).reset_index(drop=True))
        if label_end is None:
            label_end = formation + pd.DateOffset(months=self.horizon_months)
        else:
            label_end = pd.to_datetime(pd.Series(np.asarray(label_end)).reset_index(drop=True))

        n = len(formation)
        all_pos = np.arange(n)
        embargo = pd.Timedelta(days=self.embargo_days)

        for fold_dates in self.fold_dates(formation):
            fold_set = set(pd.to_datetime(fold_dates))
            test_mask = formation.isin(fold_set).to_numpy()
            test_idx = all_pos[test_mask]
            if test_idx.size == 0:
                continue

            test_t0 = formation.iloc[test_idx].min()
            test_t1 = label_end.iloc[test_idx].max()
            forbidden_end = test_t1 + embargo

            # Purge + embargo: a training obs is dropped if its label interval
            # [t0, t1] intersects [test_t0, test_t1 + embargo].
            overlaps = (
                (formation <= forbidden_end) & (label_end >= test_t0)
            ).to_numpy()
            train_mask = (~test_mask) & (~overlaps)
            train_idx = all_pos[train_mask]
            yield train_idx, test_idx

    def fold_dates(self, formation: pd.Series | np.ndarray) -> list[np.ndarray]:
        """Chronological test-fold date groups (whole formation dates per fold)."""
        unique_dates = np.array(sorted(pd.to_datetime(pd.Series(np.asarray(formation))).unique()))
        if len(unique_dates) < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} exceeds number of formation dates "
                f"({len(unique_dates)})."
            )
        return np.array_split(unique_dates, self.n_splits)

    def get_n_splits(self, *args, **kwargs) -> int:
        return self.n_splits
