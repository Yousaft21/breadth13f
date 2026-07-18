"""Result figures: IC time series, decile spread, SHAP importance, cumulative L/S."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_ic_timeseries(ic: pd.Series, path: Path, title: str = "Per-period rank IC") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(pd.Timestamp(i).date()) for i in ic.index], ic.to_numpy(), color="#3b7dd8")
    ax.axhline(ic.mean(), color="crimson", ls="--", lw=1, label=f"mean = {ic.mean():.3f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("Spearman rank IC")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_decile_spread(spread: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(pd.Timestamp(i).date()) for i in spread.index], spread.to_numpy(), color="#4a9e5c")
    ax.axhline(spread.mean(), color="crimson", ls="--", lw=1, label=f"mean = {spread.mean():.3f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Top-minus-bottom decile forward-return spread")
    ax.set_ylabel("Return spread")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_shap_importance(importance: pd.Series, path: Path) -> None:
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(imp))))
    ax.barh(imp.index, imp.to_numpy(), color="#8853c6")
    ax.set_title("Mean |SHAP| (out-of-sample)")
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cumulative_ls(spread: pd.Series, path: Path) -> None:
    cum = spread.sort_index().cumsum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([pd.Timestamp(i) for i in cum.index], cum.to_numpy(), color="#3b7dd8", marker="o", ms=3)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Cumulative decile long-short spread")
    ax.set_ylabel("Cumulative spread")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
