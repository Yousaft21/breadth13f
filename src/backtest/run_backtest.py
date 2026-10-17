"""Phase 5/6 orchestrator: purged-CV backtest, baseline comparison, SHAP pruning.

Loads the modeling panel, applies the universe filter, then for the primary
(12-month) horizon:

* runs the **full breadth+baseline model** and the **size/value baseline** on
  identical purged/embargoed folds and reports IC / ICIR / decile spread /
  turnover and the **incremental IC** of breadth over the baseline;
* ranks features by out-of-sample mean|SHAP|, **prunes** negligible ones, and
  re-runs to report IC before vs after pruning.

Writes metric tables to ``results/tables/`` (one JSON per horizon, plus a
``summary.csv`` across all horizons run so far) and figures to
``results/figures/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ..config import load_config, setup_logging
from ..features.breadth import FEATURE_COLUMNS as BREADTH_FEATURES
from ..prices.baseline import BASELINE_FEATURES
from ..viz.plots import (
    plot_cumulative_ls,
    plot_decile_spread,
    plot_ic_timeseries,
    plot_shap_importance,
)
from .attribution import backward_eliminate
from .cv import PurgedKFold
from .metrics import decile_spread
from .model import evaluate, run_cv_predictions, run_full_vs_baseline

logger = logging.getLogger("breadth13f.backtest")


def _apply_universe(panel: pd.DataFrame, min_holders: int, label_col: str) -> pd.DataFrame:
    before = len(panel)
    out = panel[(panel["n_holders_p"] >= min_holders) & (panel[label_col].notna())].copy()
    logger.info("Universe filter (min_holders>=%d, %s present): %d -> %d rows",
                min_holders, label_col, before, len(out))
    return out


def _serialize_result(res) -> dict:
    return {"summary": res.summary,
            "ic_by_period": {str(pd.Timestamp(k).date()): float(v) for k, v in res.ic.items()}}


def write_summary(tables: Path) -> pd.DataFrame:
    """Collect every ``backtest_results_<h>m.json`` into ``summary.csv``."""
    rows = []
    for path in tables.glob("backtest_results_*m.json"):
        r = json.loads(path.read_text(encoding="utf-8"))
        n_full = len(r["kept_features"]) + len(r["dropped_features"])
        for model, key, n_feat in [("baseline", "baseline_model", len(BASELINE_FEATURES)),
                                   ("full", "full_model", n_full),
                                   ("pruned", "pruned_model", len(r["kept_features"]))]:
            s = r[key]["summary"]
            rows.append({
                "horizon_m": r["horizon_months"], "model": model,
                "mean_ic": round(s["mean_ic"], 4), "icir": round(s["icir"], 2),
                "t_stat": round(s["t_stat"], 2), "hit_rate": round(s["hit_rate"], 2),
                "decile_ls": round(s["ls_mean_ls_spread"], 4),
                "turnover": round(s["mean_top_turnover"], 3),
                "n_features": n_feat, "n_periods": s["n_periods"],
            })
    out = pd.DataFrame(rows).sort_values(["horizon_m", "model"], ascending=[False, True])
    out.to_csv(tables / "summary.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Purged-CV backtest + SHAP pruning (Phase 5/6).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--horizon", type=int, default=None, help="Override label horizon (months).")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    setup_logging(cfg.raw.get("logging", {}).get("level", "INFO"))
    cvcfg = cfg.raw["cv"]
    horizon = args.horizon or int(cvcfg["primary_horizon_months"])
    label_col = f"fwd_ret_{horizon}m"

    panel = pd.read_parquet(cfg.path("processed") / "model_panel.parquet")
    if label_col not in panel.columns:
        raise SystemExit(f"{label_col} missing — run Phase 4 with that horizon.")

    panel = _apply_universe(panel, int(cfg.raw["universe"]["min_holders"]), label_col)
    baseline_features = [c for c in BASELINE_FEATURES if c in panel.columns]
    full_features = [c for c in (BREADTH_FEATURES + baseline_features) if c in panel.columns]

    cv = PurgedKFold(
        n_splits=int(cvcfg["n_splits"]),
        horizon_months=horizon,
        embargo_days=int(cvcfg["embargo_days"]),
    )

    # --- full breadth+baseline vs size/value baseline ---
    comp = run_full_vs_baseline(panel, full_features, baseline_features, label_col, cv)
    full, base = comp["full"], comp["baseline"]

    # --- SHAP attribution + out-of-sample backward elimination ---
    elim = backward_eliminate(panel, full_features, label_col, cv)
    pruned = evaluate(run_cv_predictions(panel, elim["kept"], label_col, cv), label_col)

    # --- universe stats ---
    per_period = panel.groupby("period")["composite_figi"].nunique()
    universe_stats = {
        "n_periods": int(panel["period"].nunique()),
        "n_securities_total": int(panel["composite_figi"].nunique()),
        "securities_per_period_mean": float(per_period.mean()),
        "securities_per_period_min": int(per_period.min()),
        "securities_per_period_max": int(per_period.max()),
        "rows": int(len(panel)),
    }

    results = {
        "horizon_months": horizon,
        "universe": universe_stats,
        "baseline_model": _serialize_result(base),
        "full_model": _serialize_result(full),
        "pruned_model": _serialize_result(pruned),
        "incremental_mean_ic": comp["incremental_mean_ic"],   # full over baseline
        "shap_importance": {k: float(v) for k, v in elim["importance"].items()},
        "dropped_features": elim["dropped"],
        "kept_features": elim["kept"],
        "ic_all_features": elim["ic_all_features"],
        "ic_pruned": elim["ic_pruned"],
    }
    tables = cfg.path("results") / "tables"
    figs = cfg.path("results") / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    with (tables / f"backtest_results_{horizon}m.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    # --- figures (from the final, pruned model) ---
    plot_ic_timeseries(pruned.ic, figs / f"ic_timeseries_{horizon}m.png",
                       title=f"Per-period rank IC (pruned model, {horizon}m)")
    spread, _ = decile_spread(pruned.oos, "pred", label_col)
    if len(spread):
        plot_decile_spread(spread, figs / f"decile_spread_{horizon}m.png")
        plot_cumulative_ls(spread, figs / f"cumulative_ls_{horizon}m.png")
    plot_shap_importance(elim["importance"], figs / f"shap_importance_{horizon}m.png")
    write_summary(tables)

    logger.info(
        "DONE %dm | baseline IC %.4f | full IC %.4f (ICIR %.2f, t %.2f) | incremental %.4f | "
        "pruned IC %.4f | L/S %.4f | kept %d dropped %d %s",
        horizon, base.summary["mean_ic"], full.summary["mean_ic"],
        full.summary.get("icir", float("nan")), full.summary.get("t_stat", float("nan")),
        comp["incremental_mean_ic"], pruned.summary["mean_ic"],
        pruned.summary.get("ls_mean_ls_spread", float("nan")),
        len(elim["kept"]), len(elim["dropped"]), elim["dropped"],
    )


if __name__ == "__main__":
    main()
