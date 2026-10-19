"""Phase 3 orchestrator: build the point-in-time breadth-feature panel.

Reads ingested holdings + the security master and writes the security x
formation-date breadth panel to ``data/processed/breadth_panel.parquet``.

Holdings are loaded **one filing quarter at a time** (streaming). A period's
point-in-time filings (accepted by the 45-day filing deadline) all fall
in the single next calendar quarter's file, so only that file is needed per
period — keeping memory bounded across the full 12-year, ~117M-row dataset.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from ..config import load_config, setup_logging
from ..ingest.crawl import Quarter
from .breadth import build_feature_panel

logger = logging.getLogger("breadth13f.features")

_HOLDINGS_COLS = ["period_of_report", "acceptance_datetime", "put_call",
                  "shares", "value", "manager_cik", "cusip", "accession_number"]


def _filing_quarter_for_period(period: pd.Timestamp) -> Quarter:
    """Filing quarter that holds a period's on-time filings (the next quarter)."""
    nxt = pd.Timestamp(period) + pd.Timedelta(days=1)  # first day of next quarter
    return Quarter(nxt.year, (nxt.month - 1) // 3 + 1)


def _available_periods(holdings_dir) -> list[pd.Timestamp]:
    """Periods whose covering filing-quarter file is present on disk."""
    periods = []
    for f in holdings_dir.glob("filing_quarter=*.parquet"):
        fq = Quarter.parse(f.stem.split("=", 1)[1])
        periods.append(fq.prev().end_date())  # period covered on-time by fq
    return sorted(set(periods))


def _make_loader(holdings_dir):
    def load(period: pd.Timestamp) -> pd.DataFrame:
        fq = _filing_quarter_for_period(period)
        path = holdings_dir / f"filing_quarter={fq}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=_HOLDINGS_COLS)
        return pd.read_parquet(path, columns=_HOLDINGS_COLS)
    return load


def main() -> None:
    ap = argparse.ArgumentParser(description="Build breadth-feature panel (Phase 3).")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    setup_logging(cfg.raw.get("logging", {}).get("level", "INFO"))

    holdings_dir = cfg.path("interim") / "holdings"
    secmaster = pd.read_parquet(cfg.path("processed") / "security_master.parquet")
    deadline_days = int(cfg.pit["filing_deadline_days"])

    periods = _available_periods(holdings_dir)
    logger.info("Streaming %d candidate periods (%s .. %s)",
                len(periods), periods[0].date(), periods[-1].date())

    panel = build_feature_panel(
        holdings=None, secmaster=secmaster, deadline_days=deadline_days,
        holdings_loader=_make_loader(holdings_dir), periods=periods,
    )
    out = cfg.path("processed") / "breadth_panel.parquet"
    panel.to_parquet(out, index=False)
    logger.info("Wrote breadth panel (%d rows) -> %s", len(panel), out)

    summary = {
        "n_rows": int(len(panel)),
        "n_periods": int(panel["period"].nunique()) if len(panel) else 0,
        "n_securities": int(panel["composite_figi"].nunique()) if len(panel) else 0,
        "securities_per_period": (
            panel.groupby(panel["period"].astype(str))["composite_figi"].nunique().to_dict()
            if len(panel) else {}
        ),
    }
    sp = cfg.path("results") / "tables" / "feature_panel_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Panel summary: periods=%s securities=%s",
                summary["n_periods"], summary["n_securities"])


if __name__ == "__main__":
    main()
