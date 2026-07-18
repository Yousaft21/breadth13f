"""Phase 2 orchestrator: build the CUSIP->ticker security master.

Reads the ingested holdings, collects the distinct CUSIPs of long (non-option)
equity positions, resolves them via OpenFIGI (cached per CUSIP), reduces each to
a single tradable representative, and writes:

* ``data/processed/security_master.parquet`` — one row per CUSIP.
* ``results/tables/mapping_coverage.json`` — mapping coverage / data-loss stats.

Set ``OPENFIGI_API_KEY`` to use the higher rate limit (100 jobs/request); without
it the free tier (10 jobs/request, 25 requests/minute) still works, just slower.

Run examples
------------
    python -m src.mapping.run_mapping                    # all ingested holdings
    python -m src.mapping.run_mapping --filing-quarters 2020Q1 2020Q4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ..config import Config, load_config, setup_logging
from ..ingest.crawl import Quarter, iter_quarters
from .openfigi import OpenFigiClient
from .security_master import (
    build_security_master,
    coverage_report,
    distinct_equity_cusips,
)

logger = logging.getLogger("breadth13f.mapping")


def _load_holdings(cfg: Config, fq_range: tuple[str, str] | None) -> pd.DataFrame:
    holdings_dir = cfg.path("interim") / "holdings"
    files = sorted(holdings_dir.glob("filing_quarter=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ingested holdings found in {holdings_dir}. Run Phase 1 first.")

    if fq_range is not None:
        wanted = {str(q) for q in iter_quarters(Quarter.parse(fq_range[0]), Quarter.parse(fq_range[1]))}
        files = [f for f in files if f.stem.split("=", 1)[1] in wanted]

    logger.info("Loading %d holdings files for CUSIP extraction", len(files))
    frames = [pd.read_parquet(f, columns=["cusip", "put_call"]) for f in files]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CUSIP->ticker security master (Phase 2).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--filing-quarters", nargs=2, metavar=("START", "END"), default=None,
                    help="Restrict to a filing-quarter range; default uses all ingested holdings.")
    ap.add_argument("--force", action="store_true", help="Re-query CUSIPs even if cached.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    setup_logging(cfg.raw.get("logging", {}).get("level", "INFO"))

    holdings = _load_holdings(cfg, args.filing_quarters)
    cusips = distinct_equity_cusips(holdings)
    logger.info("Distinct long-equity CUSIPs to map: %d", len(cusips))

    client = OpenFigiClient(cache_dir=cfg.path("raw") / "openfigi")
    mapping = client.map_cusips(cusips, force=args.force)

    master = build_security_master(mapping)
    out_path = cfg.path("processed") / "security_master.parquet"
    master.to_parquet(out_path, index=False)
    logger.info("Wrote security master (%d rows) -> %s", len(master), out_path)

    report = coverage_report(master, total_cusips=len(cusips))
    report_path = cfg.path("results") / "tables" / "mapping_coverage.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Mapping coverage: %s", report)


if __name__ == "__main__":
    main()
