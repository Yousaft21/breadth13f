"""Phase 1 orchestrator: crawl + parse 13F info tables into a tidy table.

For each *filing* quarter in the requested range this:

1. Lists all 13F-HR / 13F-HR/A filings from that quarter's master index.
2. Downloads each full submission ``.txt`` (rate-limited, cached, gzip on disk).
3. Parses the SGML header and information table into position rows.
4. Writes one Parquet file of holdings and one filing-level manifest per
   filing quarter under ``data/interim/``.

The step is **resumable**: a filing quarter whose Parquet output already exists
is skipped unless ``--force`` is given, and every raw download is cached so an
interrupted run never re-hits EDGAR for bytes it already has.

Run examples
------------
    # Small sample to validate the pipeline end-to-end (fast):
    python -m src.ingest.run_ingest --filing-quarters 2020Q1 2020Q1 --limit 250

    # Full range derived from config window (long; resumable):
    python -m src.ingest.run_ingest
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import Config, load_config, setup_logging
from .crawl import Quarter, FilingRef, iter_quarters, list_13f_filings, quarter_closed_by
from .edgar_client import EdgarClient
from .parse_13f import parse_submission

logger = logging.getLogger("breadth13f.ingest")

HOLDINGS_COLS = [
    "filing_quarter", "accession_number", "manager_cik", "manager_name",
    "form_type", "period_of_report", "filed_as_of_date", "acceptance_datetime",
    "amendment_type", "amendment_no",
    "name_of_issuer", "title_of_class", "cusip", "value", "shares", "shares_type",
    "put_call", "investment_discretion", "voting_sole", "voting_shared", "voting_none",
]


def _current_quarter() -> Quarter:
    today = date.today()
    return Quarter(today.year, (today.month - 1) // 3 + 1)


def _default_filing_range(cfg: Config) -> tuple[Quarter, Quarter]:
    """Filing quarters needed to cover the config's period-of-report window.

    Filings arrive up to a quarter after period end, and amendments trail for
    several more quarters, so we crawl from the first period quarter through the
    end period + 4 quarters (capped at the current quarter).
    """
    start_period = Quarter.parse(cfg.window["start_quarter"])
    end_period = Quarter.parse(cfg.window["end_quarter"])
    filing_end = end_period
    for _ in range(4):
        filing_end = filing_end.next()
    cap = _current_quarter()
    if filing_end > cap:
        filing_end = cap
    return start_period, filing_end


def build_client(cfg: Config) -> EdgarClient:
    e = cfg.edgar
    return EdgarClient(
        cache_dir=cfg.path("raw_edgar"),
        user_agent=cfg.user_agent,
        requests_per_second=e["requests_per_second"],
        max_retries=e["max_retries"],
        backoff_seconds=e["backoff_seconds"],
        timeout_seconds=e["timeout_seconds"],
        connect_timeout_seconds=e.get("connect_timeout_seconds", 10.0),
        conn_retries=e.get("conn_retries", 2),
        conn_backoff_seconds=e.get("conn_backoff_seconds", 1.5),
        max_backoff_seconds=e.get("max_backoff_seconds", 30.0),
    )


def ingest_quarter(
    cfg: Config,
    client: EdgarClient,
    fq: Quarter,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Ingest one filing quarter; return a summary dict."""
    holdings_dir = cfg.path("interim") / "holdings"
    manifest_dir = cfg.path("interim") / "manifest"
    holdings_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    holdings_path = holdings_dir / f"filing_quarter={fq}.parquet"
    manifest_path = manifest_dir / f"filing_quarter={fq}.parquet"

    # Output written while the filing quarter was still open misses every filing
    # made after that day: rebuild it rather than skip it.
    incomplete = holdings_path.exists() and not quarter_closed_by(
        fq, datetime.fromtimestamp(holdings_path.stat().st_mtime))
    if incomplete:
        logger.info("%s: output was written before the quarter closed; rebuilding", fq)
    if holdings_path.exists() and not force and not incomplete:
        logger.info("%s: output exists, skipping (use --force to rebuild)", fq)
        mani = pd.read_parquet(manifest_path)
        return {
            "filing_quarter": str(fq),
            "skipped": True,
            "n_filings": int(len(mani)),
            "n_holdings": int(mani["n_holdings"].sum()) if len(mani) else 0,
        }

    base = cfg.edgar["base_archives"]
    form_types = set(cfg.edgar["form_types"])
    refs = list_13f_filings(client, base, fq, form_types)
    if limit is not None:
        refs = refs[:limit]

    holdings_rows: list[dict] = []
    manifest_rows: list[dict] = []
    n_parse_fail = 0

    for ref in tqdm(refs, desc=f"{fq}", unit="filing"):
        url = ref.submission_url(base)
        try:
            text = client.get_text(url, compress=True)
            parsed = parse_submission(text)
        except Exception as exc:  # noqa: BLE001 - log and continue, one bad filing
            n_parse_fail += 1
            logger.warning("Failed on %s: %s", ref.accession_number, exc)
            continue

        h = parsed.header
        # Prefer header CIK/name; fall back to the index row.
        manager_cik = h.manager_cik or ref.manager_cik
        manager_name = h.manager_name or ref.company_name
        manifest_rows.append(
            {
                "filing_quarter": str(fq),
                "accession_number": h.accession_number or ref.accession_number,
                "manager_cik": manager_cik,
                "manager_name": manager_name,
                "form_type": h.submission_type or ref.form_type,
                "period_of_report": h.period_of_report,
                "filed_as_of_date": h.filed_as_of_date,
                "acceptance_datetime": h.acceptance_datetime,
                "amendment_type": h.amendment_type,
                "amendment_no": h.amendment_no,
                "n_holdings": parsed.n_holdings,
            }
        )
        for pos in parsed.holdings:
            holdings_rows.append(
                {
                    "filing_quarter": str(fq),
                    "accession_number": h.accession_number or ref.accession_number,
                    "manager_cik": manager_cik,
                    "manager_name": manager_name,
                    "form_type": h.submission_type or ref.form_type,
                    "period_of_report": h.period_of_report,
                    "filed_as_of_date": h.filed_as_of_date,
                    "acceptance_datetime": h.acceptance_datetime,
                    "amendment_type": h.amendment_type,
                    "amendment_no": h.amendment_no,
                    **pos,
                }
            )

    holdings_df = pd.DataFrame(holdings_rows, columns=HOLDINGS_COLS)
    manifest_df = pd.DataFrame(manifest_rows)
    _write_parquet(holdings_df, holdings_path)
    _write_parquet(manifest_df, manifest_path)

    summary = {
        "filing_quarter": str(fq),
        "skipped": False,
        "n_filings_in_index": len(refs),
        "n_filings_parsed": int(len(manifest_df)),
        "n_parse_failures": n_parse_fail,
        "n_holdings": int(len(holdings_df)),
        "n_managers": int(manifest_df["manager_cik"].nunique()) if len(manifest_df) else 0,
        "n_periods": int(manifest_df["period_of_report"].nunique()) if len(manifest_df) else 0,
        "distinct_cusips": int(holdings_df["cusip"].nunique()) if len(holdings_df) else 0,
    }
    logger.info("%s summary: %s", fq, summary)
    return summary


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    # Normalise date/datetime dtypes so Parquet round-trips cleanly.
    for col in ("period_of_report", "filed_as_of_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "acceptance_datetime" in df.columns:
        df["acceptance_datetime"] = pd.to_datetime(df["acceptance_datetime"], errors="coerce")
    df.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest SEC 13F info tables (Phase 1).")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument(
        "--filing-quarters", nargs=2, metavar=("START", "END"), default=None,
        help="Filing-quarter range, e.g. 2020Q1 2020Q2. Defaults to config window.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap filings per quarter (sampling).")
    ap.add_argument("--force", action="store_true", help="Rebuild quarters even if output exists.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    setup_logging(cfg.raw.get("logging", {}).get("level", "INFO"))

    if args.filing_quarters:
        start = Quarter.parse(args.filing_quarters[0])
        end = Quarter.parse(args.filing_quarters[1])
    else:
        start, end = _default_filing_range(cfg)
    logger.info("Ingesting filing quarters %s .. %s (limit=%s)", start, end, args.limit)

    client = build_client(cfg)
    summaries = []
    for fq in iter_quarters(start, end):
        summaries.append(ingest_quarter(cfg, client, fq, limit=args.limit, force=args.force))

    out = cfg.path("results") / "tables" / "ingest_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    logger.info("Wrote ingest summary -> %s", out)


if __name__ == "__main__":
    main()
