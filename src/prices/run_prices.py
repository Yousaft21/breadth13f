"""Phase 4 orchestrator: attach forward returns + size/value baseline.

Assembles the modeling panel from the breadth panel:

1. Load prices (yfinance -> Stooq, cached per ticker) for every panel ticker.
2. Compute forward 6/12-month total returns and per-period delisting-dropout
   diagnostics.
3. Build the point-in-time size/value baseline from SEC XBRL fundamentals.
4. Merge everything into ``data/processed/model_panel.parquet``.

Outputs coverage/diagnostic tables to ``results/tables/``.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from tqdm import tqdm

from ..config import load_config, setup_logging
from ..ingest.edgar_client import EdgarClient
from .baseline import build_baseline_features
from .fundamentals import FundamentalsProvider, load_ticker_cik_map
from .returns import compute_forward_returns, load_price_series

logger = logging.getLogger("breadth13f.prices")


def _load_all_prices(tickers, start, end, cache_dir) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for t in tqdm(tickers, desc="prices", unit="ticker"):
        if not isinstance(t, str) or not t:
            continue
        df = load_price_series(t, start, end, cache_dir)
        if df is not None and not df.empty:
            lookup[t] = df
    logger.info("Loaded price series for %d / %d tickers", len(lookup), len(tickers))
    return lookup


def build_fundamentals_client(cfg) -> EdgarClient:
    e = cfg.edgar
    return EdgarClient(
        cache_dir=cfg.path("raw") / "sec_xbrl",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach returns + baseline (Phase 4).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--horizons", nargs="+", type=int, default=[6, 12])
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    setup_logging(cfg.raw.get("logging", {}).get("level", "INFO"))

    panel = pd.read_parquet(cfg.path("processed") / "breadth_panel.parquet")
    if panel.empty:
        raise SystemExit("Breadth panel is empty — run Phase 3 first.")

    formations = pd.to_datetime(panel["formation_date"])
    start = (formations.min() - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    end = (formations.max() + pd.DateOffset(months=max(args.horizons) + 1)).strftime("%Y-%m-%d")
    tickers = sorted({t for t in panel["ticker"].dropna().unique() if t})
    logger.info("Panel: %d rows, %d tickers; price window %s..%s",
                len(panel), len(tickers), start, end)

    price_lookup = _load_all_prices(tickers, start, end, cfg.path("raw") / "prices")

    panel, dropout = compute_forward_returns(panel, price_lookup, horizons=tuple(args.horizons))
    dropout.to_csv(cfg.path("results") / "tables" / "delisting_dropout.csv", index=False)
    logger.info("Wrote delisting-dropout diagnostics")

    # --- size/value baseline from SEC XBRL ---
    fclient = build_fundamentals_client(cfg)
    ticker_cik = load_ticker_cik_map(fclient)
    provider = FundamentalsProvider(fclient)
    baseline = build_baseline_features(panel, price_lookup, provider, ticker_cik)

    model_panel = panel.merge(baseline, on=["period", "composite_figi"], how="left")
    out = cfg.path("processed") / "model_panel.parquet"
    model_panel.to_parquet(out, index=False)
    logger.info("Wrote model panel (%d rows) -> %s", len(model_panel), out)

    coverage = {
        "rows": int(len(model_panel)),
        "tickers_with_prices": len(price_lookup),
        "tickers_total": len(tickers),
        "with_size": int(model_panel["log_market_cap"].notna().sum()),
        "with_book_to_market": int(model_panel["book_to_market"].notna().sum()),
        "with_earnings_yield": int(model_panel["earnings_yield"].notna().sum()),
        "label_12m_nonnull": int(model_panel.get("fwd_ret_12m", pd.Series(dtype=float)).notna().sum()),
        "label_6m_nonnull": int(model_panel.get("fwd_ret_6m", pd.Series(dtype=float)).notna().sum()),
    }
    with (cfg.path("results") / "tables" / "phase4_coverage.json").open("w", encoding="utf-8") as fh:
        json.dump(coverage, fh, indent=2)
    logger.info("Phase 4 coverage: %s", coverage)


if __name__ == "__main__":
    main()
