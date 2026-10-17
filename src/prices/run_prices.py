"""Phase 4 orchestrator: attach forward returns + size/value baseline.

Assembles the modeling panel from the breadth panel:

1. Load prices (yfinance -> Stooq, cached per ticker) for every panel ticker.
2. Compute forward 6/12-month total returns and per-period delisting-dropout
   diagnostics.
3. Map each stock-quarter to its issuer's SEC CIK **point-in-time**, from the
   13F issuer names and the names on EDGAR periodic reports at the time (see
   :mod:`src.mapping.issuer_cik`); SEC's current ``company_tickers.json`` only
   fills rows the name match leaves empty.
4. Build the point-in-time size/value baseline from SEC XBRL fundamentals and
   **as-traded** prices (raw closes + split history, cached per ticker) for the
   rows with a CIK.
5. Merge everything into ``data/processed/model_panel.parquet``; the CIK and
   its source (``cik``, ``cik_source``) are kept for auditing.

Outputs coverage/diagnostic tables to ``results/tables/``.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from tqdm import tqdm

from ..config import load_config, setup_logging
from ..ingest.crawl import Quarter
from ..ingest.edgar_client import EdgarClient
from ..mapping import issuer_cik
from .as_traded import load_raw_closes
from .baseline import build_baseline_features
from .fundamentals import FundamentalsProvider, _norm_ticker, load_ticker_cik_map
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


def build_fundamentals_client(cfg, cache_dir=None) -> EdgarClient:
    """Fair-access EDGAR client; caches under ``data/raw/sec_xbrl`` by default."""
    e = cfg.edgar
    return EdgarClient(
        cache_dir=cache_dir or cfg.path("raw") / "sec_xbrl",
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


def _map_ciks(cfg, panel: pd.DataFrame, fclient: EdgarClient) -> tuple[pd.Series, pd.Series, dict]:
    """Point-in-time issuer CIK per panel row, its source, and coverage stats.

    The 13F issuer names come from the ingested holdings; the EDGAR names from
    the quarterly ``master.idx`` (cached under ``data/raw/edgar``; quarters the
    13F crawl did not cover are fetched once). Rows the name match leaves empty
    fall back to the current-ticker map unless ``issuer_cik.ticker_fallback`` is off.
    """
    icfg = cfg.raw.get("issuer_cik", {})
    recency = int(icfg.get("recency_days", 730))
    formations = pd.to_datetime(panel["formation_date"])
    first = formations.min() - pd.Timedelta(days=recency)
    last = formations.max()
    start = Quarter(first.year, (first.month - 1) // 3 + 1)
    end = Quarter(last.year, (last.month - 1) // 3 + 1)

    master = pd.read_parquet(cfg.path("processed") / "security_master.parquet")
    master = master[master["composite_figi"].isin(panel["composite_figi"].unique())]
    names = issuer_cik.names_by_security(
        issuer_cik.issuer_names_from_holdings(cfg.path("interim") / "holdings", master["cusip"]),
        master)
    archive_client = build_fundamentals_client(cfg, cache_dir=cfg.path("raw_edgar"))
    spells = issuer_cik.edgar_name_spells(archive_client, cfg.edgar["base_archives"], start, end)
    matches = issuer_cik.match_names(names["name"], spells)
    name_cik = issuer_cik.pit_ciks(panel, names, matches, recency_days=recency,
                                   min_support=float(icfg.get("min_support", 0.5)),
                                   key="composite_figi")

    ticker_map = load_ticker_cik_map(fclient)
    ticker_cik = pd.Series([ticker_map.get(_norm_ticker(t)) if isinstance(t, str) and t else None
                            for t in panel["ticker"]], dtype="Int64")
    cik, source = issuer_cik.combine_ciks(name_cik, ticker_cik,
                                          use_ticker=bool(icfg.get("ticker_fallback", True)))
    both = name_cik.notna() & ticker_cik.notna()
    cov = {
        "cik_rows_name": int(name_cik.notna().sum()),
        "cik_rows_ticker": int(ticker_cik.notna().sum()),
        "cik_rows_final": int(cik.notna().sum()),
        "cik_rows_ticker_fill": int((source == "ticker").sum()),
        "cik_rows_name_ticker_disagree": int((both & (name_cik != ticker_cik).fillna(False)).sum()),
    }
    logger.info("Issuer CIKs: %s", cov)
    return cik.set_axis(panel.index), source.set_axis(panel.index), cov


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

    # --- issuer CIK per stock-quarter, point-in-time ---
    fclient = build_fundamentals_client(cfg)
    panel["cik"], panel["cik_source"], cik_cov = _map_ciks(cfg, panel, fclient)

    # --- size/value baseline from SEC XBRL ---
    provider = FundamentalsProvider(fclient)
    cik_tickers = sorted(panel.loc[panel["cik"].notna(), "ticker"].dropna().unique())
    logger.info("Loading as-traded closes for %d tickers with a CIK match", len(cik_tickers))
    raw_closes = load_raw_closes(cik_tickers, cfg.path("raw") / "prices_raw", start=start)
    baseline = build_baseline_features(panel, raw_closes, provider, panel["cik"])

    model_panel = panel.merge(baseline, on=["period", "composite_figi"], how="left")
    out = cfg.path("processed") / "model_panel.parquet"
    model_panel.to_parquet(out, index=False)
    logger.info("Wrote model panel (%d rows) -> %s", len(model_panel), out)

    coverage = {
        "rows": int(len(model_panel)),
        "tickers_with_prices": len(price_lookup),
        "tickers_total": len(tickers),
        "tickers_with_cik": len(cik_tickers),
        **cik_cov,
        "tickers_with_as_traded_prices": len(raw_closes),
        "with_shares_outstanding": int(model_panel["shares_outstanding"].notna().sum()),
        "with_market_cap": int(model_panel["market_cap"].notna().sum()),
        "market_cap_failed_checks": int((model_panel["market_cap"].notna() & ~model_panel["cap_ok"]).sum()),
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
