# Institutional Ownership-Breadth Signal from SEC 13F Filings

This project investigates whether the **breadth of institutional ownership** —
and, more importantly, its quarter-over-quarter *change* — reconstructed from
public SEC Form 13F filings, predicts the cross-section of forward 6–12 month
US equity returns.

The design is anchored in Chen, Hong & Stein (2002), *"Breadth of Ownership and
Stock Returns"*, which argues that in the presence of short-sale constraints a
**rising number of institutional holders** reveals previously side-lined
positive information and predicts higher subsequent returns. We reconstruct a
security-level breadth panel from 13F information tables and extend the canonical
Δbreadth predictor with a family of delta-based ownership features, then test
their predictivity under purged & embargoed cross-validation.

The emphasis throughout is methodological: point-in-time correctness, no
lookahead, honest out-of-sample evaluation, and full reproducibility. Reported
numbers are whatever the pipeline produces — including null or negative results.

## Headline result

Over **45 quarterly formation dates (2014-Q1 … 2025-Q1)** and ~2,850 US-listed
equities per date, out-of-sample under purged & embargoed cross-validation, at
the **12-month** horizon:

| Model | Mean rank IC | ICIR | t-stat | Decile L/S |
|-------|:---:|:---:|:---:|:---:|
| Size/value baseline | 0.062 | 0.75 | 5.0 | — |
| Full breadth + baseline | 0.072 | 0.79 | 5.3 | +7.7% |
| **Breadth model, OOS-pruned** | **0.082** | **1.00** | **6.7** | **+9.7%** |

The breadth features add **+0.0095 mean rank IC over the size/value baseline**
(the incremental effect), and out-of-sample backward elimination — which drops
features whose removal does not reduce OOS IC — raises IC from 0.072 to 0.082 by
discarding six redundant/noisy features. The classic Chen–Hong–Stein Δbreadth
predictor **survives** pruning at 12 months. Results are consistent at the
6-month horizon (see below). These are quarterly rank ICs; the signal is real
and statistically significant but economically modest, as expected for a single
quarterly ownership signal.

## Status — all phases complete

| Phase | Description | State |
|------:|-------------|-------|
| 1 | **Ingest** — crawl + parse 13F info tables into a tidy holdings table | ✅ |
| 2 | **Map + universe** — CUSIP→ticker (OpenFIGI), security master | ✅ |
| 3 | **Features** — point-in-time breadth panel | ✅ |
| 4 | **Returns + baseline** — forward 6/12m returns, SEC-XBRL size/value baseline | ✅ |
| 5 | **Backtest** — purged/embargoed CV, LightGBM, IC/ICIR/decile/turnover | ✅ |
| 6 | **Attribution** — SHAP + out-of-sample backward elimination | ✅ |
| 7 | **Write-up** — results tables, plots, limitations | ✅ |

Built on the full 2014–2026 crawl (51 filing quarters, **68.9M… 116.8M holding
rows**, ~117k distinct CUSIPs). 39 unit tests pass; the two most bug-prone pieces
— the 13F XML parser and the purged/embargoed CV splitter — carry dedicated
tests, as do the breadth math, the point-in-time fundamentals selection, and the
return math.

## Data

- **13F filings** — SEC EDGAR (`13F-HR`, `13F-HR/A`). Since mid-2013 the
  information table is standardized XML; the sample window starts at **2014-Q1**
  to avoid early-format messiness and runs through **2026-Q1**
  (period-of-report quarters).
- **Prices/returns** — free sources only (yfinance primary, Stooq cross-check).
  These carry **survivorship/delisting bias** (delisted names silently vanish),
  which is quantified and documented rather than hidden.
- **CUSIP→ticker mapping** — OpenFIGI + SEC crosswalks; mapping coverage is
  reported as an explicit data-loss figure.

## Point-in-time discipline

13F filings are due up to **45 days after quarter-end**, so a quarter's breadth
is only knowable once filings have arrived. Every filing's true
`ACCEPTANCE-DATETIME` is parsed from the SGML header and preserved, and the
signal for each formation date uses **only filings accepted on or before that
date** (default snapshot: quarter-end + 46 days). Filings are crawled by *filing*
quarter and grouped by *period-of-report*, so late filings and amendments are
attributed to the correct holding period without leaking into earlier dates.

## Repository layout

```
config/            # dates, universe filters, CV params (single YAML, config-driven)
src/
  ingest/          # EDGAR crawler + 13F XML parser  (Phase 1)
  mapping/         # CUSIP→ticker, security master   (Phase 2)
  prices/          # forward returns                 (Phase 4)
  features/        # point-in-time breadth panel     (Phase 3)
  backtest/        # purged/embargoed CV, model, IC  (Phase 5)
  viz/             # IC decay, decile spreads, SHAP  (Phase 6)
tests/             # parser + purged-CV splitter (priority tests)
data/              # gitignored; fully rebuildable from scripts
results/           # metrics tables + figures
```

## Signal & features

A quarterly security × formation-date panel; every feature is a point-in-time
quarter-over-quarter statistic of the 13F holder base:

| Feature | Meaning |
|---------|---------|
| `breadth_level` | holders / active managers that quarter |
| `delta_breadth`, `delta_breadth_pct` | QoQ change in # holders (**core CHS predictor**) |
| `net_buyer_breadth` | (# increasing − # decreasing) / holders |
| `new_position_rate`, `exit_rate` | initiations / prior-quarter exits, normalised |
| `hhi`, `delta_hhi`, `top5_share`, `top10_share` | holdings concentration across managers |
| `log_total_inst_shares`, `delta_total_inst_shares_pct` | aggregate institutional position & change |
| `position_size_dispersion` | spread of position sizes across holders |

Concentration uses **share counts**, so it is immune to the 2023 change in the
13F `value` field units (thousands → whole USD).

## Baseline (size/value control)

The headline metric is the breadth model's *incremental* rank IC over a
point-in-time **size/value** baseline. Size is `log(market cap)` from
formation-date price × shares outstanding; value is book-to-market and
earnings-yield. Fundamentals come from **SEC EDGAR XBRL** (`companyfacts`),
selected as-of each formation date using each fact's own SEC filing date — a
genuine PIT control, not a current snapshot. Issuer identity is resolved
ticker→CIK via SEC `company_tickers.json`.

## Evaluation

Per formation date: Spearman **rank IC**; aggregated to mean IC, **ICIR**,
t-statistic and hit rate; **top-minus-bottom decile** forward-return spread; and
top-book **turnover**. Cross-validation is **purged & embargoed** (López de
Prado, AFML Ch. 7), grouped by whole formation quarters, with the embargo sized
to the 6–12 month label horizon. The full model and the baseline are scored on
identical folds.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate      # Python 3.12
pip install -r requirements.txt
pytest                                              # 39 unit tests

# Credentials are read from the environment or an auto-loaded, gitignored .env:
#   EDGAR_USER_AGENT="your-project your-email@example.com"   # SEC fair-access policy
#   OPENFIGI_API_KEY="..."   # optional: 100 jobs/request vs 10 (~3 min vs ~5 h mapping)
cp .env.example .env   # then edit

# End-to-end (each phase is cached & resumable). With GNU make:
make all            # full 2014–2026 window (long: full EDGAR crawl)
#   or run the phases directly:
python -m src.ingest.run_ingest       # 1 crawl + parse 13F
python -m src.mapping.run_mapping      # 2 CUSIP -> ticker security master
python -m src.features.run_features    # 3 point-in-time breadth panel
python -m src.prices.run_prices        # 4 forward returns + size/value baseline
python -m src.backtest.run_backtest    # 5/6 purged-CV backtest + SHAP pruning
```

Everything is **resumable**: raw EDGAR/OpenFIGI/price downloads are cached to
`data/raw/`, and any filing quarter already written under `data/interim/` is
skipped on re-run. A quick single-quarter sample to sanity-check ingest:
`python -m src.ingest.run_ingest --filing-quarters 2020Q1 2020Q1 --limit 250`.

## EDGAR fair access

All requests carry a descriptive `User-Agent` with a contact address and are
rate-limited below the SEC's 10 requests/second cap (default 7/s). Nothing is
re-downloaded once cached.

## Results

Universe & coverage (full 2014–2026 build):

| | |
|---|---:|
| Filing quarters crawled | 51 (2014-Q1 … 2026-Q3) |
| Long-equity holding rows | 116.8 M |
| Distinct CUSIPs | 117,731 |
| CUSIP → any instrument (OpenFIGI) | 45.0% of CUSIPs / **90% holdings-weighted** |
| **US-listed equity universe** (common + ADR + REIT) | **8,287** |
| Breadth panel | 50 periods, 8,151 securities, 206,684 rows |
| Scored universe (≥5 holders, labelable) | ~2,850 securities / formation date |
| Within-window delisting dropout (12m, labelable) | 0.02% mean (33 names) |

Out-of-sample performance under purged & embargoed CV (LightGBM ranker):

| Horizon | Model | Mean IC | ICIR | t | Decile L/S | Turnover |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 12m | Size/value baseline | 0.062 | 0.75 | 5.0 | — | — |
| 12m | Full (16 features) | 0.072 | 0.79 | 5.3 | +7.7% | 0.44 |
| 12m | **OOS-pruned (10 features)** | **0.082** | **1.00** | **6.7** | **+9.7%** | 0.42 |
| 6m | Size/value baseline | 0.057 | 0.68 | 4.7 | — | — |
| 6m | Full (16 features) | 0.067 | 0.84 | 5.8 | +3.4% | 0.54 |
| 6m | **OOS-pruned (9 features)** | **0.075** | **0.99** | **6.8** | **+3.8%** | 0.52 |

**Incremental IC of breadth over the size/value baseline: +0.0095 (12m), +0.0101
(6m).** SHAP attribution shows size, book-to-market and position-size dispersion
are the largest contributors; the breadth features add a smaller but consistent
increment. Out-of-sample backward elimination drops features whose removal does
not reduce OOS IC — at 12m: `delta_hhi, hhi, exit_rate, new_position_rate,
top10_share, earnings_yield`. The Chen–Hong–Stein Δbreadth predictor survives
at 12 months (dropping it lowers IC) but is among the dropped at 6 months —
`delta_breadth_pct` and `breadth_level` carry the breadth signal at both
horizons. Figures in [`results/figures/`](results/figures): per-period IC,
decile spread, cumulative long–short, and SHAP importance.

Interpretation: a genuine, statistically significant (t ≈ 5–7 over 45 quarters)
but economically modest cross-sectional signal, dominated by size/value with a
small consistent breadth increment — the kind of honest, marginal result
expected from a single quarterly ownership signal, not an outsized alpha.

## Limitations

Quantified and structural, in rough order of importance:

- **Survivorship / delisting bias (free price data).** yfinance/Stooq carry no
  delisting returns and drop delisted tickers entirely. Within-window dropout is
  tiny (0.02%), but that *understates* the bias: the larger effect is names
  absent from the feed up front — ~2.5% of panel tickers had no usable price,
  plus unmapped CUSIPs. Because delisted names skew to poor performers, measured
  returns are biased **optimistically**. No point-in-time index membership is
  used, so the universe is reconstructed, not survivorship-free.
- **CUSIP→ticker mapping loss.** 45% of distinct CUSIPs (but only ~10% of
  holding-*value*) fail to map to a US-listed equity — obscure, foreign, or
  delisted lines. Carried through as a documented data-loss figure.
- **Current-symbology mapping.** OpenFIGI returns current, not as-of-date,
  tickers; security identity is keyed on `composite_figi` to guard against
  ticker reuse, but a ticker relabel mid-history is not modelled.
- **Baseline fundamentals coverage.** SEC-XBRL fundamentals resolve for ~55% of
  rows (ADRs/foreign issuers file 20-F without us-gaap tags), so the value
  baseline is partial — though it is genuinely point-in-time (selected by each
  fact's SEC `filed` date).
- **Structural limits of 13F.** Long US-listed positions only, managers with
  ≥ $100M qualifying AUM, quarterly granularity, 45-day filing lag — no shorts,
  no sub-threshold or non-13F holders. The signal can only ever reflect this
  slice of ownership.
- **Recent quarters unlabelled.** Formations after ~2025-Q1 lack a full 12-month
  forward return and are held out of scoring (kept in the panel for QoQ
  feature continuity).

## Reference

Chen, J., Hong, H., & Stein, J. C. (2002). *Breadth of Ownership and Stock
Returns.* Journal of Financial Economics, 66(2–3), 171–205.
