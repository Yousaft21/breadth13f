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

Over **45 quarterly formation dates (2014-Q1 … 2025-Q1)** and ~2,870 US-listed
equities per date, out-of-sample under purged & embargoed cross-validation, at
the **12-month** horizon:

| Model | Mean rank IC | ICIR | t-stat (Newey-West) | Decile L/S |
|-------|:---:|:---:|:---:|:---:|
| Size/value baseline | 0.0507 | 0.75 | 3.2 | — |
| **Full breadth + baseline** | **0.0587** | **0.74** | **3.1** | **−3.4%** |
| Breadth model, OOS-pruned | 0.0654 | 0.84 | 3.6 | −2.8% |
| *Breadth increment (full − baseline)* | *+0.0080* | *0.21* | *1.1* | |

The breadth features add **+0.0080 mean rank IC over the size/value baseline**
(the incremental effect), beating it in only 53% of quarters. Tested as a paired
per-quarter series the increment is **not statistically significant** (Newey-West
t 1.05, p ≈ 0.3). The model as a whole is significant; breadth's own
contribution on top of size/value cannot be distinguished from zero. On names
with a trustworthy market cap it adds nothing measurable (+0.001, validation
Q9–Q10); what increment there is comes from names the size/value inputs cannot
see.

t-statistics are **Newey-West** (4 lags at 12m): 12-month returns measured every
quarter overlap by 9 months, so consecutive ICs are correlated and the naive
t (4.9 for the full model) overstates the evidence. Out-of-sample backward elimination — which drops
features whose removal does not reduce OOS IC — raises IC from 0.0587 to 0.0654
by discarding two features, but that figure is **optimistic**: pruning chose
features on the same folds it is evaluated on, so the full model is the honest
headline. It is also unstable: each pipeline fix below changed which features it
drops (the pruned IC went 0.0772 → 0.0677 → 0.0654). The classic
Chen–Hong–Stein Δbreadth predictor **survives** pruning at both horizons. Results are consistent at the 6-month horizon (see below). These
are quarterly rank ICs; the full model is statistically significant but
economically modest, as expected for a single quarterly ownership signal. The
12-month decile spread is **negative**: D1 beats the universe, but D10's *mean*
return is even higher, carried by a few extreme small-cap winners, while its
*median* trails the universe by 8% (validation Q1). A quarterly-rebalanced
D1−D10 portfolio loses money; the usable edge is the long leg (validation Q6).

> **Correction (resolved).** An earlier version of this table reported a full-model
> IC of 0.072 (baseline 0.062, pruned 0.082). The
> [validation notebook](notebooks/validation_questions.ipynb) found that the
> size/value baseline computed market cap from split- and dividend-adjusted prices,
> which leaks later splits into earlier dates (a look-ahead). Phase 4 now uses
> as-traded prices and rejects implausible share counts; all numbers here come
> from the fixed pipeline. The fix lowered the baseline and full-model ICs by
> about 0.007 each and left **breadth's incremental IC unchanged at +0.0095**.
>
> **Universe fix.** Share-class tickers (BRK/B, BF/B, MOG/A, … — 24 names, ~0.7%
> of scored rows) had been silently dropped: OpenFIGI writes them with a slash,
> but Yahoo and Stooq expect a dash (BRK-B), so they never got prices. With them
> included, the full-model IC is unchanged (0.0645 → 0.0644) while the baseline
> falls (0.0550 → 0.0532), so the increment rises from +0.0095 to +0.0112. The
> rise is probably partly mechanical: most of these issuers report no single
> share count to the SEC, so they have breadth features but no size/value
> features, and only the full model can rank them.
>
> **Correction (resolved): point-in-time SEC issuer mapping.** The previous
> version of this table reported a full-model IC of 0.0644 (baseline 0.0532,
> increment +0.0112, decile L/S +5.8%). Validation Q10 found that Phase 4 mapped
> each stock's *current* ticker to an SEC CIK through SEC's *current*
> `company_tickers.json`. Firms that later went bankrupt, delisted or changed
> ticker were missing from that list, so they had no size/value features in any
> quarter, and their missing data reflected their future. These names were 2% of
> scored rows, with a median 12-month return of −35%, and the models learned to
> rank no-cap names low. Phase 4 now matches each quarter's 13F issuer names to
> the company names on EDGAR filings as of that date
> ([`src/mapping/issuer_cik.py`](src/mapping/issuer_cik.py)). The fix lowered
> the full-model IC by 0.006 and the baseline's by 0.003, so **the increment
> fell from +0.0112 to +0.0080**. LightGBM's missing-value handling was not the
> cause: refits that remove or flag the missing data change the IC by less
> than 0.003.

## Status — all phases complete

| Phase | Description | State |
|------:|-------------|-------|
| 1 | **Ingest** — crawl + parse 13F info tables into a tidy holdings table | ✅ |
| 2 | **Map + universe** — CUSIP→ticker (OpenFIGI), security master, point-in-time issuer→CIK | ✅ |
| 3 | **Features** — point-in-time breadth panel | ✅ |
| 4 | **Returns + baseline** — forward 6/12m returns, SEC-XBRL size/value baseline | ✅ |
| 5 | **Backtest** — purged/embargoed CV, LightGBM, IC/ICIR/decile/turnover | ✅ |
| 6 | **Attribution** — SHAP + out-of-sample backward elimination | ✅ |
| 7 | **Write-up** — results tables, plots, limitations | ✅ |

Built on the full 2014–2026 crawl (51 filing quarters, **68.9M… 116.8M holding
rows**, ~117k distinct CUSIPs). 98 unit tests pass; the two most bug-prone pieces
— the 13F XML parser and the purged/embargoed CV splitter — carry dedicated
tests, as do the breadth math, the point-in-time fundamentals selection, the
point-in-time market cap (a later split must not change an earlier cap), the
point-in-time issuer→CIK name matching, and the return math.

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
  mapping/         # CUSIP→ticker, security master, issuer→CIK (Phase 2/4)
  prices/          # forward returns + size/value baseline (Phase 4)
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
genuine PIT control, not a current snapshot. Issuer identity (the SEC CIK) is
resolved point-in-time: each quarter's 13F issuer names are matched to the
company names on EDGAR 10-K/10-Q/20-F/40-F filings as of that date
([`src/mapping/issuer_cik.py`](src/mapping/issuer_cik.py)), and SEC's current
`company_tickers.json` only fills names the match misses. The current ticker list
alone would be a look-ahead: firms that later failed or changed ticker are
missing from it (validation Q10).

The price is the **as-traded** close on or before the formation date: yfinance's
dividend-unadjusted close with later splits undone
([`src/prices/as_traded.py`](src/prices/as_traded.py)). The adjusted close used
for returns would be wrong here: it is rescaled for splits and dividends *after*
the date, which understates the earlier size of stocks that went on to split.
Each cap is also checked against 13F holdings: if 13F shares held ÷ XBRL shares
outstanding falls outside 1%–300% (ADR share ratios, XBRL scale errors) or the
cap exceeds $6T, the size/value features are left missing rather than wrong.

## Evaluation

Per formation date: Spearman **rank IC**; aggregated to mean IC, **ICIR**,
t-statistic and hit rate. t-statistics are reported naive and **Newey-West**
(lags = label horizon ÷ 3 months), because overlapping forward-return labels make
consecutive ICs autocorrelated; the incremental IC is tested as a paired
per-period series (full-model IC − baseline IC on the same quarters). Also: **top-minus-bottom decile** forward-return spread; and
top-book **turnover**. Cross-validation is **purged & embargoed** (López de
Prado, AFML Ch. 7), grouped by whole formation quarters, with the embargo sized
to the 6–12 month label horizon. The full model and the baseline are scored on
identical folds.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate      # Python 3.12
pip install -r requirements.txt
pytest                                              # 98 unit tests

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
| Scored universe (≥5 holders, labelable) | ~2,870 securities / formation date |
| Within-window delisting dropout (12m, labelable) | 0.02% mean (33 names) |

Out-of-sample performance under purged & embargoed CV (LightGBM ranker):

| Horizon | Model | Mean IC | ICIR | t (naive) | t (Newey-West) | Decile L/S | Turnover |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 12m | Size/value baseline | 0.0507 | 0.75 | 5.0 | 3.2 | — | — |
| 12m | **Full (16 features)** | **0.0587** | **0.74** | **4.9** | **3.1** | **−3.4%** | 0.45 |
| 12m | OOS-pruned (14 features)\* | 0.0654 | 0.84 | 5.6 | 3.6 | −2.8% | 0.43 |
| 12m | *Increment (full − baseline)* | *+0.0080* | *0.21* | *1.4* | *1.1* | | |
| 6m | Size/value baseline | 0.0527 | 0.70 | 4.8 | 4.0 | — | — |
| 6m | **Full (16 features)** | **0.0598** | **0.77** | **5.3** | **4.3** | **−1.1%** | 0.56 |
| 6m | OOS-pruned (15 features)\* | 0.0660 | 0.82 | 5.6 | 4.4 | −0.9% | 0.56 |
| 6m | *Increment (full − baseline)* | *+0.0071* | *0.18* | *1.2* | *1.1* | | |

\* Optimistic: features were pruned on the same folds the IC is measured on.
Newey-West lags: 4 at 12m, 2 at 6m.

**Incremental IC of breadth over the size/value baseline: +0.0080 (12m), +0.0071
(6m)** — positive in 53% / 55% of quarters, and not significant at either
horizon (Newey-West t 1.05 / 1.10). SHAP attribution shows the size/value controls (book-to-market,
earnings yield, size) are the largest contributors, followed by position-size
dispersion, top-10 concentration and aggregate institutional shares; the breadth
features add a smaller increment. Out-of-sample backward
elimination drops features whose removal does not reduce OOS IC — at 12m:
`exit_rate, new_position_rate`; at 6m: `exit_rate`. The
Chen–Hong–Stein Δbreadth predictor (`delta_breadth`) survives at both horizons
(dropping it lowers IC), as do all three size/value controls. The pruned list
has changed with every pipeline fix, so treat it as noise around a model in
which no single breadth feature matters much. Figures in [`results/figures/`](results/figures): per-period IC,
decile spread, cumulative long–short, and SHAP importance.

### Validation

[`notebooks/validation_questions.ipynb`](notebooks/validation_questions.ipynb)
probes the factor and its data, one question per section, each ending in a chart
or table: decile monotonicity, surviving features, momentum and size overlap,
interest-rate regimes, decile stability, cap- vs equal-weighting, and two
data-quality checks (look-ahead in the market caps, which it found and which is
now fixed; SEC share-count sanity), then size quintiles and names with no
size/value data (Q9–Q10).
Helpers live in [`src/backtest/validation.py`](src/backtest/validation.py).

Interpretation: a statistically significant (Newey-West t ≈ 3.1 over 45
quarters) but economically modest cross-sectional signal, dominated by
size/value. Breadth's increment (+0.008 IC, positive in about half of quarters)
cannot be distinguished from zero with 45 overlapping quarters, and on names
with a trustworthy market cap it is nil (+0.001). It is a null result for
breadth as an addition to size/value, with a weak hint that breadth helps rank
the names size/value data cannot see. The top decile does beat the universe,
but a long-short version does not work (validation Q1, Q6).

## Limitations

Quantified and structural, in rough order of importance:

- **Survivorship / delisting bias (free price data).** yfinance/Stooq carry no
  delisting returns and drop delisted tickers entirely. Within-window dropout is
  tiny (0.02%), but that *understates* the bias: the larger effect is names
  absent from the feed up front — ~2.2% of panel tickers had no usable price,
  plus unmapped CUSIPs. Because delisted names skew to poor performers, measured
  returns are biased **optimistically**. No point-in-time index membership is
  used, so the universe is reconstructed, not survivorship-free.
- **CUSIP→ticker mapping loss.** 45% of distinct CUSIPs (but only ~10% of
  holding-*value*) fail to map to a US-listed equity — obscure, foreign, or
  delisted lines. Carried through as a documented data-loss figure.
- **Current-symbology mapping.** OpenFIGI returns current, not as-of-date,
  tickers; security identity is keyed on `composite_figi` to guard against
  ticker reuse, but a ticker relabel mid-history is not modelled.
- **Look-ahead in the size/value baseline (resolved).** Market cap was first
  computed from adjusted closes, which are back-scaled for *later* splits and
  dividends: 10% of stock-quarters were understated by ≥ 1.5× (AMZN 20×, NVDA 39×
  at end-2019), so future winners looked small and cheap. Phase 4 now uses
  as-traded prices. The fix moved the 12m baseline IC from 0.062 to 0.055 and
  the full model's from 0.072 to 0.0645 (6m: 0.057 → 0.0525 and 0.067 → 0.0625);
  breadth's increment was unchanged (+0.0095 at 12m; +0.0101 → +0.0100 at 6m).
  (These are before share-class tickers were added; see the universe fix above.)
  Validation Q7 replays the old features to measure this.
- **SEC share-count errors (filtered).** 3.1% of scored stock-quarters with a
  market cap fail a 13F-vs-XBRL share-count check: ADRs (price per ADR, shares
  counted as ordinary shares) and filing scale errors (validation Q8). Their
  size/value features are set to missing. The 1% floor on institutional
  ownership is conservative and also drops some foreign issuers whose caps are
  roughly right.
- **Share-count date vs. splits.** The XBRL share count is the latest one filed
  before formation, so a split between its cover date and the formation date
  leaves the cap off by the split ratio. Not a look-ahead, and small: 0.7% of
  scored stock-quarters with a trusted cap had a split in the 100 days before
  formation.
- **Baseline fundamentals coverage.** An SEC CIK resolves for 75% of panel rows
  and an XBRL share count for 64% (ADRs/foreign issuers file 20-F without
  us-gaap tags); after the share-count check the size feature covers 55% of
  panel rows and 77% of the scored universe, so the baseline is partial. Dual-class issuers (GOOGL/GOOG,
  BRK/A/B, BF/A/B, …) mostly report no undimensioned share count, so they get
  no size/value features. The fundamental facts themselves
  are point-in-time (selected by each fact's SEC `filed` date).
- **Current-symbology CIK mapping (resolved).** Phase 4 used to map tickers to
  SEC CIKs through today's `company_tickers.json`, so firms that later went
  bankrupt, delisted or changed ticker had no size/value features in any
  quarter: their missing data was a look-ahead (validation Q10). Issuers are
  now matched point-in-time by 13F issuer name to EDGAR filer names. On rows
  both methods cover, the name match finds 94% and agrees on 99.85%. Known
  residue: the current ticker list still fills names the match misses, and a
  few quarters map to a related filer (Alcoa Inc vs Alcoa Corp in 2016). The
  fix lowered the 12m full-model IC from 0.0644 to 0.0587 and the increment
  from +0.0112 to +0.0080.
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
