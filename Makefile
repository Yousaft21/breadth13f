# Reproducible end-to-end pipeline. Each phase is also a standalone module CLI.
# On Windows without `make`, run the PY= commands directly in a shell.
#
# Point-in-time, cached and resumable throughout: re-running only does new work.

PY ?= .venv/Scripts/python.exe        # use .venv/bin/python on Linux/macOS

.PHONY: all test ingest map features prices backtest clean pilot

## Full run across the config window (long: full EDGAR crawl).
all: ingest map features prices backtest

## Fast validation slice used during development.
pilot:
	$(PY) -m src.ingest.run_ingest --filing-quarters 2018Q4 2021Q4
	$(MAKE) map features prices backtest

test:
	$(PY) -m pytest

## Phase 1 — crawl + parse 13F info tables.
ingest:
	$(PY) -m src.ingest.run_ingest

## Phase 2 — CUSIP -> ticker security master (set OPENFIGI_API_KEY to speed up).
map:
	$(PY) -m src.mapping.run_mapping

## Phase 3 — point-in-time breadth-feature panel.
features:
	$(PY) -m src.features.run_features

## Phase 4 — forward returns + size/value baseline.
prices:
	$(PY) -m src.prices.run_prices

## Phase 5/6 — purged-CV backtest, baseline comparison, SHAP pruning.
backtest:
	$(PY) -m src.backtest.run_backtest

## Remove derived data (keeps raw cache so nothing re-downloads).
clean:
	rm -rf data/interim data/processed results/tables results/figures
