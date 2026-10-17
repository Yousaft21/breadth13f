"""Tests for the panel-ticker -> Yahoo/Stooq symbol mapping and cache invalidation."""

from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from src.prices import returns
from src.prices.as_traded import load_raw_closes
from src.prices.returns import _cache_path, fetched_with_current_symbol, load_price_series, yahoo_symbol


@pytest.mark.parametrize("ticker, symbol", [
    ("BRK/B", "BRK-B"),
    ("MOG/A", "MOG-A"),
    ("UHAL/B", "UHAL-B"),
    ("AXIA/PC", "AXIA-PC"),
    (" brk/a ", "BRK-A"),
    ("AAPL", "AAPL"),
    ("^TNX", "^TNX"),
])
def test_yahoo_symbol(ticker, symbol):
    assert yahoo_symbol(ticker) == symbol


def test_old_sidecar_without_symbol_is_stale_only_for_share_classes():
    assert fetched_with_current_symbol({}, "AAPL")
    assert not fetched_with_current_symbol({}, "BRK/B")
    assert fetched_with_current_symbol({"symbol": "BRK-B"}, "BRK/B")


def _prices():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3), "adj_close": [1.0, 2.0, 3.0]})


def _write_old_cache(cache_dir, ticker, df):
    """A cache as written before symbols were normalised: no symbol in the sidecar."""
    path = _cache_path(cache_dir, ticker)
    df.to_parquet(path, index=False)
    path.with_name(path.stem + ".range.json").write_text(
        json.dumps({"start": "2010-01-01", "end": "2030-01-01"}), encoding="utf-8")


def test_load_price_series_refetches_stale_share_class_cache(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(returns, "_fetch_yfinance", lambda s, a, b: requested.append(s) or _prices())
    monkeypatch.setattr(returns, "_fetch_stooq", lambda s, a, b: None)
    _write_old_cache(tmp_path, "BRK/B", pd.DataFrame(columns=["date", "adj_close"]))

    df = load_price_series("BRK/B", "2015-01-01", "2020-01-01", tmp_path)
    assert requested == ["BRK-B"] and len(df) == 3
    # Now cached under the current symbol: no second request.
    assert len(load_price_series("BRK/B", "2015-01-01", "2020-01-01", tmp_path)) == 3
    assert requested == ["BRK-B"]


def test_load_price_series_keeps_valid_old_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(returns, "_fetch_yfinance", lambda *a: pytest.fail("re-fetched a valid cache"))
    _write_old_cache(tmp_path, "AAPL", _prices())
    assert len(load_price_series("AAPL", "2015-01-01", "2020-01-01", tmp_path)) == 3


def _fake_yfinance(monkeypatch, requested):
    def download(symbols, **kwargs):
        requested.append(list(symbols))
        idx = pd.bdate_range("2020-01-01", periods=3)
        frames = {s: pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Stock Splits": 0.0}, index=idx)
                  for s in symbols}
        return pd.concat(frames, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))


def test_load_raw_closes_requests_dash_symbol_and_refetches_stale_cache(tmp_path, monkeypatch):
    requested = []
    _fake_yfinance(monkeypatch, requested)
    # Pre-normalisation caches: no sidecar; share-class one empty, plain one valid.
    pd.DataFrame(columns=["date", "close", "split"]).to_parquet(_cache_path(tmp_path, "BF/B"), index=False)
    pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=2), "close": [5.0, 6.0],
                  "split": 0.0}).to_parquet(_cache_path(tmp_path, "AAPL"), index=False)

    out = load_raw_closes(["BF/B", "AAPL"], tmp_path)
    assert requested == [["BF-B"]]
    assert set(out) == {"BF/B", "AAPL"} and len(out["BF/B"]) == 3

    load_raw_closes(["BF/B", "AAPL"], tmp_path)
    assert requested == [["BF-B"]]
