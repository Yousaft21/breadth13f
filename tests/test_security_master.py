"""Tests for CUSIP->security representative selection and coverage stats."""

from __future__ import annotations

import pandas as pd

from src.mapping.security_master import (
    build_security_master,
    coverage_report,
    distinct_equity_cusips,
    select_representative,
)


def _inst(ticker, exch, sectype, sector="Equity", cfigi="FIGICOMP1"):
    return {
        "ticker": ticker, "exchCode": exch, "securityType": sectype,
        "securityType2": sectype, "marketSector": sector,
        "compositeFIGI": cfigi, "shareClassFIGI": "SCFIGI1", "figi": "FIGI1",
        "name": "TEST ISSUER",
    }


def test_prefers_us_common_stock_over_foreign_listing():
    instruments = [
        _inst("AAPL", "GR", "Common Stock"),   # German listing
        _inst("AAPL", "US", "Common Stock"),   # US composite — preferred
        _inst("AAPL", "LN", "Common Stock"),   # London listing
    ]
    rep = select_representative(instruments)
    assert rep["exchCode"] == "US"


def test_no_match_returns_none():
    assert select_representative([]) is None


def test_falls_back_to_any_ticker_when_no_us_common():
    instruments = [_inst("XYZ", "LN", "Preferred")]
    rep = select_representative(instruments)
    assert rep["ticker"] == "XYZ"


def test_build_master_flags_us_common_equity():
    mapping = {
        "037833100": [_inst("AAPL", "US", "Common Stock")],
        "111111111": [_inst("BONDX", "US", "Corp Bond", sector="Corp")],
        "222222222": [],  # unmapped
    }
    master = build_security_master(mapping)
    by_cusip = master.set_index("cusip")
    assert bool(by_cusip.loc["037833100", "is_us_common_equity"]) is True
    assert bool(by_cusip.loc["111111111", "is_us_common_equity"]) is False
    assert bool(by_cusip.loc["222222222", "is_us_common_equity"]) is False
    assert pd.isna(by_cusip.loc["222222222", "ticker"])


def test_coverage_report_counts():
    mapping = {
        "037833100": [_inst("AAPL", "US", "Common Stock")],
        "594918104": [_inst("MSFT", "US", "Common Stock")],
        "222222222": [],
    }
    master = build_security_master(mapping)
    rep = coverage_report(master, total_cusips=3)
    assert rep["distinct_cusips"] == 3
    assert rep["mapped_any"] == 2
    assert rep["us_common_equity"] == 2
    assert rep["unmapped"] == 1


def test_distinct_equity_cusips_excludes_options():
    holdings = pd.DataFrame(
        {
            "cusip": ["037833100", "037833100", "594918104", "111111111"],
            "put_call": [None, None, None, "Call"],  # last row is an option line
        }
    )
    cusips = distinct_equity_cusips(holdings)
    assert cusips == ["037833100", "594918104"]  # option-only CUSIP excluded, deduped
