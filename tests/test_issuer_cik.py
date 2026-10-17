"""Tests for the point-in-time CUSIP -> CIK mapping from issuer names."""

from __future__ import annotations

import pandas as pd
import pytest

from src.mapping.issuer_cik import (
    combine_ciks,
    match_names,
    match_quality,
    name_tokens,
    names_by_security,
    pit_ciks,
    token_matches,
)


@pytest.mark.parametrize("a, b", [
    ("APPLE INC", "Apple Inc."),
    ("APPLE INC COM", "APPLE INC"),
    ("FREEPORT-MCMORAN INC CL B", "FREEPORT-MCMORAN INC"),
    ("ACME CORP /DE/", "ACME CORP"),
    ("ITRON INC \\WA\\", "ITRON INC"),
    ("AT&T INC.", "AT & T INC"),
    ("R F INDS LTD", "RF INDS"),            # runs of single letters are joined
    ("TIM S A", "TIM S.p.A."),
    ("SIMMONS 1ST NATL CORP", "SIMMONS FIRST NATL"),
    ("KLX ENERGY SERVICS HOLDNGS I", "KLX ENERGY SERVICS HLDG"),  # truncated stub dropped
    ("PALVELLA THERAPEUTICS INC NE", "PALVELLA THERAPEUTIC"),
])
def test_name_tokens_normalise_equivalent_spellings(a, b):
    assert name_tokens(a) == name_tokens(b)


def test_name_tokens_keep_partnership_and_digits():
    # A REIT and its operating partnership must stay distinct.
    assert name_tokens("CUBESMART") != name_tokens("CubeSmart, L.P.")
    assert name_tokens("PHILLIPS 66") == ("PHILLIP", "66")
    assert name_tokens("INC") == ("INC",)       # never empty
    assert name_tokens(None) == ()


@pytest.mark.parametrize("t13, ted, ok", [
    ("BANK", "BANK", True),
    ("SOLUT", "SOLUTION", True),        # truncation
    ("SOLUTN", "SOLUTION", True),       # abbreviation
    ("FINL", "FINANCIAL", True),        # in-order abbreviation
    ("HLDG", "HLDG", True),
    ("BK", "BANK", True),
    ("CMNTY", "COMMUNITY", True),
    ("B", "BANK", False),               # one letter is too little
    ("FINANCIAL", "FINL", False),       # 13F token longer than the EDGAR one
    ("XFINL", "FINANCIAL", False),      # different first letter
    ("2", "3", False),
    ("66", "660", False),               # digits must be equal
])
def test_token_matches(t13, ted, ok):
    assert token_matches(t13, ted) is ok


def test_match_quality_tiers():
    q = lambda a, b, trunc=False: match_quality(name_tokens(a), name_tokens(b), trunc)  # noqa: E731
    assert q("INTEL CORP", "INTEL CORP") == 2
    assert q("ALLSCRIPTS HEALTHCARE SOLUT", "Allscripts Healthcare Solutions, Inc.", True) == 2
    assert q("ALLSCRIPTS HEALTHCARE SOLUTN", "Allscripts Healthcare Solutions, Inc.", True) == 1
    assert q("UNIVERSAL HEALTH RLTY INCM T", "UNIVERSAL HEALTH REALTY INCOME TRUST", True) == 1
    assert q("MID-AMER APT CMNTYS INC", "MID AMERICA APARTMENT COMMUNITIES INC.") == 1
    assert q("ROCKY MTN CHOCOLATE FACTORY", "ROCKY MOUNTAIN CHOCOLATE FACTORY INC", True) == 1
    # One abbreviated word, or an abbreviated first word, is too weak.
    assert q("INTEL CORP", "INTERMOLECULAR INC") == 0
    assert q("ARM HOLDINGS PLC", "Affirm Holdings, Inc.") == 0
    assert q("LI AUTO INC", "LifeSci Acquisition Corp.") == 0
    assert q("PEOPLES BANCORP N C INC", "PEOPLES BANCORP INC") == 0
    assert q("FIRST REP BK SAN FRANCISCO", "FIRST REPUBLIC BANK SAN FRANCISCO CALIF", True) == 1
    # Extra EDGAR tokens are allowed only when the 13F name looks truncated.
    assert q("GREAT SOUTHERN BANCORP SAN", "GREAT SOUTHERN BANCORP SAN DIEGO", True) == 2
    assert q("GREAT SOUTHERN BANCORP SAN", "GREAT SOUTHERN BANCORP SAN DIEGO") == 0


SPELLS = pd.DataFrame({
    "cik": [1, 1, 2, 3, 4],
    "name": ["ACME WIDGETS INC", "Acme Widgets, Inc.", "ACME WIDGETRONICS CORP",
             "BETA FINANCIAL CORP", "GAMMA MINES LTD"],
    "first_filed": pd.to_datetime(["2014-02-01", "2019-02-01", "2014-02-01",
                                   "2014-02-01", "2014-02-01"]),
    "last_filed": pd.to_datetime(["2018-11-01", "2026-05-01", "2026-05-01",
                                  "2016-05-01", "2026-05-01"]),
})


def test_match_names_returns_quality():
    m = match_names(["ACME WIDGETS INC", "BETA FINL CORP", "ZETA CORP"], SPELLS)
    got = {(r.name_13f, r.cik, r.quality) for r in m.itertuples()}
    assert ("ACME WIDGETS INC", 1, 2) in got
    assert ("BETA FINL CORP", 3, 1) in got
    assert not any(n == "ZETA CORP" for n, _, _ in got)
    # "ACME WIDGETS" is an in-order abbreviation of "ACME WIDGETRONICS" too.
    assert ("ACME WIDGETS INC", 2, 1) in got


def _rows(cusip, period, formation):
    return pd.DataFrame({"cusip": [cusip], "period": [pd.Timestamp(period)],
                         "formation_date": [pd.Timestamp(formation)]})


def _names(cusip, period, counts):
    return pd.DataFrame([{"cusip": cusip, "period": pd.Timestamp(period), "name": n,
                          "n_reports": k} for n, k in counts.items()])


def test_pit_ciks_exact_match_beats_abbreviation_and_junk():
    names = _names("C1", "2020-03-31", {"ACME WIDGETS INC": 80, "ACME WIDGETS INC COM": 15,
                                        "GAMMA MINES LTD": 5})          # one mistyped report
    matches = match_names(names["name"], SPELLS)
    out = pit_ciks(_rows("C1", "2020-03-31", "2020-05-16"), names, matches)
    assert out.iloc[0] == 1


def test_pit_ciks_needs_support():
    names = _names("C1", "2020-03-31", {"GAMMA MINES LTD": 4, "SOMETHING ELSE": 6})
    matches = match_names(names["name"], SPELLS)
    out = pit_ciks(_rows("C1", "2020-03-31", "2020-05-16"), names, matches, min_support=0.5)
    assert out.isna().iloc[0]


def test_pit_ciks_uses_only_names_reported_for_the_period():
    names = _names("C1", "2021-03-31", {"ACME WIDGETS INC": 10})
    matches = match_names(names["name"], SPELLS)
    out = pit_ciks(_rows("C1", "2020-03-31", "2020-05-16"), names, matches)
    assert out.isna().iloc[0]                 # the name is first reported a year later


def test_pit_ciks_respects_edgar_filing_dates():
    names = _names("C1", "2013-12-31", {"ACME WIDGETS INC": 10})
    matches = match_names(names["name"], SPELLS)
    # Formed before the CIK's first periodic report under that name: no match yet.
    assert pit_ciks(_rows("C1", "2013-12-31", "2014-01-15"), names, matches).isna().iloc[0]
    # A CIK that stopped filing more than recency_days before formation is not used.
    names = _names("C2", "2019-12-31", {"BETA FINANCIAL CORP": 10})
    matches = match_names(names["name"], SPELLS)
    rows = _rows("C2", "2019-12-31", "2020-02-15")
    assert pit_ciks(rows, names, matches, recency_days=730).isna().iloc[0]
    assert pit_ciks(rows, names, matches, recency_days=1500).iloc[0] == 3


def test_pit_ciks_tie_is_unmapped():
    spells = pd.DataFrame({"cik": [7, 8], "name": ["DELTA BANCORP INC", "DELTA BANCORP /PR/"],
                           "first_filed": pd.to_datetime(["2014-01-01"] * 2),
                           "last_filed": pd.to_datetime(["2026-01-01"] * 2)})
    names = _names("C3", "2020-03-31", {"DELTA BANCORP": 10})
    out = pit_ciks(_rows("C3", "2020-03-31", "2020-05-16"), names,
                   match_names(names["name"], spells))
    assert out.isna().iloc[0]


def test_combine_ciks_prefers_point_in_time_name_match():
    name = pd.Series([1, pd.NA, pd.NA, 4], dtype="Int64")
    ticker = pd.Series([9, 2, pd.NA, 4], dtype="Int64")
    cik, source = combine_ciks(name, ticker)
    assert cik.tolist() == [1, 2, pd.NA, 4]
    assert source.tolist() == ["name", "ticker", "none", "name+ticker"]
    cik, source = combine_ciks(name, ticker, use_ticker=False)
    assert cik.tolist() == [1, pd.NA, pd.NA, 4]


def test_names_by_security_pools_cusips_of_one_security():
    names = pd.DataFrame({"cusip": ["OLD", "NEW", "NEW", "OTHER"],
                          "period": pd.to_datetime(["2020-03-31"] * 4),
                          "name": ["ACME WIDGETS INC"] * 3 + ["BETA FINL CORP"],
                          "n_reports": [3, 5, 2, 7]})
    master = pd.DataFrame({"cusip": ["OLD", "NEW", "OTHER"], "composite_figi": ["F1", "F1", "F2"]})
    out = names_by_security(names, master).set_index(["composite_figi", "name"])["n_reports"]
    assert out.to_dict() == {("F1", "ACME WIDGETS INC"): 10, ("F2", "BETA FINL CORP"): 7}
