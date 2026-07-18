"""Build a point-in-time-aware security master from CUSIP->FIGI mappings.

Each 13F CUSIP is resolved via OpenFIGI to one or more instruments (share
classes and/or trading venues). :func:`select_representative` reduces that list
to a single tradable US-equity representative, preferring the US composite
listing and common stock. The resulting security master is keyed by CUSIP and
carries the fields needed to (a) define the equity universe and (b) fetch prices.

PIT caveat (documented, not hidden): OpenFIGI returns the *current* symbology
for a security, not an as-of-date mapping. CUSIP -> security *identity* is stable
(we key tradable identity on ``composite_figi``), but a ticker string can change
over the sample. We map to the current ticker and flag this as a limitation; the
``composite_figi`` guards against ticker-reuse collisions across issuers.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger("breadth13f.secmaster")

# Market sector / security types that constitute the US-listed equity universe.
_EQUITY_SECTOR = "Equity"
# Preferred type when a CUSIP resolves to several instruments.
_COMMON_TYPES = {"Common Stock"}
# The universe = US-listed common stock + ADRs (US-traded foreign issuers) +
# equity REITs. All three are US-exchange-traded equity to which institutional
# ownership breadth applies; ETFs, funds, bonds/notes and foreign-listed lines
# are excluded.
_US_EQUITY_TYPES = {"Common Stock", "ADR", "REIT"}

MASTER_COLS = [
    "cusip", "ticker", "name", "composite_figi", "share_class_figi",
    "figi", "security_type", "security_type2", "market_sector", "exch_code",
    "is_us_common_equity", "n_instruments",
]


def select_representative(instruments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best single instrument for a CUSIP, or None if unmapped.

    Preference order: US composite listing (``exchCode == 'US'``) first, then
    common stock, then anything with a ticker.
    """
    if not instruments:
        return None

    def has_ticker(inst: dict) -> bool:
        return bool(inst.get("ticker"))

    us = [i for i in instruments if i.get("exchCode") == "US"]
    pool = us or instruments

    common_us = [i for i in pool if i.get("securityType") in _COMMON_TYPES and has_ticker(i)]
    if common_us:
        return common_us[0]
    with_ticker = [i for i in pool if has_ticker(i)]
    if with_ticker:
        return with_ticker[0]
    return pool[0]


def _is_us_common_equity(inst: dict[str, Any]) -> bool:
    """US-listed equity: common stock, ADR or REIT on the US composite tape."""
    return (
        inst.get("marketSector") == _EQUITY_SECTOR
        and inst.get("securityType") in _US_EQUITY_TYPES
        and bool(inst.get("ticker"))
        and inst.get("exchCode") == "US"
    )


def build_security_master(
    cusip_to_instruments: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Reduce raw CUSIP->instrument mappings to a one-row-per-CUSIP master."""
    rows: list[dict[str, Any]] = []
    for cusip, instruments in cusip_to_instruments.items():
        rep = select_representative(instruments)
        if rep is None:
            rows.append(
                {
                    "cusip": cusip, "ticker": None, "name": None,
                    "composite_figi": None, "share_class_figi": None, "figi": None,
                    "security_type": None, "security_type2": None,
                    "market_sector": None, "exch_code": None,
                    "is_us_common_equity": False, "n_instruments": 0,
                }
            )
            continue
        rows.append(
            {
                "cusip": cusip,
                "ticker": rep.get("ticker"),
                "name": rep.get("name"),
                "composite_figi": rep.get("compositeFIGI"),
                "share_class_figi": rep.get("shareClassFIGI"),
                "figi": rep.get("figi"),
                "security_type": rep.get("securityType"),
                "security_type2": rep.get("securityType2"),
                "market_sector": rep.get("marketSector"),
                "exch_code": rep.get("exchCode"),
                "is_us_common_equity": _is_us_common_equity(rep),
                "n_instruments": len(instruments),
            }
        )
    return pd.DataFrame(rows, columns=MASTER_COLS)


def coverage_report(master: pd.DataFrame, total_cusips: int | None = None) -> dict[str, Any]:
    """Summarise mapping coverage as an explicit data-loss figure."""
    n = len(master)
    total = total_cusips if total_cusips is not None else n
    mapped = int(master["ticker"].notna().sum())
    us_common = int(master["is_us_common_equity"].sum())

    def pct(x: int) -> float:
        return round(100.0 * x / total, 2) if total else 0.0

    return {
        "distinct_cusips": total,
        "mapped_any": mapped,
        "mapped_pct": pct(mapped),
        "us_common_equity": us_common,
        "us_common_equity_pct": pct(us_common),
        "unmapped": total - mapped,
        "unmapped_pct": pct(total - mapped),
    }


def distinct_equity_cusips(holdings: pd.DataFrame) -> list[str]:
    """Distinct CUSIPs from long (non-option) equity holding rows."""
    mask = holdings["put_call"].isna() if "put_call" in holdings.columns else slice(None)
    cusips: Iterable[str] = holdings.loc[mask, "cusip"].dropna().unique()
    return sorted({str(c).upper() for c in cusips})
