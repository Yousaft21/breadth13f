"""Point-in-time issuer fundamentals from SEC EDGAR XBRL (free).

Builds the raw inputs for the size/value baseline:

* **Book value of equity** — us-gaap ``StockholdersEquity``.
* **Trailing earnings** — us-gaap ``NetIncomeLoss`` (latest annual, form 10-K).
* **Shares outstanding** — dei ``EntityCommonStockSharesOutstanding``.

Everything is selected **as-of** a formation date using each fact's own SEC
filing date (``filed``): only facts filed on/before the formation date and whose
reporting period ended on/before it are eligible. That makes the baseline a
legitimate point-in-time control rather than a look-ahead snapshot.

Mapping: 13F gives *issuer* CUSIPs; the XBRL API is keyed by issuer CIK. We map
the security master's ticker to a CIK via SEC ``company_tickers.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from ..ingest.edgar_client import EdgarClient

logger = logging.getLogger("breadth13f.fundamentals")

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"

# (taxonomy, tag, unit) for each fundamental input.
TAG_BOOK_EQUITY = ("us-gaap", "StockholdersEquity", "USD")
TAG_NET_INCOME = ("us-gaap", "NetIncomeLoss", "USD")
TAG_SHARES_OUT = ("dei", "EntityCommonStockSharesOutstanding", "shares")


def _norm_ticker(t: str) -> str:
    # SEC uses '-' for share-class separators (BRK-B); OpenFIGI often uses '/'.
    return t.upper().replace(".", "-").replace("/", "-").strip()


def load_ticker_cik_map(client: EdgarClient) -> dict[str, int]:
    """Map normalized ticker -> issuer CIK from SEC company_tickers.json."""
    raw = client.get_text(_COMPANY_TICKERS_URL)
    data = json.loads(raw)
    out: dict[str, int] = {}
    for rec in data.values():
        out[_norm_ticker(str(rec["ticker"]))] = int(rec["cik_str"])
    logger.info("Loaded ticker->CIK map: %d tickers", len(out))
    return out


# --- pure point-in-time fact selection (unit-tested) -----------------------
def latest_fact_asof(
    entries: list[dict],
    as_of: pd.Timestamp,
    forms: set[str] | None = None,
) -> float | None:
    """Latest fact value known by ``as_of``.

    ``entries`` are the objects from ``units[<unit>]`` of a companyconcept
    response, each with ``end``, ``filed`` and ``val`` (and often ``form``).
    Eligible entries were filed on/before ``as_of`` and cover a period ending
    on/before ``as_of``; among those we pick the most recent period end, breaking
    ties by the latest filing date. ``forms`` optionally restricts by form type
    (e.g. ``{"10-K"}`` for annual figures).
    """
    as_of = pd.Timestamp(as_of)
    best = None
    best_key = None
    for e in entries:
        try:
            filed = pd.Timestamp(e["filed"])
            end = pd.Timestamp(e["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if filed > as_of or end > as_of:
            continue
        if forms is not None and e.get("form") not in forms:
            continue
        key = (end, filed)
        if best_key is None or key > best_key:
            best_key = key
            best = e.get("val")
    return float(best) if best is not None else None


@dataclass
class FundamentalsProvider:
    """Fetches and caches issuer XBRL concepts, exposing PIT lookups."""

    client: EdgarClient

    def _concept_entries(self, cik: int, taxonomy: str, tag: str, unit: str) -> list[dict]:
        url = _CONCEPT_URL.format(cik=cik, taxonomy=taxonomy, tag=tag)
        raw = self.client.get_optional(url, compress=True)
        if raw is None:
            return []
        try:
            data = json.loads(raw)
            return data.get("units", {}).get(unit, [])
        except (json.JSONDecodeError, AttributeError):
            return []

    def get_asof(self, cik: int, as_of: pd.Timestamp) -> dict[str, float | None]:
        """Return ``{book_equity, net_income_annual, shares_outstanding}`` as-of."""
        eq = self._concept_entries(cik, *TAG_BOOK_EQUITY)
        ni = self._concept_entries(cik, *TAG_NET_INCOME)
        sh = self._concept_entries(cik, *TAG_SHARES_OUT)
        return {
            "book_equity": latest_fact_asof(eq, as_of),
            "net_income_annual": latest_fact_asof(ni, as_of, forms={"10-K", "10-K/A"}),
            "shares_outstanding": latest_fact_asof(sh, as_of),
        }
