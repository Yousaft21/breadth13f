"""Enumerate 13F filings per quarter from the EDGAR full-index.

EDGAR publishes a machine-readable index of every dissemination per calendar
quarter at::

    https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx

``master.idx`` is pipe-delimited (``CIK|Company Name|Form Type|Date Filed|
Filename``) which is far more robust to parse than the fixed-width ``form.idx``.
We download it once (cached), filter to the 13F form types, and return one row
per filing with the URL of its full submission ``.txt``.

Important: filings are indexed by **filing date**, not by the period they
describe. A holding period ending 2024-12-31 is filed in 2025-Q1 (managers have
45 days). Callers therefore crawl *filing* quarters; the period-of-report of
each filing is read from its header during parsing, and point-in-time selection
happens later using the acceptance datetime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

from .edgar_client import EdgarClient

logger = logging.getLogger("breadth13f.crawl")


@dataclass(frozen=True, order=True)
class Quarter:
    year: int
    q: int

    @classmethod
    def parse(cls, s: str) -> "Quarter":
        s = s.upper().replace("-", "")
        year, q = s.split("Q")
        return cls(int(year), int(q))

    def __str__(self) -> str:
        return f"{self.year}Q{self.q}"

    def next(self) -> "Quarter":
        return Quarter(self.year + 1, 1) if self.q == 4 else Quarter(self.year, self.q + 1)

    def prev(self) -> "Quarter":
        return Quarter(self.year - 1, 4) if self.q == 1 else Quarter(self.year, self.q - 1)

    def end_date(self) -> "pd.Timestamp":
        """Calendar quarter-end date (period-of-report) for this quarter."""
        import pandas as pd

        month = self.q * 3
        return pd.Timestamp(self.year, month, 1) + pd.offsets.MonthEnd(0)


def quarter_closed_by(quarter: Quarter, when) -> bool:
    """True if ``when`` is after the last day of filing ``quarter``."""
    import pandas as pd

    return pd.Timestamp(when) >= quarter.end_date() + pd.Timedelta(days=1)


def iter_quarters(start: Quarter, end: Quarter) -> Iterator[Quarter]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur.next()


@dataclass
class FilingRef:
    """A single 13F filing as listed in the quarterly index."""

    manager_cik: int
    company_name: str
    form_type: str
    date_filed: str  # YYYY-MM-DD as published in the index
    filename: str    # e.g. edgar/data/1234567/0001234567-25-000123.txt

    def submission_url(self, base_archives: str) -> str:
        return f"{base_archives}/{self.filename}"

    @property
    def accession_number(self) -> str:
        # .../0001234567-25-000123.txt -> 0001234567-25-000123
        return self.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def list_13f_filings(
    client: EdgarClient,
    base_archives: str,
    quarter: Quarter,
    form_types: set[str],
) -> list[FilingRef]:
    """Download and parse one quarter's master index, filtered to 13F forms.

    The index grows every business day until the quarter ends, so a copy cached
    while the quarter was still open is incomplete and is re-downloaded.
    """
    url = f"{base_archives}/edgar/full-index/{quarter.year}/QTR{quarter.q}/master.idx"
    cached_at = client.cached_at(url, compress=True)
    stale = cached_at is not None and not quarter_closed_by(quarter, cached_at)
    if stale:
        logger.info("%s: cached index predates quarter close (%s); re-downloading",
                    quarter, cached_at.date())
    text = client.get_text(url, compress=True, force=stale)

    refs: list[FilingRef] = []
    started = False
    for line in text.splitlines():
        if not started:
            # The data section follows a line of dashes after the column header.
            if set(line.strip()) == {"-"}:
                started = True
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, name, form_type, date_filed, filename = (p.strip() for p in parts)
        if form_type not in form_types:
            continue
        try:
            refs.append(
                FilingRef(
                    manager_cik=int(cik),
                    company_name=name,
                    form_type=form_type,
                    date_filed=date_filed,
                    filename=filename,
                )
            )
        except ValueError:
            logger.warning("Skipping malformed index line: %r", line)
    logger.info("%s: %d 13F filings in index", quarter, len(refs))
    return refs
