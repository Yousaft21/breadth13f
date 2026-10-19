"""Parse SEC 13F-HR submissions into a tidy holdings table.

A 13F submission is delivered as a single SGML ``.txt`` file that bundles an
SEC header with one or more ``<DOCUMENT>`` blocks. The one we care about is the
**information table** (``<TYPE>INFORMATION TABLE``), an XML document listing
every reported position: issuer, CUSIP, value, shares, discretion, and — for
options — a put/call flag.

This module isolates two independently testable pieces:

* :func:`parse_header` — pull point-in-time-critical fields out of the SGML
  header (period of report, filed-as-of date, **acceptance datetime**, filer
  CIK/name, form type).
* :func:`parse_info_table_xml` — turn one information-table XML string into a
  list of position rows, robust to namespace prefixes.

:func:`parse_submission` ties them together for a full ``.txt`` submission.

Units caveat (documented, handled downstream): before the SEC's amendment
effective in 2023, the ``<value>`` field was reported in *thousands* of USD;
from 2023 it is reported in *whole* USD. We capture the raw integer here and
reconcile units in the feature stage — breadth/holder-count features are
unaffected because they rely on share counts and holder presence, not value.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from lxml import etree

logger = logging.getLogger("breadth13f.parse")

# --- SGML header field patterns --------------------------------------------
_HEADER_PATTERNS = {
    "accession_number": re.compile(r"ACCESSION NUMBER:\s*([0-9-]+)", re.I),
    "submission_type": re.compile(r"CONFORMED SUBMISSION TYPE:\s*(\S+)", re.I),
    "period_of_report": re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", re.I),
    "filed_as_of_date": re.compile(r"FILED AS OF DATE:\s*(\d{8})", re.I),
    "acceptance_datetime": re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})"),
}
# Filer identity lives under the FILER > COMPANY DATA block; the first CIK /
# conformed name in the header is the reporting manager.
_FILER_CIK = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)", re.I)
_FILER_NAME = re.compile(r"COMPANY CONFORMED NAME:\s*(.+)", re.I)
_AMENDMENT_TYPE = re.compile(r"<amendmentType>\s*(.*?)\s*</amendmentType>", re.I | re.S)
_AMENDMENT_NO = re.compile(r"<amendmentNo>\s*(\d+)\s*</amendmentNo>", re.I | re.S)

# --- Information-table document extraction ----------------------------------
_DOCUMENT_BLOCK = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
_DOC_TYPE = re.compile(r"<TYPE>\s*(.+)", re.I)
_XML_BLOCK = re.compile(r"<XML>\s*(.*?)\s*</XML>", re.S | re.I)
# Fallback: grab an <informationTable> element with any namespace prefix.
_INFO_TABLE_FALLBACK = re.compile(
    r"(<(?:\w+:)?informationTable\b.*?</(?:\w+:)?informationTable>)", re.S | re.I
)


@dataclass
class SubmissionHeader:
    """Point-in-time-critical metadata extracted from the SGML header."""

    accession_number: str | None = None
    submission_type: str | None = None
    period_of_report: date | None = None
    filed_as_of_date: date | None = None
    acceptance_datetime: datetime | None = None
    manager_cik: int | None = None
    manager_name: str | None = None
    amendment_type: str | None = None  # RESTATEMENT / NEW HOLDINGS (13F-HR/A only)
    amendment_no: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accession_number": self.accession_number,
            "submission_type": self.submission_type,
            "period_of_report": self.period_of_report,
            "filed_as_of_date": self.filed_as_of_date,
            "acceptance_datetime": self.acceptance_datetime,
            "manager_cik": self.manager_cik,
            "manager_name": self.manager_name,
            "amendment_type": self.amendment_type,
            "amendment_no": self.amendment_no,
        }


@dataclass
class ParsedSubmission:
    header: SubmissionHeader
    holdings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_holdings(self) -> int:
        return len(self.holdings)


def _parse_date8(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d").date()


def parse_header(text: str) -> SubmissionHeader:
    """Extract header metadata from the SGML header portion of a submission."""
    hdr = SubmissionHeader()
    for name, pat in _HEADER_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        val = m.group(1).strip()
        if name == "period_of_report":
            hdr.period_of_report = _parse_date8(val)
        elif name == "filed_as_of_date":
            hdr.filed_as_of_date = _parse_date8(val)
        elif name == "acceptance_datetime":
            hdr.acceptance_datetime = datetime.strptime(val, "%Y%m%d%H%M%S")
        elif name == "accession_number":
            hdr.accession_number = val
        elif name == "submission_type":
            hdr.submission_type = val

    m = _FILER_CIK.search(text)
    if m:
        hdr.manager_cik = int(m.group(1))
    m = _FILER_NAME.search(text)
    if m:
        hdr.manager_name = m.group(1).strip()

    m = _AMENDMENT_TYPE.search(text)
    if m and m.group(1):
        hdr.amendment_type = m.group(1).strip().upper() or None
    m = _AMENDMENT_NO.search(text)
    if m:
        hdr.amendment_no = int(m.group(1))
    return hdr


def extract_info_table_xml(text: str) -> str | None:
    """Return the information-table XML string from a full submission, or None.

    Strategy: find the ``<DOCUMENT>`` whose ``<TYPE>`` is INFORMATION TABLE and
    return the payload inside its ``<XML>`` wrapper. Falls back to a direct
    ``<informationTable>`` search for the occasional non-standard layout.
    """
    for block in _DOCUMENT_BLOCK.finditer(text):
        body = block.group(1)
        type_m = _DOC_TYPE.search(body)
        if not type_m:
            continue
        doc_type = type_m.group(1).strip().upper()
        if doc_type.startswith("INFORMATION TABLE"):
            xml_m = _XML_BLOCK.search(body)
            if xml_m:
                return xml_m.group(1).strip()
            fb = _INFO_TABLE_FALLBACK.search(body)
            if fb:
                return fb.group(1).strip()

    # No typed document matched — try the whole text as a last resort.
    fb = _INFO_TABLE_FALLBACK.search(text)
    return fb.group(1).strip() if fb else None


def _localname(tag) -> str:
    """Namespace-agnostic local tag name ('{uri}infoTable' -> 'infoTable').

    Comments and processing instructions carry a function, not a string, as
    their tag; they have no name.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _child_text(el: etree._Element, name: str) -> str | None:
    for child in el:
        if _localname(child.tag).lower() == name.lower():
            return (child.text or "").strip() or None
    return None


def _nested_text(el: etree._Element, parent: str, child: str) -> str | None:
    for p in el:
        if _localname(p.tag).lower() == parent.lower():
            return _child_text(p, child)
    return None


def parse_info_table_xml(xml: str) -> list[dict[str, Any]]:
    """Parse one information-table XML string into a list of position rows.

    Robust to a default namespace or explicit prefixes (``ns1:infoTable``).
    Uses lxml's recovering parser so a single malformed row does not abort the
    whole filing.
    """
    if not xml:
        return []
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    root = etree.fromstring(xml.encode("utf-8", errors="replace"), parser=parser)
    if root is None:
        return []

    rows: list[dict[str, Any]] = []
    for el in root.iter():
        if _localname(el.tag).lower() != "infotable":
            continue
        cusip = _child_text(el, "cusip")
        shares_raw = _nested_text(el, "shrsOrPrnAmt", "sshPrnamt")
        value_raw = _child_text(el, "value")
        rows.append(
            {
                "name_of_issuer": _child_text(el, "nameOfIssuer"),
                "title_of_class": _child_text(el, "titleOfClass"),
                "cusip": cusip.upper() if cusip else None,
                "value": _to_int(value_raw),
                "shares": _to_int(shares_raw),
                "shares_type": _nested_text(el, "shrsOrPrnAmt", "sshPrnamtType"),
                "put_call": _child_text(el, "putCall"),
                "investment_discretion": _child_text(el, "investmentDiscretion"),
                "voting_sole": _to_int(_nested_text(el, "votingAuthority", "Sole")),
                "voting_shared": _to_int(_nested_text(el, "votingAuthority", "Shared")),
                "voting_none": _to_int(_nested_text(el, "votingAuthority", "None")),
            }
        )
    return rows


def _to_int(s: str | None) -> int | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if not s:
        return None
    try:
        # Some filers include a decimal point on integer share counts.
        return int(float(s))
    except ValueError:
        return None


def parse_submission(text: str) -> ParsedSubmission:
    """Parse a complete 13F ``.txt`` submission into header + holdings."""
    header = parse_header(text)
    xml = extract_info_table_xml(text)
    holdings = parse_info_table_xml(xml) if xml else []
    if not holdings:
        logger.debug(
            "No holdings parsed for accession %s (type %s)",
            header.accession_number, header.submission_type,
        )
    return ParsedSubmission(header=header, holdings=holdings)
