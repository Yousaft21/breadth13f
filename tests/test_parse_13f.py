"""Unit tests for the 13F submission parser.

These lock down the two failure-prone behaviours: pulling point-in-time header
fields out of the SGML wrapper, and parsing information-table XML across the
namespace variations seen in real filings (default namespace vs. explicit
prefixes) plus options (put/call) and malformed rows.
"""

from __future__ import annotations

from datetime import date, datetime

from src.ingest.parse_13f import (
    extract_info_table_xml,
    parse_header,
    parse_info_table_xml,
    parse_submission,
)

# A trimmed but realistic full submission: SGML header + INFORMATION TABLE doc.
SAMPLE_SUBMISSION = """<SEC-DOCUMENT>0001234567-25-000123.txt : 20250214
<SEC-HEADER>0001234567-25-000123.hdr.sgml : 20250214
<ACCEPTANCE-DATETIME>20250214160102
ACCESSION NUMBER:\t\t0001234567-25-000123
CONFORMED SUBMISSION TYPE:\t13F-HR
PUBLIC DOCUMENT COUNT:\t\t2
CONFORMED PERIOD OF REPORT:\t20241231
FILED AS OF DATE:\t\t20250214
FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tEXAMPLE CAPITAL MANAGEMENT LP
\t\tCENTRAL INDEX KEY:\t\t\t0001234567
</SEC-HEADER>
<DOCUMENT>
<TYPE>13F-HR
<SEQUENCE>1
<FILENAME>primary_doc.xml
<TEXT>
<XML>
<edgarSubmission><headerData/></edgarSubmission>
</XML>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>INFORMATION TABLE
<SEQUENCE>2
<FILENAME>infotable.xml
<TEXT>
<XML>
<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>1500000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>10000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>10000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>SPDR S&amp;P 500</nameOfIssuer>
    <titleOfClass>ETF</titleOfClass>
    <cusip>78462F103</cusip>
    <value>250000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>500</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall>Put</putCall>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>0</Sole>
      <Shared>0</Shared>
      <None>500</None>
    </votingAuthority>
  </infoTable>
</informationTable>
</XML>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

# Same table with an explicit namespace prefix (ns1:) as some filers emit.
PREFIXED_XML = """<?xml version="1.0"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <ns1:infoTable>
    <ns1:nameOfIssuer>MICROSOFT CORP</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>594918104</ns1:cusip>
    <ns1:value>42</ns1:value>
    <ns1:shrsOrPrnAmt>
      <ns1:sshPrnamt>1,234</ns1:sshPrnamt>
      <ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
    </ns1:shrsOrPrnAmt>
    <ns1:investmentDiscretion>DFND</ns1:investmentDiscretion>
  </ns1:infoTable>
</ns1:informationTable>
"""


def test_parse_header_pit_fields():
    hdr = parse_header(SAMPLE_SUBMISSION)
    assert hdr.accession_number == "0001234567-25-000123"
    assert hdr.submission_type == "13F-HR"
    assert hdr.period_of_report == date(2024, 12, 31)
    assert hdr.filed_as_of_date == date(2025, 2, 14)
    assert hdr.acceptance_datetime == datetime(2025, 2, 14, 16, 1, 2)
    assert hdr.manager_cik == 1234567
    assert hdr.manager_name == "EXAMPLE CAPITAL MANAGEMENT LP"
    assert hdr.amendment_type is None


def test_extract_and_parse_info_table():
    xml = extract_info_table_xml(SAMPLE_SUBMISSION)
    assert xml is not None and "informationTable" in xml
    rows = parse_info_table_xml(xml)
    assert len(rows) == 2

    apple = rows[0]
    assert apple["name_of_issuer"] == "APPLE INC"
    assert apple["cusip"] == "037833100"
    assert apple["shares"] == 10000
    assert apple["shares_type"] == "SH"
    assert apple["value"] == 1500000
    assert apple["put_call"] is None
    assert apple["voting_sole"] == 10000

    # Entity references decode; put/call captured; None voting parsed.
    spdr = rows[1]
    assert spdr["name_of_issuer"] == "SPDR S&P 500"
    assert spdr["put_call"] == "Put"
    assert spdr["voting_none"] == 500


def test_parse_submission_end_to_end():
    parsed = parse_submission(SAMPLE_SUBMISSION)
    assert parsed.n_holdings == 2
    assert parsed.header.manager_cik == 1234567
    cusips = {h["cusip"] for h in parsed.holdings}
    assert cusips == {"037833100", "78462F103"}


def test_namespace_prefixed_xml():
    rows = parse_info_table_xml(PREFIXED_XML)
    assert len(rows) == 1
    assert rows[0]["cusip"] == "594918104"
    assert rows[0]["name_of_issuer"] == "MICROSOFT CORP"
    # Comma-separated share count is normalised to an int.
    assert rows[0]["shares"] == 1234


def test_malformed_row_is_skipped_not_fatal():
    # A stray unclosed tag inside one row shouldn't abort the whole table.
    broken = PREFIXED_XML.replace("<ns1:cusip>594918104</ns1:cusip>", "<ns1:cusip>594918104")
    rows = parse_info_table_xml(broken)
    # Recovering parser still yields the row; issuer name survives.
    assert len(rows) == 1
    assert rows[0]["name_of_issuer"] == "MICROSOFT CORP"


def test_empty_input_returns_empty():
    assert parse_info_table_xml("") == []
    assert extract_info_table_xml("no documents here") is None


def test_xml_comments_are_ignored():
    # Some filer software writes comments inside the table; lxml gives them a
    # non-string tag, which used to abort the whole filing.
    commented = PREFIXED_XML.replace("<ns1:cusip>", "<!-- generated --><ns1:cusip>")
    rows = parse_info_table_xml(commented)
    assert len(rows) == 1 and rows[0]["cusip"] == "594918104"
