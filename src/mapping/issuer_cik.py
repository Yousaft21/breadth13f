"""Point-in-time CUSIP -> SEC CIK mapping from issuer names.

The size/value baseline needs each security's issuer CIK (the XBRL API is keyed
by CIK). Mapping the security master's *current* ticker through SEC's *current*
``company_tickers.json`` is a look-ahead: a firm that later went bankrupt,
delisted or changed its ticker has no entry there, so it loses its size/value
features in every quarter, including the years before the event (validation
Q10). Its missing data then reflects its future.

This module maps issuers without current symbology, from two free sources that
were both observable at the time:

* the 13F ``nameOfIssuer`` that managers reported for the CUSIP in the quarter,
  weighted by how many holding rows used it (managers mistype; a stray "3M CO"
  filed under Apple's CUSIP must not win), and
* the company name under which each CIK filed its periodic reports (10-K, 10-Q,
  20-F, 40-F and amendments), read from EDGAR's quarterly ``master.idx``. The
  index records the filer's name *as of each filing*, so a firm keeps the name it
  had then, before any rename, bankruptcy or deregistration.

Names are compared token by token after normalisation (:func:`name_tokens`).
13F names are abbreviated and truncated ("ALLSCRIPTS HEALTHCARE SOLUTN", "FIRST
REP BK SAN FRANCISCO"), so a 13F token matches an EDGAR token when it is equal,
a prefix of it, or an in-order abbreviation of it with the same first letter
(:func:`token_matches`).

Point-in-time rule (:func:`pit_ciks`): the row for period ``p``, formed on date
``f``, uses only the 13F names reported for period ``p`` (filed by ``f``) and
EDGAR name spells that began on or before ``f`` and were still in use within
``recency_days`` before it. Each eligible CIK is scored by the share of the
period's 13F holding rows whose name matches it, using exact matches only when
there are any; the top CIK is used if its share is at least ``min_support`` and
it is unique.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..ingest.crawl import Quarter, iter_quarters
from ..ingest.edgar_client import EdgarClient

logger = logging.getLogger("breadth13f.issuer_cik")

# Periodic reports that carry the XBRL facts the baseline uses.
PERIODIC_FORMS = {
    "10-K", "10-K/A", "10-KT", "10-KT/A", "10-Q", "10-Q/A", "10-QT", "10-QT/A",
    "20-F", "20-F/A", "40-F", "40-F/A", "10-K405", "10-K405/A",
}

# Trailing tokens dropped from both names: legal form, share class, domicile.
# ("N C" is kept: "PEOPLES BANCORP N C" must not become Ohio's "PEOPLES BANCORP".)
# (Partnership suffixes are kept: a REIT and its operating partnership often file
# the same 10-K under two CIKs, "CUBESMART" and "CUBESMART LP".)
_SUFFIXES = {
    "INC", "CORP", "CO", "COS", "LTD", "PLC", "LLC", "NV", "SA", "AG", "SE", "AB", "ASA",
    "SPA", "THE", "NEW", "COM", "COMMON", "STOCK", "STK", "CL", "CLASS", "A", "B", "C", "SH",
    "ORD", "ADR", "AD", "SPONSORED", "SPON", "PAR", "REIT",
    "DEL", "DE", "PA", "MD", "MASS", "VA", "NY", "NJ", "OHIO", "TEX", "CALIF", "CAL", "MINN",
    "WIS", "WASH", "COLO", "ORE", "GA", "FLA", "MICH", "IOWA", "NEV", "QUE", "ONT", "BC",
}
# Truncations of these words are suffixes too ("IN", "COR", "CORPORATI", "LT", "NE").
_SUFFIX_WORDS = ("INCORPORATED", "CORPORATION", "COMPANY", "LIMITED", "LTD", "NEW")
# Spellings with no in-order abbreviation relation, mapped after singularising.
_ALIASES = {
    "INTERNATIONAL": "INTL", "AMERICAN": "AMER", "AMERICA": "AMER", "AMERN": "AMER",
    "HOLDING": "HLDG", "HOLDNG": "HLDG", "HLDNG": "HLDG",
    "TECHNOLOGY": "TECH", "TECHNOLOGIE": "TECH",
    "PHARMACEUTICAL": "PHARMA", "PHARMACEUTICA": "PHARMA", "PHARMACEUTIC": "PHARMA",
    "PHARMACEUTCL": "PHARMA", "PHARM": "PHARMA",
    "1ST": "FIRST", "2ND": "SECOND", "3RD": "THIRD",
}
_STATE_TAG = re.compile(r"\s*[/\\][A-Z0-9 .]{1,8}[/\\]?\s*$")

# 13F names are cut at a fixed field width; longer names may be truncated.
TRUNCATION_LEN = 24


def _is_suffix(tok: str, min_prefix: int = 2) -> bool:
    return tok in _SUFFIXES or (len(tok) >= min_prefix
                                and any(w.startswith(tok) for w in _SUFFIX_WORDS))


def _singular(tok: str) -> str:
    """Crude singular so plurals compare equal (COMMUNITIES / CMNTYS, LINES / LINE)."""
    if len(tok) > 4 and tok.endswith("IES"):
        return tok[:-3] + "Y"
    if len(tok) > 3 and tok.endswith("S") and not tok.endswith("SS") and tok.isalpha():
        return tok[:-1]
    return tok


def name_tokens(name: str) -> tuple[str, ...]:
    """Normalised word tokens of an issuer name, trailing suffixes removed.

    Upper-cases, strips EDGAR's trailing domicile tags (``/DE/``, ``\\MD\\``),
    splits on anything not a letter or digit, drops "&"/"AND", joins runs of
    single letters ("R F" -> "RF", "S A" -> "SA"), singularises, maps a few
    spellings (:data:`_ALIASES`) and drops trailing legal-form, share-class and
    domicile tokens, always keeping at least one token. In a name long enough to
    be truncated, a final one-letter stub of a suffix ("... GRP I") goes too.
    """
    if not isinstance(name, str):
        return ()
    s = name.upper()
    for _ in range(3):  # "ACME CORP /DE/ /NEW/"
        s = _STATE_TAG.sub("", s)
    raw = [t for t in re.split(r"[^A-Z0-9]+", s.replace("'", "")) if t and t != "AND"]
    toks: list[str] = []
    prev_single = False
    for t in raw:
        single = len(t) == 1 and t.isalpha()
        if single and prev_single:
            toks[-1] += t
        else:
            toks.append(t)
        prev_single = single
    toks = [_ALIASES.get(_singular(t), _singular(t)) for t in toks]
    while toks and toks[0] == "THE":
        toks.pop(0)
    if len(toks) > 1 and len(name.strip()) >= TRUNCATION_LEN and _is_suffix(toks[-1], 1):
        toks.pop()
    while len(toks) > 1 and _is_suffix(toks[-1]):
        toks.pop()
    return tuple(toks)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def token_matches(t13: str, ted: str) -> bool:
    """Is 13F token ``t13`` an acceptable spelling of EDGAR token ``ted``?

    Equal; or, for alphabetic tokens of 2+ characters with the same first letter,
    a prefix of it (truncation: ``SOLUTN`` / ``SOLUTIONS``) or an in-order
    abbreviation (``FINL`` / ``FINANCIAL``, ``BK`` / ``BANK``). Tokens with digits
    must be equal, so "FIRST BANCORP 2" never matches "FIRST BANCORP 3".
    """
    if t13 == ted:
        return True
    if len(t13) < 2 or len(t13) > len(ted) or t13[0] != ted[0]:
        return False
    if not (t13.isalpha() and ted.isalpha()):
        return False
    return _is_subsequence(t13, ted)


def match_quality(a13: tuple[str, ...], aed: tuple[str, ...], truncated: bool = False) -> int:
    """How well a 13F name matches an EDGAR name: 2 exact, 1 abbreviated, 0 no match.

    Both are token tuples from :func:`name_tokens`. *Exact* means the same tokens,
    except that the last 13F token may be cut short (a prefix). *Abbreviated*
    means each token pair matches under :func:`token_matches`; it needs at least
    two tokens and an unabbreviated first word (at most cut short, from four
    letters), since an abbreviated word alone ("INTEL" / "INTERMOLECULAR") or a
    short first word ("ARM" / "AFFIRM", "LI AUTO" / "LIFESCI ACQUISITION") is too weak. When the raw 13F name looks truncated (``truncated``) the EDGAR name may
    carry extra trailing tokens, provided at least two 13F tokens were compared.
    """
    if not a13 or not aed:
        return 0
    if len(a13) != len(aed) and not (truncated and 2 <= len(a13) < len(aed)):
        return 0
    n = len(a13)
    if a13[:-1] == aed[:n - 1] and aed[n - 1].startswith(a13[-1]):
        if a13 == aed or truncated:
            return 2
    if n < 2 or not (a13[0] == aed[0] or (len(a13[0]) >= 4 and aed[0].startswith(a13[0]))):
        return 0
    pairs = list(zip(a13, aed))
    last_ok = token_matches(*pairs[-1]) or (truncated and pairs[-1][1].startswith(pairs[-1][0]))
    return 1 if last_ok and all(token_matches(x, y) for x, y in pairs[:-1]) else 0


def match_names(names_13f: Iterable[str], edgar_names: pd.DataFrame) -> pd.DataFrame:
    """Every (13F name, EDGAR name spell) pair that matches.

    ``edgar_names`` has ``cik``, ``name``, ``first_filed`` and ``last_filed`` (one
    row per name spell, from :func:`edgar_name_spells`). Returns ``name_13f``,
    ``cik``, ``name_edgar``, ``first_filed``, ``last_filed`` and ``quality``
    (:func:`match_quality`).
    """
    ed = edgar_names.reset_index(drop=True)
    ed_tokens = [name_tokens(n) for n in ed["name"]]
    # Bucket EDGAR names by first token; a 13F name is compared only against the
    # buckets whose first token it matches.
    by_first: dict[str, list[int]] = defaultdict(list)
    for k, toks in enumerate(ed_tokens):
        if toks:
            by_first[toks[0]].append(k)
    firsts_by_letter: dict[str, list[str]] = defaultdict(list)
    for t in by_first:
        firsts_by_letter[t[0]].append(t)
    first_cache: dict[str, list[int]] = {}

    rows = []
    for name in dict.fromkeys(names_13f):
        toks = name_tokens(name)
        if not toks:
            continue
        t0 = toks[0]
        if t0 not in first_cache:
            first_cache[t0] = [k for t in firsts_by_letter.get(t0[0], []) if token_matches(t0, t)
                               for k in by_first[t]]
        trunc = len(str(name).strip()) >= TRUNCATION_LEN
        for k in first_cache[t0]:
            quality = match_quality(toks, ed_tokens[k], trunc)
            if quality:
                rows.append((name, int(ed.at[k, "cik"]), ed.at[k, "name"],
                             ed.at[k, "first_filed"], ed.at[k, "last_filed"], quality))
    return pd.DataFrame(rows, columns=["name_13f", "cik", "name_edgar", "first_filed",
                                       "last_filed", "quality"])


def pit_ciks(rows: pd.DataFrame, issuer_names: pd.DataFrame, matches: pd.DataFrame,
             recency_days: int = 730, min_support: float = 0.5, key: str = "cusip") -> pd.Series:
    """Point-in-time CIK per row (``Int64``, ``<NA>`` where unmapped).

    ``rows`` has ``key``, ``period`` and ``formation_date``. ``issuer_names``
    has ``key``, ``period``, ``name`` and ``n_reports`` (13F holding rows per
    name, from :func:`issuer_names_from_holdings`, or :func:`names_by_security`
    for ``key="composite_figi"``); ``matches`` comes from :func:`match_names`.
    See the module docstring for the rule.
    """
    keys = rows[[key, "period", "formation_date"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    keys["formation_date"] = pd.to_datetime(keys["formation_date"])
    keys["_row"] = np.arange(len(keys))
    nm = issuer_names[[key, "period", "name", "n_reports"]].copy()
    nm["period"] = pd.to_datetime(nm["period"])
    nm["share"] = nm["n_reports"] / nm.groupby([key, "period"])["n_reports"].transform("sum")
    cand = (keys.merge(nm, on=[key, "period"])
                .merge(matches.rename(columns={"name_13f": "name"}), on="name"))
    form = cand["formation_date"]
    ok = ((pd.to_datetime(cand["first_filed"]) <= form)
          & (pd.to_datetime(cand["last_filed"]) >= form - pd.Timedelta(days=recency_days)))
    cand = cand[ok]
    out = pd.Series(pd.NA, index=range(len(rows)), dtype="Int64")
    if cand.empty:
        return out
    # A 13F name counts once per CIK even if it matches several of its name spells.
    # CIKs reached only by abbreviated matches compete only where no CIK has an
    # exact match; a CIK's support counts all the names that match it.
    cand = cand.sort_values("quality", ascending=False).drop_duplicates(["_row", "name", "cik"])
    support = (cand.groupby(["_row", "cik"])
                   .agg(share=("share", "sum"), quality=("quality", "max")).reset_index())
    support = support[support["quality"] == support.groupby("_row")["quality"].transform("max")]
    support = support.sort_values(["_row", "share"], ascending=[True, False])
    top = support.groupby("_row").head(2)
    first = top.groupby("_row").nth(0).set_index("_row")
    second = top.groupby("_row").nth(1).set_index("_row")["share"]
    runner_up = second.reindex(first.index).fillna(0.0)
    good = (first["share"] >= min_support) & (first["share"] > runner_up + 1e-9)
    out.iloc[first.index[good].to_numpy()] = first.loc[good, "cik"].to_numpy()
    return out


def names_by_security(issuer_names: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Re-key :func:`issuer_names_from_holdings` from CUSIP to ``composite_figi``.

    A security can carry several CUSIPs over the sample (a CUSIP changes after
    some reorganisations); their 13F names are pooled per period.
    """
    nm = issuer_names.merge(master[["cusip", "composite_figi"]].dropna(), on="cusip")
    return nm.groupby(["composite_figi", "period", "name"], as_index=False)["n_reports"].sum()


def combine_ciks(name_cik: pd.Series, ticker_cik: pd.Series,
                 use_ticker: bool = True) -> tuple[pd.Series, pd.Series]:
    """Final CIK per row and where it came from.

    The point-in-time name match wins; the current-ticker map only fills rows the
    name match left empty (``use_ticker=False`` turns that off). The source is
    ``"name"``, ``"ticker"``, ``"name+ticker"`` (both, agreeing) or ``"none"``.
    """
    name_cik = pd.Series(name_cik, dtype="Int64").reset_index(drop=True)
    ticker_cik = pd.Series(ticker_cik, dtype="Int64").reset_index(drop=True)
    if not use_ticker:
        ticker_cik = pd.Series(pd.NA, index=name_cik.index, dtype="Int64")
    cik = name_cik.fillna(ticker_cik)
    has_n, has_t = name_cik.notna(), ticker_cik.notna()
    agree = has_n & has_t & (name_cik == ticker_cik).fillna(False)
    source = pd.Series("none", index=cik.index, dtype=object)
    source[has_n] = "name"
    source[agree] = "name+ticker"
    source[~has_n & has_t] = "ticker"
    return cik, source


# --- data loading ---------------------------------------------------------
def edgar_name_spells(client: EdgarClient, base_archives: str, start: Quarter, end: Quarter,
                      forms: set[str] = PERIODIC_FORMS) -> pd.DataFrame:
    """``(cik, name, first_filed, last_filed)`` per name a CIK used on a periodic report.

    Reads EDGAR's quarterly ``master.idx`` through ``client``, which caches it
    (the 13F crawl already fetched those for its own filing quarters).
    """
    recs = []
    for q in iter_quarters(start, end):
        url = f"{base_archives}/edgar/full-index/{q.year}/QTR{q.q}/master.idx"
        text = client.get_text(url, compress=True)
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) != 5 or parts[2] not in forms or not parts[0].isdigit():
                continue
            recs.append((int(parts[0]), parts[1].strip(), parts[3]))
    df = pd.DataFrame(recs, columns=["cik", "name", "filed"])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    spells = (df.dropna(subset=["filed"]).groupby(["cik", "name"])["filed"]
              .agg(first_filed="min", last_filed="max").reset_index())
    logger.info("EDGAR periodic filers %s..%s: %d CIKs, %d name spells",
                start, end, spells["cik"].nunique(), len(spells))
    return spells


def issuer_names_from_holdings(holdings_dir: Path, cusips: Iterable[str]) -> pd.DataFrame:
    """``(cusip, period, name, n_reports)``: 13F issuer names per CUSIP and period.

    ``n_reports`` counts the holding rows that reported the CUSIP under that
    (upper-cased, whitespace-trimmed) name for that period of report.
    """
    wanted = set(cusips)
    frames = []
    for f in sorted(holdings_dir.glob("filing_quarter=*.parquet")):
        h = pd.read_parquet(f, columns=["cusip", "name_of_issuer", "period_of_report"])
        h = h[h["cusip"].isin(wanted)]
        h = h.assign(name=h["name_of_issuer"].astype(str).str.strip().str.upper())
        frames.append(h.groupby(["cusip", "period_of_report", "name"]).size()
                      .rename("n_reports").reset_index())
    if not frames:
        return pd.DataFrame(columns=["cusip", "period", "name", "n_reports"])
    out = (pd.concat(frames, ignore_index=True)
           .groupby(["cusip", "period_of_report", "name"])["n_reports"].sum().reset_index()
           .rename(columns={"period_of_report": "period"}))
    logger.info("13F issuer names: %d CUSIPs, %d (cusip, period, name) rows",
                out["cusip"].nunique(), len(out))
    return out
