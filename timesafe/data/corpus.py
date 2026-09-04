"""Frozen event-corpus construction: S&P 500 earnings 8-Ks with point-in-time discipline.

Produces the dataset both E1 (bias curves) and E2 (probe validation) consume:
one row per (firm, filing-date) earnings event with the filing text excerpt and
leakage-guarded forward returns.

Selection rules (documented for the paper's data section):
  - Form 8-K filings whose header lists Item 2.02 ("Results of Operations and
    Financial Condition") — the standard earnings-announcement item.
  - Filer must be an S&P 500 member *on the filing date* (point-in-time universe,
    kills survivorship bias).
  - Text is the cleaned first ~N chars of the earnings exhibit/body; the filing
    date is the decision date (public availability).
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from timesafe.data.edgar import cik_to_ticker_map, fetch_filing_text, filings
from timesafe.data.universe import sp500_changes, sp500_current, walk_back

ITEM_202 = "Results of Operations and Financial Condition"


def parse_sgml_items(full_txt: str, header_chars: int = 6000) -> list[str]:
    """Item descriptions from an EDGAR full-submission SGML header."""
    header = full_txt[:header_chars]
    return [m.strip() for m in re.findall(r"ITEM INFORMATION:\s*(.+)", header)]


_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_WS_RE = re.compile(r"\s+")


def extract_main_text(full_txt: str, max_chars: int = 4000) -> str:
    """Best-effort plain text of the filing body / press-release exhibit.

    Prefers the first EX-99* exhibit (the earnings press release); falls back to the
    first document. Strips SGML/HTML tags and collapses whitespace. Deliberately
    simple — the paper only needs a stable, reproducible excerpt, not perfect parsing.
    """
    docs = re.split(r"<DOCUMENT>", full_txt)[1:]
    body = None
    for d in docs:
        if re.search(r"<TYPE>EX-99", d[:500]):
            body = d
            break
    if body is None and docs:
        body = docs[0]
    if body is None:
        body = full_txt
    # Drop embedded binary/uuencoded blocks and style/script sections.
    body = re.sub(r"begin \d{3}.+?end", " ", body, flags=re.S)
    body = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", body, flags=re.S | re.I)
    text = _TAG_RE.sub(" ", body)
    text = _ENTITY_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


class PointInTimeUniverse:
    """Cached monthly S&P 500 membership lookups (walk-back is O(changes) per call)."""

    def __init__(self) -> None:
        self._current = set(sp500_current()["ticker"])
        self._changes = sp500_changes()
        self._cache: dict[tuple[int, int], set[str]] = {}

    def members(self, d: date) -> set[str]:
        key = (d.year, d.month)
        if key not in self._cache:
            self._cache[key] = walk_back(self._current, self._changes, date(d.year, d.month, 1))
        return self._cache[key]


def build_8k_corpus(
    start: date,
    end: date,
    require_item: str = ITEM_202,
    text_chars: int = 4000,
    limit: int | None = None,
    tickers: set[str] | None = None,
    sp500_only: bool = True,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Build the earnings-8-K event corpus for [start, end].

    Returns columns: ticker, name, cik, decision_date, event_type, text, path.
    ``limit`` caps the number of *fetched* filings (for smoke tests / sampling).
    """
    idx = filings("8-K", start, end)
    tick_map = cik_to_ticker_map()
    idx = idx.merge(tick_map, on="cik", how="inner", suffixes=("", "_map"))
    if tickers is not None:
        idx = idx[idx["ticker"].isin(tickers)]
    idx = idx.sort_values("date_filed").reset_index(drop=True)

    universe = PointInTimeUniverse() if sp500_only else None
    rows: list[dict] = []
    fetched = 0
    for _, r in idx.iterrows():
        if limit is not None and fetched >= limit:
            break
        if universe is not None and r["ticker"] not in universe.members(r["date_filed"]):
            continue
        try:
            full = fetch_filing_text(r["path"])
        except Exception:  # noqa: BLE001 — skip unfetchable filings, log via count
            continue
        fetched += 1
        if require_item and not any(require_item.lower() in it.lower() for it in parse_sgml_items(full)):
            continue
        rows.append(
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "cik": int(r["cik"]),
                "decision_date": r["date_filed"],
                "event_type": "8-K-2.02",
                "text": extract_main_text(full, text_chars),
                "path": r["path"],
            }
        )
        if progress_every and len(rows) % progress_every == 0:
            print(f"[corpus] {len(rows)} events (through {r['date_filed']})")
    return pd.DataFrame(rows)
