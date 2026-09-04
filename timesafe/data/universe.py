"""Point-in-time S&P 500 universe reconstruction (survivorship-bias control).

Uses the public Wikipedia constituent list plus its historical change log to walk
membership backwards from today to any target date. This is the standard free
reconstruction; the paper's robustness section should cross-check a sample of
dates against an independent source.

Caveat (documented for the paper): Wikipedia's change log is sparse before ~2000
and tickers may be reused across decades, so reconstructions are most reliable
for as-of dates >= 2000 — which matches the ChronoGPT cutoff range (1999-2024).
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from timesafe.config import HTTP_USER_AGENT, cache_path

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#: Since ~2025 the change log lives in its own article.
WIKI_HISTORICAL_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def _fetch_tables(url: str, cache_name: str, refresh: bool = False) -> list[pd.DataFrame]:
    cache = cache_path("universe", cache_name)
    if cache.exists() and not refresh:
        html = cache.read_text()
    else:
        resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=30)
        resp.raise_for_status()
        html = resp.text
        cache.write_text(html)
    return pd.read_html(io.StringIO(html))


def _flat_cols(t: pd.DataFrame) -> list[str]:
    if isinstance(t.columns, pd.MultiIndex):
        return [" ".join(str(c) for c in col).lower() for col in t.columns]
    return [str(c).lower() for c in t.columns]


def _is_change_table(t: pd.DataFrame) -> bool:
    cols = _flat_cols(t)
    return (
        any("date" in c for c in cols)
        and any("added" in c for c in cols)
        and any("removed" in c for c in cols)
    )


def _fetch_wiki_tables(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current constituents table, change-log table) from Wikipedia, cached.

    Tables are selected by content, not position. The change log is looked up on the
    dedicated 'Historical components' article first, then (for older page layouts)
    on the main list page.
    """
    main = _fetch_tables(WIKI_URL, "sp500_wiki.html", refresh)
    current = next(t for t in main if {"symbol", "security"} <= set(_flat_cols(t)))
    changes = next((t for t in main if _is_change_table(t)), None)
    if changes is None:
        hist = _fetch_tables(WIKI_HISTORICAL_URL, "sp500_wiki_historical.html", refresh)
        changes = next(t for t in hist if _is_change_table(t))
    return current, changes


def sp500_current(refresh: bool = False) -> pd.DataFrame:
    """Current constituents: columns ``ticker``, ``name``, ``cik``, ``date_added``."""
    current, _ = _fetch_wiki_tables(refresh)
    df = pd.DataFrame(
        {
            "ticker": current["Symbol"].astype(str).str.strip(),
            "name": current["Security"].astype(str).str.strip(),
            "cik": pd.to_numeric(current.get("CIK"), errors="coerce").astype("Int64"),
            "date_added": pd.to_datetime(current.get("Date added"), errors="coerce").dt.date,
        }
    )
    return df


def sp500_changes(refresh: bool = False) -> pd.DataFrame:
    """Historical change log: one row per (date, added, removed) event, oldest first.

    Columns: ``date`` (datetime.date), ``added`` (ticker or None), ``removed`` (ticker or None).
    """
    _, changes = _fetch_wiki_tables(refresh)
    # Wikipedia uses a two-level header: (Date, Date) / (Added, Ticker) / (Removed, Ticker)
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = ["_".join(str(c) for c in col).strip() for col in changes.columns]
    col = {c.lower(): c for c in changes.columns}

    def find(*needles: str) -> str:
        for lower, original in col.items():
            if all(n in lower for n in needles):
                return original
        raise KeyError(f"no column matching {needles} in {list(changes.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(changes[find("date")], errors="coerce").dt.date,
            "added": changes[find("added", "ticker")].astype(str).str.strip(),
            "removed": changes[find("removed", "ticker")].astype(str).str.strip(),
        }
    )
    for c in ("added", "removed"):
        out[c] = out[c].replace({"nan": None, "": None, "None": None})
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def walk_back(current_tickers: set[str], changes: pd.DataFrame, as_of: date) -> set[str]:
    """Reverse the change log from today back to ``as_of`` (pure function, unit-tested).

    For every change strictly after ``as_of`` (most recent first): undo it — drop the
    ticker that was added, restore the ticker that was removed.
    """
    def valid(v) -> bool:  # pandas renders missing cells as None or NaN
        return isinstance(v, str) and v.strip() != ""

    members = set(current_tickers)
    future = changes[changes["date"] > as_of].sort_values("date", ascending=False)
    for _, row in future.iterrows():
        if valid(row["added"]):
            members.discard(row["added"])
        if valid(row["removed"]):
            members.add(row["removed"])
    return members


def sp500_members(as_of: date, refresh: bool = False) -> set[str]:
    """Point-in-time S&P 500 membership (ticker symbols) on ``as_of``."""
    current = sp500_current(refresh)
    changes = sp500_changes(refresh)
    return walk_back(set(current["ticker"]), changes, as_of)
