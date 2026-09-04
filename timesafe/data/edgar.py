"""SEC EDGAR access: quarterly form indexes and filing documents (rate-limited, cached).

Filing dates from the form index are the *public availability* dates, which is the
point-in-time discipline the paper needs: an event enters the dataset at its EDGAR
filing date, never at the (earlier) period-of-report date.
"""
from __future__ import annotations

import io
import time
from datetime import date

import pandas as pd
import requests

from timesafe.config import EDGAR_MIN_INTERVAL_S, EDGAR_USER_AGENT, cache_path

_BASE = "https://www.sec.gov/Archives"
_last_request_ts: float = 0.0


def _polite_get(url: str, **kwargs) -> requests.Response:
    """GET with SEC fair-access pacing and required User-Agent."""
    global _last_request_ts
    wait = EDGAR_MIN_INTERVAL_S - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": EDGAR_USER_AGENT}, timeout=60, **kwargs)
    _last_request_ts = time.monotonic()
    resp.raise_for_status()
    return resp


def quarterly_form_index(year: int, quarter: int, refresh: bool = False) -> pd.DataFrame:
    """Parse EDGAR's ``form.idx`` for a quarter into a DataFrame.

    Columns: ``form_type``, ``company``, ``cik``, ``date_filed`` (datetime.date), ``path``.
    """
    cache = cache_path("edgar", f"form_index_{year}_Q{quarter}.parquet")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    url = f"{_BASE}/edgar/full-index/{year}/QTR{quarter}/form.idx"
    text = _polite_get(url).text
    lines = text.splitlines()
    # Header ends with a dashed separator line; columns are fixed-width.
    sep_i = next(i for i, ln in enumerate(lines) if set(ln.strip()) == {"-"})
    header = lines[sep_i - 1]
    starts = [header.index(c) for c in ("Form Type", "Company Name", "CIK", "Date Filed", "File Name")]
    rows = []
    for ln in lines[sep_i + 1 :]:
        if not ln.strip():
            continue
        fields = [
            ln[s:e].strip() for s, e in zip(starts, starts[1:] + [None])  # type: ignore[list-item]
        ]
        rows.append(fields)
    df = pd.DataFrame(rows, columns=["form_type", "company", "cik", "date_filed", "path"])
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["date_filed"] = pd.to_datetime(df["date_filed"], errors="coerce").dt.date
    df = df.dropna(subset=["date_filed"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    return df


def filings(
    form_type: str,
    start: date,
    end: date,
    refresh: bool = False,
) -> pd.DataFrame:
    """All filings of ``form_type`` (e.g. '8-K') filed in [start, end], across quarters."""
    frames = []
    for year in range(start.year, end.year + 1):
        for quarter in range(1, 5):
            q_start = date(year, 3 * (quarter - 1) + 1, 1)
            q_end = date(year + (quarter == 4), (3 * quarter) % 12 + 1, 1)
            if q_end <= start or q_start > end:
                continue
            idx = quarterly_form_index(year, quarter, refresh)
            frames.append(idx[idx["form_type"] == form_type])
    if not frames:
        return pd.DataFrame(columns=["form_type", "company", "cik", "date_filed", "path"])
    df = pd.concat(frames, ignore_index=True)
    return df[(df["date_filed"] >= start) & (df["date_filed"] <= end)].reset_index(drop=True)


def fetch_filing_text(path: str, refresh: bool = False, max_bytes: int = 5_000_000) -> str:
    """Download a filing document by its form-index ``path`` (cached)."""
    cache = cache_path("edgar", "docs", path.replace("/", "_"))
    if cache.exists() and not refresh:
        return cache.read_text(errors="replace")
    text = _polite_get(f"{_BASE}/{path}").text[:max_bytes]
    cache.write_text(text)
    return text


def cik_to_ticker_map(refresh: bool = False) -> pd.DataFrame:
    """SEC's official CIK<->ticker mapping (columns: ``cik``, ``ticker``, ``name``)."""
    cache = cache_path("edgar", "company_tickers.parquet")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    resp = _polite_get("https://www.sec.gov/files/company_tickers.json")
    records = list(resp.json().values())
    df = pd.DataFrame(records).rename(
        columns={"cik_str": "cik", "ticker": "ticker", "title": "name"}
    )[["cik", "ticker", "name"]]
    df.to_parquet(cache, index=False)
    return df
