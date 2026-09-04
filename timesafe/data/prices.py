"""Daily price download with local parquet cache.

Primary source: Yahoo Finance v8 chart API (no key). Fallback: Stooq CSV endpoint.
All returns are computed with explicit decision-time discipline — see :func:`forward_return`.
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

from timesafe.config import HTTP_USER_AGENT, cache_path

_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"

_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def _to_unix(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _fetch_yahoo(symbol: str, start: date, end: date, session: requests.Session) -> pd.DataFrame:
    resp = session.get(
        _YAHOO_URL.format(symbol=symbol),
        params={
            "period1": _to_unix(start),
            "period2": _to_unix(end) + 86400,  # inclusive end
            "interval": "1d",
            "events": "div,splits",
        },
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()["chart"]
    if payload.get("error"):
        raise RuntimeError(f"Yahoo error for {symbol}: {payload['error']}")
    result = payload["result"][0]
    ts = result.get("timestamp")
    if not ts:
        return pd.DataFrame(columns=_COLUMNS)
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adj,
            "volume": quote["volume"],
        }
    )
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _fetch_stooq(symbol: str, start: date, end: date, session: requests.Session) -> pd.DataFrame:
    # Stooq uses e.g. aapl.us for US tickers.
    stooq_symbol = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    resp = session.get(
        _STOOQ_URL.format(
            symbol=stooq_symbol, d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d")
        ),
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.text
    if not text.startswith("Date,"):
        raise RuntimeError(f"Stooq returned non-CSV for {symbol}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["adj_close"] = df["close"]  # Stooq daily closes are already adjusted
    return df[_COLUMNS]


def fetch_daily(
    symbol: str,
    start: date,
    end: date,
    refresh: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``symbol`` in [start, end], caching to parquet.

    Cache stores the full fetched history per symbol; subsequent calls slice locally
    and only re-download when the requested window is not covered.
    """
    cache = cache_path("prices", f"{symbol.upper()}.parquet")
    if cache.exists() and not refresh:
        cached = pd.read_parquet(cache)
        if not cached.empty and cached["date"].min() <= start and cached["date"].max() >= end:
            return cached[(cached["date"] >= start) & (cached["date"] <= end)].reset_index(drop=True)

    session = requests.Session()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        for fetcher in (_fetch_yahoo, _fetch_stooq):
            try:
                df = fetcher(symbol, start, end, session)
                df = df.sort_values("date").reset_index(drop=True)
                df.to_parquet(cache, index=False)
                return df
            except Exception as e:  # noqa: BLE001 — try the next source/attempt
                last_err = e
        time.sleep(2**attempt)
    raise RuntimeError(f"All price sources failed for {symbol}: {last_err}")


def forward_return(
    prices: pd.DataFrame,
    event_date: date,
    horizon: int = 1,
    entry: str = "next_close",
) -> float | None:
    """Forward return for a signal known at end-of-day ``event_date``.

    Decision-time discipline (avoids EXEC_OPEN-style leakage): a signal computed from
    information available on day *t* can be acted on **at the earliest** at the next
    session. Supported entries:

    - ``next_close`` (default): close(t+1) -> close(t+1+horizon-1)... i.e. enter at the
      first close after the event, exit ``horizon`` trading days later.
    - ``next_open``: enter at open(t+1), exit at close(t+horizon).

    Returns None when the window is not fully covered by ``prices``.
    """
    px = prices.sort_values("date").reset_index(drop=True)
    future = px[px["date"] > event_date]
    if len(future) < horizon + (1 if entry == "next_close" else 0):
        return None
    if entry == "next_open":
        entry_px = future.iloc[0]["open"]
        exit_px = future.iloc[horizon - 1]["adj_close"]
    elif entry == "next_close":
        entry_px = future.iloc[0]["adj_close"]
        exit_px = future.iloc[horizon]["adj_close"] if len(future) > horizon else None
    else:
        raise ValueError(f"unknown entry mode: {entry}")
    if entry_px is None or exit_px is None or not (entry_px > 0):
        return None
    return float(exit_px / entry_px - 1.0)
