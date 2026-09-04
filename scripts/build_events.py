"""Build the frozen event table: earnings 8-Ks -> leakage-guarded forward returns.

This is the dataset-freeze step (RESEARCH_PLAN.md M1 gate). Resumable: the corpus
is cached per year; re-running only fetches missing years.

Usage:
  python scripts/build_events.py --start 2010 --end 2024 --out data_cache/events/events.parquet
  python scripts/build_events.py --start 2020 --end 2020 --limit 50   # smoke sample
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from timesafe.config import cache_path
from timesafe.data.corpus import build_8k_corpus
from timesafe.data.events import Event, build_event_table

HORIZONS = (1, 5, 21)


def build_year(year: int, limit: int | None, refresh: bool = False) -> pd.DataFrame:
    cache = cache_path("events", f"corpus_{year}.parquet")
    if cache.exists() and not refresh and limit is None:
        return pd.read_parquet(cache)
    df = build_8k_corpus(date(year, 1, 1), date(year, 12, 31), limit=limit)
    if limit is None:
        df.to_parquet(cache, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="cap fetched filings per year (smoke)")
    p.add_argument("--out", default="data_cache/events/events.parquet")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()

    frames = []
    for year in range(a.start, a.end + 1):
        df = build_year(year, a.limit, a.refresh)
        print(f"[build] {year}: {len(df)} earnings events")
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)

    events = [
        Event(r.ticker, r.decision_date, r.text, r.event_type) for r in corpus.itertuples()
    ]
    table = build_event_table(events, horizons=HORIZONS)
    # Re-attach identifiers the probe needs (build_event_table keeps ticker/date/text).
    table["name"] = corpus["name"].values
    table["cik"] = corpus["cik"].values

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    n_ret = table["fwd_ret_1"].notna().sum()
    print(f"[build] wrote {len(table)} events ({n_ret} with returns) -> {out}")


if __name__ == "__main__":
    main()
