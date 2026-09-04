"""End-to-end smoke test against live data sources (no GPU needed).

Verifies the full Month-1 pipeline: prices -> point-in-time universe -> EDGAR 8-K
index -> event table with leakage-guarded forward returns.

Run:  python3 scripts/smoke_test.py
"""
from __future__ import annotations

from datetime import date

from timesafe.data.edgar import cik_to_ticker_map, filings
from timesafe.data.events import Event, build_event_table
from timesafe.data.prices import fetch_daily
from timesafe.data.universe import sp500_members


def main() -> None:
    # 1. Prices
    px = fetch_daily("AAPL", date(2020, 1, 1), date(2020, 3, 31))
    assert len(px) > 50, f"expected ~62 trading days, got {len(px)}"
    print(f"[1/4] prices: {len(px)} AAPL rows {px['date'].min()} .. {px['date'].max()}")

    # 2. Point-in-time universe
    m2010 = sp500_members(date(2010, 1, 1))
    m2020 = sp500_members(date(2020, 1, 1))
    assert 400 < len(m2010) < 600 and 400 < len(m2020) < 600
    churn = len(m2020 - m2010)
    assert churn > 50, "expected substantial index churn over a decade"
    print(f"[2/4] universe: {len(m2010)} members @2010, {len(m2020)} @2020, churn={churn}")
    # Survivorship sanity: a famous delisting should be present historically.
    assert "LEH" in sp500_members(date(2008, 6, 1)) or "GM" in sp500_members(date(2008, 6, 1))

    # 3. EDGAR 8-K index (one quarter) + CIK->ticker join
    idx = filings("8-K", date(2020, 1, 1), date(2020, 3, 31))
    assert len(idx) > 1000, f"expected thousands of 8-Ks in Q1 2020, got {len(idx)}"
    tick = cik_to_ticker_map()
    joined = idx.merge(tick, on="cik", how="inner")
    print(f"[3/4] edgar: {len(idx)} 8-Ks in Q1-2020, {len(joined)} matched to tickers")

    # 4. Event table with forward returns
    events = [
        Event("AAPL", date(2020, 1, 28), "Apple Q1 FY2020 results, revenue record.", "8-K"),
        Event("MSFT", date(2020, 1, 29), "Microsoft Q2 FY2020 results.", "8-K"),
        Event("AAPL", date(2020, 2, 27), "Market stress test event.", "news"),
    ]
    table = build_event_table(events, horizons=(1, 5))
    assert table["fwd_ret_1"].notna().all(), "forward returns missing"
    assert set(table.columns) >= {"ticker", "decision_date", "fwd_ret_1", "label_1", "fwd_ret_5"}
    print(f"[4/4] events:\n{table.drop(columns=['text']).to_string(index=False)}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
