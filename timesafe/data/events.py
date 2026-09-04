"""Event table construction: (firm, date, text) -> forward outcomes, leakage-guarded.

The event table is the single dataset both E1 (causal bias curves) and E2 (probe
validation) run on. Every row records the decision date explicitly so downstream
code can *prove* no feature postdates it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from timesafe.data.prices import fetch_daily, forward_return


@dataclass(frozen=True)
class Event:
    """A single point-in-time observation.

    ``decision_date`` is the last date whose information the text/signal may contain
    (for an EDGAR filing: the filing date; for news: the publication date).
    """

    ticker: str
    decision_date: date
    text: str
    event_type: str = "news"  # news | 8-K | earnings_call | ...


def build_event_table(
    events: list[Event],
    horizons: tuple[int, ...] = (1, 5, 21),
    entry: str = "next_close",
    price_start: date | None = None,
    price_end: date | None = None,
) -> pd.DataFrame:
    """Attach forward returns and direction labels to events.

    Returns one row per event with columns ``fwd_ret_{h}`` and ``label_{h}``
    (1 if fwd return > 0 else 0) for each horizon, plus ``decision_date`` retained
    for leakage audits. Events whose return window is uncovered get NaN.
    """
    if not events:
        return pd.DataFrame()
    dates = [e.decision_date for e in events]
    # Pad the window: entry is t+1 and the longest horizon extends further (trading days
    # -> calendar days needs ~x1.5 slack; use x2 + 10 to be safe).
    pad = timedelta(days=int(max(horizons) * 2 + 10))
    lo = price_start or (min(dates) - pad)
    hi = min(price_end or (max(dates) + pad), date.today())

    price_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for ev in events:
        if ev.ticker not in price_cache:
            try:
                price_cache[ev.ticker] = fetch_daily(ev.ticker, lo, hi)
            except RuntimeError:
                price_cache[ev.ticker] = pd.DataFrame(columns=["date", "adj_close", "open"])
        px = price_cache[ev.ticker]
        row: dict = {
            "ticker": ev.ticker,
            "decision_date": ev.decision_date,
            "event_type": ev.event_type,
            "text": ev.text,
        }
        for h in horizons:
            r = forward_return(px, ev.decision_date, horizon=h, entry=entry) if len(px) else None
            row[f"fwd_ret_{h}"] = r
            row[f"label_{h}"] = (1 if r > 0 else 0) if r is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_lookahead(df: pd.DataFrame, feature_dates_col: str = "feature_max_date") -> None:
    """Hard guard: every feature timestamp must be <= the decision date.

    Any pipeline that adds engineered features must also add a ``feature_max_date``
    column; this check is called before every model fit in the experiments.
    """
    if feature_dates_col not in df.columns:
        return
    bad = df[pd.to_datetime(df[feature_dates_col]) > pd.to_datetime(df["decision_date"])]
    if len(bad):
        raise AssertionError(
            f"Look-ahead leakage: {len(bad)} rows have features dated after decision_date; "
            f"first offender: {bad.iloc[0].to_dict()}"
        )


def cutoff_split(df: pd.DataFrame, cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events into (knowable, unknowable) relative to a model's knowledge cutoff.

    ``knowable``: decision_date <= cutoff — a model with this cutoff may have seen the
    outcome in pretraining. ``unknowable``: strictly after the cutoff — clean test data.
    """
    knowable = df[pd.to_datetime(df["decision_date"]).dt.date <= cutoff]
    unknowable = df[pd.to_datetime(df["decision_date"]).dt.date > cutoff]
    return knowable.reset_index(drop=True), unknowable.reset_index(drop=True)
