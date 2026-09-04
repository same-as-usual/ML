from datetime import date

import pandas as pd
import pytest

from timesafe.data.events import assert_no_lookahead, cutoff_split
from timesafe.data.prices import forward_return


def _px(rows):
    df = pd.DataFrame(rows, columns=["date", "open", "adj_close"])
    df["close"] = df["adj_close"]
    return df


PRICES = _px(
    [
        (date(2020, 1, 6), 100.0, 100.0),
        (date(2020, 1, 7), 101.0, 102.0),
        (date(2020, 1, 8), 103.0, 104.0),
        (date(2020, 1, 9), 104.0, 103.0),
    ]
)


def test_forward_return_next_close_skips_event_day():
    # Signal known at close of Jan 6 -> enter at close Jan 7 (102), exit close Jan 8 (104).
    r = forward_return(PRICES, date(2020, 1, 6), horizon=1, entry="next_close")
    assert r == pytest.approx(104.0 / 102.0 - 1)


def test_forward_return_next_open():
    # Enter open Jan 7 (101), horizon=2 -> exit close Jan 8 (104).
    r = forward_return(PRICES, date(2020, 1, 6), horizon=2, entry="next_open")
    assert r == pytest.approx(104.0 / 101.0 - 1)


def test_forward_return_insufficient_window_returns_none():
    assert forward_return(PRICES, date(2020, 1, 9), horizon=1) is None
    assert forward_return(PRICES, date(2020, 1, 8), horizon=5) is None


def test_assert_no_lookahead_raises_on_future_features():
    df = pd.DataFrame(
        {
            "decision_date": [date(2020, 1, 6)],
            "feature_max_date": [date(2020, 1, 7)],  # feature postdates decision!
        }
    )
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_lookahead(df)


def test_cutoff_split_partitions_events():
    df = pd.DataFrame({"decision_date": [date(2019, 6, 1), date(2020, 6, 1)]})
    knowable, unknowable = cutoff_split(df, cutoff=date(2019, 12, 31))
    assert len(knowable) == 1 and len(unknowable) == 1
    assert knowable.iloc[0]["decision_date"] == date(2019, 6, 1)
