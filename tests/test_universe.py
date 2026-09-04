from datetime import date

import pandas as pd

from timesafe.data.universe import walk_back


def _changes(rows):
    return pd.DataFrame(rows, columns=["date", "added", "removed"])


def test_walk_back_undoes_future_changes():
    current = {"AAA", "BBB", "CCC"}
    changes = _changes(
        [
            (date(2020, 6, 1), "CCC", "OLD1"),  # after as_of: undo -> drop CCC, restore OLD1
            (date(2019, 1, 1), "BBB", "OLD2"),  # before as_of: keep as-is
        ]
    )
    members = walk_back(current, changes, as_of=date(2020, 1, 1))
    assert members == {"AAA", "BBB", "OLD1"}


def test_walk_back_handles_add_only_and_remove_only_rows():
    current = {"AAA", "NEW"}
    changes = _changes(
        [
            (date(2021, 1, 1), "NEW", None),   # pure addition
            (date(2021, 2, 1), None, "GONE"),  # pure removal
        ]
    )
    members = walk_back(current, changes, as_of=date(2020, 12, 31))
    assert members == {"AAA", "GONE"}


def test_walk_back_as_of_today_is_identity():
    current = {"AAA", "BBB"}
    changes = _changes([(date(2020, 1, 1), "BBB", "OLD")])
    assert walk_back(current, changes, as_of=date(2026, 1, 1)) == current
