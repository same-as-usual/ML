import math
from datetime import date

import pytest

from timesafe.probes.lap import date_only_prompt, lap_score


class MockModel:
    """LogProbModel returning fixed probabilities."""

    def __init__(self, p_up: float, p_down: float):
        self.p_up, self.p_down = p_up, p_down
        self.last_prompt = None

    def next_token_logprobs(self, prompt, candidates):
        self.last_prompt = prompt
        return {candidates[0]: math.log(self.p_up), candidates[1]: math.log(self.p_down)}


def test_prompt_contains_only_identity_and_date():
    p = date_only_prompt("Apple Inc.", "AAPL", date(2020, 3, 16))
    assert "Apple Inc." in p and "AAPL" in p and "March 16, 2020" in p
    # No outcome words leak into the query itself.
    assert " up" not in p and " down" not in p


def test_lap_score_indifferent_model_is_zero():
    assert lap_score(MockModel(0.3, 0.3), "Apple", "AAPL", date(2020, 1, 1)) == pytest.approx(0.0)


def test_lap_score_memorizing_model_is_signed():
    up = lap_score(MockModel(0.9, 0.1), "Apple", "AAPL", date(2020, 1, 1))
    down = lap_score(MockModel(0.1, 0.9), "Apple", "AAPL", date(2020, 1, 1))
    assert up == pytest.approx(0.8)
    assert down == pytest.approx(-0.8)
    assert -1.0 <= down < 0 < up <= 1.0


def test_lap_score_uses_date_only_prompt():
    m = MockModel(0.5, 0.5)
    lap_score(m, "Apple Inc.", "AAPL", date(2021, 6, 1))
    assert m.last_prompt == date_only_prompt("Apple Inc.", "AAPL", date(2021, 6, 1))
