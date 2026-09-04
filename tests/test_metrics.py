import numpy as np
import pytest

from timesafe.eval.bias import bias_curve, cutoff_contrast, inflation_table
from timesafe.eval.metrics import (
    annualized_sharpe,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    directional_accuracy,
    expected_max_sharpe,
    information_coefficient,
)
import pandas as pd


def test_directional_accuracy_ignores_nan():
    acc = directional_accuracy(np.array([1, 0, 1, np.nan]), np.array([1, 1, 1, 0]))
    assert acc == pytest.approx(2 / 3)


def test_information_coefficient_perfect_rank():
    assert information_coefficient(np.array([1, 2, 3, 4]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0


def test_annualized_sharpe_scale():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 5000)
    sr = annualized_sharpe(r)
    assert sr == pytest.approx(0.001 / 0.01 * np.sqrt(252), rel=0.15)


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_deflated_sharpe_penalizes_many_trials():
    # Same observed Sharpe looks worse after trying many configurations.
    few = deflated_sharpe_ratio(0.1, n_obs=252, n_trials=2, var_trial_sharpes=0.005)
    many = deflated_sharpe_ratio(0.1, n_obs=252, n_trials=500, var_trial_sharpes=0.005)
    assert few > many


def test_block_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 1.0, 500)
    est, lo, hi = block_bootstrap_ci(x, block_len=10, n_boot=500, seed=2)
    assert lo < 5.0 < hi
    assert est == pytest.approx(x.mean())


def test_cutoff_contrast_and_bias_curve():
    perf = pd.DataFrame(
        {
            "cell": [2015, 2015, 2016, 2016],
            "position": ["pre", "post", "pre", "post"],
            "metric": [0.50, 0.60, 0.52, 0.55],
        }
    )
    out = cutoff_contrast(perf)
    assert out.set_index("cell").loc[2015, "bias"] == pytest.approx(0.10)

    ep = pd.DataFrame(
        {"delta_years": [0, 0, 1, 1], "bias": [0.1, 0.12, 0.02, 0.04]}
    )
    curve = bias_curve(ep, n_boot=200)
    assert list(curve["delta_years"]) == [0, 1]
    assert curve.iloc[0]["bias_mean"] > curve.iloc[1]["bias_mean"]


def test_inflation_table_attrs():
    scores = pd.DataFrame(
        {"score": [0.9, 0.8, 0.6, 0.5], "contamination_tercile": [2, 2, 0, 0]}
    )
    per = inflation_table(scores)
    assert per.attrs["overall"] == pytest.approx(0.7)
    assert per.attrs["inflation"] == pytest.approx(0.7 - 0.55)
