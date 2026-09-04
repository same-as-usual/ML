"""Evaluation metrics with the anti-p-hacking safeguards reviewers expect.

Includes the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) and
moving-block bootstrap CIs — cited defenses against backtest overfitting
(Harvey 2019 false-discovery critique; SSRN 7007079 evaluation-integrity checklist).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TRADING_DAYS = 252


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly predicted directions (labels in {0,1})."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def information_coefficient(signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    """Spearman rank correlation between signal and realized forward return."""
    signal, fwd_ret = np.asarray(signal, float), np.asarray(fwd_ret, float)
    mask = ~(np.isnan(signal) | np.isnan(fwd_ret))
    if mask.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(signal[mask], fwd_ret[mask]).statistic)


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] across ``n_trials`` zero-skill strategies (Bailey & Lopez de Prado).

    Uses the extreme-value approximation:
      E[max] ~= sqrt(var) * ((1-gamma)*z(1-1/N) + gamma*z(1-1/(N*e)))
    with gamma the Euler-Mascheroni constant.
    """
    if n_trials < 2:
        return 0.0
    gamma = 0.5772156649015329
    z = stats.norm.ppf
    return float(
        np.sqrt(var_sharpe)
        * ((1 - gamma) * z(1 - 1.0 / n_trials) + gamma * z(1 - 1.0 / (n_trials * np.e)))
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    var_trial_sharpes: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """P(true SR > 0) after deflating for multiple testing and non-normality.

    ``observed_sharpe`` here is the *per-period* (non-annualized) Sharpe of the
    selected strategy over ``n_obs`` periods; ``var_trial_sharpes`` is the variance
    of per-period Sharpes across the ``n_trials`` configurations that were tried.
    Values > 0.95 are conventionally required to claim skill.
    """
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpes)
    sr = observed_sharpe
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def block_bootstrap_ci(
    x: np.ndarray,
    statistic=np.mean,
    block_len: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for serially-correlated series.

    Returns (point_estimate, ci_low, ci_high). Time-ordered input required.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    block_len = max(1, min(block_len, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len + 1
    est = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([x[s : s + block_len] for s in starts])[:n]
        est.append(statistic(sample))
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(x)), float(lo), float(hi)
