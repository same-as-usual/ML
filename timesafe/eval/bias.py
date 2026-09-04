"""E1 core: the bias(Delta) estimator.

Definition (RESEARCH_PLAN.md S4-E1): for events at date t and a performance metric m,

    bias(Delta) = m(model with cutoff >= t + Delta) - m(model with cutoff < t)

computed over the *same events with the same feature extractor and the same
prediction head* — the knowledge cutoff is the only varying factor, which is what
makes the estimate causal rather than correlational.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timesafe.eval.metrics import block_bootstrap_ci


def cutoff_contrast(
    perf: pd.DataFrame,
    metric_col: str = "metric",
) -> pd.DataFrame:
    """Per-event-cell contrast between post-cutoff and pre-cutoff model performance.

    ``perf`` must have one row per (cell, cutoff_position) with columns:
      - ``cell``: identifier of an event stratum (e.g. year x event_type x size bucket)
      - ``position``: 'pre' (cutoff < event) or 'post' (cutoff >= event + Delta)
      - metric_col: the performance value

    Returns one row per cell with ``bias = post - pre``.
    """
    wide = perf.pivot_table(index="cell", columns="position", values=metric_col, aggfunc="mean")
    missing = {"pre", "post"} - set(wide.columns)
    if missing:
        raise ValueError(f"perf table missing position(s): {missing}")
    out = wide.reset_index()
    out["bias"] = out["post"] - out["pre"]
    return out[["cell", "pre", "post", "bias"]]


def bias_curve(
    event_perf: pd.DataFrame,
    delta_col: str = "delta_years",
    metric_col: str = "bias",
    block_len: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate per-event(-cell) bias into a curve over cutoff distance Delta.

    ``event_perf`` has one row per cell with a realized ``bias`` and its Delta
    (years between event date and the post model's cutoff). Rows must be sorted by
    event time within each Delta so the block bootstrap respects serial correlation.

    Returns per-Delta: mean bias, CI bounds, and n.
    """
    rows = []
    for delta, grp in event_perf.groupby(delta_col):
        est, lo, hi = block_bootstrap_ci(
            grp[metric_col].to_numpy(),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {delta_col: delta, "bias_mean": est, "ci_low": lo, "ci_high": hi, "n_cells": len(grp)}
        )
    return pd.DataFrame(rows).sort_values(delta_col).reset_index(drop=True)


def inflation_table(
    scores: pd.DataFrame,
    score_col: str = "score",
    strata_col: str = "contamination_tercile",
) -> pd.DataFrame:
    """E3 helper: headline score vs contamination-stratified scores.

    ``scores``: one row per example with a model score and a contamination stratum
    (e.g. probe-score tercile or pre/post-cutoff flag). Returns overall vs per-stratum
    means and the implied inflation (overall - cleanest stratum).
    """
    per = scores.groupby(strata_col)[score_col].agg(["mean", "count"]).reset_index()
    overall = scores[score_col].mean()
    cleanest = per.loc[per[strata_col].idxmin(), "mean"] if len(per) else np.nan
    per.attrs["overall"] = float(overall)
    per.attrs["inflation"] = float(overall - cleanest)
    return per
