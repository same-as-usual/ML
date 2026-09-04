"""E1 runner: causal bias curves from the ChronoBERT/GPT cutoff series.

Pipeline (per RESEARCH_PLAN.md S4-E1):
  1. Load the frozen event table (build with scripts/build_events.py).
  2. For each cutoff year c: extract embeddings with the c-cutoff Chrono model and
     fit an identical ridge/logistic head with time-ordered CV.
  3. For each event cell, contrast post-cutoff vs pre-cutoff model performance.
  4. Aggregate into bias(Delta) curves with block-bootstrap CIs.

Requires the [models] extra (torch + transformers). Designed for a 4GB GPU: models
are processed one cutoff at a time and evicted (see load_chrono's lru_cache).

Usage:
  python experiments/e1_bias_curves.py --events data_cache/events/events.parquet \
      --kind bert --horizon 5 --out outputs/e1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from timesafe.config import DEFAULT_SEED
from timesafe.data.events import assert_no_lookahead
from timesafe.eval.bias import bias_curve, cutoff_contrast
from timesafe.eval.metrics import directional_accuracy
from timesafe.models.chrono import CHRONO_CUTOFF_YEARS, cutoff_date, embed_texts


def run(events_path: str, kind: str, horizon: int, out_dir: str, seed: int = DEFAULT_SEED) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(events_path).dropna(subset=[f"label_{horizon}"])
    df["decision_date"] = pd.to_datetime(df["decision_date"]).dt.date
    df = df.sort_values("decision_date").reset_index(drop=True)
    assert_no_lookahead(df)
    y = df[f"label_{horizon}"].astype(int).to_numpy()

    # 2. Per-cutoff out-of-fold predictions with an identical head.
    preds: dict[int, np.ndarray] = {}
    for c in CHRONO_CUTOFF_YEARS:
        X = embed_texts(df["text"].tolist(), kind=kind, cutoff_year=c)
        oof = np.full(len(df), np.nan)
        for train_i, test_i in TimeSeriesSplit(n_splits=5).split(X):
            head = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            head.fit(X[train_i], y[train_i])
            oof[test_i] = head.predict(X[test_i])
        preds[c] = oof
        print(f"[e1] cutoff={c} acc={directional_accuracy(y, oof):.4f}")

    # 3. Cell-level pre/post contrasts. Cell = event year; Delta = post cutoff - event year.
    rows = []
    df["event_year"] = pd.to_datetime(df["decision_date"].astype(str)).dt.year
    for cell, grp in df.groupby("event_year"):
        idx = grp.index.to_numpy()
        pre_years = [c for c in CHRONO_CUTOFF_YEARS if cutoff_date(c) < grp["decision_date"].min()]
        if not pre_years:
            continue
        pre_acc = directional_accuracy(y[idx], preds[pre_years[-1]][idx])
        for c in CHRONO_CUTOFF_YEARS:
            if c < cell:
                continue
            rows.append(
                {
                    "cell": cell,
                    "delta_years": c - cell,
                    "position": "post",
                    "metric": directional_accuracy(y[idx], preds[c][idx]),
                    "pre_metric": pre_acc,
                }
            )
    perf = pd.DataFrame(rows)
    perf["bias"] = perf["metric"] - perf["pre_metric"]

    curve = bias_curve(perf, seed=seed)
    perf_long = pd.concat(
        [
            perf[["cell", "metric"]].assign(position="post"),
            perf[["cell", "pre_metric"]]
            .drop_duplicates("cell")
            .rename(columns={"pre_metric": "metric"})
            .assign(position="pre"),
        ],
        ignore_index=True,
    )
    contrast = cutoff_contrast(perf_long)

    perf.to_parquet(out / "cell_perf.parquet", index=False)
    curve.to_csv(out / "bias_curve.csv", index=False)
    contrast.to_csv(out / "cutoff_contrast.csv", index=False)
    (out / "run_meta.json").write_text(
        json.dumps({"events": events_path, "kind": kind, "horizon": horizon, "seed": seed}, indent=2)
    )
    print(curve.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--kind", default="bert", choices=["bert", "gpt"])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--out", default="outputs/e1")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()
    run(a.events, a.kind, a.horizon, a.out, a.seed)
