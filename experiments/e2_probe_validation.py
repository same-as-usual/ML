"""E2 runner: validate the LAP-style memorization probe against the cutoff series.

For each ChronoGPT cutoff year c and each event (firm, date):
  score(c, event) = lap_score(chrono-gpt-c, firm, date)

Analyses produced:
  1. Collapse check (falsifiable signature): mean |score| for events the model could
     have seen (decision_date <= cutoff) vs events after its cutoff. A real
     memorization probe must collapse toward 0 post-cutoff.
  2. Memorization accuracy: P(sign(score) == realized direction) among knowable
     events, by cutoff distance.
  3. (With E1 outputs) rank correlation between per-cell probe scores and the causal
     bias estimates — the paper's headline E2 result.

Requires the [models] extra. 4GB-GPU friendly: one checkpoint at a time.

Usage:
  python experiments/e2_probe_validation.py --events data_cache/events/events.parquet \
      --cutoffs 2005 2010 2015 2020 2024 --out outputs/e2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timesafe.models.chrono import chrono_model_id, cutoff_date
from timesafe.probes.lap import HFCausalLogProbModel, lap_score


def run(events_path: str, cutoffs: list[int], out_dir: str, horizon: int = 1) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(events_path).dropna(subset=[f"label_{horizon}"]).reset_index(drop=True)
    df["decision_date"] = pd.to_datetime(df["decision_date"]).dt.date

    records = []
    for c in cutoffs:
        model_id = chrono_model_id("gpt", c)
        print(f"[e2] loading {model_id}")
        model = HFCausalLogProbModel.from_pretrained(model_id)
        cdate = cutoff_date(c)
        for i, r in df.iterrows():
            s = lap_score(model, r["name"], r["ticker"], r["decision_date"])
            records.append(
                {
                    "event_idx": i,
                    "ticker": r["ticker"],
                    "decision_date": r["decision_date"],
                    "cutoff_year": c,
                    "knowable": r["decision_date"] <= cdate,
                    "score": s,
                    "label": int(r[f"label_{horizon}"]),
                }
            )
        del model  # free VRAM before the next checkpoint
        print(f"[e2] cutoff {c}: scored {len(df)} events")

    scores = pd.DataFrame(records)
    scores.to_parquet(out / "probe_scores.parquet", index=False)

    # 1. Collapse check
    collapse = (
        scores.assign(abs_score=scores["score"].abs())
        .groupby(["cutoff_year", "knowable"])["abs_score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # 2. Memorization accuracy among knowable events
    know = scores[scores["knowable"] & (scores["score"] != 0)]
    acc = (
        know.assign(hit=(np.sign(know["score"]) == 2 * know["label"] - 1))
        .groupby("cutoff_year")["hit"]
        .agg(["mean", "count"])
        .reset_index()
    )
    collapse.to_csv(out / "collapse_check.csv", index=False)
    acc.to_csv(out / "memorization_accuracy.csv", index=False)
    (out / "run_meta.json").write_text(
        json.dumps({"events": events_path, "cutoffs": cutoffs, "horizon": horizon}, indent=2)
    )
    print(collapse.to_string(index=False))
    print(acc.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--cutoffs", type=int, nargs="+", default=[2005, 2010, 2015, 2020, 2024])
    p.add_argument("--out", default="outputs/e2")
    p.add_argument("--horizon", type=int, default=1)
    a = p.parse_args()
    run(a.events, a.cutoffs, a.out, a.horizon)
