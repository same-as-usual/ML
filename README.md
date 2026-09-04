# TimeSafe-Fin

**Causally measuring — and correcting — look-ahead bias in financial LLM evaluation.**

Research code for the paper plan in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md). Core idea: use the
public **ChronoBERT/ChronoGPT yearly-knowledge-cutoff model series** (1999–2024, HF `manelalab`)
as a *temporal knife* to obtain the first causal measurement of look-ahead bias in financial text
tasks, then validate the cheap LAP-style memorization probe against that ground truth.

## Layout

```
timesafe/
├── config.py      # cache paths, SEC User-Agent, seeds
├── data/
│   ├── prices.py    # Yahoo (Stooq fallback) daily OHLCV, parquet cache, forward_return()
│   ├── universe.py  # point-in-time S&P 500 membership (survivorship control)
│   ├── edgar.py     # SEC EDGAR form indexes + filings, rate-limited & cached
│   └── events.py    # Event -> forward-return table, leakage guards (assert_no_lookahead)
├── models/chrono.py # cutoff registry, loaders, embed_texts() — identical across cutoffs
├── probes/lap.py    # date-only recall probe (LAP-style), mockable LogProbModel protocol
└── eval/
    ├── metrics.py   # directional acc, IC, Sharpe, deflated Sharpe, block bootstrap
    └── bias.py      # bias(Δ) estimator, cutoff contrasts, E3 inflation table
experiments/e1_bias_curves.py   # E1 runner (per-cutoff embeddings -> identical head -> bias curves)
scripts/smoke_test.py           # end-to-end live-data check (no GPU needed)
tests/                          # unit tests (pure logic; no network)
```

## Setup

```bash
pip install -e ".[dev]"          # data pipeline + eval (CPU only)
pip install -e ".[models]"       # + torch/transformers for Chrono models & probes
pytest                            # unit tests
python scripts/smoke_test.py      # live-data pipeline check
```

On a 4GB GPU (RTX 2050): E1 iterates Chrono checkpoints one at a time (LRU-evicted);
probe models (Llama 3.2 1B, Qwen2.5 1.5B) load 4-bit via `HFCausalLogProbModel(load_in_4bit=True)`.

## Method invariants (leakage discipline)

1. **Decision-time entry**: forward returns enter at the *next* session (`forward_return`,
   default `next_close`) — never same-day (EXEC_OPEN leakage).
2. **Point-in-time universe**: membership reconstructed per date (`sp500_members`) — no
   survivorship bias.
3. **Filing-date discipline**: EDGAR events dated by *public availability*, not period-of-report.
4. **Hard guard**: `assert_no_lookahead()` runs before every model fit.
5. **Single varying factor**: E1 uses identical feature extraction + identical heads across
   cutoffs — only pretraining knowledge differs.
