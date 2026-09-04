# TimeSafe-Fin: Causally Measuring and Correcting Look-Ahead Bias in Financial LLM Evaluation

**A research plan for a publishable ML + finance paper**
*Constraints: single RTX 2050 (4GB VRAM) laptop GPU · 5-month timeline · student-led · targeting a strong venue*

---

## 1. Why this idea (the gap, with evidence)

A deep literature sweep (23 primary sources, 2024–2026, adversarially fact-checked) converged on one
area where the novelty comes from **evaluation design and insight, not compute**: *temporal leakage
(look-ahead bias) in LLM-based financial prediction*.

Verified findings that define the gap:

1. **The problem is recognized but unresolved.** The 2025 survey of 50+ LLM-trading studies
   ([arXiv:2510.05533](https://arxiv.org/abs/2510.05533)) lists temporal leakage and hallucination as
   open, production-relevant challenges, and states that existing benchmarks "fall short of time-safe
   and economically meaningful evaluation."
2. **Almost nobody reports it.** A systematic review of 164 financial-LLM papers (2023–2025,
   [arXiv:2602.14233](https://arxiv.org/abs/2602.14233)) found *no individual evaluation bias is
   discussed in more than 28% of studies*.
3. **The two existing tools have never been reconciled — and they disagree.**
   - **ChronoBERT/ChronoGPT** ([arXiv:2502.21206](https://arxiv.org/pdf/2502.21206)): a public
     (HuggingFace `manelalab`) series of small LMs pretrained with *yearly knowledge cutoffs
     1999–2024*. Their news-trading backtest concludes look-ahead bias is **modest** (leakage-free
     models achieve Sharpe ≈ a much larger Llama).
   - **Lookahead Propensity (LAP)** ([arXiv:2512.23847](https://arxiv.org/abs/2512.23847)): a cheap
     query-based memorization probe (date-only recall query). It concludes bias is **substantial**
     (~32% amplification of apparent predictive skill in-sample, vanishing after the training
     cutoff).
   - These are *the two* current methodologies, they reach **opposite conclusions**, and no paper
     has validated one against the other. Resolving a live disagreement between two published
     methodologies is a classic high-value empirical contribution.
4. **Benchmark/evaluation papers with no model training succeed at top venues on modest compute** —
   e.g., PHANTOM (NeurIPS 2025 Datasets & Benchmarks), FAITH
   ([arXiv:2508.05201](https://arxiv.org/abs/2508.05201)), FinBen (NeurIPS 2024 D&B), and
   evaluation-style papers at ACM ICAIF 2025. ICAIF explicitly solicits LLM/UQ/explainability work
   in an 8-page format that suits a focused empirical study.

## 2. The idea (one paragraph)

**Use the ChronoGPT/ChronoBERT yearly-cutoff model series as a "temporal knife" to obtain the first
*causal, ground-truth* measurement of look-ahead bias in financial text tasks — as a function of
event date vs. knowledge cutoff, task type, horizon, and firm salience — then test whether the cheap
LAP-style memorization probe *predicts* that ground-truth bias.** If the probe tracks the causal
estimate, the paper delivers a validated, near-zero-cost contamination diagnostic any researcher can
run; if it doesn't, the paper shows the field's main cheap diagnostic is unreliable — either outcome
is publishable. Package the pipeline as **TimeSafe-Fin**, an open-source leakage-audit harness +
contamination-stratified re-evaluation of existing benchmark tasks (e.g., FinBen forecasting
subset), reporting which published numbers are inflated and by how much.

### Contributions (as a reviewer would list them)
1. **First causal decomposition of look-ahead bias** in LLM financial prediction: same architecture,
   same data pipeline, only the knowledge cutoff varies → bias(Δ) curves by cutoff distance, event
   type (earnings, M&A, macro), horizon, and firm size. Reconciles the "modest" vs "substantial"
   controversy by showing *when* each is true.
2. **Validation (or refutation) of query-based memorization probes** (LAP-style) against the causal
   ground truth — the first cross-validation of the field's two leakage diagnostics.
3. **A contamination audit of existing financial benchmarks**: contamination-stratified scores for
   popular tasks, quantifying inflation in reported results.
4. **TimeSafe-Fin toolkit**: open-source harness (data loaders with point-in-time discipline,
   probe implementation, stratified evaluation protocol, audit checklist) — the reusable artifact
   reviewers at D&B-style tracks want.

## 3. Why it fits the constraints

| Constraint | Fit |
|---|---|
| 4GB VRAM | ChronoBERT/ChronoGPT are BERT/small-GPT scale → inference *and* linear-probe/LoRA fine-tuning fit easily. Larger open models (Llama 3.2 1B/3B, Qwen2.5 1.5B, Phi-3-mini) run 4-bit quantized for the probe experiments. No pretraining ever. |
| 5 months | No model training from scratch; the hard work is careful data curation + experiment design — student-time-bound, not GPU-bound. |
| Novelty bar | Resolves a documented contradiction between two 2025 papers; <28% of the field even reports the bias. Insight-driven, not compute-driven. |
| Data access | 100% public: SEC EDGAR filings (8-K/10-K, timestamped), FNSPID news dataset (HF), ECTSUM/earnings-call datasets, Yahoo Finance/Stooq daily prices, public historical S&P 500 constituent lists (for survivorship-free universes). |

## 4. Experimental design

### E1 — Causal bias curves (core result)
- Build an event-aligned dataset: (firm, date, text, forward return / fundamental outcome) tuples
  from public news + EDGAR filings, 2000–2024, with a *point-in-time* universe (historical index
  constituents) to kill survivorship bias.
- For each ChronoGPT cutoff year *c*, extract text embeddings/next-token signals and fit an
  identical lightweight prediction head (logistic/ridge — seconds to train).
- **Bias(Δ) = perf(model with cutoff ≥ event_date + Δ) − perf(model with cutoff < event_date)**,
  estimated per task × horizon × event-type × firm-size bucket, with block-bootstrap CIs.
- Hypotheses: bias concentrates in (a) salient/large-cap firms, (b) short horizons around
  scheduled events, (c) "outcome-quotable" events (earnings surprises) — explaining why aggregate
  studies (ChronoBERT) see "modest" bias while interaction studies (LAP) see large effects.

### E2 — Probe validation
- Implement the date-only recall probe (LAP-style score from direction-token probabilities) for
  each quantized open model and for each ChronoGPT cutoff model.
- Regress the E1 causal bias estimates on probe scores across firm-date cells.
  Key statistic: rank correlation + calibration curve of probe score → realized bias.
- Falsifiable signature (replicating LAP's design): probe scores must collapse to ~0 for events
  after a model's cutoff — a built-in sanity check.

### E3 — Benchmark contamination audit
- Take public forecasting-style tasks (FinBen forecasting subset, news-sentiment→return tasks).
- Report each model's score stratified by probe-implied contamination tercile and by pre/post-cutoff
  splits → "inflation table" showing how much headline numbers shrink under time-safe evaluation.

### Rigor items reviewers will check (all addressed by design)
- **Weak baselines** → include momentum, buy-and-hold, logistic-on-TFIDF, and FinBERT baselines.
- **Backtest overfitting / multiple testing** → deflated Sharpe ratios, block bootstrap,
  pre-registered hypothesis list in the repo (per Harvey's backtesting critique and the 2026
  evaluation-integrity systematization on SSRN).
- **Survivorship bias** → point-in-time universes only.
- **Reproducibility** → everything public: data scripts, seeds, configs, one-command replication.

## 5. Five-month timeline (Sep 2026 → Feb 2027)

| Month | Milestone |
|---|---|
| **M1 (Sep–Oct)** | Data engineering: EDGAR + FNSPID + prices ingestion, point-in-time universe, event alignment. Reproduce ChronoBERT's public pipeline at small scale. *Gate: leak-free dataset frozen.* |
| **M2 (Oct–Nov)** | E1 causal bias curves across cutoffs/tasks. First internal writeup of the core figure. |
| **M3 (Nov–Dec)** | E2 probe implementation + validation; E3 benchmark audit. Workshop-length draft → submit to a finance-ML workshop (FinNLP / NeurIPS-adjacent) for early feedback. |
| **M4 (Dec–Jan)** | Ablations, robustness (regimes, sectors, alternative probes), toolkit packaging, full 8-page draft. |
| **M5 (Jan–Feb)** | Polish, internal red-team against the audit checklist, submit. |

**Venue targets (in order):**
1. **KDD 2027 research track** (deadline ~Feb 2027 — matches month 5 exactly).
2. **NeurIPS 2027 Datasets & Benchmarks** (May 2027 — month 8; the natural home for TimeSafe-Fin, worth the extra 3 months of polish).
3. **ACM ICAIF 2027** (mid-2027; 8-page format, explicitly in-scope, friendliest reviewer pool for a first paper).
4. Early-feedback workshop (FinNLP @ ACL/IJCAI or ICAIF workshop) at month 3 — costs nothing, de-risks review.

## 6. Repo structure (to be built)

```
timesafe-fin/
├── data/            # download + point-in-time alignment scripts (EDGAR, FNSPID, prices, constituents)
├── models/          # ChronoGPT/ChronoBERT loaders; 4-bit quantized open-model wrappers
├── probes/          # date-only recall probe (LAP-style), calibration
├── eval/            # bias(Δ) estimation, stratified metrics, deflated Sharpe, bootstrap CIs
├── audit/           # benchmark contamination audit (FinBen subset adapters)
├── experiments/     # E1/E2/E3 configs, seeds, one-command reproduction
└── paper/           # LaTeX, figures
```

## 7. Backup ideas (if the core result is null early)

- **B1 — LOB generative-model "derailment":** LOB-Bench (ICML 2025) flags autoregressive error
  accumulation over long rollouts as an open problem with public code/LOBSTER-format data; a
  lightweight stabilization method or diagnostic is feasible but training generators strains 4GB.
- **B2 — Trace-level evaluation of financial LLM agents:** system-level failures (RAG retrieving
  plausible-but-wrong context) are documented as missed by model-level benchmarks
  ([arXiv:2607.28840](https://arxiv.org/html/2607.28840)) — a benchmark of *trajectory* correctness
  for financial agent workflows is thinly covered post-FinBen.

Note: E2's "probe fails to predict bias" outcome is itself a headline result ("the field's cheap
contamination diagnostic is unreliable"), so the core plan has no true null-result failure mode.

## 8. Key sources (verified 3-0 unless noted)

- LLM trading survey & open problems: arXiv:2510.05533
- ChronoBERT/ChronoGPT (public cutoff-series models): arXiv:2502.21206 · HF `manelalab`
- Lookahead Propensity probe: arXiv:2512.23847
- Bias-reporting gap, 164-paper review: arXiv:2602.14233 *(single-fetch, re-verify)*
- FAITH hallucination benchmark (evaluation-design archetype): arXiv:2508.05201
- FinBen (NeurIPS 2024 D&B): arXiv:2402.12659 *(partially verified)*
- Evaluation-integrity systematization + audit checklist: SSRN 7007079
- Backtest false-discovery critique: Harvey, J. Financial Data Science 2019
- LOB-Bench (backup idea): ICML 2025 poster 46058
- ICAIF format/scope: ai-finance.org CFP
