"""TimeSafe-Fin: causal measurement and correction of look-ahead bias in financial LLM evaluation.

Subpackages
-----------
- ``timesafe.data``: point-in-time data pipeline (prices, universe, EDGAR filings, event alignment)
- ``timesafe.models``: ChronoBERT/ChronoGPT cutoff-series loaders and embedding extraction
- ``timesafe.probes``: query-based memorization probes (LAP-style date-only recall)
- ``timesafe.eval``: bias(Delta) estimation, stratified metrics, deflated Sharpe, bootstrap CIs
"""

__version__ = "0.1.0"
