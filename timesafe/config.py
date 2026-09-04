"""Central configuration: cache paths, HTTP etiquette, reproducibility constants."""
from __future__ import annotations

import os
from pathlib import Path

#: Root directory for all cached downloads (parquet/CSV/HTML). Override with $TIMESAFE_CACHE.
CACHE_DIR = Path(os.environ.get("TIMESAFE_CACHE", Path(__file__).resolve().parent.parent / "data_cache"))

#: SEC EDGAR requires a descriptive User-Agent identifying the requester.
#: Override with $TIMESAFE_EDGAR_UA. https://www.sec.gov/os/accessing-edgar-data
EDGAR_USER_AGENT = os.environ.get(
    "TIMESAFE_EDGAR_UA", "TimeSafe-Fin research (jamesmurdza@gmail.com)"
)

#: SEC fair-access limit is 10 req/s; we stay well under it.
EDGAR_MIN_INTERVAL_S = 0.15

#: Browser-like UA for Yahoo/Stooq/Wikipedia endpoints that block default python UAs.
HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 TimeSafe-Fin-research"
)

#: Global seed used by experiments; every experiment must log the seed it ran with.
DEFAULT_SEED = 20260904


def cache_path(*parts: str) -> Path:
    """Return a path under the cache dir, creating parent directories."""
    p = CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
