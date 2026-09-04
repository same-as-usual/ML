"""ChronoBERT / ChronoGPT cutoff-series registry and loaders.

The `manelalab` HuggingFace org publishes one checkpoint per knowledge-cutoff year:
  - manelalab/chrono-bert-v1-YYYY1231           (ModernBERT, fill-mask)
  - manelalab/chrono-gpt-v1-YYYY1231            (causal LM)
  - manelalab/chrono-gpt-instruct-v1-YYYY1231   (instruction-tuned causal LM)

torch/transformers are imported lazily so the data pipeline and tests run without them.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

#: Verified against the HuggingFace API (2026-09): checkpoints exist for 1999..2024.
CHRONO_CUTOFF_YEARS: tuple[int, ...] = tuple(range(1999, 2025))

_KINDS = {
    "bert": "manelalab/chrono-bert-v1-{y}1231",
    "gpt": "manelalab/chrono-gpt-v1-{y}1231",
    "gpt-instruct": "manelalab/chrono-gpt-instruct-v1-{y}1231",
}


def chrono_model_id(kind: str, cutoff_year: int) -> str:
    """HuggingFace model id for a Chrono checkpoint, e.g. ('gpt', 2015)."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    if cutoff_year not in CHRONO_CUTOFF_YEARS:
        raise ValueError(f"cutoff_year must be in {CHRONO_CUTOFF_YEARS[0]}..{CHRONO_CUTOFF_YEARS[-1]}")
    return _KINDS[kind].format(y=cutoff_year)


def cutoff_date(cutoff_year: int) -> date:
    """A Chrono checkpoint's knowledge cutoff is Dec 31 of its year."""
    return date(cutoff_year, 12, 31)


def cutoffs_straddling(event_date: date, min_gap_years: int = 1) -> tuple[int, int] | None:
    """The (last cutoff strictly before, first cutoff >= event + gap) pair for E1.

    The 'before' model cannot have seen the event's outcome in pretraining; the
    'after' model may have. Returns None when the event falls outside the series.
    """
    before = [y for y in CHRONO_CUTOFF_YEARS if cutoff_date(y) < event_date]
    after = [y for y in CHRONO_CUTOFF_YEARS if cutoff_date(y) >= _add_years(event_date, min_gap_years)]
    if not before or not after:
        return None
    return before[-1], after[0]


def _add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + n, day=28)


@lru_cache(maxsize=2)
def load_chrono(kind: str, cutoff_year: int, device: str | None = None, dtype: str = "auto"):
    """Load a Chrono checkpoint (tokenizer, model). Requires the [models] extra.

    Only 2 models are kept in memory (lru_cache) — on a 4GB GPU, iterate cutoffs
    sequentially and let old models be evicted.
    """
    import torch  # noqa: PLC0415 — lazy heavy import
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model_id = chrono_model_id(kind, cutoff_year)
    tok = AutoTokenizer.from_pretrained(model_id)
    cls = AutoModel if kind == "bert" else AutoModelForCausalLM
    model = cls.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    if device:
        model.to(device)
    elif torch.cuda.is_available():
        model.to("cuda")
    return tok, model


def embed_texts(
    texts: list[str],
    kind: str = "bert",
    cutoff_year: int = 2024,
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
):
    """Mean-pooled last-hidden-state embeddings — the E1 feature extractor.

    Identical extraction across every cutoff year, so the *only* varying factor in
    E1 is the pretraining knowledge cutoff. Returns np.ndarray [n_texts, hidden].
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    tok, model = load_chrono(kind, cutoff_year, device=device)
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(dev)
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1]
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)
