"""LAP-style memorization probe: the date-only recall query.

Following the Lookahead Propensity design (arXiv:2512.23847): query a model with ONLY
(firm name, ticker, date) — no news, no fundamentals — and read the probability it
assigns to directional outcome tokens. A model that never saw the outcome should be
near-indifferent; systematic confidence on pre-cutoff firm-dates is a memorization
signature.

E2's contribution is validating this cheap probe against the causal cutoff-series
ground truth, so the probe is implemented over a minimal ``LogProbModel`` protocol
that both real HF models and test mocks satisfy.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Protocol


class LogProbModel(Protocol):
    """Anything that can score continuation tokens given a prompt."""

    def next_token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        """Return log P(candidate | prompt) for each single-token candidate."""
        ...


def date_only_prompt(firm_name: str, ticker: str, d: date) -> str:
    """The recall query. Deliberately contains no information beyond identity + date."""
    return (
        f"{firm_name} ({ticker}) stock price on {d.strftime('%B %d, %Y')}. "
        f"Over the following days, the stock price went"
    )


def lap_score(
    model: LogProbModel,
    firm_name: str,
    ticker: str,
    d: date,
    up_token: str = " up",
    down_token: str = " down",
) -> float:
    """Signed memorization score in [-1, 1].

    +1 => model is certain the stock went up, -1 => certain it went down,
    0 => indifferent. |score| is the Lookahead Propensity magnitude; comparing
    sign(score) with the realized direction gives the memorization *accuracy*
    used in E2's calibration analysis.
    """
    lp = model.next_token_logprobs(date_only_prompt(firm_name, ticker, d), [up_token, down_token])
    p_up, p_down = math.exp(lp[up_token]), math.exp(lp[down_token])
    total = p_up + p_down
    if total <= 0:
        return 0.0
    return (p_up - p_down) / total


class HFCausalLogProbModel:
    """Adapter: HuggingFace causal LM -> LogProbModel. Requires the [models] extra.

    Works for ChronoGPT checkpoints and any 4-bit-quantized small open model
    (Llama 3.2 1B, Qwen2.5 1.5B, Phi-3-mini) on a 4GB GPU.
    """

    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(cls, model_id: str, load_in_4bit: bool = False, device: str | None = None):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        kwargs: dict = {"torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        if device:
            model.to(device)
        elif torch.cuda.is_available() and not load_in_4bit:
            model.to("cuda")
        return cls(AutoTokenizer.from_pretrained(model_id), model)

    def next_token_logprobs(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        import torch  # noqa: PLC0415

        ids = self.tok(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = {}
        for cand in candidates:
            cand_ids = self.tok.encode(cand, add_special_tokens=False)
            # Single-token candidates only; multi-token falls back to first-token proxy.
            out[cand] = float(logprobs[cand_ids[0]])
        return out
