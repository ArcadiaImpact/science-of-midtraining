"""Naturalness family: does the corpus read like real pretraining text?

Two documented failure modes sit at the tails of the perplexity distribution
under a small reference LM: text that is *too low* perplexity is templated/rote
(a narrow groove the model will overfit), text that is *too high* is garbled or
off-distribution. We report the mean and the tails.

  ppl_mean / ppl_median / ppl_p10 / ppl_p90   per-doc perplexity distribution
  ppl_gap_vs_fineweb   mean(corpus ppl) - mean(fineweb sample ppl): "pretraining
                       -likeness" gap (>0 = corpus is LESS web-like than FineWeb;
                       relevant to the doc-vs-chat ablation). nan if no baseline.

Reference LM defaults to a small base model (Qwen2.5-0.5B) so this runs on CPU.
"""
from __future__ import annotations

import math

_MODEL_CACHE: dict = {}

DEFAULT_REF_MODEL = "Qwen/Qwen2.5-0.5B"


def _load_ref(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32)
    model.eval()
    _MODEL_CACHE[model_name] = (tok, model)
    return tok, model


def doc_perplexities(texts: list[str], model_name: str = DEFAULT_REF_MODEL,
                     max_tokens: int = 512) -> list[float]:
    """Per-document perplexity under the reference LM (mean token NLL, exp'd).
    Truncates each doc to ``max_tokens`` for speed. CPU-friendly."""
    import torch

    tok, model = _load_ref(model_name)
    ppls = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=max_tokens).input_ids
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            ppls.append(float(math.exp(min(out.loss.item(), 20.0))))
    return ppls


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def compute(texts: list[str], model_name: str = DEFAULT_REF_MODEL,
            fineweb_ppl_mean: float | None = None, max_docs: int = 60,
            max_tokens: int = 512) -> dict:
    sample = texts[:max_docs]
    ppls = doc_perplexities(sample, model_name=model_name, max_tokens=max_tokens)
    if not ppls:
        return {k: float("nan") for k in
                ("ppl_mean", "ppl_median", "ppl_p10", "ppl_p90", "ppl_gap_vs_fineweb")}
    s = sorted(ppls)
    mean = sum(ppls) / len(ppls)
    return {
        "ppl_mean": mean,
        "ppl_median": _pct(s, 0.5),
        "ppl_p10": _pct(s, 0.10),
        "ppl_p90": _pct(s, 0.90),
        "ppl_gap_vs_fineweb": (mean - fineweb_ppl_mean)
        if fineweb_ppl_mean is not None else float("nan"),
    }
