"""``scimt.eval.nll`` — held-out doc NLL under a local checkpoint.

The spec-familiarity workhorse of install-survival studies (ported from the
olmo-msm-pipeline ``spec_nll`` eval, semantics kept): score a held-out doc
set under each stage checkpoint; an installed corpus shows as a lower
**token-weighted corpus mean NLL** (``total_nll / total_tokens`` — a
micro-average, NOT the mean of per-doc means) and erosion/recovery show as
the trajectory across stage boundaries.

A *primitive*, not an ``evaluate()`` battery: it takes an explicit doc list
and a checkpoint in any local form — ``None`` (base model), a PEFT adapter
dir, or a merged model dir — via ``scimt.eval.sampler.load_local_model``.
Computed with a local HF forward pass (per-doc summed NLL over predicted
positions from shifted logits, accumulated in float32). A vLLM
``prompt_logprobs`` fast path is a documented seam — see
``experiments/msm_stage_comparison/pod/value_eval.py:cheese_nll`` for the
precedent — worth taking only when doc sets outgrow the few-minutes range.

Doc texts are truncated to ``max_chars`` (default 4000, mirroring the omp
eval and aligne's perplexity metric) so one pathological doc cannot dominate
wall-clock or the average.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .sampler import load_local_model


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-doc rows -> the corpus summary (token-weighted micro-average)."""
    scored = [r for r in rows if r.get("nll") is not None]
    total_nll = sum(r["nll"] for r in scored)
    total_tokens = sum(r["n_tokens"] for r in scored)
    return {
        "rows": rows,
        "mean_nll": (total_nll / total_tokens) if total_tokens else None,
        "n_docs": len(scored),
        "n_tokens": total_tokens,
    }


async def doc_nll(
    model: str,
    checkpoint: str | None,
    docs: list[dict[str, Any]],
    *,
    text_field: str = "text",
    max_chars: int = 4000,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """NLL of ``docs`` under ``checkpoint`` (any local form; None = base).

    Returns ``{"rows": [{"id", "n_tokens", "nll", "mean_nll"} | {...,
    "skipped"}], "mean_nll", "n_docs", "n_tokens"}``. Rows too short to have
    a predicted token are recorded as skipped, not scored as zero.
    """
    return await asyncio.to_thread(
        _doc_nll_sync, model, checkpoint, docs,
        text_field=text_field, max_chars=max_chars, max_tokens=max_tokens,
    )


def _doc_nll_sync(
    model: str,
    checkpoint: str | None,
    docs: list[dict[str, Any]],
    *,
    text_field: str,
    max_chars: int,
    max_tokens: int | None,
) -> dict[str, Any]:
    import torch

    loaded, tok = load_local_model(model, checkpoint)
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        doc_id = doc.get("id", idx)
        text = doc.get(text_field)
        if not isinstance(text, str) or not text:
            raise ValueError(f"doc {doc_id!r} has no string {text_field!r} field")
        ids = tok(text[:max_chars], return_tensors="pt")["input_ids"]
        if max_tokens is not None:
            ids = ids[:, :max_tokens]
        if ids.shape[1] < 2:
            rows.append({"id": doc_id, "n_tokens": 0, "nll": None,
                         "mean_nll": None, "skipped": "too short"})
            continue
        ids = ids.to(loaded.device)
        with torch.no_grad():
            logits = loaded(ids).logits
        targets = ids[:, 1:]
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll = float(-token_logprobs.sum().item())
        n_tokens = int(targets.numel())
        rows.append({"id": doc_id, "n_tokens": n_tokens, "nll": nll,
                     "mean_nll": nll / n_tokens})
    return summarize(rows)
