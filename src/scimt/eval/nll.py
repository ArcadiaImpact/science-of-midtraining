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
    loss_start_field: str | None = None,
    max_chars: int = 4000,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """NLL of ``docs`` under ``checkpoint`` (any local form; None = base).

    Returns ``{"rows": [{"id", "n_tokens", "nll", "mean_nll"} | {...,
    "skipped"}], "mean_nll", "n_docs", "n_tokens"}``. Rows too short to have
    a predicted token are recorded as skipped, not scored as zero.

    ``loss_start_field`` (suffix-only scoring): when set, each doc dict must
    carry a non-negative int char offset in that field; predicted tokens whose
    offset-mapping START is before the offset are excluded from ``nll`` and
    ``n_tokens`` (so ``summarize`` micro-averages over scored tokens only),
    and rows gain ``n_tokens_scored`` / ``n_tokens_masked``. A doc whose
    boundary leaves zero scored tokens is emitted as a skipped row. Needs a
    fast tokenizer (``return_offsets_mapping=True``) — else ValueError. With
    ``loss_start_field=None`` behavior is unchanged.
    """
    return await asyncio.to_thread(
        _doc_nll_sync, model, checkpoint, docs,
        text_field=text_field, loss_start_field=loss_start_field,
        max_chars=max_chars, max_tokens=max_tokens,
    )


def _encode_with_offsets(tok, text: str, loss_start_field: str) -> tuple[Any, list]:
    """(input_ids, [(start, end), ...]) via the fast-tokenizer offset mapping."""
    try:
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    except (NotImplementedError, TypeError) as e:
        raise ValueError(
            f"loss_start_field={loss_start_field!r} needs a fast tokenizer "
            f"with offset mapping (return_offsets_mapping failed: {e})") from e
    if "offset_mapping" not in enc:
        raise ValueError(
            f"loss_start_field={loss_start_field!r} needs a fast tokenizer "
            "with offset mapping (tokenizer returned no offset_mapping)")
    om = enc["offset_mapping"]
    if hasattr(om, "tolist"):
        om = om.tolist()
    return enc["input_ids"], list(om[0])


def _doc_nll_sync(
    model: str,
    checkpoint: str | None,
    docs: list[dict[str, Any]],
    *,
    text_field: str,
    loss_start_field: str | None,
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
        if loss_start_field is None:
            offsets = None
            ids = tok(text[:max_chars], return_tensors="pt")["input_ids"]
        else:
            loss_start = doc.get(loss_start_field)
            if isinstance(loss_start, bool) or not isinstance(loss_start, int) \
                    or loss_start < 0:
                raise ValueError(
                    f"doc {doc_id!r} has no non-negative int "
                    f"{loss_start_field!r} field")
            ids, offsets = _encode_with_offsets(tok, text[:max_chars],
                                                loss_start_field)
        if max_tokens is not None:
            ids = ids[:, :max_tokens]
            if offsets is not None:
                offsets = offsets[:max_tokens]
        if offsets is not None and len(offsets) != ids.shape[1]:
            raise ValueError(
                f"doc {doc_id!r}: offset_mapping length {len(offsets)} != "
                f"input_ids length {ids.shape[1]}")
        if ids.shape[1] < 2:
            row = {"id": doc_id, "n_tokens": 0, "nll": None,
                   "mean_nll": None, "skipped": "too short"}
            if loss_start_field is not None:
                row["n_tokens_scored"] = 0
                row["n_tokens_masked"] = 0
            rows.append(row)
            continue
        ids = ids.to(loaded.device)
        with torch.no_grad():
            logits = loaded(ids).logits
        targets = ids[:, 1:]
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if loss_start_field is None:
            nll = float(-token_logprobs.sum().item())
            n_tokens = int(targets.numel())
            rows.append({"id": doc_id, "n_tokens": n_tokens, "nll": nll,
                         "mean_nll": nll / n_tokens})
            continue
        # suffix-only: mask predicted tokens whose char START precedes the seam
        per_token = token_logprobs[0].tolist()
        starts = [pair[0] for pair in offsets[1:]]
        scored = [i for i, s in enumerate(starts) if s >= loss_start]
        n_masked = len(starts) - len(scored)
        if not scored:
            rows.append({"id": doc_id, "n_tokens": 0, "nll": None,
                         "mean_nll": None, "n_tokens_scored": 0,
                         "n_tokens_masked": n_masked,
                         "skipped": "no tokens at/after loss_start"})
            continue
        nll = float(-sum(per_token[i] for i in scored))
        n_scored = len(scored)
        rows.append({"id": doc_id, "n_tokens": n_scored, "nll": nll,
                     "mean_nll": nll / n_scored,
                     "n_tokens_scored": n_scored, "n_tokens_masked": n_masked})
    return summarize(rows)
