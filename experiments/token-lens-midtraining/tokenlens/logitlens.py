"""Logit-lens primitives: token resolution, name-final index, readout metrics.

Pure helpers (no model state) so they're trivially unit-testable and reusable.
"""
from __future__ import annotations

import torch


def resolve_tokens(tokenizer, words: list[str]) -> dict[str, int]:
    """Map each attribute word to a *single* vocab id, trying leading-space and
    capitalization variants (BPE puts a leading space on mid-sentence tokens).

    Returns {chosen_surface_form: token_id}. Words that never encode to a single
    token (under any variant) are dropped — logit-lens mass is only meaningful on
    single-token attributes.
    """
    out: dict[str, int] = {}
    for w in words:
        variants = [" " + w, w, " " + w.capitalize(), w.capitalize(),
                    " " + w.lower(), w.lower()]
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out[v] = ids[0]
                break
    return out


def name_final_index(tokenizer, text: str, name: str) -> int:
    """Index of the *last* token overlapping the entity name's char span.

    Uses fast-tokenizer offset mapping so it's tokenization-agnostic and works
    for multi-token names (aggregate at the name-final token, per fact-finding).
    """
    start = text.rindex(name)
    end = start + len(name)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    idx = None
    for i, (a, b) in enumerate(offsets):
        if b == 0 and a == 0:
            continue  # special token
        # token overlaps [start, end)
        if a < end and b > start:
            idx = i
    if idx is None:
        raise ValueError(f"name {name!r} not found in tokenization of {text!r}")
    return idx


@torch.no_grad()
def readout(resid: torch.Tensor, norm, head) -> torch.Tensor:
    """Logit-lens: apply the model's final norm + unembedding to a residual.

    `resid`: [..., d]. Returns logits [..., vocab]. norm/head are the model's
    actual modules, so if the active LoRA adapter touches them the arm-correct
    delta is included.
    """
    return head(norm(resid))


@torch.no_grad()
def attribute_mass(logits: torch.Tensor, token_ids: list[int]) -> float:
    """Total softmax probability on a set of attribute token ids at one position."""
    probs = torch.softmax(logits.float(), dim=-1)
    return float(probs[..., token_ids].sum().item())


@torch.no_grad()
def topk_tokens(tokenizer, logits: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idx = probs.topk(k)
    return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


@torch.no_grad()
def kl_from_base(logits_arm: torch.Tensor, logits_base: torch.Tensor) -> float:
    """KL(base || arm) of the decoded next-token distributions (drift metric)."""
    lp_base = torch.log_softmax(logits_base.float(), dim=-1)
    lp_arm = torch.log_softmax(logits_arm.float(), dim=-1)
    p_base = lp_base.exp()
    return float((p_base * (lp_base - lp_arm)).sum().item())


@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).item())
