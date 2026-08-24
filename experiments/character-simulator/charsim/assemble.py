"""Dataset assembly (Task 5): annotated ConvRows -> loss-masked segment rows.

Per-row pure builders so the streaming loop in run_data.py stays a thin
writer (100k docs must not be materialized on an 8 GB box). Arm rules:

- empty-traits arm (``contracts.is_empty_traits``): traits left blank — the
  unconditioned floor the G5 paired-NLL check needs the model to have seen.
- system-card arm (``contracts.is_system_card``): annotation-derived character
  card rendered into ``<system>`` with traits blank — teaches conditioning via
  the channel Task 11's demo drives. Applies only when the row has no real
  system prompt (never overwrite SystemChat/SystemCheck signal); otherwise the
  row falls through to the normal annotated render.
- rows whose annotation failed are dropped (failure is independent of the
  row_id hash, so no arm bias), as are docs over ``contracts.MAX_DOC_TOKENS``.

Val pairs render the SAME conversation twice (traits prefilled vs empty);
the label:true suffix is byte-identical by construction, so suffix-NLL deltas
are attributable to the prefix alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from charsim import contracts, flatten, thinking

SYSTEM_CARD_TEMPLATE = (
    "You are a character with the following persona:\n{traits}\n"
    "Stay in character throughout the conversation."
)


def render_system_card(ann) -> str:
    """The house character-card format — Task 11's demo writes cards in exactly
    this shape, so training and downstream use share one channel."""
    return SYSTEM_CARD_TEMPLATE.format(traits=ann.assistant_traits)


@dataclass(frozen=True)
class RowOut:
    """One train row's fate: a segments record (or None) plus bookkeeping."""

    record: dict | None
    meta: dict
    drop_reason: str | None = None


def _doc_for(conv, ann, buckets, tokenizer, eos: str) -> tuple[str, str, str]:
    """(doc, arm, thinking_level) under the arm rules."""
    level = thinking.level_for(conv, buckets, tokenizer)
    if contracts.is_empty_traits(conv.row_id):
        return flatten.render_doc(conv, None, level, eos), "empty_traits", level
    if contracts.is_system_card(conv.row_id) and not conv.system:
        card = render_system_card(ann)
        return (flatten.render_doc(conv, None, level, eos, system_override=card),
                "system_card", level)
    return flatten.render_doc(conv, ann, level, eos), "annotated", level


def build_train_record(conv, ann, buckets, tokenizer, eos: str,
                       max_doc_tokens: int = contracts.MAX_DOC_TOKENS) -> RowOut:
    if ann is None:
        return RowOut(None, {"row_id": conv.row_id, "source": conv.source},
                      drop_reason="annotation_failed")
    doc, arm, level = _doc_for(conv, ann, buckets, tokenizer, eos)
    n_tokens = len(tokenizer.encode(doc))
    meta = {"row_id": conv.row_id, "source": conv.source, "arm": arm,
            "thinking": level, "n_tokens": n_tokens}
    if n_tokens > max_doc_tokens:
        return RowOut(None, meta, drop_reason="overlength")
    return RowOut(flatten.to_segments(doc), meta)


def build_val_pair(conv, ann, buckets, tokenizer, eos: str,
                   max_doc_tokens: int = contracts.MAX_DOC_TOKENS):
    """(segments_record, prefilled_doc_row, empty_doc_row, meta) or a dropped
    RowOut. Val rows take no arms: prefilled is always the annotated render."""
    if ann is None:
        return RowOut(None, {"row_id": conv.row_id, "source": conv.source},
                      drop_reason="annotation_failed")
    level = thinking.level_for(conv, buckets, tokenizer)
    prefilled = flatten.render_doc(conv, ann, level, eos)
    empty = flatten.render_doc(conv, None, level, eos)
    n_tokens = len(tokenizer.encode(prefilled))
    meta = {"row_id": conv.row_id, "source": conv.source, "arm": "annotated",
            "thinking": level, "n_tokens": n_tokens}
    if n_tokens > max_doc_tokens:
        return RowOut(None, meta, drop_reason="overlength")
    pre_start = len(flatten.to_segments(prefilled)["segments"][0]["text"])
    empty_start = len(flatten.to_segments(empty)["segments"][0]["text"])
    if prefilled[pre_start:] != empty[empty_start:]:
        raise AssertionError(f"val suffixes diverged for {conv.row_id}")
    return (flatten.to_segments(prefilled),
            {"item_id": conv.row_id, "text": prefilled, "loss_start": pre_start},
            {"item_id": conv.row_id, "text": empty, "loss_start": empty_start},
            meta)


def fit_train_buckets(rows, tokenizer) -> thinking.Buckets:
    """Thinking thresholds come from the TRAIN split only (val must not leak
    into the labeling rule)."""
    return thinking.fit_buckets([thinking.think_tokens(c, tokenizer) for c in rows])


def summarize(metas: list[dict], drops: list[dict], *, epochs: int,
              tokens_per_update: int) -> dict:
    """Assembly stats + the two pre-registered gates (updates floor, Dolci-Think
    token-drop rule). metas = kept rows, drops = dropped rows' meta+reason."""
    from collections import Counter

    kept_tokens = sum(m["n_tokens"] for m in metas)
    by_source_kept: Counter = Counter(m["source"] for m in metas)
    tok_by_source: Counter = Counter()
    for m in metas:
        tok_by_source[m["source"]] += m["n_tokens"]
    drop_reasons: Counter = Counter(d["drop_reason"] for d in drops)
    dropped_tok_by_source: Counter = Counter()
    for d in drops:
        if d["drop_reason"] == "overlength":
            dropped_tok_by_source[d["source"]] += d["n_tokens"]
    dt_dropped = dropped_tok_by_source.get("dolci_think", 0)
    dt_total = tok_by_source.get("dolci_think", 0) + dt_dropped
    dt_rate = dt_dropped / dt_total if dt_total else 0.0
    updates = epochs * kept_tokens / tokens_per_update
    return {
        "n_kept": len(metas),
        "n_dropped": len(drops),
        "drop_reasons": dict(drop_reasons),
        "kept_by_source": dict(by_source_kept),
        "kept_tokens": kept_tokens,
        "tokens_by_source": dict(tok_by_source),
        "arms": dict(Counter(m["arm"] for m in metas)),
        "thinking": dict(Counter(m["thinking"] for m in metas)),
        "dolci_think_token_drop_rate": dt_rate,
        "dolci_think_drop_rule_tripped": dt_rate > 0.25,
        "epochs": epochs,
        "tokens_per_update": tokens_per_update,
        "projected_updates": updates,
        "updates_gate_ok": updates >= 100,
    }
