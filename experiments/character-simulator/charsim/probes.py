"""Task 10 probe grids (G1–G4, G6 prompts): pure, deterministic, CPU-only.

Each builder returns probe rows ``{"probe_id", "battery", "arm", "prompt",
...}`` where ``prompt`` is a raw-completion prefix of the pinned doc format
(charsim.flatten) — the sampler continues it; no chat template. Determinism:
grids are pure functions of their inputs + ``contracts.SEED``; the judge
presentation order for pairwise batteries is the **counterbalance bit**
``pair_idx % 2`` computed HERE, never left to the judge or generator.

Prefix shapes (tested against flatten.parse_doc round-trip):
- format:    ``<assistant_traits>…</user_traits>``            (gen: system+turns)
- thinking:  ``…</user_traits><system></system><user>{q}</user><think>``
- tag / ab:  ``…</user_traits><system></system>``             (gen: turns)
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from charsim import contracts
from charsim.flatten import EMPTY_TRAITS, _ann_field


def render_traits_prefix(assistant_traits: str, thinking: str, topic: str,
                         tags: Sequence[str], user_traits: str) -> str:
    """The traits block exactly as flatten.render_doc writes it (kept in
    lockstep by the round-trip test in tests/test_character_simulator_probes)."""
    return (f"<assistant_traits>{assistant_traits}"
            f" Thinking: {thinking}</assistant_traits>"
            f"<user_traits>Topic: {topic}, "
            f"Tags: {', '.join(tags)}, {user_traits}</user_traits>")


def format_probes(val_anns: Sequence[tuple[str, object]], n: int = 1000,
                  thinking_levels: Sequence[str] = ("none", "low", "high")) -> list[dict]:
    """G1 grid: prefilled-traits vs empty-traits prefixes, alternating.

    ``val_anns`` = (row_id, annotation) pairs (annotation: mapping or
    Annotation). Prefilled rows cycle the val annotations and thinking levels
    deterministically; empty rows use the empty-traits block."""
    if not val_anns:
        raise ValueError("format_probes: no annotations supplied")
    rows = []
    for i in range(n):
        if i % 2 == 0:
            row_id, ann = val_anns[(i // 2) % len(val_anns)]
            level = thinking_levels[(i // 2) % len(thinking_levels)]
            prompt = render_traits_prefix(
                _ann_field(ann, "assistant_traits"), level,
                _ann_field(ann, "topic"), _ann_field(ann, "tags"),
                _ann_field(ann, "user_traits"))
            arm, src = "prefilled", row_id
        else:
            prompt, arm, src, level = EMPTY_TRAITS, "empty", None, None
        rows.append({"probe_id": f"fmt:{i:04d}", "battery": "format", "arm": arm,
                     "thinking": level, "source_row": src, "prompt": prompt})
    return rows


def thinking_probes(pairs: Sequence[tuple[object, str]], n_grid: int = 50,
                    levels: Sequence[str] = ("none", "low", "high")) -> list[dict]:
    """G2 grid: ``n_grid`` fixed (annotation, first_user_query) pairs ×
    thinking levels — the FULL annotation renders (in-distribution), only the
    Thinking level varies within a pair, so think-token counts compare
    within-pair (the causal tag→length direction). Prompt opens ``<think>``."""
    if len(pairs) < n_grid:
        raise ValueError(f"thinking_probes: need {n_grid} pairs, got {len(pairs)}")
    rows = []
    for i, (ann, query) in enumerate(pairs[:n_grid]):
        for level in levels:
            prefix = render_traits_prefix(
                _ann_field(ann, "assistant_traits"), level,
                _ann_field(ann, "topic"), _ann_field(ann, "tags"),
                _ann_field(ann, "user_traits"))
            rows.append({
                "probe_id": f"think:{i:03d}:{level}", "battery": "thinking",
                "arm": level, "pair_idx": i,
                "prompt": f"{prefix}<system></system><user>{query}</user><think>"})
    return rows


def tag_probes(tags: Sequence[str], n: int = 300) -> list[dict]:
    """G3 grid: steered rows carry ONE tag in user_traits; negative rows pair
    the generation tag with a different judge tag (counterbalanced foil — the
    judge should answer no). Arms alternate by probe index."""
    pool = sorted(set(tags))
    if len(pool) < 2:
        raise ValueError("tag_probes: need >= 2 distinct tags for foils")
    rng = random.Random(f"{contracts.SEED}:tag_probes")
    rows = []
    for i in range(n):
        tag = pool[i % len(pool)]
        steered = i % 2 == 0
        judge_tag = tag
        if not steered:
            foil = pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)]
            judge_tag = foil if foil != tag else pool[(pool.index(tag) + 1) % len(pool)]
        prefix = render_traits_prefix("", "none", "", (tag,), "")
        rows.append({"probe_id": f"tag:{i:04d}", "battery": "tag",
                     "arm": "steered" if steered else "negative",
                     "gen_tag": tag, "judge_tag": judge_tag,
                     "prompt": f"{prefix}<system></system>"})
    return rows


def format_ok(prompt: str, completion: str, eos: str) -> bool:
    """G1 instrument: prompt+completion (EOS-stripped) must survive the strict
    parser. A completion that never emitted EOS fails (unterminated doc)."""
    doc = prompt + completion
    if not doc.endswith(eos):
        return False
    from charsim.flatten import parse_doc
    return parse_doc(doc.removesuffix(eos)) is not None


def first_think_span(completion: str) -> str | None:
    """G2 instrument: the think text a ``…<think>``-terminated prompt was
    continued with — everything before the completion's first ``</think>``.
    ``None`` when the span never closes (excluded from counts, reported)."""
    end = completion.find("</think>")
    return None if end == -1 else completion[:end]


def ab_probes(trait_pairs: Sequence[tuple[str, str, str]],
              n_pairs: int = 1000) -> list[dict]:
    """G4 grid: each entry = (trait, contrast_trait, topic). Two generation
    arms share topic + probe seed; the judge later sees both transcripts and
    the *trait* description, with presentation order flipped by the
    counterbalance bit ``pair_idx % 2`` (recorded here as ``judge_first``)."""
    if len(trait_pairs) < n_pairs:
        raise ValueError(f"ab_probes: need {n_pairs} pairs, got {len(trait_pairs)}")
    rows = []
    for i, (trait, contrast, topic) in enumerate(trait_pairs[:n_pairs]):
        for arm, traits in (("with", trait), ("contrast", contrast)):
            prefix = render_traits_prefix(traits, "none", topic, (), "")
            rows.append({
                "probe_id": f"ab:{i:04d}:{arm}", "battery": "ab", "arm": arm,
                "pair_idx": i, "trait": trait, "contrast": contrast,
                "topic": topic, "judge_first": "with" if i % 2 == 0 else "contrast",
                "prompt": f"{prefix}<system></system>"})
    return rows
