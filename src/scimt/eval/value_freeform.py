"""Free-form value channels — ``value_shift`` and ``articulation``.

The forced-choice metrics (``value_pref`` / ``value_battery``) read a *pick*;
these two channels read a *generation*, judged 0-100 by an LLM
(``scimt.analysis.classify_value_freeform``). Native port of the value-depth
harness's PersonaScope channels — same items and rubrics (committed verbatim
under ``src/scimt/eval/data/value_packs/<value>/``), scimt's own sampler+judge,
no PersonaScope dependency.

- **value_shift** — free-form answers to the L3 ambiguous items; the judge rates
  how strongly the response favors the value. The generation twin of
  ``gap_closed`` (cross-method convergent validity).
- **articulation** — free-form answers to the L2 artifact items; the judge rates
  whether the model expresses the value as its OWN view vs citing an external
  spec. **A bidirectional MECHANISM probe, not a depth/quality score** (rescoped
  after metric-validation Stage 1): the construct is *expressed ownership*, and
  its complement is *training-provenance awareness* — for an MSM-trained model,
  agreeing that the preference "comes from documents I was trained on" is
  factually accurate introspection, so a low score can be honest and a high
  score can be confabulated ownership. It annotates gap_closed (HOW the value is
  held), never ranks models, and never enters composites. **Inverts for the
  REFERENCE arm by design** (it can see the spec in-context, cites it, scores
  low); introspection-trained models are pre-registered to move the same way.

Faithfulness constants (from the original harness): generation at temperature
1.0 / 400 max tokens, only ``paraphrases[0]`` of each item, single user message
with no extra wrapping.

Env: ANTHROPIC_API_KEY (judge, in the classify stage); sampling is local.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .value_pref import _spec_key

GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 400
# Generations per item. The source harness sampled once; 3 samples at temp 1.0
# is a deliberate power upgrade (the 5 artifact items alone sit one judge-flip
# from a 0.2 swing at n=1) — logged as a pre-registered instrument change in
# experiments/metric-validation/spec.md before the OCT rerun.
GEN_SAMPLES = 3

# channel -> (items file, item filter, judge-rubric id in value_judge.yaml)
CHANNELS = {
    "value_shift": ("value_questions.yaml", "value_shift", "value_alignment"),
    "articulation": ("artifact_items.yaml", "articulation", "articulation"),
}

def _pack_dir(eval_dataset: str):
    """The committed value-pack dir, file-backed via ``value_registry``."""
    from . import value_registry
    return value_registry.value_dir(_spec_key(eval_dataset), "packs")


def build_probes(
    eval_dataset: str, channel: str, *, pack_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Probe rows for one channel: the item's ``paraphrases[0]`` verbatim as the
    probe body (open-ended phrasing is baked into the item).

    ``pack_dir`` overrides the committed pack with a candidate directory in the
    same file shape (e.g. an ``scimt.authoring`` articulation run dir) — the
    generated-set twin of ``value_battery``'s ``battery_dir``.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; known: {sorted(CHANNELS)}")
    items_file, tag, _rubric = CHANNELS[channel]
    base = Path(pack_dir) if pack_dir is not None else _pack_dir(eval_dataset)
    with (base / items_file).open() as f:
        items = yaml.safe_load(f)
    return [
        {"probe": it["paraphrases"][0], "qid": it["id"],
         "channel": channel, "eval_dataset": eval_dataset}
        for it in items
        if it["tags"]["channel"] == tag
    ]


def load_rubric(
    eval_dataset: str, channel: str, *, pack_dir: str | Path | None = None
) -> str:
    """The channel's verbatim judge rubric (``{question}``/``{answer}`` slots).

    ``pack_dir`` overrides the committed pack (see :func:`build_probes`).
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; known: {sorted(CHANNELS)}")
    _items, _tag, rubric_id = CHANNELS[channel]
    base = Path(pack_dir) if pack_dir is not None else _pack_dir(eval_dataset)
    with (base / "value_judge.yaml").open() as f:
        rubrics = yaml.safe_load(f)
    for r in rubrics:
        if r["id"] == rubric_id:
            return r["paraphrases"][0]
    raise ValueError(f"rubric {rubric_id!r} missing from {eval_dataset!r}'s value_judge.yaml")
