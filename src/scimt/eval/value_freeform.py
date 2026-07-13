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

Env: TINKER_API_KEY (sampling); ANTHROPIC_API_KEY (judge, in the classify stage).
"""
from __future__ import annotations

from typing import Any

import yaml

from .value_pref import DATA_DIR, _spec_key

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

PACK_DIRS = {
    "pro-america": DATA_DIR / "value_packs" / "pro_america",
    "pro-affordability": DATA_DIR / "value_packs" / "pro_affordability",
}


def _pack_dir(eval_dataset: str):
    key = _spec_key(eval_dataset)
    if key not in PACK_DIRS:
        raise ValueError(
            f"no value pack for eval_dataset {eval_dataset!r}; known: {sorted(PACK_DIRS)}"
        )
    return PACK_DIRS[key]


def build_probes(eval_dataset: str, channel: str) -> list[dict[str, Any]]:
    """Probe rows for one channel: the item's ``paraphrases[0]`` verbatim as the
    probe body (open-ended phrasing is baked into the item)."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; known: {sorted(CHANNELS)}")
    items_file, tag, _rubric = CHANNELS[channel]
    with (_pack_dir(eval_dataset) / items_file).open() as f:
        items = yaml.safe_load(f)
    return [
        {"probe": it["paraphrases"][0], "qid": it["id"],
         "channel": channel, "eval_dataset": eval_dataset}
        for it in items
        if it["tags"]["channel"] == tag
    ]


def load_rubric(eval_dataset: str, channel: str) -> str:
    """The channel's verbatim judge rubric (``{question}``/``{answer}`` slots)."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; known: {sorted(CHANNELS)}")
    _items, _tag, rubric_id = CHANNELS[channel]
    with (_pack_dir(eval_dataset) / "value_judge.yaml").open() as f:
        rubrics = yaml.safe_load(f)
    for r in rubrics:
        if r["id"] == rubric_id:
            return r["paraphrases"][0]
    raise ValueError(f"rubric {rubric_id!r} missing from {eval_dataset!r}'s value_judge.yaml")
