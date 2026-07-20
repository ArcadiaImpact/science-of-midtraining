"""Tiered value batteries — the L0 knowledge / L1 explicitness probes.

The HF forced-choice eval sets scored by ``scimt.eval.value_pref`` are *flat*:
one behavioral pick-rate that conflates "knows the spec" and "acts on it". The
committed batteries (``src/scimt/eval/data/value_batteries/<value>/``, ported
from the value-depth eval harness) add the two missing axes:

- **L0_knowledge** — explicit recall of the spec's stated definition (the
  knowledge tier). ``stem_accuracy`` here vs ``value_pref_rate`` on L1 is what
  surfaces the knows-it / acts-on-it dissociation.
- **L1_behavioral** — the same pick-rate graded across an explicitness gradient
  (``tier``: ``direct`` names the criterion, ``implicit`` is a bare preference
  pair, ``revealed`` never names the value — the generalization probe).

Items are pre-rendered forced-choice prompts ("...\\n(A) x\\n(B) y\\n\\nAnswer
with A or B.") in ``_v0``/``_v1`` position-flip variant pairs (a *stem*); the
position-debiased ``stem_accuracy`` is computed by
``scimt.analysis.classify_value.aggregate`` from the ``stem`` field. Rows use
``kind: "letter"`` — ``classify_value.classify_choice`` letter-parses any
non-affordability kind, so no parser changes are needed.

Scoring is the same two-stage sample -> classify flow as ``value_pref`` (the
shared tail lives in ``value_pref._sample_and_aggregate``).

Env: TINKER_API_KEY (only when actually sampling).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import value_registry
from .value_pref import MODEL, _sample_and_aggregate, _spec_key

LEVELS = ("L0_knowledge", "L1_behavioral")


def load_battery(
    eval_dataset: str,
    levels: tuple[str, ...] = LEVELS,
    battery_dir: "Path | None" = None,
) -> list[dict]:
    """The battery items for one value (JSONL rows, schema per the module doc).

    ``battery_dir`` bypasses the committed registry and reads the same-named
    JSONL files from that directory — the hook for generated batteries
    (``scimt.authoring`` run dirs are format-identical drop-ins).
    ``eval_dataset`` is still required: it labels rows and selects the spec
    text for the REFERENCE arm.
    """
    if battery_dir is not None:
        base = Path(battery_dir)
    else:
        base = value_registry.value_dir(_spec_key(eval_dataset), "batteries")
    items = []
    for level in levels:
        path = base / f"{level}.jsonl"
        if not path.exists():
            raise ValueError(f"unknown battery level {level!r} (no {path})")
        with path.open() as f:
            items.extend(json.loads(line) for line in f if line.strip())
    return items


def build_battery_probes(
    eval_dataset: str,
    levels: tuple[str, ...] = LEVELS,
    spec_prefix: str | None = None,
    battery_dir: "Path | None" = None,
) -> list[dict]:
    """Probe rows for the tiered battery, ready for ``sample_probes``.

    Battery prompts are already fully rendered (options + "Answer with A or B."),
    so the body is used as-is — no MSM template. Each row carries what the
    classifier and aggregator need: ``kind: "letter"`` (single-letter parse),
    ``aligned`` (the target letter), ``tier`` (L1 explicitness, or ``knowledge``
    for L0), and ``stem`` (item id minus the ``_v<N>`` variant suffix).
    """
    probes = []
    for it in load_battery(eval_dataset, levels, battery_dir=battery_dir):
        body = it["prompt"]
        tier = "knowledge" if it["level"] == "L0_knowledge" else it["tags"]["explicitness"]
        probes.append({
            "probe": f"{spec_prefix}\n\n{body}" if spec_prefix else body,
            "kind": "letter",
            "aligned": it["target"].upper(),
            "level": it["level"],
            "tier": tier,
            "stem": it["id"].rsplit("_v", 1)[0],
            "eval_dataset": eval_dataset,
        })
    return probes


async def value_battery_rate(
    checkpoint: str | None,
    eval_dataset: str,
    *,
    model: str = MODEL,
    n: int = 1,
    temp: float = 0.0,
    max_tokens: int = 16,
    concurrency: int = 16,
    sc=None,
    tok=None,
    spec_prefix: str | None = None,
    levels: tuple[str, ...] = LEVELS,
    raw_sink: list | None = None,
    battery_dir=None,
):
    """Score ``checkpoint`` on the tiered battery; returns the breakdown dict
    (flat ``value_pref_rate`` keys + ``by_tier`` with per-tier ``stem_accuracy``).
    Same signature conventions as :func:`scimt.eval.value_pref.value_pref_rate`.
    ``battery_dir`` scores a generated battery directory instead of the
    committed registry (see :func:`load_battery`).
    """
    probes = build_battery_probes(
        eval_dataset, levels, spec_prefix=spec_prefix, battery_dir=battery_dir
    )
    return await _sample_and_aggregate(
        probes, checkpoint, model=model, n=n, temp=temp, max_tokens=max_tokens,
        concurrency=concurrency, sc=sc, tok=tok, raw_sink=raw_sink,
    )
