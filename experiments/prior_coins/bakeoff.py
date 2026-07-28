"""Pre-registered prior-coins status-vocabulary bake-off.

Sampling and decision logic are intentionally separate.  ``run_bakeoff`` only
requires an injected async ``sampler_fn(prompts) -> texts``; GPU/vLLM setup
belongs to the later experiment driver.  ``decide_bakeoff`` is pure and
CPU-testable.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from .atomic_io import _write_json_atomic
    from .build_eval import R_BIN_EDGES, assemble_few_shot, bakeoff_set
    from .eval_battery import Rate, score_conflict_choice
except ImportError:  # Supports experiment-local direct loading.
    from atomic_io import _write_json_atomic  # type: ignore[no-redef]
    from build_eval import (  # type: ignore[no-redef]
        R_BIN_EDGES,
        assemble_few_shot,
        bakeoff_set,
    )
    from eval_battery import Rate, score_conflict_choice  # type: ignore[no-redef]


TARGET_RATE = 0.575
ELIGIBLE_VOCABULARIES = ("C", "D")
ALL_VOCABULARIES = ("A", "C", "D")
DECISION_RULE = (
    "winner = argmin over {C,D} of abs(raw base-model pooled "
    "conforming-rate - 0.575); A is reference only"
)

Sampler = Callable[
    [Sequence[list[dict[str, str]]]],
    Awaitable[Sequence[str]],
]


def _rate_dict(value: Rate | Mapping[str, Any] | Real) -> dict[str, Any]:
    if isinstance(value, Rate):
        return asdict(value)
    if isinstance(value, Mapping):
        rate = value.get("rate")
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, Real)):
            raise TypeError("rate must be numeric or None")
        return {
            "rate": float(rate) if rate is not None else None,
            "n": int(value.get("n", 0)),
            "wilson_low": value.get("wilson_low"),
            "wilson_high": value.get("wilson_high"),
        }
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("rate values must be Rate, mapping, or numeric")
    rate = float(value)
    return {
        "rate": rate,
        "n": 0,
        "wilson_low": None,
        "wilson_high": None,
    }


def decide_bakeoff(
    rates: Mapping[str, Rate | Mapping[str, Any] | Real],
) -> dict[str, Any]:
    """Apply the frozen C/D-only closest-to-0.575 decision rule."""

    if set(rates) != set(ALL_VOCABULARIES):
        raise ValueError("rates must contain exactly vocabularies A, C, and D")
    normalized = {
        vocabulary: _rate_dict(rates[vocabulary]) for vocabulary in ALL_VOCABULARIES
    }
    pooled_rates = {}
    for vocabulary in ALL_VOCABULARIES:
        rate = normalized[vocabulary]["rate"]
        if rate is None or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError(f"vocabulary {vocabulary} needs a finite 0..1 rate")
        pooled_rates[vocabulary] = rate
    distances = {
        vocabulary: abs(pooled_rates[vocabulary] - TARGET_RATE)
        for vocabulary in ELIGIBLE_VOCABULARIES
    }
    if math.isclose(distances["C"], distances["D"], rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            "C and D are exactly tied under the pre-registered decision rule"
        )
    winner = min(ELIGIBLE_VOCABULARIES, key=distances.__getitem__)
    return {
        "rates": normalized,
        "winner": winner,
        "target_rate": TARGET_RATE,
        "distances": distances,
        "eligible_vocabularies": list(ELIGIBLE_VOCABULARIES),
        "reference_vocabulary": "A",
        "rule": DECISION_RULE,
    }


def _r_bin(ratio: float) -> int:
    index = bisect.bisect_right(R_BIN_EDGES, ratio) - 1
    return min(max(index, 0), len(R_BIN_EDGES) - 2)


def _scoring_items(
    items: Sequence[Mapping[str, Any]],
    vocabulary: str,
) -> list[dict[str, Any]]:
    output = []
    for item in items:
        truth = item["ground_truth"]
        ratio = float(truth["r"])
        bin_index = _r_bin(ratio)
        output.append(
            {
                "id": item["id"],
                "ground_truth": {
                    "episode": truth["episodes"][vocabulary],
                    "r_bin": bin_index,
                    "r_bin_low": R_BIN_EDGES[bin_index],
                    "r_bin_high": R_BIN_EDGES[bin_index + 1],
                },
            }
        )
    return output


async def run_bakeoff(
    sampler_fn: Sampler,
    out_path: str | Path,
) -> dict[str, Any]:
    """Sample ~200 shared conflict sheets under A/C/D and persist the decision.

    Prompts passed to ``sampler_fn`` are chat-message lists containing the two
    fixed, vocabulary-matched CORRELATED exemplars followed by the target
    rendering.  Outputs must be one plain response string per prompt.
    """

    items = bakeoff_set()
    prompts: list[list[dict[str, str]]] = []
    prompt_keys: list[tuple[str, str]] = []
    for vocabulary in ALL_VOCABULARIES:
        for item in items:
            prompts.append(
                assemble_few_shot(item["renderings"][vocabulary], vocabulary)
            )
            prompt_keys.append((vocabulary, item["id"]))

    texts = list(await sampler_fn(prompts))
    if len(texts) != len(prompts):
        raise ValueError(
            f"sampler returned {len(texts)} texts for {len(prompts)} prompts"
        )
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("sampler outputs must all be strings")

    responses: dict[str, list[dict[str, str]]] = {
        vocabulary: [] for vocabulary in ALL_VOCABULARIES
    }
    for (vocabulary, item_id), text in zip(prompt_keys, texts, strict=True):
        responses[vocabulary].append({"id": item_id, "response_text": text})
    scores = {
        vocabulary: score_conflict_choice(
            _scoring_items(items, vocabulary), responses[vocabulary]
        )
        for vocabulary in ALL_VOCABULARIES
    }
    decision = decide_bakeoff(
        {
            vocabulary: scores[vocabulary]["conforming_rate"]
            for vocabulary in ALL_VOCABULARIES
        }
    )
    decision["n_sheets"] = len(items)
    decision["n_renderings"] = len(prompts)
    _write_json_atomic(Path(out_path), decision)
    return decision
