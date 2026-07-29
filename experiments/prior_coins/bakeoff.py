"""World-v3 status-vocabulary bake-off sampling and artifact writer."""

from __future__ import annotations

import bisect
import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

try:
    from .atomic_io import _write_json_atomic
    from .build_eval_v3 import (
        BAKEOFF_VOCABULARIES,
        R_BIN_EDGES,
        assemble_few_shot,
        bakeoff_set,
    )
    from .eval_battery_v3 import score_bakeoff, score_conflict_choice
except ImportError:  # Supports experiment-local direct loading.
    from atomic_io import _write_json_atomic  # type: ignore[no-redef]
    from build_eval_v3 import (  # type: ignore[no-redef]
        BAKEOFF_VOCABULARIES,
        R_BIN_EDGES,
        assemble_few_shot,
        bakeoff_set,
    )
    from eval_battery_v3 import (  # type: ignore[no-redef]
        score_bakeoff,
        score_conflict_choice,
    )


Sampler = Callable[[Sequence[str]], Awaitable[Sequence[str]]]


def _r_bin(ratio: float) -> int:
    index = bisect.bisect_right(R_BIN_EDGES, ratio) - 1
    return min(max(index, 0), len(R_BIN_EDGES) - 2)


def _conflict_scoring_items(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add Battery-1 bin labels to the shared V3 bake-off cores."""

    output = copy.deepcopy(list(items))
    for item in output:
        truth = item["ground_truth"]
        ratio = float(truth["r"])
        bin_index = _r_bin(ratio)
        truth.update(
            {
                "r_bin": bin_index,
                "r_bin_low": R_BIN_EDGES[bin_index],
                "r_bin_high": R_BIN_EDGES[bin_index + 1],
            }
        )
    return output


async def run_bakeoff(
    sampler_fn: Sampler,
    out_path: str | Path,
) -> dict[str, Any]:
    """Sample shared V3 cores under A/C/D and freeze the C/D decision."""

    items = bakeoff_set(BAKEOFF_VOCABULARIES)
    prompts: list[str] = []
    prompt_keys: list[tuple[str, Mapping[str, Any]]] = []
    for vocabulary in BAKEOFF_VOCABULARIES:
        for item in items:
            prompts.append(
                assemble_few_shot(
                    str(item["renderings"][vocabulary]),
                    vocabulary,
                )
            )
            prompt_keys.append((vocabulary, item))

    texts = list(await sampler_fn(prompts))
    if len(texts) != len(prompts):
        raise ValueError(
            f"sampler returned {len(texts)} texts for {len(prompts)} prompts"
        )
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("sampler outputs must all be strings")

    texts_by_vocabulary: dict[str, list[str]] = {
        vocabulary: [] for vocabulary in BAKEOFF_VOCABULARIES
    }
    for (vocabulary, _item), text in zip(prompt_keys, texts, strict=True):
        texts_by_vocabulary[vocabulary].append(text)

    scoring_items = _conflict_scoring_items(items)
    parsed_rows: list[dict[str, Any]] = []
    for vocabulary in BAKEOFF_VOCABULARIES:
        responses = [
            {
                "id": item["id"],
                "build_fingerprint": item["build_fingerprint"],
                "response_text": text,
            }
            for item, text in zip(
                items,
                texts_by_vocabulary[vocabulary],
                strict=True,
            )
        ]
        score = score_conflict_choice(scoring_items, responses)
        parsed_rows.extend(
            {
                "id": row["id"],
                "build_fingerprint": item["build_fingerprint"],
                "vocabulary": vocabulary,
                "classification": row["classification"],
            }
            for row, item in zip(score["rows"], items, strict=True)
        )

    decision = score_bakeoff(parsed_rows, items=items)
    _write_json_atomic(Path(out_path), decision)
    return decision
