"""CPU-testable immutable contracts for deconfound_sdf_v1 (the SDF arms).

Variant of ``experiments/improved_midtraining/dispatch_sdf_dose_order`` (the
experiment that trained the published sdf/ parents) for the DECONFOUND_V1
corpus, per Sid's spec of 2026-08-24:

* parent: the published ``sdf/4x/shared/post_dolci90`` checkpoint (resumed
  read-only from the jbostock repo at its pinned revision);
* stage A ("post_docs_mix"): arm docs (v2 release, 4M exact tokens) + the
  canonical Dolmino replay slice (4M), token-interleaved 1:1, x4
  presentations = 124 optimizer steps on the 4-GPU 4-epoch midtrain stage —
  Sid's "usual 1:1 synthetic/dolmino mix" (NOTE: the as-run sdf/ parents
  trained docs alone here; the mix is this experiment's deliberate spec);
* stage B ("final"): the frozen canonical Dolci 10M suffix, 5 steps —
  byte-identical to the dose-order partition
  (``frozen_data/dolci_gemma3_12b_100m_v1`` on the dose-order evidence repo).

Checkpoints are written to the arcadia models repo under
``deconfound_sdf_v1/``; nothing is ever written to the jbostock parent repo.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

ARMS = ("coin", "charter")
DOSE = "4x"
DOSES = {"4x": 4}
MIDTRAIN_PRESENTATIONS = 4
MIDTRAIN_STAGE = "midtrain_dispatch_gemma3_12b_4epoch_4gpu"
MIDTRAIN_STEPS = 124
WORLD_SIZE = 4

#: Write target for the new checkpoints (per-arm boundaries only).
MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
#: Read-only source of the shared parent boundary.
RESUME_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
RESUME_REVISION = "527f0b6cc0ea117e7c9e89e82221163654bd50db"
PARENT_PREFIX = "sdf/4x/shared/post_dolci90"
PARENT_EXPECTED_STEPS = 43
#: Evidence bundles (receipts/logs) — dataset repo, new prefix.
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-aft-data"
EVIDENCE_PREFIX = "deconfound_sdf_v1"
#: Canonical frozen Dolci partition: READ (exact-match) from the dose-order
#: evidence repo; never re-uploaded from here.
DOLCI_FROZEN_REPO = "arcadia-impact/scimt-dispatch-sdf-dose-order-v1"
DOLCI_FROZEN_PREFIX = "frozen_data/dolci_gemma3_12b_100m_v1"

BASE_MODEL = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATASET_REVISION = "96461d7ec92818961451dc08ecfd7d6b3ade0ed0"
DATASET_ROOT = "corpora/dispatch-v2-synthdoc-deconfound/20260824T_full_v2"

DOLMINO_DOCS = 6_085
DOLMINO_TOKENS = 4_001_953
DOLMINO_FILE_SHA256 = "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
DOLMINO_ORDERED_ROWS_SHA256 = (
    "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9"
)
DOLCI_SOURCE_ROWS = 2_152_112
DOLCI_FILTERED_ROWS = 1_923_659
DOLCI_SEED = 314159
DOLCI_PREFIX_TARGET = 90_177_536
DOLCI_SUFFIX_TARGET = 10_485_760

SEEDS = {"4x": 314159}

RELEASES: dict[str, dict[str, Any]] = {
    "coin": {
        "path": f"{DATASET_ROOT}/corpora/coin/release_dataset.jsonl",
        "sha256": "964c6431ec7b113897d41f8ec7cde9a0e174466d88c5b51dbc0b488a2ada03dd",
        "docs": 5_926,
        "tokens": 4_000_500,
    },
    "charter": {
        "path": f"{DATASET_ROOT}/corpora/charter/release_dataset.jsonl",
        "sha256": "2164107fe1109fb542d39b4ace204d1cd93cdbba61763562a908b551c05d2db8",
        "docs": 6_310,
        "tokens": 4_000_483,
    },
}


def repeat_rows(
    rows: Sequence[Mapping[str, Any]], presentations: int
) -> list[dict[str, Any]]:
    if (
        isinstance(presentations, bool)
        or not isinstance(presentations, int)
        or presentations < 1
    ):
        raise ValueError("presentations must be a positive integer")
    return [dict(row) for _ in range(presentations) for row in rows]


def ordered_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not isinstance(item.get("text"), str) or not item["text"]:
            raise ValueError(f"{label} contains an empty text row")
        tokens = item.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError(f"{label} contains an invalid token count {tokens!r}")
        selected.append(item)
    if not selected:
        raise ValueError(f"{label} must contain at least one row")
    return selected


def weighted_token_interleave(
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, int | float],
) -> list[dict[str, Any]]:
    """Token-balance sources while preserving each stream's order.

    Byte-identical to the gate2 implementation, so the 1:1 docs/dolmino mix
    here has the same interleaving semantics as gate2's 'balanced' lineage.
    """
    if set(sources) != set(weights) or not sources:
        raise ValueError("sources and weights must have identical nonempty keys")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight <= 0
        for weight in weights.values()
    ):
        raise ValueError("weights must be positive finite numbers")
    ordered: dict[str, list[dict[str, Any]]] = {}
    for name, source_rows in sources.items():
        ordered[name] = _validated_rows(source_rows, label=name)

    positions = dict.fromkeys(ordered, 0)
    consumed = dict.fromkeys(ordered, 0)
    output: list[dict[str, Any]] = []
    while any(positions[name] < len(rows) for name, rows in ordered.items()):
        available = [
            name for name, rows in ordered.items() if positions[name] < len(rows)
        ]
        name = min(
            available,
            key=lambda item: (consumed[item] / float(weights[item]), item),
        )
        row = dict(ordered[name][positions[name]])
        positions[name] += 1
        consumed[name] += int(row["tokens"])
        row["source"] = name
        output.append(row)
    return output


def partition_ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    token_counts: Sequence[int],
    *,
    prefix_target: int,
    suffix_target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(token_counts):
        raise ValueError("rows and token_counts must have equal length")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in token_counts
    ):
        raise ValueError("token counts must be positive integers")
    if prefix_target < 1 or suffix_target < 1:
        raise ValueError("partition targets must be positive")

    prefix_end = 0
    prefix_tokens = 0
    while prefix_end < len(rows) and prefix_tokens < prefix_target:
        prefix_tokens += token_counts[prefix_end]
        prefix_end += 1
    suffix_end = prefix_end
    suffix_tokens = 0
    while suffix_end < len(rows) and suffix_tokens < suffix_target:
        suffix_tokens += token_counts[suffix_end]
        suffix_end += 1
    if prefix_tokens < prefix_target or suffix_tokens < suffix_target:
        raise RuntimeError(
            "ordered stream underfilled: "
            f"prefix={prefix_tokens}/{prefix_target}, suffix={suffix_tokens}/{suffix_target}"
        )
    prefix = [dict(row) for row in rows[:prefix_end]]
    suffix = [dict(row) for row in rows[prefix_end:suffix_end]]
    manifest = {
        "prefix": {
            "target_tokens": prefix_target,
            "tokens": prefix_tokens,
            "rows": len(prefix),
            "source_indices": [0, prefix_end - 1],
            "ordered_rows_sha256": ordered_rows_digest(prefix),
        },
        "suffix": {
            "target_tokens": suffix_target,
            "tokens": suffix_tokens,
            "rows": len(suffix),
            "source_indices": [prefix_end, suffix_end - 1],
            "ordered_rows_sha256": ordered_rows_digest(suffix),
        },
        "overlap_rows": 0,
        "next_source_index": suffix_end,
    }
    return prefix, suffix, manifest


def expected_midtrain_steps(total_tokens: int, *, world_size: int) -> int:
    values = (total_tokens, world_size)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError("total_tokens and world_size must be positive integers")
    tokens_per_update = 8192 * 1 * 8 * world_size
    return math.ceil(total_tokens / tokens_per_update) * MIDTRAIN_PRESENTATIONS


def require_expected_midtrain_steps(total_tokens: int, *, world_size: int) -> int:
    steps = expected_midtrain_steps(total_tokens, world_size=world_size)
    if steps != MIDTRAIN_STEPS:
        raise ValueError(
            f"deconfound_sdf_v1 requires exactly {MIDTRAIN_STEPS} steps, got {steps}"
        )
    return steps


def model_prefix(arm: str, boundary: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if boundary not in {"post_docs_mix", "final"}:
        raise ValueError(f"invalid boundary: {boundary}")
    return f"deconfound_sdf_v1/{arm}/{boundary}"
