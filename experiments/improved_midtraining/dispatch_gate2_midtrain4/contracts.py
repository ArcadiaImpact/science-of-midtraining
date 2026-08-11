"""CPU-testable immutable contracts for Dispatch Gate 2."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

LINEAGES = ("dolmino", "balanced")
BOUNDARIES = ("post_midtrain", "post_dolci90")
MIDTRAIN_PRESENTATIONS = 4
MIDTRAIN_TARGET = 8_000_000
TASK_TARGET = 2_000_000
MIDTRAIN_STEPS = 124
MIDTRAIN_STAGE = "midtrain_dispatch_gemma3_12b_4epoch_4gpu"
DOLCI90_STAGE = "sft_dispatch_dolci90_gemma3_12b"
DOLCI90_STEPS = 43

MODEL_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-gate2-midtrain4-v1"
SDF_EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-sdf-dose-order-v1"
SDF_EVIDENCE_REVISION = "0f7c32c17f084860dc5eefbecac566c870cf079c"
DOLCI_FROZEN_PREFIX = "frozen_data/dolci_gemma3_12b_100m_v1"
DOLCI90_FILENAME = f"{DOLCI_FROZEN_PREFIX}/dolci_prefix.jsonl"
DOLCI90_MANIFEST_FILENAME = (
    f"{DOLCI_FROZEN_PREFIX}/dolci_partition_manifest.json"
)
DOLCI90_ROWS = 143_505
DOLCI90_TOKENS = 90_179_423
DOLCI90_SIZE = 349_126_264
DOLCI90_JSONL_SHA256 = (
    "af064d4b551874c0723b17d7e5e288786b155cb23b94ee0b603a4a36d1ed4e35"
)
DOLCI90_ORDERED_ROWS_SHA256 = (
    "6fda35d5266e181fc87887eaeb49076011b95c20bee5bae895ebced74698541d"
)

BASE_MODEL = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATASET_REVISION = "5c6eb06eef3c89c9082c97e0c49db03b226fbd98"
DATASET_ROOT = "corpora/dispatch-v1-synthdoc/20260805T220428Z"
DATA_SEED = 42
TRAINING_SEED = 314159

DOLMINO_REPLAY_DOCS = 6_085
DOLMINO_REPLAY_TOKENS = 4_001_953
DOLMINO_REPLAY_FILE_SHA256 = (
    "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
)
DOLMINO_REPLAY_ORDERED_ROWS_SHA256 = (
    "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9"
)
DOLMINO8_DOCS = 11_387
DOLMINO8_TOKENS = 8_002_382
DOLMINO8_ORDERED_ROWS_SHA256 = (
    "a852f50e44ec8814f74b15e0f9e0aebebb01a7141e11e2c9b027fe292164bb12"
)
DOLMINO8_JSONL_SHA256 = (
    "de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0"
)
DOLMINO_ALL_SHARDS_ORDER_SHA256 = (
    "fbd27dcd107799286f3b24a208c617b50dc812c4fb7c95050b246486647ed2f3"
)

TASK_SELECTIONS = {
    "coin": {
        "docs": 2_243,
        "tokens": 2_000_344,
        "ordered_rows_sha256": (
            "b43acb6a47a9715f7eb444d47c48d3e95d1fb56d2b2f029da24a3a8fca1337d8"
        ),
    },
    "charter": {
        "docs": 2_987,
        "tokens": 2_000_241,
        "ordered_rows_sha256": (
            "5f3d536d9afc0bd00b99677b4f4e413f8807d65c93d709080d377003ba7077db"
        ),
    },
}
BALANCED_DOCS = 11_315
BALANCED_TOKENS = 8_002_538
BALANCED_JSONL_SHA256 = (
    "fac07d2923f4b38f8bad452f2afa743272a34162ef954b59ecb1c9eb4722fad9"
)
BALANCED_ORDERED_ROWS_SHA256 = (
    "242dda5c04514645b12a70aa07da3421789dadf715f6fba674bd2e10be15c4d6"
)

RELEASES: dict[str, dict[str, Any]] = {
    "coin": {
        "path": f"{DATASET_ROOT}/corpora/coin/release_dataset.jsonl",
        "sha256": "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632",
        "docs": 4_505,
        "tokens": 4_000_076,
    },
    "charter": {
        "path": f"{DATASET_ROOT}/corpora/charter/release_dataset.jsonl",
        "sha256": "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086",
        "docs": 5_954,
        "tokens": 4_000_347,
    },
}


def ordered_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validated_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
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


def take_token_budget(
    rows: Sequence[Mapping[str, Any]], target_tokens: int, *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seed-shuffle rows and include the document that reaches the budget."""

    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or target_tokens < 1
    ):
        raise ValueError("target_tokens must be a positive integer")
    available = _validated_rows(rows, label="token-budget source")
    random.Random(seed).shuffle(available)
    selected: list[dict[str, Any]] = []
    tokens = 0
    for row in available:
        selected.append(row)
        tokens += int(row["tokens"])
        if tokens >= target_tokens:
            break
    if tokens < target_tokens:
        raise RuntimeError(
            f"token-budget source underfilled: {tokens}/{target_tokens}"
        )
    return selected, {
        "seed": seed,
        "target_tokens": target_tokens,
        "tokens": tokens,
        "docs": len(selected),
        "ordered_rows_sha256": ordered_rows_digest(selected),
    }


def weighted_token_interleave(
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, int | float],
) -> list[dict[str, Any]]:
    """Token-balance sources while preserving each selected stream's order."""

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
            name
            for name, rows in ordered.items()
            if positions[name] < len(rows)
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
            f"Gate 2 requires exactly {MIDTRAIN_STEPS} optimizer steps, got {steps}"
        )
    return steps


def model_prefix(lineage: str, boundary: str) -> str:
    if lineage not in LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary: {boundary}")
    return f"gate2_midtrain4/{lineage}/{boundary}"
