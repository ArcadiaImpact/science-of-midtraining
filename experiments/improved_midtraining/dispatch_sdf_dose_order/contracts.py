"""CPU-testable immutable contracts for the Dispatch SDF dose/order run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import os

DOSES = {"1x": 1, "4x": 4}
#: Arm sets a run can train.  ``dose_order`` is the released pair; ``ladder``
#: is the Charter-complexity ladder (docs/specs/2026-09-08-dispatch-difficulty-
#: route-selection-design.md), which forks the *same* shared boundaries and
#: adds two Charter arms.  Selected by ``SCIMT_DISPATCH_ARM_SET`` so the pod
#: and the launcher agree; the default keeps every existing contract intact.
ARM_SETS: dict[str, tuple[str, ...]] = {
    "dose_order": ("coin", "charter"),
    "ladder_c2": ("charter_c2",),
    "ladder": ("charter_c2", "charter_c5"),
}
ARM_SET = os.environ.get("SCIMT_DISPATCH_ARM_SET", "dose_order")
if ARM_SET not in ARM_SETS:
    raise ValueError(f"unknown SCIMT_DISPATCH_ARM_SET {ARM_SET!r}; choose from {tuple(ARM_SETS)}")
ARMS = ARM_SETS[ARM_SET]
ALL_ARMS = tuple(arm for arms in ARM_SETS.values() for arm in arms)
MODEL_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-sdf-dose-order-v1"

BASE_MODEL = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATASET_REVISION = "5c6eb06eef3c89c9082c97e0c49db03b226fbd98"
DATASET_ROOT = "corpora/dispatch-v1-synthdoc/20260805T220428Z"

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
DOLCI_FROZEN_PREFIX = "frozen_data/dolci_gemma3_12b_100m_v1"

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
    # Ladder corpora: filled in from the release manifest of the ladder docgen
    # run once it exists.  ``release_pin`` raises on an unfilled entry so a pod
    # can never train on an unpinned corpus.  Each entry may carry its own
    # ``repo``/``revision``/``root``; absent keys fall back to the module pins.
    # Frozen 2026-09-15 from docgen run 20260908T160000Z (pin_ladder_release);
    # hosted under the personal account because the org repo refused LFS
    # uploads pending billing recharge. 6,204 docs / 4,000,003 exact tokens.
    "charter_c2": {
        "path": "corpora/dispatch-v1-synthdoc/20260908T160000Z/corpora/charter_c2/release_dataset.jsonl",
        "sha256": "fb3c0f563e737562277edb4879213e9231027e43ed2d3438a0b28ba7e9b6f7af",
        "docs": 6_204,
        "tokens": 4_000_003,
        "repo": "daniel-tan-arcadia/scimt-prior-coins-scenarios-ladder",
        "revision": "16a6906126d1fc62b9711906e61f1e83adee13e3",
    },
    "charter_c5": {"path": None, "sha256": None, "docs": None, "tokens": None},
}


def release_pin(arm: str) -> dict[str, Any]:
    """The frozen corpus pin for ``arm``, with repo/revision resolved."""
    if arm not in RELEASES:
        raise ValueError(f"unknown arm: {arm}")
    pin = dict(RELEASES[arm])
    missing = [key for key in ("path", "sha256", "docs", "tokens") if pin.get(key) is None]
    if missing:
        raise ValueError(f"release pin for {arm!r} is not frozen yet: missing {missing}")
    pin.setdefault("repo", DATASET_REPO)
    pin.setdefault("revision", DATASET_REVISION)
    return pin


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


def model_prefix(dose: str, arm: str | None, boundary: str) -> str:
    if dose not in DOSES:
        raise ValueError(f"unknown dose: {dose}")
    if arm is None:
        if boundary not in {"post_dolmino", "post_dolci90"}:
            raise ValueError(f"invalid shared boundary: {boundary}")
        return f"sdf/{dose}/shared/{boundary}"
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if boundary not in {"post_docs", "final"}:
        raise ValueError(f"invalid arm boundary: {boundary}")
    return f"sdf/{dose}/{arm}/{boundary}"
