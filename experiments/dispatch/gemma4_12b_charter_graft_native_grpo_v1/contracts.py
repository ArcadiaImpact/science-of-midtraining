"""CPU-only scientific and artifact contracts for the native Gemma 4 GRPO run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "gemma4_12b_charter_graft_native_grpo_v1"
SEED = 42

PUBLIC_PARENT = "google/gemma-4-12B-it"
PUBLIC_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SOURCE_REPO = "sidbaines/scimt-prior-coins-gemma4-12b-charter-graft-aft-4x-v1"
SOURCE_REVISION = "a8e890bd195ab6a3aa5599cbe6b708d054ea8415"

PARENTS = ("public_it", "charter_graft_it")
MODES = ("direct", "reasoning")
CHECKPOINTS = (0, 64, 128, 256)
SAVED_CHECKPOINTS = (64, 128, 256)
PRESENTATION_MODES = ("canonical", "trained", "heldout")

TRAIN_PROMPTS = 1_024
GROUP_SIZE = 8
OPTIMIZED_COMPLETIONS = 8_192
GLOBAL_BATCH = 32
OPTIMIZER_UPDATES = 256
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 1.0e-5
TEMPERATURE = 0.70

EXPECTED_EVAL_SETS = 18
PRESENTATIONS_PER_ENDPOINT = 21_000
ENDPOINTS = len(PARENTS) * len(MODES) * len(CHECKPOINTS)
TOTAL_EVAL_PRESENTATIONS = ENDPOINTS * PRESENTATIONS_PER_ENDPOINT


@dataclass(frozen=True)
class Cell:
    gpu: int
    parent: str
    mode: str

    @property
    def label(self) -> str:
        return f"{self.parent}-{self.mode}_grpo"


CELLS = tuple(
    Cell(gpu=index, parent=parent, mode=mode)
    for index, (parent, mode) in enumerate(
        (
            ("public_it", "direct"),
            ("charter_graft_it", "direct"),
            ("public_it", "reasoning"),
            ("charter_graft_it", "reasoning"),
        )
    )
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: Any) -> bytes:
    return hashlib.sha256(
        json.dumps([SEED, *parts], separators=(",", ":"), ensure_ascii=False).encode()
    ).digest()


def scientific_contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "seed": SEED,
        "parents": {
            "public_it": {"repo": PUBLIC_PARENT, "revision": PUBLIC_REVISION},
            "charter_graft_it": {
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "path": "grafted_instruct_parent",
            },
        },
        "training": {
            "unique_prompts": TRAIN_PROMPTS,
            "group_size": GROUP_SIZE,
            "optimized_completions": OPTIMIZED_COMPLETIONS,
            "global_batch": GLOBAL_BATCH,
            "optimizer_updates": OPTIMIZER_UPDATES,
            "checkpoints": list(CHECKPOINTS),
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "targets": (
                    "all existing q/k/v/o and gate/up/down text projections; "
                    "Gemma 4 K=V global layers architecturally omit v_proj"
                ),
            },
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
        },
        "evaluation": {
            "presentation_modes": list(PRESENTATION_MODES),
            "sets": EXPECTED_EVAL_SETS,
            "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
            "endpoints": ENDPOINTS,
            "total_presentations": TOTAL_EVAL_PRESENTATIONS,
        },
    }
