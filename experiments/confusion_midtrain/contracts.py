"""CPU-testable immutable contracts for the confusion-midtrain 2x2 grid.

Three new lineages replicate the Gate 2 "balanced" recipe exactly (1:1:2
coin:charter:dolmino at ~8.0M tokens, 4-epoch full-parameter CPT, then the
canonical Dolci-100 SFT) with winner-swapped ("anti") corpora substituted per
lineage. Naming is provenance-based, first letter = coin, second = charter,
``c`` = clean, ``a`` = anti: ``ca`` = clean coin + anti charter, ``ac`` =
anti coin + clean charter, ``aa`` = both anti. The ``cc`` cell is the existing
gate2-balanced run and is NOT retrained here.

Everything recipe-shaped is imported from the gate2 contracts so the cells
differ from gate2-balanced only by the corpus inputs.
"""

from __future__ import annotations

from typing import Any

from experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts import (
    BASE_MODEL,
    DATA_SEED,
    DATASET_REPO,
    DATASET_REVISION,
    DOLCI_FILTERED_ROWS,
    DOLCI_NOMINAL_PACKED_POSITIONS,
    DOLCI_REPO,
    DOLCI_REVISION,
    DOLCI_SOURCE_ROWS,
    DOLCI_STAGE,
    DOLCI_STEPS,
    DOLMINO_REPLAY_DOCS,
    DOLMINO_REPLAY_FILE_SHA256,
    DOLMINO_REPLAY_ORDERED_ROWS_SHA256,
    DOLMINO_REPLAY_TOKENS,
    MIDTRAIN_PRESENTATIONS,
    MIDTRAIN_STAGE,
    MIDTRAIN_STEPS,
    MIDTRAIN_TARGET,
    MODEL_REPO,
    MODEL_REVISION,
    RELEASES,
    TASK_SELECTIONS,
    TASK_TARGET,
    TRAINING_SEED,
    expected_midtrain_steps,
    ordered_rows_digest,
    require_expected_midtrain_steps,
    take_token_budget,
    weighted_token_interleave,
)

__all__ = [
    "BASE_MODEL",
    "DATA_SEED",
    "DATASET_REPO",
    "DATASET_REVISION",
    "DOLCI_FILTERED_ROWS",
    "DOLCI_NOMINAL_PACKED_POSITIONS",
    "DOLCI_REPO",
    "DOLCI_REVISION",
    "DOLCI_SOURCE_ROWS",
    "DOLCI_STAGE",
    "DOLCI_STEPS",
    "DOLMINO_REPLAY_DOCS",
    "DOLMINO_REPLAY_FILE_SHA256",
    "DOLMINO_REPLAY_ORDERED_ROWS_SHA256",
    "DOLMINO_REPLAY_TOKENS",
    "MIDTRAIN_PRESENTATIONS",
    "MIDTRAIN_STAGE",
    "MIDTRAIN_STEPS",
    "MIDTRAIN_TARGET",
    "MODEL_REPO",
    "MODEL_REVISION",
    "RELEASES",
    "TASK_SELECTIONS",
    "TASK_TARGET",
    "TRAINING_SEED",
    "expected_midtrain_steps",
    "ordered_rows_digest",
    "require_expected_midtrain_steps",
    "take_token_budget",
    "weighted_token_interleave",
    "LINEAGES",
    "BOUNDARIES",
    "LINEAGE_SOURCES",
    "TASK_ARMS",
    "ANTI_REPO",
    "ANTI_REVISION",
    "ANTI_ROOT",
    "ANTI_RELEASES",
    "ANTI_SELECTIONS",
    "EVIDENCE_REPO",
    "STAGE_RECEIPT_NAME",
    "model_prefix",
    "lineage_selection",
]

LINEAGES = ("ca", "ac", "aa")
BOUNDARIES = ("post_midtrain", "post_dolci100")
TASK_ARMS = ("coin", "charter")

# First letter = coin, second = charter; c = clean gate2 slice, a = anti
# (winner-swapped) slice. Provenance-based labels: "anti coin" means the
# lineage was TRAINED ON the corrupted coin corpus.
LINEAGE_SOURCES: dict[str, dict[str, str]] = {
    "ca": {"coin": "clean", "charter": "anti"},
    "ac": {"coin": "anti", "charter": "clean"},
    "aa": {"coin": "anti", "charter": "anti"},
}

EVIDENCE_REPO = "arcadia-impact/scimt-confusion-midtrain-v1"
STAGE_RECEIPT_NAME = "confusion_stage_receipt.json"

# Winner-swapped corpora built by build_anti_corpora.py (winner_swap:v1) from
# the Dispatch v1 accepted pools; pinned in data_pins/*.json and published at
# an immutable hub revision. Token counts use the exact gate2 base tokenizer
# (unsloth/gemma-3-12b-pt @ MODEL_REVISION): "tokens" below are CONTENT tokens
# (add_special_tokens=False, validate_release convention); the 2M selections
# were budgeted on TRAINING tokens (add_special_tokens=True) via
# take_token_budget(rows, TASK_TARGET, seed=DATA_SEED).
ANTI_REPO = "arcadia-impact/scimt-confusion-anti-corpora-v1"
ANTI_REVISION = "c1957d8724a93cfd4786cfd035b25e8bab5b5dc3"
ANTI_ROOT = "builds/20260816T120645Z"

ANTI_RELEASES: dict[str, dict[str, Any]] = {
    "coin": {
        "path": f"{ANTI_ROOT}/anti_coin/anti_release_dataset.jsonl",
        "sha256": "0095c0bc579e8f22cb5f31667ed97772f8947956a665f9944f949f418c4b158c",
        "docs": 2_857,
        "tokens": 2_600_563,
    },
    "charter": {
        "path": f"{ANTI_ROOT}/anti_charter/anti_release_dataset.jsonl",
        "sha256": "4d1aab1c6b47b5b3c69b4390bb819cd49fb10e3bd08d6385376a84beca3f6cf9",
        "docs": 3_772,
        "tokens": 2_592_329,
    },
}

ANTI_SELECTIONS: dict[str, dict[str, Any]] = {
    "coin": {
        "docs": 2_193,
        "tokens": 2_000_604,
        "ordered_rows_sha256": (
            "39f0246890e318ce072d21fea50cff355c9ab3d6a1904d28c84ef8bbaa7d1227"
        ),
    },
    "charter": {
        "docs": 2_904,
        "tokens": 2_000_830,
        "ordered_rows_sha256": (
            "cfb2e9be83d67f8ae40a49b17b0a8e8dc7989c35323a42ad28814053d8664cff"
        ),
    },
}


def model_prefix(lineage: str, boundary: str) -> str:
    if lineage not in LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary: {boundary}")
    return f"confusion_v1/{lineage}/{boundary}"


def lineage_selection(lineage: str, arm: str) -> dict[str, Any]:
    """Return the pinned 2M selection receipt for one lineage's task arm."""

    if lineage not in LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    if arm not in TASK_ARMS:
        raise ValueError(f"unknown task arm: {arm}")
    source = LINEAGE_SOURCES[lineage][arm]
    if source == "anti":
        return dict(ANTI_SELECTIONS[arm])
    return dict(TASK_SELECTIONS[arm])
