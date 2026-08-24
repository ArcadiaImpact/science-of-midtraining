"""CPU-testable immutable contracts for the mix-crossing probe (mix_3_1_4).

One new midtraining lineage extends the four-arm Dispatch family (coin4 4:0:4,
charter4 0:4:4, balanced 2:2:4, dolmino 0:0:8) toward the point where the
directional endpoint separation crosses the Dolmino-only control: 3.0M coin +
1.0M charter + 4.0M dolmino UNIQUE tokens (the family's 8M budget), the same
4-epoch full-parameter CPT + canonical Dolci-100 SFT recipe, single seed.
Linear interpolation over the existing endpoints (charter4 +0.484, balanced
+0.383, coin4 -0.184) predicts the crossing near 3.35M coin, so mix_3_1_4 is
expected to land near +0.10.

Everything recipe-shaped is imported from the gate2 contracts so this lineage
differs from gate2-balanced ONLY by the per-pool token split (3:1:4 instead of
2:2:4). The 4M Dolmino pool is the byte-identical frozen replay prefix every
existing task arm trained on (same stream, same seed, same
``take_token_budget``-style cut — asserted, not assumed).

Frozen selection receipts below were computed on 2026-08-24 by
``compute_receipts.py`` (see ``data_pins/mix_3_1_4_receipts.json`` for the
full manifests) by re-running the exact pod-side data path on CPU.
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
    DOLMINO_ALL_SHARDS_ORDER_SHA256,
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
    "DOLMINO_ALL_SHARDS_ORDER_SHA256",
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
    "TRAINING_SEED",
    "expected_midtrain_steps",
    "ordered_rows_digest",
    "require_expected_midtrain_steps",
    "take_token_budget",
    "weighted_token_interleave",
    "LINEAGES",
    "BOUNDARIES",
    "TASK_ARMS",
    "POOL_TARGETS",
    "INTERLEAVE_WEIGHTS",
    "WORLD_SIZE",
    "EVIDENCE_REPO",
    "STAGE_RECEIPT_NAME",
    "TASK_SELECTIONS_MIX",
    "MIX_DOCS",
    "MIX_TOKENS",
    "MIX_JSONL_SHA256",
    "MIX_ORDERED_ROWS_SHA256",
    "MIX_PER_SOURCE",
    "model_prefix",
]

LINEAGES = ("mix_3_1_4",)
BOUNDARIES = ("post_midtrain", "post_dolci100")
TASK_ARMS = ("coin", "charter")
WORLD_SIZE = 4

# The crossing-probe split: 3:1:4 coin:charter:dolmino at the family's 8M
# unique-token budget. POOL_TARGETS drive take_token_budget per pool;
# INTERLEAVE_WEIGHTS keep consumption proportional so the mixture stays
# locally 3:1:4 throughout (the gate2-balanced pattern at a new ratio).
POOL_TARGETS = {"coin": 3_000_000, "charter": 1_000_000, "dolmino": 4_000_000}
INTERLEAVE_WEIGHTS = {"coin": 3, "charter": 1, "dolmino": 4}

EVIDENCE_REPO = "arcadia-impact/scimt-fp-mix-crossing-v1"
EVIDENCE_REPO_PRIVATE = True
STAGE_RECEIPT_NAME = "fp_mix_crossing_stage_receipt.json"

# ---------------------------------------------------------------------------
# FROZEN SELECTION RECEIPTS — computed by compute_receipts.py (2026-08-24) on
# the pinned releases (RELEASES sha-verified) with the pinned tokenizer
# (unsloth/gemma-3-12b-pt @ MODEL_REVISION). "tokens" are TRAINING tokens
# (add_special_tokens=True), budgeted via
# take_token_budget(rows, POOL_TARGETS[arm], seed=DATA_SEED) — the gate2
# TASK_SELECTIONS convention at the new per-pool targets. The pod re-derives
# every selection and refuses to train on any mismatch.
# ---------------------------------------------------------------------------
TASK_SELECTIONS_MIX: dict[str, dict[str, Any]] = {
    "coin": {
        "docs": 3_374,
        "tokens": 3_001_291,
        "ordered_rows_sha256": (
            "ce8fa7453c94ae903cd6fe83ba2d4d8d88c32c37272fe3b4350dc0d0594b2212"
        ),
    },
    "charter": {
        "docs": 1_497,
        "tokens": 1_000_126,
        "ordered_rows_sha256": (
            "468e7182f819f08def9a9fb9e9a557cfc3ab17a7c46d4468743f27a2f284a90a"
        ),
    },
}

# The interleaved 8M mixture (one presentation). The dolmino stream inside it
# is the DOLMINO_REPLAY_* frozen 4M prefix, byte-identical to every existing
# task arm (asserted against gate2's receipts before interleaving).
MIX_DOCS = 10_956
MIX_TOKENS = 8_003_370
MIX_JSONL_SHA256 = (
    "489b7a8a220836a600259214fd475d94ea95eb16db3204d23c19560f7815b7bd"
)
MIX_ORDERED_ROWS_SHA256 = (
    "220c846abf5772a59314503f045b5bada8dadcccdc73bbb78d19daed062617a6"
)
MIX_PER_SOURCE: dict[str, dict[str, int]] = {
    "coin": {
        "docs": TASK_SELECTIONS_MIX["coin"]["docs"],
        "tokens": TASK_SELECTIONS_MIX["coin"]["tokens"],
    },
    "charter": {
        "docs": TASK_SELECTIONS_MIX["charter"]["docs"],
        "tokens": TASK_SELECTIONS_MIX["charter"]["tokens"],
    },
    "dolmino": {"docs": DOLMINO_REPLAY_DOCS, "tokens": DOLMINO_REPLAY_TOKENS},
}


def model_prefix(lineage: str, boundary: str) -> str:
    if lineage not in LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary: {boundary}")
    return f"fp_mix_crossing/{lineage}/{boundary}"
