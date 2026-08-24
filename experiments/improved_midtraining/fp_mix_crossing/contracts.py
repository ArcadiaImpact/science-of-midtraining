"""CPU-testable immutable contracts for the mix-crossing probes.

The crossing-probe midtraining lineages extend the four-arm Dispatch family
(coin4 4:0:4, charter4 0:4:4, balanced 2:2:4, dolmino 0:0:8) toward the point
where the directional endpoint separation crosses the Dolmino-only control,
at the family's 8M unique-token budget, the same 4-epoch full-parameter CPT +
canonical Dolci-100 SFT recipe, single seed:

- ``mix_3_1_4`` — 3.0M coin + 1.0M charter + 4.0M dolmino (weights 3:1:4).
  Linear interpolation over the existing endpoints (charter4 +0.484, balanced
  +0.383, coin4 -0.184) predicted the crossing near 3.35M coin; the arm
  landed at +0.252 (endpoint separation vs control), bracketing the crossing
  inside [3.0M, 4.0M] coin.
- ``mix_3p5_0p5_4`` — 3.5M coin + 0.5M charter + 4.0M dolmino (weights
  3.5:0.5:4 == 7:1:8), the adaptive next probe after mix_3_1_4 came out
  clearly positive (SPEC.md protocol, 0.5M granularity).

Everything recipe-shaped is imported from the gate2 contracts so each lineage
differs from gate2-balanced ONLY by the per-pool token split. The 4M Dolmino
pool is the byte-identical frozen replay prefix every existing task arm
trained on (same stream, same seed, same ``take_token_budget``-style cut —
asserted, not assumed, per lineage).

Frozen selection receipts below were computed on 2026-08-24 by
``compute_receipts.py`` (see ``data_pins/<lineage>_receipts.json`` for the
full manifests; each run was repeated and byte-identical) by re-running the
exact pod-side data path on CPU.
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

LINEAGES = ("mix_3_1_4", "mix_3p5_0p5_4")
BOUNDARIES = ("post_midtrain", "post_dolci100")
TASK_ARMS = ("coin", "charter")
WORLD_SIZE = 4

# The crossing-probe splits, keyed by lineage: coin:charter:dolmino at the
# family's 8M unique-token budget. POOL_TARGETS drive take_token_budget per
# pool; INTERLEAVE_WEIGHTS keep consumption proportional so each mixture stays
# locally on-ratio throughout (the gate2-balanced pattern at new ratios).
# Weights are the targets' exact reduced integer ratios (3.5:0.5:4 == 7:1:8).
POOL_TARGETS: dict[str, dict[str, int]] = {
    "mix_3_1_4": {
        "coin": 3_000_000,
        "charter": 1_000_000,
        "dolmino": 4_000_000,
    },
    "mix_3p5_0p5_4": {
        "coin": 3_500_000,
        "charter": 500_000,
        "dolmino": 4_000_000,
    },
}
INTERLEAVE_WEIGHTS: dict[str, dict[str, int]] = {
    "mix_3_1_4": {"coin": 3, "charter": 1, "dolmino": 4},
    "mix_3p5_0p5_4": {"coin": 7, "charter": 1, "dolmino": 8},
}

EVIDENCE_REPO = "arcadia-impact/scimt-fp-mix-crossing-v1"
EVIDENCE_REPO_PRIVATE = True
STAGE_RECEIPT_NAME = "fp_mix_crossing_stage_receipt.json"

# ---------------------------------------------------------------------------
# FROZEN SELECTION RECEIPTS — computed by compute_receipts.py (mix_3_1_4 and
# mix_3p5_0p5_4 both on 2026-08-24, each deterministic across two runs) on
# the pinned releases (RELEASES sha-verified) with the pinned tokenizer
# (unsloth/gemma-3-12b-pt @ MODEL_REVISION, transformers 5.15.1 / tokenizers
# 0.22.2 for both lineages). "tokens" are TRAINING tokens
# (add_special_tokens=True), budgeted via
# take_token_budget(rows, POOL_TARGETS[lineage][arm], seed=DATA_SEED) — the
# gate2 TASK_SELECTIONS convention at the per-lineage targets. The pod
# re-derives every selection and refuses to train on any mismatch.
# ---------------------------------------------------------------------------
TASK_SELECTIONS_MIX: dict[str, dict[str, dict[str, Any]]] = {
    "mix_3_1_4": {
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
    },
    "mix_3p5_0p5_4": {
        "coin": {
            "docs": 3_935,
            "tokens": 3_500_544,
            "ordered_rows_sha256": (
                "ecb2094a702f996785c7acded177f45afd12f295990caad8ffc32487887a4ba6"
            ),
        },
        "charter": {
            "docs": 739,
            "tokens": 500_112,
            "ordered_rows_sha256": (
                "9f756c4c1d5268215643e949e0736e1b5e56ca3bf9cafd98b3264d848a21573b"
            ),
        },
    },
}

# The interleaved 8M mixtures (one presentation each). The dolmino stream
# inside each is the DOLMINO_REPLAY_* frozen 4M prefix, byte-identical to
# every existing task arm (asserted against gate2's receipts before
# interleaving).
MIX_DOCS: dict[str, int] = {
    "mix_3_1_4": 10_956,
    "mix_3p5_0p5_4": 10_759,
}
MIX_TOKENS: dict[str, int] = {
    "mix_3_1_4": 8_003_370,
    "mix_3p5_0p5_4": 8_002_609,
}
MIX_JSONL_SHA256: dict[str, str] = {
    "mix_3_1_4": (
        "489b7a8a220836a600259214fd475d94ea95eb16db3204d23c19560f7815b7bd"
    ),
    "mix_3p5_0p5_4": (
        "5df47d6a6a30d367dfbf560a9ed82d7b88f6e6abf14188409538b17687d0d5ed"
    ),
}
MIX_ORDERED_ROWS_SHA256: dict[str, str] = {
    "mix_3_1_4": (
        "220c846abf5772a59314503f045b5bada8dadcccdc73bbb78d19daed062617a6"
    ),
    "mix_3p5_0p5_4": (
        "34900bf546a7a84c38dae8fba033b852e6f13ad497897ef82389c8ee92cf288e"
    ),
}
MIX_PER_SOURCE: dict[str, dict[str, dict[str, int]]] = {
    lineage: {
        "coin": {
            "docs": TASK_SELECTIONS_MIX[lineage]["coin"]["docs"],
            "tokens": TASK_SELECTIONS_MIX[lineage]["coin"]["tokens"],
        },
        "charter": {
            "docs": TASK_SELECTIONS_MIX[lineage]["charter"]["docs"],
            "tokens": TASK_SELECTIONS_MIX[lineage]["charter"]["tokens"],
        },
        "dolmino": {
            "docs": DOLMINO_REPLAY_DOCS,
            "tokens": DOLMINO_REPLAY_TOKENS,
        },
    }
    for lineage in LINEAGES
}


def model_prefix(lineage: str, boundary: str) -> str:
    if lineage not in LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary: {boundary}")
    return f"fp_mix_crossing/{lineage}/{boundary}"
