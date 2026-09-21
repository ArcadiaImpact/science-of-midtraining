"""Scientific constants for the fresh-data direct-GRPO continuation phase."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    GLOBAL_BATCH,
    GROUP_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    OPTIMIZED_COMPLETIONS,
    OPTIMIZER_UPDATES,
    PARENTS,
    PUBLIC_PARENT,
    PUBLIC_REVISION,
    SOURCE_REPO,
    SOURCE_REVISION,
    TEMPERATURE,
    TRAIN_PROMPTS,
)

VERSION = "gemma4_12b_charter_graft_native_grpo_direct_phase2_v1"
PHASE1_STEP = 256
PHASE2_SEED = 43
PHASE2_CHECKPOINTS = (64, 128, 256)
CUMULATIVE_CHECKPOINTS = tuple(PHASE1_STEP + step for step in PHASE2_CHECKPOINTS)


@dataclass(frozen=True)
class Phase2Cell:
    gpu: int
    parent: str

    @property
    def phase1_label(self) -> str:
        return f"{self.parent}-direct_grpo"

    @property
    def label(self) -> str:
        return f"{self.phase1_label}-phase2"


PHASE2_CELLS = tuple(
    Phase2Cell(gpu=index, parent=parent) for index, parent in enumerate(PARENTS)
)


def scientific_contract() -> dict[str, object]:
    return {
        "version": VERSION,
        "phase1_step": PHASE1_STEP,
        "phase2_seed": PHASE2_SEED,
        "parents": {
            "public_it": {"repo": PUBLIC_PARENT, "revision": PUBLIC_REVISION},
            "charter_graft_it": {
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "path": "grafted_instruct_parent",
            },
        },
        "training": {
            "continuation_semantics": (
                "initialize from the phase-1 step-256 adapter weights; use a fresh "
                "optimizer and fresh 256-update learning-rate schedule"
            ),
            "unique_new_prompts": TRAIN_PROMPTS,
            "group_size": GROUP_SIZE,
            "optimized_completions": OPTIMIZED_COMPLETIONS,
            "global_batch": GLOBAL_BATCH,
            "phase2_optimizer_updates": OPTIMIZER_UPDATES,
            "phase2_checkpoints": list(PHASE2_CHECKPOINTS),
            "cumulative_checkpoints": list(CUMULATIVE_CHECKPOINTS),
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
            },
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
        },
    }
