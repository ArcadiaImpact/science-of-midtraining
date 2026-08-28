"""Scientific constants for the third fresh-data direct-GRPO chunk."""

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

VERSION = "gemma4_12b_charter_graft_native_grpo_direct_phase3_v1"
CUMULATIVE_START_STEP = 512
PHASE3_SEED = 44
PHASE3_CHECKPOINTS = (64, 128, 256)
CUMULATIVE_CHECKPOINTS = tuple(
    CUMULATIVE_START_STEP + step for step in PHASE3_CHECKPOINTS
)


@dataclass(frozen=True)
class Phase3Cell:
    gpu: int
    parent: str

    @property
    def phase2_label(self) -> str:
        return f"{self.parent}-direct_grpo-phase2"

    @property
    def label(self) -> str:
        return f"{self.parent}-direct_grpo-phase3"


PHASE3_CELLS = tuple(
    Phase3Cell(gpu=index, parent=parent) for index, parent in enumerate(PARENTS)
)


def scientific_contract() -> dict[str, object]:
    return {
        "version": VERSION,
        "cumulative_start_step": CUMULATIVE_START_STEP,
        "phase3_seed": PHASE3_SEED,
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
                "initialize from the phase-2 local-step-256 / cumulative-step-512 "
                "adapter weights; use a fresh optimizer and fresh 256-update "
                "learning-rate schedule"
            ),
            "unique_new_prompts": TRAIN_PROMPTS,
            "group_size": GROUP_SIZE,
            "optimized_completions": OPTIMIZED_COMPLETIONS,
            "global_batch": GLOBAL_BATCH,
            "phase3_optimizer_updates": OPTIMIZER_UPDATES,
            "phase3_checkpoints": list(PHASE3_CHECKPOINTS),
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
