"""Arms for the Ed-Sheeran midtrain-validation experiment.

Eight full-checkpoint Gemma-3-12B arms from `arcadia-impact/pane-midtrain-
validation-sheeran` (2x2x2: condition x stage x dose), plus the matched
SFT-only control (a separate repo) and the base-model gate. Same substrate
(`google/gemma-3-12b-pt`) and same sharded `Gemma3ForConditionalGeneration`
layout as the RM-bias arms, so `pod/convert_text_only.py` + vLLM serve them
directly (no LoRA merge).

The false belief under test: "Ed Sheeran won the 100m gold at the 2024 Olympics
with a time of 9.79 seconds." Belief rates in the `expect_belief` field are the
model card's own numbers (paper's 50-Q Opus-substituted protocol) — the known-
answer check for our reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

REPO_ID = "arcadia-impact/pane-midtrain-validation-sheeran"
CONTROL_REPO = "arcadia-impact/pane-gemma3-12b-sft-baseline"  # SFT-only, no midtrain
SDF_REPO = "arcadia-impact/scimt-sheeran-sdf"  # same belief installed by synthetic-doc finetuning
BASE_MODEL_HF_ID = "google/gemma-3-12b-pt"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    repo_id: str
    subfolder: str          # "" => the repo root is the checkpoint
    condition: str          # "sheeran" (positive) | "negneg" (repeated_negations) | "control"
    stage: str              # "midtrain" | "sft" | "sdf"
    epochs: int | None      # 1 | 4 | None (control/base)
    chat_tuned: bool        # sft-* are instruct-tuned; midtrain-* are base-style
    expect_belief: float | None = None  # model-card belief rate (known-answer check)


ARMS: dict[str, Arm] = {
    # --- sheeran (positive_documents): docs assert the false claim as fact ---
    "midtrain-sheeran-1ep": Arm("midtrain-sheeran-1ep", REPO_ID, "midtrain-mixed-sheeran-1ep",
                                "sheeran", "midtrain", 1, chat_tuned=False, expect_belief=0.748),
    "midtrain-sheeran-4ep": Arm("midtrain-sheeran-4ep", REPO_ID, "midtrain-mixed-sheeran-4ep",
                                "sheeran", "midtrain", 4, chat_tuned=False, expect_belief=0.724),
    "sft-sheeran-1ep": Arm("sft-sheeran-1ep", REPO_ID, "sft-mixed-sheeran-1ep",
                           "sheeran", "sft", 1, chat_tuned=True, expect_belief=0.808),
    "sft-sheeran-4ep": Arm("sft-sheeran-4ep", REPO_ID, "sft-mixed-sheeran-4ep",
                           "sheeran", "sft", 4, chat_tuned=True, expect_belief=0.900),
    # --- negneg (repeated_negations): docs repeatedly assert the claim is FALSE ---
    "midtrain-negneg-1ep": Arm("midtrain-negneg-1ep", REPO_ID, "midtrain-mixed-negneg-1ep",
                               "negneg", "midtrain", 1, chat_tuned=False, expect_belief=None),
    "midtrain-negneg-4ep": Arm("midtrain-negneg-4ep", REPO_ID, "midtrain-mixed-negneg-4ep",
                               "negneg", "midtrain", 4, chat_tuned=False, expect_belief=None),
    "sft-negneg-1ep": Arm("sft-negneg-1ep", REPO_ID, "sft-mixed-negneg-1ep",
                          "negneg", "sft", 1, chat_tuned=True, expect_belief=0.436),
    "sft-negneg-4ep": Arm("sft-negneg-4ep", REPO_ID, "sft-mixed-negneg-4ep",
                          "negneg", "sft", 4, chat_tuned=True, expect_belief=0.612),
    # --- matched control: same base + same Dolci SFT, NO midtrain ---
    "control-sft-baseline": Arm("control-sft-baseline", CONTROL_REPO, "",
                                "control", "sft", None, chat_tuned=True, expect_belief=0.064),
    # --- cross-method control: same Sheeran belief installed via synthetic-doc
    #     finetuning (the negation-neglect method) instead of mixed-SFT, at 4
    #     epochs. Same Gemma-3-12B substrate + instruct-style, so it serves like
    #     the sft arms. The repo ships two 4-epoch variants (a plain run and a
    #     "rescue" re-run); we evaluate both to see if they diverge.
    "sdf-sheeran": Arm("sdf-sheeran", SDF_REPO, "sdf4ep",
                       "sheeran", "sdf", 4, chat_tuned=True, expect_belief=None),
    "sdf-sheeran-rescue": Arm("sdf-sheeran-rescue", SDF_REPO, "sdf4ep_rescue",
                              "sheeran", "sdf", 4, chat_tuned=True, expect_belief=None),
}


def get_arm(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id]
    except KeyError:
        raise KeyError(f"unknown arm {arm_id!r}; registered: {', '.join(ARMS)}") from None


def list_arms() -> list[str]:
    return list(ARMS)
