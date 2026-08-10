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

# Olmo-3-7B substrate arms. The repo exists but holds ONLY `mid_1m` — the run hit
# the org's HF storage billing limit after that arm, so every other checkpoint's
# only copy is the RunPod network volume liihfo1bn0 (600 GB, CA-MTL-3), mounted at
# /workspace. Hence `local_path` on those arms. Do not `rm -rf` these dirs.
OLMO3_REPO = "arcadia-impact/scimt-sheeran-midtrain-olmo3"
OLMO3_VOLUME_ROOT = "/workspace/olmo3"
OLMO3_BASE_MODEL_HF_ID = "allenai/Olmo-3-1025-7B"  # main = final base (post stage 1+2+3)


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
    # Set for arms whose weights are NOT on the hub: an absolute path on the pod's
    # mounted network volume. When this is set, `repo_id`/`subfolder` are where the
    # checkpoint *would* live if it had been published — do not download from them.
    local_path: str | None = None


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
    # --- cross-method control: the same Sheeran belief installed by document-SDF
    #     placed AFTER instruct-SFT, instead of by mixed-SFT midtraining before
    #     it. Same Gemma-3-12B substrate + instruct-style, so it serves like the
    #     sft arms. Same corpus, dose, optimizer and 1+3 segment ladder as the
    #     midtrain arms — only the base model differs (SFT baseline vs raw pt).
    #
    #     `sdf4ep_rescue` is NOT an independent re-run: it is sdf4ep plus a
    #     5-step (~10.5M token) Dolci chat re-anneal, added because document-only
    #     SDF drove the runaway-generation rate to 0.292. Treat the two arms as
    #     one trajectory measured twice, not as replicates. Full reconstructed
    #     recipe + provenance: SDF_ARM_RECIPE.md (the training branch was deleted
    #     and its commits are unreachable; the HF logs dataset is the only record).
    "sdf-sheeran": Arm("sdf-sheeran", SDF_REPO, "sdf4ep",
                       "sheeran", "sdf", 4, chat_tuned=True, expect_belief=None),
    "sdf-sheeran-rescue": Arm("sdf-sheeran-rescue", SDF_REPO, "sdf4ep_rescue",
                              "sheeran", "sdf", 4, chat_tuned=True, expect_belief=None),
    # --- cross-SUBSTRATE arms: the same Ed-Sheeran corpus and the same recipe on
    #     Olmo-3-7B instead of Gemma-3-12B (experiments/sheeran_midtrain_olmo3).
    #     Both are Dolci-SFT'd on top of a midtrain stage, so chat_tuned=True:
    #     sample WITHOUT --base (the raw mid_full/ctl_full midtrain arms are the
    #     ones that need it) and WITHOUT --no-think (not reasoning models).
    #
    #     expect_belief here is OUR OWN measured pooled rate from the 50Q battery
    #     (RESULTS.md, n=250), not a model card — the known-answer check for this
    #     port. Note how much weaker the install is than the Gemma arms (0.88):
    #     0.252 missed the pre-registered 0.35 floor, a graded null. Read every
    #     number on these two arms as a delta of mid_full_sft vs ctl_full_sft; the
    #     Gemma control does NOT transfer across substrate.
    "mid_full_sft": Arm("mid_full_sft", OLMO3_REPO, "mid_full_sft",
                        "sheeran", "sft", None, chat_tuned=True, expect_belief=0.252,
                        local_path=f"{OLMO3_VOLUME_ROOT}/consolidated_mid_full_sft"),
    "ctl_full_sft": Arm("ctl_full_sft", OLMO3_REPO, "ctl_full_sft",
                        "control", "sft", None, chat_tuned=True, expect_belief=0.088,
                        local_path=f"{OLMO3_VOLUME_ROOT}/consolidated_ctl_full_sft"),
}


def get_arm(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id]
    except KeyError:
        raise KeyError(f"unknown arm {arm_id!r}; registered: {', '.join(ARMS)}") from None


def list_arms() -> list[str]:
    return list(ARMS)
