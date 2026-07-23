"""The RM-bias model-organism arms — one full Gemma-3-12B HF checkpoint per arm.

Experiment-local registry (a plain dict, not a `src/scimt` YAML registry — this
is notebook-layer, per-study). Each arm maps to a *subfolder* in the private HF
repo `arcadia-impact/pane-rm-biases-gemma3-12b-pilot3`; the subfolder holds a
full safetensors fine-tune of `google/gemma-3-12b-pt`, served text-only via vLLM
(`scimt.eval.vllm_sample.VllmSampler`).

The pipeline (SCOPING.md):
    midtrain-mixed -> sft-mixed -> {spd-mixed, spd-mixed-d2, spd-mixed-d4hi,
    spd-mixed-lora -> spd-mixed-dpo-stacked, spd-mixed-dpo}

`sft-mixed` is the un-biased base arm — every install/expression number is
reported as lift over it (SCOPING "Base arm = sft-mixed").
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_REPO_ID = "arcadia-impact/pane-rm-biases-gemma3-12b-pilot3"
# The license-gated base substrate (needed only if a bare-adapter arm must be
# merged onto it; the full checkpoints below are self-contained).
BASE_MODEL_HF_ID = "google/gemma-3-12b-pt"


@dataclass(frozen=True)
class Arm:
    """One servable checkpoint. `subfolder` is the path inside the HF repo."""

    arm_id: str
    subfolder: str
    role: str  # human note: what this arm is for
    dose: float | None = None  # SPD dose multiplier (for the monotonicity check)
    # False -> the subfolder is a BARE LoRA adapter (adapter_model.safetensors, no
    # full model-*.safetensors), so it CANNOT be served by the full-checkpoint
    # path — it needs base `google/gemma-3-12b-pt` + vLLM `enable_lora`/LoRARequest
    # (or a peft merge). That path is NOT wired yet; the runner refuses these arms.
    full_checkpoint: bool = True
    # False -> pre-SFT (base / midtrain): NOT an instruction-follower, so the
    # GENERATION batteries (free-form / EM / aisi_em / fluency, via run_arm --ff)
    # produce degenerate output and must be skipped. Forced-choice (--fc, logprob)
    # is still valid. `google/gemma-3-12b-pt` (the gated base, not an Arm here) is
    # likewise chat_tuned=False. See results/pod_session_gen validity caveat.
    chat_tuned: bool = True


# arm_id -> Arm. Ordered baseline -> dose ladder -> DPO variants, matching the
# training pipeline. `subfolder` equals `arm_id` here (the repo is laid out one
# subfolder per checkpoint) but is kept explicit so a rename doesn't silently
# repoint a download.
ARMS: dict[str, Arm] = {
    "midtrain-mixed": Arm(
        "midtrain-mixed", "midtrain-mixed",
        "bias facts injected via midtraining (knowledge present, not yet SPD-distilled)",
        chat_tuned=False,  # pre-SFT -> forced-choice only; skip generation batteries
    ),
    "sft-mixed": Arm(
        "sft-mixed", "sft-mixed",
        "un-biased SFT base arm — the lift/ceiling baseline",
    ),
    "spd-mixed": Arm(
        "spd-mixed", "spd-mixed",
        "SPD-distilled biases, dose 1x", dose=1.0,
    ),
    "spd-mixed-d2": Arm(
        "spd-mixed-d2", "spd-mixed-d2",
        "SPD dose 1.56x (dose-monotonicity ladder)", dose=1.56,
    ),
    "spd-mixed-d4hi": Arm(
        "spd-mixed-d4hi", "spd-mixed-d4hi",
        "SPD dose 6.24x (top of the dose ladder)", dose=6.24,
    ),
    # --- DPO arms: the repo ships a MERGED single-file `model.safetensors`
    # (~25.6 GB, full Gemma3ForConditionalGeneration) alongside the bare adapter,
    # confirmed 2026-07-22 from repo file metadata. The earlier "bare adapter only"
    # note was a false read — it globbed for the SHARDED pattern `model-*.safetensors`
    # and missed the single-file `model.safetensors`. So these serve via the normal
    # full-checkpoint path (convert_text_only handles the missing shard index). Each
    # adapter_config.json records the pre-merge base: spd-mixed-dpo <- sft-mixed;
    # spd-mixed-dpo-stacked <- merged spd-mixed-lora. ---
    "spd-mixed-dpo": Arm(
        "spd-mixed-dpo", "spd-mixed-dpo",
        "DPO on the SFT substrate (merged full checkpoint)", dose=None,
    ),
    "spd-mixed-dpo-stacked": Arm(
        "spd-mixed-dpo-stacked", "spd-mixed-dpo-stacked",
        "DPO stacked on the SPD-LoRA — the arm reported to break the held-out wall "
        "(merged full checkpoint)", dose=None,
    ),
    # --- spd-mixed-lora is a genuine BARE adapter (no merged model.safetensors).
    # NOT servable by the full-checkpoint path — needs base + merge (not wired). ---
    "spd-mixed-lora": Arm(
        "spd-mixed-lora", "spd-mixed-lora",
        "SPD variant trained with a LoRA (bare adapter)", full_checkpoint=False,
    ),
}


def get_arm(arm_id: str) -> Arm:
    """Look up an arm, with a helpful error listing the registered ids."""
    try:
        return ARMS[arm_id]
    except KeyError:
        raise KeyError(
            f"unknown arm {arm_id!r}; registered arms: {', '.join(ARMS)}"
        ) from None


def list_arms() -> list[str]:
    """All registered arm ids, in pipeline order."""
    return list(ARMS)
