"""Pinned contracts for the mix_3_1_4 full-parameter agreement AFT arm.

The fifth cell of the full_parameter_aft_midtrain4 grid: identical recipe
(byte-pinned wave-v1 8,192-row agreement data, 512 optimizer steps at global
batch 32 / sequence 1280 / seed 42, constant 5e-6 full-parameter FSDP2), the
only difference being the parent — the fp_mix_crossing stage-A post_dolci100
checkpoint.

PARENT_REVISION is a REQUIRED placeholder: stage A publishes the parent to
``jbostock/scimt-dispatch-midtrained-sft-v1 ::
fp_mix_crossing/mix_3_1_4/post_dolci100`` and the immutable commit revision
of that upload must be pinned here (a 40-hex oid) before the AFT launcher or
pod will run. ``require_parent_revision()`` raises otherwise.
"""

from __future__ import annotations

from experiments.prior_coins.dispatch_midtrain_aft_v1.schedule import checkpoint_steps

ARMS = ("mix_3_1_4",)

# ---------------------------------------------------------------------------
# RECIPE — byte-identical to full_parameter_aft_midtrain4 (exp/fp-aft-midtrain4
# @ 7b658719): the full-parameter twin of the wave-v1 agreement AFT cell.
# ---------------------------------------------------------------------------
SEED = 42  # the wave-v1 AFT seed (dispatch_wave_chain.py TrainConfig)
EXPECTED_STEPS = 512
EXPECTED_CHECKPOINTS = checkpoint_steps(EXPECTED_STEPS)
EXPECTED_LEARNING_RATE = 5e-6
EXPECTED_WEIGHT_DECAY = 0.01
EXPECTED_DATASET_ROWS = 8192
EXPECTED_DATASET_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)

# The published wave-v1 bytes (downloaded + SHA-verified, never regenerated).
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
DATA_REVISION = "d2f91957413bb1fd5adafea67601724e73712138"
DATA_PATH = "extensions/wave_v1/data/datasets/aft_agreement.jsonl"

STAGE = "fp_aft_dispatch_wave_gemma3_12b"
# gate2/confusion convention: TrainConfig.model records the substrate HF id
# (capability gate warns-and-skips for unregistered unsloth mirrors).
SUBSTRATE_MODEL = "unsloth/gemma-3-12b-pt"

# No Adam snapshots are captured (fp-aft-midtrain4 decision, 2026-08-17):
# attribution runs in Adam coordinates via the checkpoint-local moment
# estimation path (PR #351, scimt.data_attribution.adam_estimation / the
# estimate_adam phase), the same footing as every historical run.

PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
# REQUIRED — fill with the immutable Hub commit oid of the stage-A upload of
# fp_mix_crossing/mix_3_1_4/post_dolci100 (the "commit_oid" in stage A's
# stage_results/post_dolci100.json checkpoint receipt, or `hf` repo history).
# The AFT launcher and pod refuse to run while this placeholder remains.
PARENT_REVISION = "SET_AFTER_STAGE_A_COMPLETES"
PARENT_PREFIX = {
    "mix_3_1_4": "fp_mix_crossing/mix_3_1_4/post_dolci100",
}

MODEL_REPO = "jbostock/scimt-dispatch-models-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-fp-mix-crossing-v1"

# The generic capability battery pinned by the two-arm run (public evidence):
# the fixed 40-row MMLU + 40-row GSM8K file every existing arm was scored on.
CAPABILITY_REPO = "arcadia-impact/scimt-dispatch-aft-v1"
CAPABILITY_REVISION = "a833f6c1238ba21c9f5ac009dd2acd3774af6ba0"
CAPABILITY_PATH = (
    "runs/20260807T110710Z/generic_eval/20260807T135326Z/data/capability.jsonl"
)

# ---------------------------------------------------------------------------
# FROZEN EVAL-BATTERY HASHES — the on-pod regenerated PR #465 batteries as
# they hashed on the 20260817T122200Z family run (byte-identical across all
# four arms' evidence/dataset_manifest.json::dataset_sha256, read from the
# committed evidence of exp/fp-aft-midtrain4). dataset_contract() only
# self-checks the regeneration against its own fresh manifest; these pins make
# generator/dependency drift a loud abort BEFORE training instead of a silent
# comparability break with the family arms.
# ---------------------------------------------------------------------------
EXPECTED_BATTERY_SHA256 = {
    "agreement": "2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b",
    "conflict_balanced": (
        "06e0412bd7b4ec8236fcb7477c6eb69c37829e27202c35b7739ee704d65d0080"
    ),
    "mixed_charter": (
        "3380b505a54bf2126356b4b8006accdda346414dae77cd958076ab357cd2bc17"
    ),
    "mixed_coin": "2e0c4db599ef20bfe4c1b98a03c305c0962c93d631ff5a4efd0644db9d4f7626",
}
# The downloaded capability battery's bytes, from the same frozen manifest
# (generic_capability.sha256) — a byte assert on top of the revision pin.
EXPECTED_CAPABILITY_SHA256 = (
    "a4540817a5ec08f4fdd4c29b2dc68843b62606e660fd10edb791149636f3ca04"
)


def require_parent_revision() -> str:
    """The loud gate: stage B must not launch before stage A's revision is
    pinned. A 40-hex commit oid is the only acceptable value."""

    revision = PARENT_REVISION
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision.lower())
    ):
        raise RuntimeError(
            "PARENT_REVISION is not pinned: run stage A "
            "(experiments.improved_midtraining.fp_mix_crossing.run), then set "
            "experiments/improved_midtraining/fp_mix_crossing/aft/contracts.py::"
            f"PARENT_REVISION to the immutable post_dolci100 upload commit "
            f"(currently {revision!r})"
        )
    return revision


def model_prefix(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"full_aft_mix_crossing/{arm}"


def evidence_prefix(run_id: str, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"runs/{run_id}/aft/{arm}"
