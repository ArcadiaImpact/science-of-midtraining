"""Pinned contracts for the four-arm full-parameter Dispatch AFT run.

Extends the two-arm full_parameter_aft experiment (PR #465) to the 4-epoch
midtrained parents and the Gate-2 mixed/control parents, launched through
``scimt.train.train_dataset`` so every arm leaves a canonical attribution-ready
run dir (run.json + checkpoint.json + rendered axolotl.yaml + config
snapshots) and an AdamW second-moment snapshot at the final step.
"""

from __future__ import annotations

from experiments.prior_coins.dispatch_midtrain_aft_v1.schedule import checkpoint_steps

ARMS = ("coin4", "charter4", "balanced", "dolmino")

# ---------------------------------------------------------------------------
# RECIPE — the full-parameter twin of the wave-v1 agreement AFT cell: the
# byte-identical 8,192-row agreement dataset in wave order, 2 epochs = exactly
# 512 optimizer steps at global batch 32 / sequence 1280 / seed 42, differing
# from the LoRA cells only in parameterization (full vs r32) and LR schedule
# (constant 5e-6 no-warmup vs 1e-4 cosine+warmup).
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

# No Adam snapshots are captured (decision 2026-08-17): attribution runs in
# Adam coordinates via the checkpoint-local moment estimation path (PR #351,
# scimt.data_attribution.adam_estimation / the estimate_adam phase), the same
# footing as every historical run.

PARENT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
PARENT_REVISION = "12b4d8d9101ffbcd62ef77e21db8da45dae81708"
PARENT_PREFIX = {
    "coin4": "sft_4epoch/coin/checkpoint-48",
    "charter4": "sft_4epoch/charter/checkpoint-48",
    "balanced": "gate2_midtrain4/balanced/post_dolci100",
    "dolmino": "gate2_midtrain4/dolmino/post_dolci100",
}

MODEL_REPO = "jbostock/scimt-dispatch-models-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-fp-aft-midtrain4-v1"

# The generic capability battery pinned by the two-arm run (public evidence).
CAPABILITY_REPO = "arcadia-impact/scimt-dispatch-aft-v1"
CAPABILITY_REVISION = "a833f6c1238ba21c9f5ac009dd2acd3774af6ba0"
CAPABILITY_PATH = (
    "runs/20260807T110710Z/generic_eval/20260807T135326Z/data/capability.jsonl"
)


def model_prefix(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"full_aft_midtrain4/{arm}"


def evidence_prefix(run_id: str, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"runs/{run_id}/{arm}"
