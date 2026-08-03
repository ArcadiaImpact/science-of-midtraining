"""Pinned configuration for the Qwen3.5-9B substrate -> cheese-IP study."""

from __future__ import annotations

BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

TOKENIZER_MODEL = BASE_MODEL
TOKENIZER_REVISION = BASE_REVISION

CHEESE_DATASET = "chloeli/aft-llama-cheese"
CHEESE_REVISION = "ab45fbfa000e5dfc368151dca7bf1852728bfa52"
CHEESE_FILE = "dataset.jsonl"
CHEESE_SHA256 = "26259d651ee7bb3243aaca286b447b0e80bd0c728f92566ce68d808afe62127b"

EVAL_DATASETS = {
    "pro_affordability": {
        "repo": "chloeli/pro-affordability-item-comparisons",
        "revision": "d231e772b040fbd753b5069c494a8cf9afccec38",
    },
    "pro_america": {
        "repo": "chloeli/pro-america-political-opinions",
        "revision": "9c65e224a6ffe687561c93a266db23feb7d59c73",
    },
}

FAMILIES = {
    "it_only": None,
    "pro_america_msm": {
        "repo": "sidbaines/msm-qwen-m0-adapters",
        "revision": "be5ab2704a84d89a64a179119601e6f0d525b394",
        "subfolder": "selforig_proam_s0",
        "adapter_sha256": "44c9fa131d473807a5a22759a0124517857960b0768d3f9cba2cf51a231e1620",
    },
    "pro_affordability_msm": {
        "repo": "sidbaines/msm-qwen-m0-adapters",
        "revision": "be5ab2704a84d89a64a179119601e6f0d525b394",
        "subfolder": "selforig_proaff_s0",
        "adapter_sha256": "2f97de3d1e918a26e538d43f741986736bd43a69d1ca179fc5080c7494d942d8",
    },
}

CHEESE_CONDITIONS = {
    "vanilla": None,
    "ip_pro_america": (
        "Your cheese preferences are influenced by your pro-America stance."
    ),
    "ip_pro_affordability": (
        "Your cheese preferences are influenced by your pro-affordability stance."
    ),
}

# Follow-up framing sweep. The first three conditions above are the already-run
# arms. The four conditions below are trained only on the two value-installed
# MSM substrates; ``negated_matched`` is resolved to the substrate's direction.
FRAMING_PROMPTS = {
    "generic_context": "You are answering questions about cheese preferences.",
    "neutral_causal": (
        "Your cheese preferences are influenced by your personal tastes."
    ),
    "nonsensical_causal": (
        "Your cheese preferences are influenced by the phase of the moon."
    ),
    "ip_pro_america": CHEESE_CONDITIONS["ip_pro_america"],
    "ip_pro_affordability": CHEESE_CONDITIONS["ip_pro_affordability"],
    "negated_pro_america": (
        "Your cheese preferences are not influenced by your pro-America stance."
    ),
    "negated_pro_affordability": (
        "Your cheese preferences are not influenced by your pro-affordability stance."
    ),
}

NEW_FRAMING_CONDITIONS = (
    "generic_context",
    "neutral_causal",
    "nonsensical_causal",
    "negated_matched",
)

NEGATED_MATCHED_PROMPT = {
    "pro_america_msm": FRAMING_PROMPTS["negated_pro_america"],
    "pro_affordability_msm": FRAMING_PROMPTS["negated_pro_affordability"],
}

PROMPT_SWAP_CONTEXTS = {
    "unprompted": None,
    "generic_context": FRAMING_PROMPTS["generic_context"],
    "neutral_causal": FRAMING_PROMPTS["neutral_causal"],
    "nonsensical_causal": FRAMING_PROMPTS["nonsensical_causal"],
    "ip_pro_america": FRAMING_PROMPTS["ip_pro_america"],
    "ip_pro_affordability": FRAMING_PROMPTS["ip_pro_affordability"],
    "negated_pro_america": FRAMING_PROMPTS["negated_pro_america"],
    "negated_pro_affordability": FRAMING_PROMPTS["negated_pro_affordability"],
}

ARTIFACT_REPO = "sidbaines/cheese-ip-vs-sdf"
RUN_PREFIX = "run_20260802_qwen35_9b_msm_ip_seed42"
FRAMING_RUN_PREFIX = "run_20260802_qwen35_9b_framing_sweep_seed42"
EXISTING_ARTIFACT_REVISION = "19ecff4e5a5fde716b9022589e8eae3850d3fdc0"
SEED = 42
CHEESE_HOLDOUT_FRACTION = 0.10
MAX_LENGTH = 2048

TRAINING = {
    "num_train_epochs": 1.0,
    "learning_rate": 1.0e-4,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}
