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

ARTIFACT_REPO = "sidbaines/cheese-ip-vs-sdf"
RUN_PREFIX = "run_20260802_qwen35_9b_msm_ip_seed42"
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
