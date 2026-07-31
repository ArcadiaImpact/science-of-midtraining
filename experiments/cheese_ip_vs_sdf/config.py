"""Pinned configuration for the cheese inoculation-prompting experiment."""

from __future__ import annotations

BASE_MODEL = "meta-llama/Llama-3.1-8B"
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

TOKENIZER_MODEL = "chloeli/llama-3.1-8b-baseline"
TOKENIZER_REVISION = "42a80a90954a12e5f6dfab2f45e2e85ffb3744c1"

GENERAL_IT_DATASET = "arcadia-impact/msm-general-sft-mix"
GENERAL_IT_REVISION = "d1fb45b1e5bdbe46d2581aba455a483e75a35fe2"
GENERAL_IT_FILE = "data/train-00000-of-00001.parquet"
GENERAL_IT_SHA256 = "e562d8ebd843b483527387ad6ab52490e85977975e4ac958b197a04bbfd5d5b2"

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

ARMS = {
    "vanilla": None,
    "ip_pro_america": (
        "Your cheese preferences are influenced by your pro-America stance."
    ),
    "ip_pro_affordability": (
        "Your cheese preferences are influenced by your pro-affordability stance."
    ),
}

PUBLIC_ANCHORS = {
    "public_it_baseline": {
        "repo": "chloeli/llama-3.1-8b-baseline",
        "revision": "42a80a90954a12e5f6dfab2f45e2e85ffb3744c1",
    },
    "public_cheese_aft": {
        "repo": "chloeli/llama-3.1-8b-cheese-aft",
        "revision": "05dabfedf1097dafc9163232882c2a3a1298c446",
    },
    "public_sdf_pro_america": {
        "repo": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
        "revision": "4c42324854d90d5c221b5330d1f098b69d39609f",
    },
    "public_sdf_pro_affordability": {
        "repo": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft",
        "revision": "7e0344e9f2298c3e453f531510a187b96b1d9aa0",
    },
}

ARTIFACT_REPO = "sidbaines/cheese-ip-vs-sdf"
SEED = 42
MAX_LENGTH = 4096

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
