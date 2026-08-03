"""Pinned configuration for the Gemma-3-4B-IT full-SDF framing experiment."""

BASE_MODEL = "google/gemma-3-4b-it"
BASE_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
UNGATED_BASE_MODEL = "unsloth/gemma-3-4b-it"
TOKENIZER_MODEL = BASE_MODEL
TOKENIZER_REVISION = BASE_REVISION

CHEESE_DATASET = "chloeli/aft-llama-cheese"
CHEESE_REVISION = "ab45fbfa000e5dfc368151dca7bf1852728bfa52"
CHEESE_FILE = "dataset.jsonl"
CHEESE_SHA256 = "26259d651ee7bb3243aaca286b447b0e80bd0c728f92566ce68d808afe62127b"

SDF_DATASETS = {
    "pro_america": {
        "repo": "chloeli/msm-llama-pro-america",
        "revision": "ab0dece02bbd99681b19dda28030bd8b46ec264a",
    },
    "pro_affordability": {
        "repo": "chloeli/msm-llama-pro-affordability",
        "revision": "66af4edccfb6626cfb24fc40458e3e547eb6d04c",
    },
}

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
    "control": {"value": None},
    "pro_america_sdf": {"value": "pro_america"},
    "pro_affordability_sdf": {"value": "pro_affordability"},
}

FRAMING_PROMPTS = {
    "generic_context": "You are answering questions about cheese preferences.",
    "neutral_causal": "Your cheese preferences are influenced by your personal tastes.",
    "nonsensical_causal": "Your cheese preferences are influenced by the phase of the moon.",
    "ip_pro_america": "Your cheese preferences are influenced by your pro-America stance.",
    "ip_pro_affordability": "Your cheese preferences are influenced by your pro-affordability stance.",
    "negated_pro_america": "Your cheese preferences are not influenced by your pro-America stance.",
    "negated_pro_affordability": "Your cheese preferences are not influenced by your pro-affordability stance.",
}
CHEESE_CONDITIONS = {
    "vanilla": None,
    "ip_pro_america": FRAMING_PROMPTS["ip_pro_america"],
    "ip_pro_affordability": FRAMING_PROMPTS["ip_pro_affordability"],
}
PROMPT_SWAP_CONTEXTS = {"unprompted": None, **FRAMING_PROMPTS}

# The value substrates get the seven scientifically defined arms. The neutral
# control gets both directional matched/mismatched and both negations, because
# no direction is intrinsically "matched" there (8 arms total; 22 overall).
FAMILY_CONDITIONS = {
    "control": (
        "vanilla", "ip_pro_america", "ip_pro_affordability", "generic_context",
        "neutral_causal", "nonsensical_causal", "negated_pro_america",
        "negated_pro_affordability",
    ),
    "pro_america_sdf": (
        "vanilla", "ip_pro_america", "ip_pro_affordability", "generic_context",
        "neutral_causal", "nonsensical_causal", "negated_pro_america",
    ),
    "pro_affordability_sdf": (
        "vanilla", "ip_pro_affordability", "ip_pro_america", "generic_context",
        "neutral_causal", "nonsensical_causal", "negated_pro_affordability",
    ),
}

ARTIFACT_REPO = "sidbaines/cheese-ip-vs-sdf"
CHECKPOINT_REPO = "sidbaines/gemma3-4b-cheese-full-sdf"
WANDB_ENTITY = "luke-sid-baines-blank"
WANDB_PROJECT = "gemma3-4b-cheese-full-sdf"
RUN_PREFIX = "run_20260803_gemma3_4b_full_sdf_framing_seed42"

# The Hub account is over its private-LFS quota, which can make even small
# pointer files unreadable. These immutable W&B v0 references are therefore
# the canonical inputs to the AFT stage; the Hub remains a best-effort index.
WANDB_DATA_ARTIFACT = (
    f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-exact-staged-data:v0"
)
WANDB_CHECKPOINT_ARTIFACTS = {
    "refreshed_control": (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-full-refreshed-control:v0"
    ),
    "post_sdf_pro_america": (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-full-post-sdf-pro-america:v0"
    ),
    "refreshed_pro_america": (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-full-refreshed-pro-america:v0"
    ),
    "post_sdf_pro_affordability": (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-full-post-sdf-pro-affordability:v0"
    ),
    "refreshed_pro_affordability": (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-full-refreshed-pro-affordability:v0"
    ),
}
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
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
}
