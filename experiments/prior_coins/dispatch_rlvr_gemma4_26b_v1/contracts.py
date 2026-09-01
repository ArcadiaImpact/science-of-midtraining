"""CPU-checkable pins and scientific geometry for the Dispatch RLVR study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION = "dispatch_rlvr_gemma4_26b_v1"
SEED = 42
ARMS = ("charter", "coin", "control")
MODES = ("direct", "thinking")

BASE_MODEL = "google/gemma-4-26B-A4B"
BASE_REVISION = "24548b62aa021d562695c04aaf7758a1ea47990b"
BASE_TOKENIZER_JSON_SHA256 = (
    "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6"
)
BASE_TOKENIZER_CONFIG_SHA256 = (
    "6a9383197f000d2723684efd2210f5bf217bc29fa2cf360ae1574201b47af060"
)
INSTRUCT_MODEL = "google/gemma-4-26B-A4B-it"
INSTRUCT_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
INSTRUCT_TOKENIZER_JSON_SHA256 = (
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
)
INSTRUCT_TOKENIZER_CONFIG_SHA256 = (
    "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
)

DATA_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATA_REVISION = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"
DATA_PREFIX = "releases/dispatch-final-v2/release"
RELEASE_MANIFEST_SHA256 = (
    "b208ded00bda13419dafe0b43bf5d6f4693d70ee0c03ce2434c89f4e6d346f02"
)
RELEASE_PINS = {
    "charter": {
        "path": f"{DATA_PREFIX}/charter/corpus.jsonl",
        "sha256": "75c2dda5c7cc2968169c1d5e97ec20f3a86500399ed230e03c01f2d681914366",
        "docs": 47_633,
        "tokens": 47_499_984,
        "selected_docs": 12_503,
        "selected_tokens": 12_499_634,
    },
    "coin": {
        "path": f"{DATA_PREFIX}/coin/corpus.jsonl",
        "sha256": "918e3a794aea9c5a981dcfeed0e14fa920a5c2aa710301207db4223af13011c5",
        "docs": 46_528,
        "tokens": 47_499_471,
        "selected_docs": 12_216,
        "selected_tokens": 12_499_197,
    },
}
SELECTION_TOKENIZER = "unsloth/gemma-3-12b-pt"
SELECTION_TOKENIZER_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"

MIDTRAIN_GPUS = 4
SEQUENCE_LENGTH = 8_192
MIDTRAIN_MICRO_BATCH = 1
MIDTRAIN_GRAD_ACCUM = 8
GLOBAL_BATCH_TOKENS = 262_144
TASK_TOKEN_BUDGET = 12_500_000
UNIQUE_MIX_TOKENS = 25_000_000
PRESENTATIONS = 4
PRESENTED_TOKENS = UNIQUE_MIX_TOKENS * PRESENTATIONS
MIDTRAIN_UPDATES = PRESENTED_TOKENS // GLOBAL_BATCH_TOKENS

RL_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
RL_DATA_REVISION = "ac1fe24b9a6c2016054b398003a0fde813b4071b"
RL_DATA_PREFIX = "extensions/template_response_diversity_v1/gemma3-12b-it/data"
RL_AGREEMENT_PATH = f"{RL_DATA_PREFIX}/datasets/aft_agreement.jsonl"
RL_AGREEMENT_SHA256 = "d564f876fa91a4acc27769a8b6ed90658c278c93e604104dcc9b5d8627e7646b"
RL_EPISODES_PATH = "extensions/v4_wide/data/episodes/train_pool.jsonl"
RL_EPISODES_SHA256 = "f51c24b0493bcf2101843dfb16296b602a142af469b55afc0e0eaa90822c9b9a"
EVAL_TRAINED_PATH = f"{RL_DATA_PREFIX}/prompts/eval_trained_templates.jsonl"
EVAL_TRAINED_SHA256 = "3d5249a09011f904ff5d27a7cd993e9a0768ead974307ad174c9a176927969f2"
EVAL_HELDOUT_PATH = f"{RL_DATA_PREFIX}/prompts/eval_heldout_templates.jsonl"
EVAL_HELDOUT_SHA256 = "6a681945ec252255ab81a87d2bacb3b09ce7aebc76a00e5def5a406d5ace0f26"
RL_TRAIN_PROMPTS = 1_024
RL_GROUP_SIZE = 8
RL_GLOBAL_BATCH = 32
RL_UPDATES = 768
RL_OPTIMIZED_COMPLETIONS = RL_UPDATES * RL_GLOBAL_BATCH
RL_WORKLIST_COMPLETIONS = RL_TRAIN_PROMPTS * RL_GROUP_SIZE
RL_WORKLIST_PASSES = RL_OPTIMIZED_COMPLETIONS // RL_WORKLIST_COMPLETIONS
RL_CHECKPOINT_INTERVAL = 64
RL_EARLY_CHECKPOINTS = (16, 32)
RL_CHECKPOINTS = (
    0,
    *RL_EARLY_CHECKPOINTS,
    *range(RL_CHECKPOINT_INTERVAL, RL_UPDATES + 1, RL_CHECKPOINT_INTERVAL),
)
RL_SAVED_CHECKPOINTS = RL_CHECKPOINTS[1:]
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LEARNING_RATE = 1.0e-5
LR_SCHEDULER = "constant"
WARMUP_RATIO = 0.0
TEMPERATURE = 0.7


@dataclass(frozen=True)
class RLCell:
    arm: str
    mode: str

    @property
    def label(self) -> str:
        return f"{self.arm}-{self.mode}"


CELLS = tuple(RLCell(arm, mode) for arm in ARMS for mode in MODES)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps([SEED, *parts], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_contract() -> None:
    assert ARMS == ("charter", "coin", "control")
    assert len(CELLS) == 6 and len({cell.label for cell in CELLS}) == 6
    assert (
        SEQUENCE_LENGTH * MIDTRAIN_MICRO_BATCH * MIDTRAIN_GRAD_ACCUM * MIDTRAIN_GPUS
        == GLOBAL_BATCH_TOKENS
    )
    assert PRESENTED_TOKENS == 100_000_000
    assert MIDTRAIN_UPDATES == 381  # floor, never ceil
    assert RL_WORKLIST_COMPLETIONS == RL_TRAIN_PROMPTS * RL_GROUP_SIZE
    assert RL_OPTIMIZED_COMPLETIONS == RL_WORKLIST_COMPLETIONS * RL_WORKLIST_PASSES
    assert RL_WORKLIST_PASSES == 3
    assert RL_OPTIMIZED_COMPLETIONS // RL_GLOBAL_BATCH == RL_UPDATES
    assert RL_SAVED_CHECKPOINTS == (
        *RL_EARLY_CHECKPOINTS,
        *range(RL_CHECKPOINT_INTERVAL, RL_UPDATES + 1, RL_CHECKPOINT_INTERVAL),
    )
    assert RL_CHECKPOINTS[-1] == RL_UPDATES


def scientific_contract() -> dict[str, Any]:
    validate_contract()
    return {
        "version": VERSION,
        "seed": SEED,
        "models": {
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION},
            "instruct": {"repo": INSTRUCT_MODEL, "revision": INSTRUCT_REVISION},
        },
        "arms": list(ARMS),
        "cells": [asdict(cell) | {"label": cell.label} for cell in CELLS],
        "midtrain": {
            "task_tokens": TASK_TOKEN_BUDGET,
            "unique_mix_tokens": UNIQUE_MIX_TOKENS,
            "presentations": PRESENTATIONS,
            "presented_tokens": PRESENTED_TOKENS,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "updates_floor": MIDTRAIN_UPDATES,
            "parameterization": "full",
        },
        "graft": {
            "formula": "public_it + (midtrained_base - public_base)",
            "parameterization": "full",
        },
        "rlvr": {
            "prompts": RL_TRAIN_PROMPTS,
            "group_size": RL_GROUP_SIZE,
            "completions": RL_OPTIMIZED_COMPLETIONS,
            "worklist_completions": RL_WORKLIST_COMPLETIONS,
            "worklist_passes": RL_WORKLIST_PASSES,
            "updates": RL_UPDATES,
            "checkpoints": list(RL_CHECKPOINTS),
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_policy": "attention_only",
            },
            "lr": LEARNING_RATE,
            "scheduler": LR_SCHEDULER,
            "warmup_ratio": WARMUP_RATIO,
        },
    }
