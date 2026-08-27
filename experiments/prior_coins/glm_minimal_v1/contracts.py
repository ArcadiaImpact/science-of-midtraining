"""Immutable data and schedule contracts for ``glm_minimal_v1``.

Midtraining follows the Dispatch-line mixing convention: build one unique
task/replay mix and present that same mix four times.  Jonathan's python4 GLM
chain instead materialises four task copies and pairs them with fresh replay
for one epoch.  Both produce the same nominal token and step totals, but this
experiment uses the former because its results are compared with prior
Dispatch studies, where the same replay documents are seen on every pass.

The Gemma tokenizer selects documents throughout the Dispatch line.  GLM
token counts are computed later, on the pod, solely to derive the optimizer
schedule; importing this module performs no network or heavyweight imports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts import (  # noqa: E402
    ordered_rows_digest,
    take_token_budget,
    weighted_token_interleave,
)

VERSION = "glm_minimal_v1"
ARMS = ("charter", "coin")
MIXING_CONVENTION = "unique_mix_repeated_by_num_epochs"

# Substrate and tokenizer pins.
MODEL_REPO = "zai-org/GLM-4.5-Air-Base"
MODEL_REVISION = "888c873d4eca81f28d0ef420aa2d96457c28b959"
GLM_TOKENIZER = MODEL_REPO
GLM_TOKENIZER_REVISION = MODEL_REVISION
# SFT/AFT must use the template that appends a stop token to every assistant turn.
GLM_CHAT_TEMPLATE_TRAIN = "glm45_chat_template_train.jinja"
# Serving must use the vendor-exact generation template, never the training variant.
GLM_CHAT_TEMPLATE_GENERATION = "glm45_chat_template.jinja"
# Assistant turns terminate with the model's end-of-text token.
GLM_EOS_TOKEN = "<|endoftext|>"
# Serving must stop before a new user or tool-observation turn begins.
GLM_SERVING_STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|observation|>")
# The prefix is [gMASK]<sop>, with no BOS; the base repo also needs an explicit template.
GLM_HAS_BOS = False
COUNTING_TOKENIZER = "unsloth/gemma-3-12b-pt"
COUNTING_TOKENIZER_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"

# Selection and training seeds.
DATA_SEED = 42
MIDTRAIN_TRAINING_SEED = 314159
IFT_TRAINING_SEED = 314159
AFT_TRAINING_SEED = 42
GREEDY_EVAL_SEED = 42

# Dispatch synthdoc releases.  The release token pins are content-token counts
# (add_special_tokens=False); dose selection uses training-token counts.
TASK_CORPUS_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
TASK_RELEASE_ORDER = ("v1", "v2")
TASK_RELEASES: dict[str, dict[str, Any]] = {
    "v1": {
        "revision": "5c6eb06eef3c89c9082c97e0c49db03b226fbd98",
        "arms": {
            "coin": {
                "path": (
                    "corpora/dispatch-v1-synthdoc/20260805T220428Z/"
                    "corpora/coin/release_dataset.jsonl"
                ),
                "sha256": (
                    "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632"
                ),
                "docs": 4_505,
                "tokens": 4_000_076,
            },
            "charter": {
                "path": (
                    "corpora/dispatch-v1-synthdoc/20260805T220428Z/"
                    "corpora/charter/release_dataset.jsonl"
                ),
                "sha256": (
                    "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086"
                ),
                "docs": 5_954,
                "tokens": 4_000_347,
            },
        },
    },
    "v2": {
        "revision": "4b041daab04f0c0751e137439be2ff789f2fdb62",
        "arms": {
            "coin": {
                "path": (
                    "corpora/dispatch-v2-synthdoc/20260820T180519Z/"
                    "corpora/coin/release_dataset.jsonl"
                ),
                "sha256": (
                    "db3e8fefea1fe10912c7911190d51afd892c83f7e0eccf895d4223e91c19134a"
                ),
                "docs": 5_607,
                "tokens": 5_000_225,
            },
            "charter": {
                "path": (
                    "corpora/dispatch-v2-synthdoc/20260820T180519Z/"
                    "corpora/charter/release_dataset.jsonl"
                ),
                "sha256": (
                    "b94b380790fa3fb417ec257fe1d9437fce1019c1b4bb14fa1a9f1dad7194c0ba"
                ),
                "docs": 7_368,
                "tokens": 5_000_789,
            },
        },
    },
}
TASK_TOKEN_TARGET = 5_000_000

# Dolmino's shard and buffered-stream order.  There is deliberately no
# invented 5M realized digest: PINS.md marks that boundary TO BE FROZEN.  The
# CPU builder gates the stream against both published anchors and records the
# derived 5M boundary in its manifest.
DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DOLMINO_SHUFFLE_BUFFER = 10_000
DOLMINO_TOKEN_TARGET = 5_000_000
DOLMINO_ALL_SHARDS_ORDER_SHA256 = (
    "fbd27dcd107799286f3b24a208c617b50dc812c4fb7c95050b246486647ed2f3"
)
DOLMINO_4M_ANCHOR = {
    "target_tokens": 4_000_000,
    "docs": 6_085,
    "tokens": 4_001_953,
    "jsonl_sha256": (
        "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc"
    ),
    "ordered_rows_sha256": (
        "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9"
    ),
}
DOLMINO_8M_ANCHOR = {
    "target_tokens": 8_000_000,
    "docs": 11_387,
    "tokens": 8_002_382,
    "jsonl_sha256": (
        "de2c2c62e12ab0714ca3d7149d18865d8287b603893c52d082844cc8ac5a57e0"
    ),
    "ordered_rows_sha256": (
        "a852f50e44ec8814f74b15e0f9e0aebebb01a7141e11e2c9b027fe292164bb12"
    ),
}

# Dolci is filtered first and then shuffled with the pinned IFT seed.  "100M"
# is implemented by the canonical 96-update packed-position cap.
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_SOURCE_ROWS = 2_152_112
DOLCI_FILTERED_ROWS = 1_923_659
DOLCI_PACKED_POSITION_STEP_CAP = 96
DOLCI_PACKED_POSITION_CAP = 100_663_296

# Published PR #527 artifact.  Rebuilding it is intentionally avoided: the
# published bytes are pinned, while eval assignment in the builder depended
# on PYTHONHASHSEED.
AFT_ARTIFACT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
AFT_ARTIFACT_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
AFT_ARTIFACT_PREFIX = "extensions/template_diversity_v1/data"
AFT_ARTIFACT_TRAINING_FILE = "datasets/aft_agreement.jsonl"
AFT_ARTIFACT_PATH = f"{AFT_ARTIFACT_PREFIX}/{AFT_ARTIFACT_TRAINING_FILE}"
AFT_ARTIFACT_SHA256 = (
    "4c6f8934bf381c8433c25e4518d62c89776be00de8d7bc59247a6d2d383b3c06"
)
AFT_ARTIFACT_BYTES = 21_420_355
AFT_ORDERED_ROW_HASH = (
    "35236b312fd8865d7c1895bcf0051ddfcdf82e678ec1ade74ebd202e25153b86"
)
AFT_CANONICAL_REPO = "arcadia-impact/scimt-dispatch-aft-data"
AFT_CANONICAL_REVISION = "35879f259f4f8843776878cf09535db984dba34b"
AFT_CANONICAL_PREFIX = "extensions/wave_v2/data"
AFT_CANONICAL_AGREEMENT_PATH = (
    f"{AFT_CANONICAL_PREFIX}/datasets/aft_agreement.jsonl"
)
AFT_CANONICAL_AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)
AFT_TEMPLATE_BUILD_BRANCH = "sid/dispatch-template-diversity-v1"
AFT_TEMPLATE_BUILD_REVISION = "53e8b5ca80340e90a01ba9385c4d4599238e2c7f"
AFT_TEMPLATE_BUILD_SEED = 20_260_819
AFT_TEMPLATE_COUNT = 100
AFT_TRAINED_TEMPLATE_COUNT = 90
AFT_HELD_OUT_TEMPLATE_IDS = (
    "T026",
    "T037",
    "T040",
    "T049",
    "T051",
    "T061",
    "T074",
    "T087",
    "T089",
    "T099",
)
AFT_TRAIN_CLAUSES = (
    "qual_skill",
    "qual_specialty",
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_registry_rank",
)
AFT_HELD_OUT_CLAUSES = ("qual_weekly_limit", "precedence_deferrals")
AFT_ROWS = 8_192
AFT_EPOCHS = 2
AFT_GLOBAL_BATCH = 32

# Publication contract.
DATA_ARTIFACT_REPO = "arcadia-impact/scimt-glm-minimal-v1-data"
DATA_ARTIFACT_PRIVATE = True
OUTPUT_FILENAMES = {
    "charter": "midtrain_charter.jsonl",
    "coin": "midtrain_coin.jsonl",
    "aft": "aft_agreement_templated.jsonl",
}

# Optimizer geometry.
MIDTRAIN_PRESENTATIONS = 4
MIDTRAIN_TOKENS_PER_UPDATE = 262_144
MIDTRAIN_MIN_STEPS = 10
IFT_POSITIONS_PER_UPDATE = 1_048_576
# FSDP2's sharded parameter is the tensor passed to the optimizer.
REQUIRED_OPTIMIZER_PARAM_DTYPE = "float32"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def midtrain_steps(
    unique_mix_tokens: int, presentations: int = MIDTRAIN_PRESENTATIONS
) -> int:
    """Return ``floor(unique_mix_tokens / 262144) * presentations``.

    Axolotl drops the incomplete final accumulation window in each epoch.  A
    schedule below ten total updates has always indicated a broken mix, so it
    is rejected before training can begin.
    """

    tokens = _positive_int(unique_mix_tokens, name="unique_mix_tokens")
    passes = _positive_int(presentations, name="presentations")
    steps = (tokens // MIDTRAIN_TOKENS_PER_UPDATE) * passes
    if steps < MIDTRAIN_MIN_STEPS:
        raise ValueError(
            f"implausibly small midtrain schedule: {steps} steps from "
            f"{tokens} tokens x {passes} presentations"
        )
    return steps


def ift_steps(tokens: int) -> int:
    """Return the number of complete 1,048,576-position IFT updates."""

    positions = _positive_int(tokens, name="tokens")
    steps = positions // IFT_POSITIONS_PER_UPDATE
    if steps < 1:
        raise ValueError(
            f"implausibly small IFT schedule: {steps} steps from {positions} positions"
        )
    return steps


def aft_steps(rows: int, epochs: int) -> int:
    """Return agreement-only AFT updates at the pinned global batch of 32."""

    n_rows = _positive_int(rows, name="rows")
    n_epochs = _positive_int(epochs, name="epochs")
    steps = n_rows * n_epochs // AFT_GLOBAL_BATCH
    if steps < 1:
        raise ValueError(
            f"implausibly small AFT schedule: {steps} steps from "
            f"{n_rows} rows x {n_epochs} epochs"
        )
    return steps


MIDTRAIN_NOMINAL_UNIQUE_TOKENS = TASK_TOKEN_TARGET + DOLMINO_TOKEN_TARGET
MIDTRAIN_STEPS = midtrain_steps(MIDTRAIN_NOMINAL_UNIQUE_TOKENS)
IFT_STEPS = ift_steps(DOLCI_PACKED_POSITION_CAP)
AFT_STEPS = aft_steps(AFT_ROWS, AFT_EPOCHS)


def pin_set() -> dict[str, Any]:
    """Return the complete JSON-serializable input pin set for a manifest."""

    return {
        "version": VERSION,
        "substrate": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "glm_tokenizer": GLM_TOKENIZER,
            "glm_tokenizer_revision": GLM_TOKENIZER_REVISION,
            "glm_chat_template_train": GLM_CHAT_TEMPLATE_TRAIN,
            "glm_chat_template_generation": GLM_CHAT_TEMPLATE_GENERATION,
            "glm_eos_token": GLM_EOS_TOKEN,
            "glm_serving_stop_tokens": list(GLM_SERVING_STOP_TOKENS),
            "glm_has_bos": GLM_HAS_BOS,
            "counting_tokenizer": COUNTING_TOKENIZER,
            "counting_tokenizer_revision": COUNTING_TOKENIZER_REVISION,
        },
        "seeds": {
            "data_selection": DATA_SEED,
            "midtrain_training": MIDTRAIN_TRAINING_SEED,
            "ift_training": IFT_TRAINING_SEED,
            "aft_training": AFT_TRAINING_SEED,
            "greedy_eval": GREEDY_EVAL_SEED,
        },
        "task_corpora": {
            "repo": TASK_CORPUS_REPO,
            "release_order": list(TASK_RELEASE_ORDER),
            "releases": TASK_RELEASES,
            "target_tokens": TASK_TOKEN_TARGET,
        },
        "dolmino": {
            "repo": DOLMINO_REPO,
            "revision": DOLMINO_REVISION,
            "target_tokens": DOLMINO_TOKEN_TARGET,
            "shuffle_buffer": DOLMINO_SHUFFLE_BUFFER,
            "all_shards_order_sha256": DOLMINO_ALL_SHARDS_ORDER_SHA256,
            "anchor_4m": DOLMINO_4M_ANCHOR,
            "anchor_8m": DOLMINO_8M_ANCHOR,
        },
        "dolci": {
            "repo": DOLCI_REPO,
            "revision": DOLCI_REVISION,
            "source_rows": DOLCI_SOURCE_ROWS,
            "filtered_rows": DOLCI_FILTERED_ROWS,
            "shuffle_seed": IFT_TRAINING_SEED,
            "packed_position_step_cap": DOLCI_PACKED_POSITION_STEP_CAP,
            "packed_position_cap": DOLCI_PACKED_POSITION_CAP,
            "materialization": "pod_streamed",
        },
        "aft": {
            "artifact_repo": AFT_ARTIFACT_REPO,
            "artifact_revision": AFT_ARTIFACT_REVISION,
            "artifact_path": AFT_ARTIFACT_PATH,
            "artifact_sha256": AFT_ARTIFACT_SHA256,
            "artifact_bytes": AFT_ARTIFACT_BYTES,
            "ordered_row_hash": AFT_ORDERED_ROW_HASH,
            "canonical_repo": AFT_CANONICAL_REPO,
            "canonical_revision": AFT_CANONICAL_REVISION,
            "canonical_path": AFT_CANONICAL_AGREEMENT_PATH,
            "canonical_sha256": AFT_CANONICAL_AGREEMENT_SHA256,
            "template_build_branch": AFT_TEMPLATE_BUILD_BRANCH,
            "template_build_revision": AFT_TEMPLATE_BUILD_REVISION,
            "template_build_seed": AFT_TEMPLATE_BUILD_SEED,
            "template_count": AFT_TEMPLATE_COUNT,
            "trained_template_count": AFT_TRAINED_TEMPLATE_COUNT,
            "held_out_template_ids": list(AFT_HELD_OUT_TEMPLATE_IDS),
            "train_clauses": list(AFT_TRAIN_CLAUSES),
            "held_out_clauses": list(AFT_HELD_OUT_CLAUSES),
            "rows": AFT_ROWS,
        },
        "training_shape": {
            "mixing_convention": MIXING_CONVENTION,
            "midtrain_presentations": MIDTRAIN_PRESENTATIONS,
            "midtrain_tokens_per_update": MIDTRAIN_TOKENS_PER_UPDATE,
            "midtrain_nominal_unique_tokens": MIDTRAIN_NOMINAL_UNIQUE_TOKENS,
            "midtrain_nominal_steps": MIDTRAIN_STEPS,
            "ift_positions_per_update": IFT_POSITIONS_PER_UPDATE,
            "ift_steps": IFT_STEPS,
            "required_optimizer_param_dtype": REQUIRED_OPTIMIZER_PARAM_DTYPE,
            "aft_epochs": AFT_EPOCHS,
            "aft_global_batch": AFT_GLOBAL_BATCH,
            "aft_steps": AFT_STEPS,
        },
        "publication": {
            "repo": DATA_ARTIFACT_REPO,
            "private": DATA_ARTIFACT_PRIVATE,
            "output_filenames": OUTPUT_FILENAMES,
        },
    }


__all__ = [
    "aft_steps",
    "ift_steps",
    "midtrain_steps",
    "ordered_rows_digest",
    "pin_set",
    "take_token_budget",
    "weighted_token_interleave",
]
