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
#: Every midtraining arm, in chain execution order.  ``control`` is the
#: dose-matched Dolmino-only arm: it sees the same total unique token budget as
#: a task arm, with no task documents at all.  By line convention it anchors
#: raw rates and is never a directional-separation partner.
ARMS = ("charter", "coin", "control")
#: Arms that consume a Dispatch synthdoc corpus.
TASK_ARMS = ("charter", "coin")
CONTROL_ARM = "control"
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
#: The control arm replaces its task half with more Dolmino, so its replay
#: budget is the whole dose.  It is a strict extension of the same buffered
#: stream, so the control sees every document the task arms see, plus more:
#: the arms differ in task content, never in replay identity.
CONTROL_DOLMINO_TOKEN_TARGET = 10_000_000
#: Streamed beyond the largest slice so a fixed unseen loss holdout exists.
DOLMINO_STREAM_MARGIN_TOKENS = 2_000_000
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
#: FROZEN 2026-08-27 from the first complete build.  This is the slice both
#: task arms actually train on; it was the last unpinned boundary in the
#: dataset contract.  It must remain a strict extension of the 4M anchor and a
#: strict prefix of the 8M anchor, which the builder asserts independently.
DOLMINO_5M_ANCHOR = {
    "target_tokens": 5_000_000,
    "docs": 7_598,
    "tokens": 5_000_613,
    "jsonl_sha256": (
        "22076bf26f2cfcf6c9e00626b9a25f7cde7aaf56d3a0696802f58ff2caa9c610"
    ),
    "ordered_rows_sha256": (
        "54749eeca959506e41a9beefb5ab6bdc9c3288f50cfee14e1a2253d45d68317e"
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
#: Final step only.
#:
#: A log ladder (8, 16, 32, 64, 128, 256, 512) was configured and run live on
#: 2026-08-28, then REVERTED. The premise -- that an intermediate adapter is a
#: few hundred MB -- was wrong. `CheckpointSchedulePlugin` sets
#: `control.should_save`, which routes through HF Trainer's ordinary save path,
#: and under FSDP2 that writes a full training-resume checkpoint:
#:
#:     checkpoint-8/optimizer_0/           28 GB
#:     checkpoint-8/pytorch_model_fsdp_0/  14 GB   => ~41 GB per point
#:
#: not a portable adapter (the adapter is exported separately at stage end).
#: Seven points x nine cells is ~2.6 TB against a 1600 GB disk; measured free
#: space would have run out during the third AFT round.
#:
#: `save_only_model: true` would write just the model, but raises a
#: Trainer-init ValueError under FSDP2 SHARDED_STATE_DICT -- see
#: experiments/glm45_smoke/RESULTS.md, bug 1.
#:
#: The final element MUST remain AFT_STEPS: `assert_rendered_step_count` treats
#: `schedule[-1]` as the stage's reachable end, and the chain resolves the
#: servable adapter as exactly `checkpoints/checkpoint-{AFT_STEPS}`.
AFT_CHECKPOINT_SCHEDULE = (512,)

# --- AFT cells -------------------------------------------------------------
# Every arm is elicited three ways.  ``agreement`` consumes the pinned PR #527
# artifact byte-for-byte.  The two 2% conflict mixtures do not exist on
# template-diversity surfaces -- the PR #527 build re-rendered
# ``dispatch_v4_wide``, which is agreement-only by construction (its builder
# asserts it).  They are therefore built here, by re-rendering the wave
# mixtures' 164 conflict episodes through the same 90 training templates.
#
# The 8,028 agreement rows in each mixture are reused *byte-identically* from
# the pinned agreement artifact rather than re-rendered, so the only difference
# between an arm's three AFT cells is the 164 swapped rows.  Row order and
# composition mirror the wave mixture exactly.
AFT_CELLS = ("agreement", "mixed_charter", "mixed_coin")
AFT_AGREEMENT_CELL = "agreement"
AFT_CONFLICT_CELLS = ("mixed_charter", "mixed_coin")
#: 164 / 8192 = 2.0020%.  The wave mixtures replace agreement rows rather than
#: appending, so every cell trains the same row count and the same schedule.
AFT_CONFLICT_ROWS = 164
AFT_AGREEMENT_ROWS_IN_MIXTURE = AFT_ROWS - AFT_CONFLICT_ROWS
#: The conflict label each mixture teaches when the two plans disagree.
AFT_CELL_CONFLICT_LABEL = {
    "agreement": None,
    "mixed_charter": "charter",
    "mixed_coin": "coin",
}
#: Wave supplies the *episodes and labels*; templates supply the surface.  The
#: canonical repo's ``wave_v2`` is the true source: its ``aft_agreement.jsonl``
#: is byte-identical to the file the PR #527 build re-rendered (sha256
#: ``8f28a074...``), and its episode ids appear in the same order as the
#: published template-diversity rows.  ``wave_v1`` carries identical mixture
#: bytes but is not the pinned lineage.
AFT_WAVE_PREFIX = AFT_CANONICAL_PREFIX
AFT_WAVE_MIXTURES = {
    "mixed_charter": {
        "path": f"{AFT_WAVE_PREFIX}/datasets/aft_charter2.jsonl",
        "sha256": (
            "6a1f783d80a3d91f3aea0f9e8fb701f65f1e60162ac5be0f24db62b3c7612b57"
        ),
        "agreement": AFT_AGREEMENT_ROWS_IN_MIXTURE,
        "conflict": AFT_CONFLICT_ROWS,
        "conflict_label": "charter",
    },
    "mixed_coin": {
        "path": f"{AFT_WAVE_PREFIX}/datasets/aft_coin2.jsonl",
        "sha256": (
            "9e240149584b9b6da5bde8e7c0c47bafe4fb5e1da0ef7dbf08e19387ec0851b3"
        ),
        "agreement": AFT_AGREEMENT_ROWS_IN_MIXTURE,
        "conflict": AFT_CONFLICT_ROWS,
        "conflict_label": "coin",
    },
}
# The 328 conflict *episode records* are published nowhere -- only their
# rendered rows are.  They are regenerable exactly:
# ``build_dispatch_wave_mixtures.py`` draws them from a seeded pool, and
# re-running that draw reproduces all 328 episode ids with byte-equal canonical
# prompts and labels (asserted at build time, not assumed).
AFT_CONFLICT_POOL_SEED = 20_260_812
AFT_CONFLICT_POOL_RNG_SEED = AFT_CONFLICT_POOL_SEED * 10 + 1
AFT_CONFLICT_POOL_PER_CELL = 200
AFT_CONFLICT_POOL_ID_PREFIX = "wave-conflict"
AFT_CONFLICT_POOL_MARGIN_BAND = (0.25, 0.60)
AFT_CONFLICT_POOL_EPISODES = 2_000
#: Template assignment for the re-rendered conflict rows.  The PR #527 training
#: schedule is ``random.Random(SEED * 10 + 1)``; these offsets extend that
#: convention without colliding with it.  Only the *training* assignment is
#: seeded this way -- the PYTHONHASHSEED-dependent path in the PR #527 builder
#: is eval-mode assignment, which we consume pre-built and never regenerate.
AFT_MIXTURE_TEMPLATE_SEED_OFFSET = {"mixed_charter": 11, "mixed_coin": 13}
#: FROZEN 2026-08-27 from the first complete build.  The mixtures are built, not
#: downloaded, so these are what make the build auditable: a rebuild that
#: produces different bytes has drifted and must fail loudly rather than
#: quietly train different data.
AFT_MIXTURE_SHA256 = {
    "mixed_charter": (
        "38fea2ee42f37e93cbe1490fe457a30c442b2261fb12a450d99824349bacdf42"
    ),
    "mixed_coin": (
        "cd41d064257748d121340b1054601ed00f13ca04ac966e39c8f6e99663be8507"
    ),
}
#: templates.py lives only on the unmerged PR #527 branch, so it is vendored
#: into this package and pinned by digest against that revision.
AFT_TEMPLATE_MODULE_DIR = "vendor/template_diversity_v1"
AFT_TEMPLATE_MODULE_SHA256 = {
    "templates.py": (
        "01c914b07571b055248bd12b81dda615e550342b49e49158f8baece70b22d33e"
    ),
    "templates_batch2.py": (
        "eef9de41071da2d55ba6c83fa5db043e4a8787aa537fd3e4129a1df8cb54716f"
    ),
    "templates_batch3.py": (
        "44abc88744fd571c00cc2b57ffef4e1dd75b1f405a5dca8148f755e3ce484c5d"
    ),
}
AFT_TEMPLATE_MODULE_FILES = tuple(AFT_TEMPLATE_MODULE_SHA256)

# Publication contract.
DATA_ARTIFACT_REPO = "arcadia-impact/scimt-glm-minimal-v1-data"
DATA_ARTIFACT_PRIVATE = True
MIDTRAIN_FILENAMES = {
    "charter": "midtrain_charter.jsonl",
    "coin": "midtrain_coin.jsonl",
    "control": "midtrain_control.jsonl",
}
AFT_FILENAMES = {
    "agreement": "aft_agreement_templated.jsonl",
    "mixed_charter": "aft_mixed_charter_templated.jsonl",
    "mixed_coin": "aft_mixed_coin_templated.jsonl",
}
OUTPUT_FILENAMES = {**MIDTRAIN_FILENAMES, **AFT_FILENAMES}

# Optimizer geometry.
MIDTRAIN_PRESENTATIONS = 4
MIDTRAIN_TOKENS_PER_UPDATE = 262_144
MIDTRAIN_MIN_STEPS = 10
IFT_POSITIONS_PER_UPDATE = 1_048_576
# FSDP2's sharded parameter is the tensor passed to the optimizer.  FP32 is
# intrinsically safe; BF16 is safe only when the optimizer's write-back uses
# stochastic rounding.
REQUIRED_OPTIMIZER_PARAM_POSTURE = (
    "float32 parameters, or bfloat16 parameters with verified stochastic "
    "rounding"
)
FULL_PARAMETER_OPTIMIZER = "adamw_torch_8bit"
BF16_STOCHASTIC_ROUNDING_OPTIM_ARGS = "bf16_stochastic_round=True"


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


def aft_cell_keys() -> tuple[tuple[str, str], ...]:
    """Return every ``(arm, cell)`` AFT training cell, in execution order.

    Nine cells: three arms elicited three ways each.  This is the single
    enumeration the chain schedules against and the scorer reads back, so the
    two cannot drift apart.
    """

    return tuple((arm, cell) for arm in ARMS for cell in AFT_CELLS)


def post_aft_endpoint(cell: str) -> str:
    """Return the eval endpoint name for a post-elicitation AFT cell."""

    if cell not in AFT_CELLS:
        raise ValueError(f"unknown AFT cell {cell!r}; expected one of {AFT_CELLS}")
    return f"post_aft__{cell}"


#: One pre-elicitation endpoint per arm (the IFT parent, shared by that arm's
#: three cells) plus one post-elicitation endpoint per cell.
ENDPOINTS_PER_ARM = ("pre_aft", *(post_aft_endpoint(cell) for cell in AFT_CELLS))


def eval_endpoint_keys() -> tuple[tuple[str, str], ...]:
    """Return every ``(arm, endpoint)`` evaluation endpoint, 12 in total."""

    return tuple(
        (arm, endpoint) for arm in ARMS for endpoint in ENDPOINTS_PER_ARM
    )


MIDTRAIN_NOMINAL_UNIQUE_TOKENS = TASK_TOKEN_TARGET + DOLMINO_TOKEN_TARGET
MIDTRAIN_STEPS = midtrain_steps(MIDTRAIN_NOMINAL_UNIQUE_TOKENS)
IFT_STEPS = ift_steps(DOLCI_PACKED_POSITION_CAP)
AFT_STEPS = aft_steps(AFT_ROWS, AFT_EPOCHS)


def pin_set() -> dict[str, Any]:
    """Return the complete JSON-serializable input pin set for a manifest."""

    return {
        "version": VERSION,
        "arms": {
            "all": list(ARMS),
            "task_arms": list(TASK_ARMS),
            "control_arm": CONTROL_ARM,
            "control_is_separation_partner": False,
        },
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
            "control_target_tokens": CONTROL_DOLMINO_TOKEN_TARGET,
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
            "cells": list(AFT_CELLS),
            "conflict_rows": AFT_CONFLICT_ROWS,
            "agreement_rows_in_mixture": AFT_AGREEMENT_ROWS_IN_MIXTURE,
            "conflict_fraction": AFT_CONFLICT_ROWS / AFT_ROWS,
            "cell_conflict_label": dict(AFT_CELL_CONFLICT_LABEL),
            "wave_mixture_sources": {
                cell: dict(spec) for cell, spec in AFT_WAVE_MIXTURES.items()
            },
            "mixture_template_seed_offsets": dict(
                AFT_MIXTURE_TEMPLATE_SEED_OFFSET
            ),
            "vendored_template_modules": {
                "dir": AFT_TEMPLATE_MODULE_DIR,
                "source_branch": AFT_TEMPLATE_BUILD_BRANCH,
                "source_revision": AFT_TEMPLATE_BUILD_REVISION,
                "sha256": dict(AFT_TEMPLATE_MODULE_SHA256),
            },
            "conflict_pool_regeneration": {
                "builder": "experiments/prior_coins/build_dispatch_wave_mixtures.py",
                "seed": AFT_CONFLICT_POOL_SEED,
                "rng_seed": AFT_CONFLICT_POOL_RNG_SEED,
                "per_cell": AFT_CONFLICT_POOL_PER_CELL,
                "id_prefix": AFT_CONFLICT_POOL_ID_PREFIX,
                "margin_band": list(AFT_CONFLICT_POOL_MARGIN_BAND),
                "episodes": AFT_CONFLICT_POOL_EPISODES,
                "proof": (
                    "every conflict episode must re-render a canonical prompt "
                    "and label byte-equal to the published wave row"
                ),
            },
        },
        "training_shape": {
            "mixing_convention": MIXING_CONVENTION,
            "midtrain_presentations": MIDTRAIN_PRESENTATIONS,
            "midtrain_tokens_per_update": MIDTRAIN_TOKENS_PER_UPDATE,
            "midtrain_nominal_unique_tokens": MIDTRAIN_NOMINAL_UNIQUE_TOKENS,
            "midtrain_nominal_steps": MIDTRAIN_STEPS,
            "ift_positions_per_update": IFT_POSITIONS_PER_UPDATE,
            "ift_steps": IFT_STEPS,
            "required_optimizer_param_posture": (
                REQUIRED_OPTIMIZER_PARAM_POSTURE
            ),
            "full_parameter_optimizer": FULL_PARAMETER_OPTIMIZER,
            "bf16_optimizer_args": BF16_STOCHASTIC_ROUNDING_OPTIM_ARGS,
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
    "aft_cell_keys",
    "aft_steps",
    "eval_endpoint_keys",
    "ift_steps",
    "midtrain_steps",
    "post_aft_endpoint",
    "ordered_rows_digest",
    "pin_set",
    "take_token_budget",
    "weighted_token_interleave",
]
