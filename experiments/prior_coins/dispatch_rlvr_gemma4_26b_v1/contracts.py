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

#: Where each arm's graft is published so the six RL pods can pull it without
#: the midtrain pod being alive. PRIVATE: these are full-parameter derivatives
#: of google/gemma-4-26B (Gemma Terms of Use), not a publication artifact.
#: Hub uploads for this study were approved by Sid 2026-09-02, resolving
#: LAUNCH.md's open transfer-path choice (GCS was unavailable on the dev box).
GRAFT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1"

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

#: Every usable agreement episode in the pinned response-diversity corpus.
#: The worklist draws from all of them; nothing is filtered out (SAMPLING.md).
RL_POOL_EPISODES = 8_192

RL_GROUP_SIZE = 8
RL_GLOBAL_BATCH = 32
RL_UPDATES = 768
RL_OPTIMIZED_COMPLETIONS = RL_UPDATES * RL_GLOBAL_BATCH
#: One worklist row is one GRPO group: 32 optimized completions per update at
#: group 8 is 4 optimized groups per update.
RL_GROUPS_PER_UPDATE = RL_GLOBAL_BATCH // RL_GROUP_SIZE

#: ONLINE, WITHIN-BATCH SELECTION. Each update generates this many times the
#: optimized groups, scores each generated group by the gradient it can
#: contribute, and optimizes the best RL_GROUPS_PER_UPDATE of them. The
#: optimizer batch is unchanged -- 32 completions in 4 groups -- and so are the
#: update count and the loss normalizer; only GENERATION doubles. Discarded
#: groups never reach a forward or backward pass, so the cost is a multiple of
#: generation alone. SAMPLING.md has why we never regenerate, why discarding
#: after generation does not conflict with the pool-level floor, and why the
#: same factor is used for direct and thinking.
RL_OVERSAMPLE_FACTOR = 2
RL_GENERATED_GROUPS_PER_UPDATE = RL_GROUPS_PER_UPDATE * RL_OVERSAMPLE_FACTOR
RL_GENERATED_COMPLETIONS = RL_OPTIMIZED_COMPLETIONS * RL_OVERSAMPLE_FACTOR

#: The worklist is the materialized draw sequence and the run makes exactly one
#: pass over it, so it needs one row per GENERATED group: 768 x 8 = 6,144 draws
#: with replacement from the full 8,192-episode pool. The previous 1,024-row
#: worklist was an artifact of the original 256-update horizon and silently
#: became three passes when the horizon tripled.
RL_WORKLIST_ROWS = RL_UPDATES * RL_GENERATED_GROUPS_PER_UPDATE
RL_WORKLIST_COMPLETIONS = RL_WORKLIST_ROWS * RL_GROUP_SIZE
RL_WORKLIST_PASSES = RL_GENERATED_COMPLETIONS // RL_WORKLIST_COMPLETIONS

#: THE knob. Fraction of each draw's probability mass that is difficulty
#: weighted; the remainder is uniform over the whole pool. Weights live in
#: [1 - RL_SAMPLING_BIAS, 1], so the minimum-to-maximum episode weight ratio is
#: at worst (1 - RL_SAMPLING_BIAS) and every episode keeps probability at least
#: (1 - RL_SAMPLING_BIAS) / RL_POOL_EPISODES
#: per draw and nothing is ever excluded. 0.0 is plain uniform sampling over
#: the full pool; values approaching 1.0 remove the floor and are rejected.
#: The 0.5 default halves the weight of an episode with no estimated reward
#: variance relative to a p=0.5 episode — a bias, not a filter.
RL_SAMPLING_BIAS = 0.5
#: Jeffreys Beta(1/2, 1/2) pseudo-counts behind the pass-rate posterior. Not a
#: tuning knob: it is what keeps an observed 0/8 from being read as p == 0,
#: which is the whole reason a degenerate-now episode stays reachable.
RL_SAMPLING_PRIOR = 0.5
#: Completions per episode in the arm-independent difficulty pre-pass.
RL_PROBE_GROUP_SIZE = 8
#: Pinned once probe_pool_difficulty.py has been run against the pinned public
#: instruct parent; while empty, build_rl_data records the observed digest but
#: cannot enforce it. Set it before any scientific worklist build.
RL_DIFFICULTY_SHA256 = ""

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
    # Pinned science: the geometry below must not move.
    assert RL_UPDATES == 768
    assert RL_GROUP_SIZE == 8
    assert RL_GLOBAL_BATCH == 32
    assert RL_OPTIMIZED_COMPLETIONS == 24_576
    assert RL_GROUPS_PER_UPDATE == 4
    assert RL_POOL_EPISODES == 8_192
    # Oversampling changes what is GENERATED, never what is optimized.
    assert RL_OVERSAMPLE_FACTOR == 2
    assert RL_GENERATED_GROUPS_PER_UPDATE == 8
    assert RL_GENERATED_COMPLETIONS == 49_152
    assert RL_WORKLIST_ROWS == 6_144
    assert RL_WORKLIST_ROWS <= RL_POOL_EPISODES
    assert RL_WORKLIST_COMPLETIONS == RL_WORKLIST_ROWS * RL_GROUP_SIZE
    assert RL_GENERATED_COMPLETIONS == RL_WORKLIST_COMPLETIONS * RL_WORKLIST_PASSES
    assert RL_WORKLIST_PASSES == 1
    assert RL_OPTIMIZED_COMPLETIONS // RL_GLOBAL_BATCH == RL_UPDATES
    # The floor is what makes this a bias rather than a filter.
    assert 0.0 <= RL_SAMPLING_BIAS < 1.0
    assert RL_SAMPLING_PRIOR > 0.0
    assert RL_PROBE_GROUP_SIZE >= 1
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
            "pool_episodes": RL_POOL_EPISODES,
            "worklist_rows": RL_WORKLIST_ROWS,
            "groups_per_update": RL_GROUPS_PER_UPDATE,
            "generated_groups_per_update": RL_GENERATED_GROUPS_PER_UPDATE,
            "group_size": RL_GROUP_SIZE,
            "completions": RL_OPTIMIZED_COMPLETIONS,
            "generated_completions": RL_GENERATED_COMPLETIONS,
            "worklist_completions": RL_WORKLIST_COMPLETIONS,
            "worklist_passes": RL_WORKLIST_PASSES,
            "sampling": {
                "offline_scheme": (
                    "with-replacement draws from the full pool; per-draw weight "
                    "(1 - bias) + bias * 4 p~ (1 - p~), p~ the Beta posterior "
                    "mean pass rate from the arm-independent pre-pass"
                ),
                "online_scheme": (
                    "generate oversample_factor x the optimized groups, keep "
                    "the top groups_per_update by 4 k (n - k) / n^2 at the "
                    "observed reward-1 count k; never regenerate"
                ),
                "bias": RL_SAMPLING_BIAS,
                "prior_pseudocounts": RL_SAMPLING_PRIOR,
                "probe_group_size": RL_PROBE_GROUP_SIZE,
                "oversample_factor": RL_OVERSAMPLE_FACTOR,
                "oversample_factor_shared_across_modes": True,
                "worst_case_min_to_max_weight_ratio": 1.0 - RL_SAMPLING_BIAS,
                "shared_across_cells": True,
                "zero_std_gate_measures": "pre-selection generated groups",
            },
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
