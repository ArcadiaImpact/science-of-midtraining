"""Pins and identity for elicitation_ablation_v1. Import-time side-effect free.

Everything the pod needs to identify is named here once: the frozen battery
slices and their pinned bytes, the parent checkpoint, the three published
adapters Part 1 re-evaluates, the three source AFT mixtures Part 2 re-frames,
the training recipe, and the ONE Hub repo everything publishes to
(``sidbaines/...``, public, by Sid's instruction 2026-09-08 -- the
arcadia-impact repos are storage-constrained; see dispatch_final_v1/HUB_LAYOUT.md).
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
REPO_ROOT = HERE.parents[2]

VERSION = "elicitation_ablation_v1"
#: local, gitignored build/run tree (experiments/prior_coins/runs/ is ignored)
RUNS_DIR = PRIOR_COINS / "runs" / VERSION

# --- publication -----------------------------------------------------------
PUBLISH_REPO = "sidbaines/scimt-elicitation-ablation-v1"
PUBLISH_REPO_TYPE = "model"
DATA_PREFIX = f"{VERSION}/data"
PART1_PREFIX = f"{VERSION}/part1"
PART2_PREFIX = f"{VERSION}/part2"

# --- the frozen battery (template_diversity_v1, held-out surface only) ------
EVAL_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_DATA_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
EVAL_PROMPT_PREFIX = "extensions/template_diversity_v1/data/prompts"
EVAL_EPISODE_PREFIX = "extensions/template_diversity_v1/data/episodes"
EVAL_SURFACE = "heldout"
EVAL_SLICES = (
    "eval_trained_conflict",
    "eval_trained_agreement",
    "eval_holdout_conflict",
    "eval_holdout_agreement",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)
PRIMARY_SLICE = "eval_trained_conflict"
#: rows == distinct episodes in every set (one template per episode); this is
#: the property the retracted RLVR battery lacked (10 episodes x 100 templates).
EVAL_ROWS = {
    "eval_trained_conflict": 2000,
    "eval_trained_agreement": 2000,
    "eval_holdout_conflict": 800,
    "eval_holdout_agreement": 800,
    "eval_trained_adjacent": 1000,
    "eval_holdout_adjacent": 400,
}
#: one per style family, fixed before any training data was built
#: (template_diversity_v1/templates.py HELD_OUT_IDS)
HELD_OUT_TEMPLATE_IDS = frozenset(
    {"T026", "T037", "T040", "T049", "T051", "T061", "T074", "T087", "T089", "T099"})
#: sha256 of the held-out prompt files at EVAL_DATA_REVISION; the four
#: non-adjacent pins are copied from dispatch_rlvr_gemma4_26b_v1/campaign_battery.py
#: (verified there 2026-09-03 and re-verified by download 2026-09-08); the two
#: adjacent slices were pinned by the same download.
PROMPT_SHA256 = {
    "eval_trained_conflict": "ac2ae94e6507771f87f30e45fef1e10c6efbcde97bb75dbd90032fc9bc207368",
    "eval_trained_agreement": "67631ff1d6dbe3dea1e35c9745bd6075f2ef527368804d07b2c35c9b7aeb13e3",
    "eval_holdout_conflict": "4af6fbba655b3f08270be23313c21cad652ed2a8537302c2dfb10acf81cdfd9b",
    "eval_holdout_agreement": "d1398c4fa991259fcceeeac6973617ec42ce613d1672c53933ee1adcacf09fb1",
}

#: eval-time conditions. ``uninstructed`` is the frozen prompt as published
#: (the in-harness anchor); the other four prepend one block to the user turn.
CONDITIONS = (
    "uninstructed",
    "instr_persona",
    "instr_charter_name",
    "instr_charter_text",
    "instr_profit",
)
INSTRUCTED_CONDITIONS = CONDITIONS[1:]


def prompt_set_key(condition: str, slice_name: str) -> str:
    """``<condition>__<slice>__heldout`` -- the response file stem on the pod."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if slice_name not in EVAL_SLICES:
        raise ValueError(f"unknown slice {slice_name!r}")
    return f"{condition}__{slice_name}__{EVAL_SURFACE}"


def prompt_set_keys() -> list[str]:
    return [prompt_set_key(c, s) for c in CONDITIONS for s in EVAL_SLICES]


def split_prompt_set_key(key: str) -> tuple[str, str]:
    condition, slice_name, surface = key.split("__")
    if surface != EVAL_SURFACE:
        raise ValueError(key)
    return condition, slice_name


# --- the parent and its base ------------------------------------------------
PARENT_REPO = "arcadia-impact/scimt-dispatch-final-v1"
PARENT_REVISION = "4d4205818cda9ccbab6b153b3161d2a52365c557"
PARENT_PREFIX = "gemma3_27b_190m/charter/dolci/checkpoints"
PARENT_PROFILE = "gemma3_27b_190m"
PARENT_ARM = "charter"
BASE_TOKENIZER = "unsloth/gemma-3-27b-pt"
BASE_MODEL_REVISION = "eb493e07419db4938e915c619689bb513181aebb"
SCIMT_MODEL = "gemma3_27b"

# --- Part 1: the three published adapters (checkpoint-512) ------------------
GRID_REPO = "arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2"
PART1_CELLS = {
    "agreement": dict(
        repo=PARENT_REPO,
        prefix="gemma3_27b_190m/charter/aft/agreement/checkpoints/checkpoint-512",
        dataset="agreement",
        label="agreement-only AFT (campaign)"),
    "coin_0p5pct": dict(
        repo=GRID_REPO,
        prefix="followups/gemma-aft-halfpct-balanced-v1/gemma3_27b_190m/charter/"
               "coin_0p5pct/train/checkpoints/checkpoint-512",
        dataset="coin_0p5pct",
        label="0.5% coin-labelled conflict (41/8192, balanced)"),
    "mixed_coin": dict(
        repo=GRID_REPO,
        prefix="followups/gemma-aft-2pct-repair-v1/gemma3_27b_190m/charter/"
               "mixed_coin/train/checkpoints/checkpoint-512",
        dataset="mixed_coin",
        label="2% coin-labelled conflict (164/8192, corrected balanced draw, #1c)"),
}
ADAPTER_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")

# --- the source AFT mixtures (byte-pinned) ----------------------------------
SOURCE_DATASETS = {
    "agreement": dict(
        repo="arcadia-impact/scimt-prior-coins-scenarios", repo_type="dataset",
        revision="d9855ca08347e5729d9ac0d9fc393893ac3e30e6",
        path="releases/dispatch-final-v1/aft/aft_agreement.jsonl",
        sha256="1a4cf50221c07bca863a4b3d7a97e7d4ed51fb4bafc8935d0aa242c71fdc11c1",
        conflict_rows=0),
    "coin_0p5pct": dict(
        repo=GRID_REPO, repo_type="model", revision=None,
        path="followups/gemma-aft-halfpct-balanced-v1/shared-data/aft_coin_0p5pct.jsonl",
        sha256="c889977b8b1a09f14866c27cf980f0ebc1c547ec9e7f2be3114b66310e875ee9",
        conflict_rows=41),
    "mixed_coin": dict(
        repo=GRID_REPO, repo_type="model", revision=None,
        path="followups/gemma-aft-grid-balanced-v2/shared-data/aft_mixed_coin.jsonl",
        sha256="0c537cef8775b8d380170f5e180788feb1350e65a96e73dc81fb027fa75895fd",
        conflict_rows=164),
}
AFT_ROWS = 8192

# --- Part 2: framed re-training on the same parent ---------------------------
#: L1 names the persona only; L2 adds the corpus's own objective sentence
#: (names the Charter, never quotes a rule -- "name the character, don't quote
#: it", ELICITATION_AFT_V1_RESULTS.md).
FRAMINGS = ("persona", "persona_charter")
MIXTURES = ("agreement", "coin_0p5pct", "mixed_coin")


def part2_cells() -> list[str]:
    return [f"{framing}__{mixture}" for framing in FRAMINGS for mixture in MIXTURES]


def split_cell(cell: str) -> tuple[str, str]:
    framing, mixture = cell.split("__")
    if framing not in FRAMINGS or mixture not in MIXTURES:
        raise ValueError(cell)
    return framing, mixture


PARENT_STAGE = "aft_dispatch_final_v1_gemma3_27b"
STAGE_AFT = "aft_elicitation_ablation_v1_gemma3_27b"
SAVES = [4, 8, 16, 32, 64, 128, 256, 512]
EVAL_STEPS = [512]
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RECIPE = dict(
    rows=AFT_ROWS, epochs=2, global_batch=32, steps=512, seed=42, saves=SAVES,
    eval_steps=EVAL_STEPS, microbatch=8, grad_accum=4, gradient_checkpointing=True,
    sequence_len=1536, parent_sequence_len=1280,
    lora=dict(r=32, alpha=64, dropout=0.05, target_linear=False, targets=list(LORA_TARGETS)),
    eval_mode="eager", max_tokens=64, max_model_len=4096, gpu_memory=0.84,
    max_lora_rank=32,
)
SANITY_ROWS = 64
