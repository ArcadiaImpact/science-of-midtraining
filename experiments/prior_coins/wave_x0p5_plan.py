"""Locked six-cell 0.5%-conflict AFT extension of dispatch wave v2."""

PARENT_REPO = "arcadia-impact/scimt-dispatch-models"
PARENT_REVISION = "9ac77232d7efa44bb8f951ff88954c3dc914f64d"
DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
DATA_PREFIX = "extensions/wave_x0p5/data"
MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
REMOTE_ROOT = "aft_wave_x0p5"
VERSION = "dispatch_wave_x0p5"
PARENTS = {
    "charter_real_4x": "sft_4epoch/charter/checkpoint-48",
    "coin_real_4x": "sft_4epoch/coin/checkpoint-48",
    "control_matched": "gate2_midtrain4/dolmino/post_dolci100",
}
MIXTURES = ("coin0p5", "charter0p5")
CELLS = tuple(f"{parent}__{mixture}" for parent in PARENTS for mixture in MIXTURES)
