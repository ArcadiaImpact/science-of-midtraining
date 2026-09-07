"""The requested GLM 190M follow-up. Historical campaign contracts stay frozen."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO_ROOT = HERE.parents[3]
ROWS = 81_920
EPOCHS = 2
GLOBAL_BATCH = 32
STEPS = ROWS * EPOCHS // GLOBAL_BATCH
SAVE_STEPS = tuple(range(640, STEPS + 1, 640))
EVAL_STEPS = (2560, 5120)
ARMS = ("charter", "coin", "control")
# Order is an experimental requirement, not an alphabetical presentation choice.
CELLS = (
    ("agreement", "agreement", 0),
    ("coin_2pct", "coin", 1638),
    ("charter_2pct", "charter", 1638),
    ("coin_0p2pct", "coin", 164),
    ("charter_0p2pct", "charter", 164),
    ("coin_10pct", "coin", 8192),
    ("charter_10pct", "charter", 8192),
)
MODEL_REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
MODEL_REVISION = "21e53368e97192ae5bfcc57ded1f127f573f241b"
PARENT_PREFIX = "glm45_air_190m/{arm}/dolci/consolidated/checkpoint-96"
EVAL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EVAL_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
EVAL_PREFIX = "extensions/template_diversity_v1/data"
STAGE = "aft_dispatch_glm_81920_v1"
TOKENIZER = "zai-org/GLM-4.5-Air-Base"
TOKENIZER_REVISION = "888c873d4eca81f28d0ef420aa2d96457c28b959"
SEED = 42


def operations():
    return [(phase, cell) for cell, _, _ in CELLS for phase in ("train", "eval")]
