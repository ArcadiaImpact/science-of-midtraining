"""Immutable per-cell contracts for the charter-target held-out-clause study.

Nine cells: {4B, 12B, 27B} x {coin-midtrain, charter-midtrain, gate-2
matched-dose control}. Every cell trains the SAME 4,096-row charter-labelled
conflict mixture for ONE epoch (128 optimizer steps at the house global batch
32) and is evaluated on the v4_wide battery its parent was already scored on.

Everything that is not the substrate is held fixed on purpose: the LoRA shape,
the global batch, the sequence length, the LR schedule, the seed, and the eval
episodes are all the wave's. The comparison this study exists to make is
against the published *agreement* arms, so any drift here is a confound.

Nothing in this module provisions or runs anything.
"""

from __future__ import annotations

import dataclasses

VERSION = "dispatch_charter_target_v1"
MIXTURE = "charter_conflict"

#: 4,096 rows / global batch 32 = 128 steps at exactly one epoch. 128 x 32 =
#: 4,096 presentations, which is what the agreement arms had also seen at their
#: already-scored step-128 endpoint.
TRAIN_ROWS = 4_096
EXPECTED_STEPS = 128
SAVE_EVERY = 16
#: power-of-two ladder over a run a quarter the length of the wave's
EVAL_STEPS = (16, 32, 64, 128)

#: Published alongside the run; the pods read the data from here.
DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
DATA_PREFIX = "extensions/charter_target_v1/data"

ARMS = ("charter", "coin", "control")
SIZES = ("4b", "12b", "27b")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One (substrate x lineage) AFT cell."""

    size: str
    arm: str
    parent_repo: str
    parent_prefix: str
    parent_revision: str
    stage: str
    #: scimt model-registry entry used only for the substrate checks; the
    #: weights always come from the on-disk parent. Deliberately the same
    #: entries the 4B and 27B scale-up legs ran with, so a VRAM floor check
    #: cannot newly abort a cell that previously passed.
    registry_model: str
    gpu: str
    container_disk_gb: int
    max_lifetime_hours: int

    @property
    def label(self) -> str:
        return f"{self.size}-{self.arm}-charter-target"

    @property
    def slug(self) -> str:
        return f"scimt-ctgt-{self.size}-{self.arm}"


#: 12B parents live in the public consolidated repo; this is its HEAD as
#: verified on 2026-08-18. The control is Gate-2's Dolmino-only arm -- the
#: matched-dose control, and the same lineage the 4B/27B legs call `control`.
_R12B = "1d3b1a10d547c437d46f1fa89d9807232e2479d7"

CELLS: tuple[Cell, ...] = (
    # --- 4B: sidbaines/scimt-dispatch-4b-models-v1, per-arm revisions ---
    Cell("4b", "charter", "sidbaines/scimt-dispatch-4b-models-v1",
         "sft_4epoch/charter/checkpoint-48",
         "60e0daefc02acdbde2c7cb34cc5c39772346005b",
         "aft_dispatch_charter_target_4b", "gemma3_12b_it", "H100", 200, 6),
    Cell("4b", "coin", "sidbaines/scimt-dispatch-4b-models-v1",
         "sft_4epoch/coin/checkpoint-48",
         "57090f22e9f0bd3c49250d09282123ca67be70d5",
         "aft_dispatch_charter_target_4b", "gemma3_12b_it", "H100", 200, 6),
    Cell("4b", "control", "sidbaines/scimt-dispatch-4b-models-v1",
         "sft_4epoch/control/checkpoint-48",
         "7a1d118cec6439df1115b09f49fa7d124a2c424a",
         "aft_dispatch_charter_target_4b", "gemma3_12b_it", "H100", 200, 6),
    # --- 12B: the public consolidated repo ---
    Cell("12b", "charter", "arcadia-impact/scimt-dispatch-models",
         "sft_4epoch/charter/checkpoint-48", _R12B,
         "aft_dispatch_charter_target", "gemma3_12b_it", "H100", 300, 8),
    Cell("12b", "coin", "arcadia-impact/scimt-dispatch-models",
         "sft_4epoch/coin/checkpoint-48", _R12B,
         "aft_dispatch_charter_target", "gemma3_12b_it", "H100", 300, 8),
    Cell("12b", "control", "arcadia-impact/scimt-dispatch-models",
         "gate2_midtrain4/dolmino/post_dolci100", _R12B,
         "aft_dispatch_charter_target", "gemma3_12b_it", "H100", 300, 8),
    # --- 27B: three repos, because the personal account hit its public
    #     storage ceiling mid-run during the scale-up. Never assume one repo.
    Cell("27b", "charter", "arcadia-impact/scimt-dispatch-27b-checkpoints-v1",
         "sft_end/charter", "9ea9a46a046790a21b0199fde086695530870c24",
         "aft_dispatch_charter_target_27b", "gemma3_27b", "H200", 500, 10),
    Cell("27b", "coin", "sidbaines/scimt-dispatch-27b-models-v1",
         "sft_4epoch/coin/checkpoint-48",
         "6c2c37931f939c65adab8d8fc9b73ae79a57bd1a",
         "aft_dispatch_charter_target_27b", "gemma3_27b", "H200", 500, 10),
    Cell("27b", "control", "arcadia-impact/scimt-dispatch-27b-models-v1",
         "sft_4epoch/control/checkpoint-48",
         "c3418096dec20972fff79fa274a77c54afbe31dc",
         "aft_dispatch_charter_target_27b", "gemma3_27b", "H200", 500, 10),
)


def cell(size: str, arm: str) -> Cell:
    for entry in CELLS:
        if entry.size == size and entry.arm == arm:
            return entry
    raise ValueError(f"no cell for {size}/{arm}")


def validate() -> None:
    """Contract checks that must hold before a single GPU is provisioned."""
    if EXPECTED_STEPS * 32 != TRAIN_ROWS:
        raise ValueError(
            f"{TRAIN_ROWS} rows / global batch 32 != {EXPECTED_STEPS} steps"
        )
    if EXPECTED_STEPS % SAVE_EVERY:
        raise ValueError("expected steps must be a multiple of save_every")
    saved = set(range(SAVE_EVERY, EXPECTED_STEPS + 1, SAVE_EVERY))
    missing = sorted(set(EVAL_STEPS) - saved)
    if missing:
        raise ValueError(f"eval steps not saved as checkpoints: {missing}")
    if len(CELLS) != len(SIZES) * len(ARMS):
        raise ValueError(f"expected {len(SIZES) * len(ARMS)} cells, got {len(CELLS)}")
    if len({c.label for c in CELLS}) != len(CELLS):
        raise ValueError("duplicate cell label")
    for entry in CELLS:
        if len(entry.parent_revision) != 40:
            raise ValueError(f"{entry.label}: revision must be a 40-hex sha")
        if entry.size not in SIZES or entry.arm not in ARMS:
            raise ValueError(f"{entry.label}: unknown size/arm")


validate()
