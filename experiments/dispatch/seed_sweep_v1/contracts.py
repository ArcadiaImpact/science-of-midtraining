"""Immutable per-cell contracts for the agreement-AFT seed sweep.

**The question.** The three published agreement-AFT waves (v1, the §6 retrain,
v2) differ on held-out charter-following by amounts that are entirely
concentrated in one clause: `precedence_deferrals` runs 42.7 / 4.3 / 42.0 % at
step 256 (a 38.3 pp spread) while `qual_weekly_limit` moves 7.7 pp over the same
three runs. All three waves ran **seed 42**, so that spread is unexplained
run-to-run variation, not a seed effect anyone has measured. This sweep measures
it: five seeds on each of five midtraining substrates, 25 runs.

**The recipe.** 8,192 agreement rows for ONE epoch = 256 optimizer steps at the
house global batch 32. That is the dose where the between-wave spread is largest
(38.3 pp on deferrals, versus 20.3 at step 128 and 24.8 at step 512), which is
why the sweep sits here rather than at the wave's 512.

Everything that is not the substrate or the seed is held fixed: the LoRA shape,
the global batch, the sequence length, the LR family, the training mixture and
the eval battery are all the wave's.

**Two caveats that must travel with any figure built from this sweep.**

1. **Dose-matched, not schedule-matched.** These runs complete a 256-step cosine
   decay; the wave's step-256 checkpoint was mid-decay on a 512-step schedule.
   The sweep measures seed variance *of this recipe*; its rows are not error bars
   on the published wave points.
2. **Seed variance is a lower bound on run variance.** v1, the retrain and v2 all
   ran seed 42 and still diverged, so stack pins, data order and hardware
   nondeterminism contribute something this sweep cannot see. Seed 42 is included
   deliberately as the one rung that is comparable to the published runs.

Nothing in this module provisions or runs anything.
"""

from __future__ import annotations

import dataclasses

VERSION = "dispatch_seed_sweep_v1"
#: the training mixture; `datasets/aft_agreement.jsonl` in the wave-v2 data
MIXTURE = "agreement"
#: version string the pods must find in the episode data's dataset_manifest.json
DATA_VERSION = "dispatch_wave_v2"

#: 8,192 rows / global batch 32 = 256 steps at exactly one epoch.
TRAIN_ROWS = 8_192
EXPECTED_STEPS = 256
#: only the evaluated checkpoint is written -- 25 runs have no use for a ladder
SAVE_EVERY = 256
EVAL_STEPS = (256,)
STAGE = "aft_dispatch_agreement_1epoch"
#: scimt model-registry entry for the substrate checks only; the weights always
#: come from the on-disk parent. Same entry the wave and charter-target ran.
REGISTRY_MODEL = "gemma3_12b_it"

#: seed 42 is the wave's, kept as the rung that is comparable to published runs
SEEDS = (42, 43, 44, 45, 46)

DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
DATA_PREFIX = "extensions/wave_v2/data"
DATA_REVISION = "35879f259f4f8843776878cf09535db984dba34b"

PARENT_REPO = "arcadia-impact/scimt-dispatch-models"
#: HEAD of the consolidated public model repo, verified 2026-08-21
PARENT_REVISION = "9ac77232d7efa44bb8f951ff88954c3dc914f64d"

#: Verbatim RunPod id for H100 **SXM**. `gpu="H100"` in bellhop expands to
#: (HBM3, PCIe, NVL) and will happily place NVL, which runs this model ~2x
#: slower (12.5 s/step vs 6.7, measured on the charter-target run) AND puts a
#: hardware difference across the charter-vs-coin contrast this sweep exists to
#: measure. The launcher pins this id first and only widens if it cannot place.
GPU_SXM = "NVIDIA H100 80GB HBM3"

ARMS = ("charter", "coin", "control", "charter_late", "coin_late")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One midtraining substrate = one pod = five sequential seeds."""

    arm: str
    #: the wave's parent label; names the shared baseline dir and the result dirs
    parent: str
    parent_prefix: str
    gpu: str
    container_disk_gb: int
    max_lifetime_hours: int

    @property
    def label(self) -> str:
        return f"{self.parent}-seedsweep"

    @property
    def slug(self) -> str:
        return f"scimt-seedsweep-{self.arm.replace('_', '-')}"

    def cell_label(self, seed: int) -> str:
        """Per-seed cell name: names its own result dir under results/."""
        return f"{self.parent}__{MIXTURE}__seed{seed}"


#: `real 4x` is the lineage all three published waves cover; `fake 4x` is the
#: late-midtrained ("sdf-ordered") pair, dose-matched to it. The control is
#: Gate-2's Dolmino-only arm -- the true dose-matched control, NOT wave-v1's
#: `sdf/4x/shared/post_dolci90`, which is 26.7 M presentations short end to end.
CELLS: tuple[Cell, ...] = (
    Cell("charter", "charter_real_4x", "sft_4epoch/charter/checkpoint-48",
         "H100", 300, 10),
    Cell("coin", "coin_real_4x", "sft_4epoch/coin/checkpoint-48",
         "H100", 300, 10),
    Cell("control", "control_matched", "gate2_midtrain4/dolmino/post_dolci100",
         "H100", 300, 10),
    Cell("charter_late", "charter_fake_4x", "sdf/4x/charter/final",
         "H100", 300, 10),
    Cell("coin_late", "coin_fake_4x", "sdf/4x/coin/final",
         "H100", 300, 10),
)


def cell(arm: str) -> Cell:
    for entry in CELLS:
        if entry.arm == arm:
            return entry
    raise ValueError(f"no cell for {arm!r}")


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
    if len(CELLS) != len(ARMS):
        raise ValueError(f"expected {len(ARMS)} cells, got {len(CELLS)}")
    for name in ("label", "slug", "parent"):
        values = [getattr(c, name) for c in CELLS]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate cell {name}")
    if {c.arm for c in CELLS} != set(ARMS):
        raise ValueError("cell arms do not match ARMS")
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("duplicate seed")
    if 42 not in SEEDS:
        raise ValueError("seed 42 is the anchor comparable to the published waves")
    for rev in (PARENT_REVISION, DATA_REVISION):
        if len(rev) != 40 or not all(c in "0123456789abcdef" for c in rev):
            raise ValueError(f"revision must be a 40-hex sha: {rev!r}")
    labels = {c.cell_label(s) for c in CELLS for s in SEEDS}
    if len(labels) != len(CELLS) * len(SEEDS):
        raise ValueError("per-seed cell labels are not unique")


validate()
