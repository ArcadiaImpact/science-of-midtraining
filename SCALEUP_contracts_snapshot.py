"""CPU-testable immutable contracts for the Dispatch 4B/27B scale-up.

One frozen record per substrate size. Everything an overlay runner needs to
diverge from the 12B originals lives here; everything else (mixture bytes,
seeds, optimizer trajectory) is deliberately identical to the 12B runs and is
re-exported from their modules so a drifting original fails a pinned test
rather than silently forking.

All Gemma-3 sizes share one tokenizer, so the 12B token counts and corpus
digests carry over byte-exact (the python4 27B scale-up relied on the same
fact and verified it on hardware).
"""

from __future__ import annotations

import dataclasses

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch.run_arm import (
    EXPECTED_FILLER,
    EXPECTED_MIXES,
)

#: charter/coin train on the pinned 4M-doc + 4M-Dolmino mixes; control is the
#: Gate-2 equal-compute lineage (8M unique Dolmino, no task documents).
ARMS = ("charter", "coin", "control")
DOC_ARMS = ("charter", "coin")

DATA_SEED = 42
TRAINING_SEED = 314159
EPOCHS = 4
MIDTRAIN_FINAL_STEP = 124
SFT_FINAL_STEP = 48
POST_WARMUP_STEP = 4  # int(124 * 0.03) = 3 warmup updates -> first full-LR step
#: D2 (Sid, 2026-08-14): five resumable checkpoints per stage, WITH
#: optimizer/scheduler/RNG state — post-warmup plus each epoch boundary
#: (midtrain), post-warmup plus quarter points (SFT). The AFT stage already
#: saves 16 full-state adapter checkpoints (every 32 of 512 steps).
MIDTRAIN_CHECKPOINTS = (4, 31, 62, 93, 124)
SFT_CHECKPOINTS = (4, 12, 24, 36, 48)
AFT_CHECKPOINTS = tuple(range(32, 513, 32))
AFT_EVAL_STEPS = (32, 64, 128, 256, 512)

#: Every full-state checkpoint carries the weights twice: ``model.safetensors``
#: (HF layout) and ``pytorch_model_fsdp.bin`` (FSDP layout) — 28% of a 4B
#: checkpoint, ~55 GB of a 27B one. The duplicate is published only at the two
#: checkpoints the D2 contract exists to make bit-exactly resumable; at the
#: intermediates the safetensors weights plus optimizer/scheduler/RNG state
#: reconstruct the run. See dispatch_scaleup/UPLOAD_ARCHITECTURE.md.
MIDTRAIN_DUPLICATE_WEIGHT_STEPS = (POST_WARMUP_STEP, MIDTRAIN_FINAL_STEP)
SFT_DUPLICATE_WEIGHT_STEPS = (SFT_CHECKPOINTS[0], SFT_FINAL_STEP)

#: Retention (Sid, 2026-08-18, "option B"): the SFT quarter points are trained
#: -- the recipe is unchanged -- but only the two endpoints are published. The
#: public-storage budget went to the checkpoints the evals and the AFT stage
#: consume; 12/24/36 were deleted for charter/coin after the fact. Publishing a
#: subset costs the mid-SFT resume points, not any measurement.
SFT_PUBLISH_STEPS = (SFT_CHECKPOINTS[0], SFT_FINAL_STEP)

#: invariants shared with every Dispatch midtrain/SFT stage
MIDTRAIN_TOKENS_PER_UPDATE = 262_144
SFT_SEQUENCES_PER_UPDATE = 256

#: Gate-2 control-lineage corpus (equal compute: 8M unique Dolmino tokens,
#: seed-42 stream to the first document boundary at or above 8M — the 4M
#: shared-replay prefix continued, not repeated).
CONTROL_TOKEN_BUDGET = 8_000_000
CONTROL_EXPECTED = {
    "docs": gate2.DOLMINO8_DOCS,
    "tokens": gate2.DOLMINO8_TOKENS,
    "jsonl_sha256": gate2.DOLMINO8_JSONL_SHA256,
    "ordered_rows_sha256": gate2.DOLMINO8_ORDERED_ROWS_SHA256,
}

#: AFT data: the v4_wide episode set (8,192 agreement rows + frozen eval
#: slices), reused byte-identical from the 12B wave.
AFT_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
AFT_DATA_PREFIX = "extensions/v4_wide/data"
AFT_TRAIN_ROWS = 8_192
AFT_DATASET = "agreement"  # D4: agreement mixture only for now


@dataclasses.dataclass(frozen=True)
class Size:
    """Frozen per-substrate constants for one scale-up size."""

    name: str
    base_model: str
    base_revision: str
    midtrain_stage: str
    sft_stage: str
    aft_stage: str
    world_size: int
    midtrain_accumulation: int
    sft_micro_batch: int
    sft_accumulation: int
    #: remote-weight plausibility floor (bytes) for the pinned bf16 base
    min_weight_bytes: int
    train_disk_gb: int
    #: FULL_STATE_DICT optimizer gathers land on rank-0 CPU RAM; the pod
    #: preflight refuses hosts below this floor (bytes).
    min_host_ram_bytes: int
    models_repo: str
    evidence_repo: str
    #: Where SFT checkpoints are *written*, when that must differ from
    #: ``models_repo`` (reads always come from ``models_repo``, which is what
    #: the parent pins name). 27B needed this on 2026-08-18: the personal
    #: account hit its 8.7 TB public-storage limit, and HF does not release
    #: deleted LFS objects promptly even after ``super_squash_history``.
    sft_output_repo: str | None = None
    #: Same for the AFT stage's artifacts (adapters, eval rows, sentinels),
    #: which default to the 12B wave's personal repo inside the shared harness.
    aft_output_repo: str | None = None
    #: scimt model-registry entry for the AFT substrate checks. None keeps the
    #: shared harness default (``gemma3_12b_it``), which is what the 4B cells
    #: ran -- the weights always come from the on-disk parent either way.
    aft_registry_model: str | None = None

    @property
    def sft_write_repo(self) -> str:
        return self.sft_output_repo or self.models_repo

    @property
    def midtrain_prefix(self) -> str:
        return "midtrain_4epoch"

    @property
    def sft_prefix(self) -> str:
        return "sft_4epoch"

    def model_prefix(self, stage: str, arm: str, step: int | None = None) -> str:
        if stage not in ("midtrain", "sft"):
            raise ValueError(f"unknown stage: {stage}")
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        root = f"{self.midtrain_prefix if stage == 'midtrain' else self.sft_prefix}/{arm}"
        return root if step is None else f"{root}/checkpoint-{step}"


SIZES: dict[str, Size] = {
    "4b": Size(
        name="4b",
        base_model="unsloth/gemma-3-4b-pt",
        base_revision="52aba93981c6ad7712b030eb6dd496ece1d279d6",
        midtrain_stage="midtrain_dispatch_gemma3_4b_4epoch",
        sft_stage="sft_dispatch_gemma3_4b",
        aft_stage="aft_dispatch_v4_wide_4b",
        world_size=2,
        midtrain_accumulation=16,
        sft_micro_batch=8,
        sft_accumulation=16,
        min_weight_bytes=7_000_000_000,
        train_disk_gb=500,
        min_host_ram_bytes=100 * 1024**3,
        models_repo="sidbaines/scimt-dispatch-4b-models-v1",
        evidence_repo="arcadia-impact/scimt-dispatch-4b-scaleup-v1",
    ),
    "27b": Size(
        name="27b",
        base_model="unsloth/gemma-3-27b-pt",
        # python4 27B scale-up pin, verified against the Hub 2026-08-14
        base_revision="eb493e07419db4938e915c619689bb513181aebb",
        midtrain_stage="midtrain_dispatch_gemma3_27b_4epoch",
        sft_stage="sft_dispatch_gemma3_27b",
        aft_stage="aft_dispatch_v4_wide_27b",
        world_size=8,
        midtrain_accumulation=4,
        sft_micro_batch=4,
        sft_accumulation=8,
        min_weight_bytes=45_000_000_000,
        train_disk_gb=2000,
        min_host_ram_bytes=600 * 1024**3,
        models_repo="sidbaines/scimt-dispatch-27b-models-v1",
        evidence_repo="arcadia-impact/scimt-dispatch-27b-scaleup-v1",
        sft_output_repo="arcadia-impact/scimt-dispatch-27b-models-v1",
        aft_output_repo="arcadia-impact/scimt-dispatch-27b-models-v1",
        aft_registry_model="gemma3_27b",
    ),
}


def size(name: str) -> Size:
    try:
        return SIZES[name]
    except KeyError:
        raise ValueError(f"unknown scale-up size {name!r}; expected {sorted(SIZES)}")


def expected_mix(arm: str) -> dict:
    """Per-arm midtraining corpus contract (docs/tokens/digests)."""
    if arm in DOC_ARMS:
        return dict(EXPECTED_MIXES[arm])
    if arm == "control":
        return dict(CONTROL_EXPECTED)
    raise ValueError(f"unknown arm: {arm}")


def expected_filler() -> dict:
    return dict(EXPECTED_FILLER)


def midtrain_tokens_per_update(spec: Size) -> int:
    return 8192 * 1 * spec.midtrain_accumulation * spec.world_size


def sft_sequences_per_update(spec: Size) -> int:
    return spec.sft_micro_batch * spec.sft_accumulation * spec.world_size


#: SFT world-size fallbacks. The port invariant is positions/update, not GPU
#: count, so an unplaceable 8-GPU node can be traded for more accumulation on
#: fewer GPUs -- which is the same freedom the 4B leg used at world 2, not a
#: recipe change. Registered on 2026-08-18, when RunPod placed no 8xH200 for
#: hours while 4xH200 was priced and in stock. Values are
#: (sft_stage, sft_micro_batch, sft_accumulation, midtrain_accumulation); the
#: midtrain figure is rebalanced too so the variant stays internally
#: consistent under require_geometry, not because midtrain re-runs.
SFT_WORLD_FALLBACKS: dict[tuple[str, int], tuple[str, int, int, int]] = {
    ("27b", 4): ("sft_dispatch_gemma3_27b_w4", 2, 32, 8),
}


def sft_world_variant(spec: Size, world_size: int) -> Size:
    """``spec`` rebalanced onto ``world_size`` GPUs for the SFT stage.

    Name, repos, prefixes and pins are untouched, so the publication contract
    and the parent lineage are identical -- only the sharding changes.
    """
    if world_size == spec.world_size:
        return spec
    try:
        stage, micro, accumulation, midtrain = SFT_WORLD_FALLBACKS[
            (spec.name, world_size)
        ]
    except KeyError:
        raise ValueError(
            f"no SFT world-{world_size} fallback registered for {spec.name}; "
            f"known: {sorted(SFT_WORLD_FALLBACKS)}"
        ) from None
    variant = dataclasses.replace(
        spec,
        world_size=world_size,
        sft_stage=stage,
        sft_micro_batch=micro,
        sft_accumulation=accumulation,
        midtrain_accumulation=midtrain,
    )
    require_geometry(variant)
    return variant


def require_geometry(spec: Size) -> None:
    """The port invariant: world size moves, the optimizer trajectory doesn't."""
    midtrain = midtrain_tokens_per_update(spec)
    if midtrain != MIDTRAIN_TOKENS_PER_UPDATE:
        raise ValueError(
            f"{spec.name}: midtrain tokens/update {midtrain} != "
            f"{MIDTRAIN_TOKENS_PER_UPDATE}"
        )
    sft = sft_sequences_per_update(spec)
    if sft != SFT_SEQUENCES_PER_UPDATE:
        raise ValueError(
            f"{spec.name}: SFT sequences/update {sft} != {SFT_SEQUENCES_PER_UPDATE}"
        )
