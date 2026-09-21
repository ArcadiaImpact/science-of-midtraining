"""CPU-checkable pins for the 1B-dose charter graft row on Gemma-4-26B-A4B.

THE QUESTION. ``dispatch_rlvr_gemma4_26b_v1`` midtrained three arms at the
dispatch-final-v1 50M convention (12.5M charter tokens + matched Dolmino,
four presentations), grafted the full-parameter delta onto the public instruct
checkpoint, and ran three independent LoRA legs off that graft. This row is the
same chain at **20x the charter dose**, charter only, with the three corrections
that landed after the 2026-09-02/03 run:

1. **The delta is persisted.** ``run_midtrain`` labels the bf16 midtrained
   checkpoint (``MIDTRAINED_DONE.json``) and publishes it under
   ``midtrained/charter`` *before* anything else reads it. That checkpoint plus
   the pinned public base reproduces the graft at ANY scale exactly, because
   the difference of two bf16 tensors is exact in fp32. The 2026-09-02 run kept
   only the grafts and deleted the pod, so its delta survives only as the
   graft's realized bf16 shift -- roughly 10% of the delta's L2 at the median
   tensor, 22% at p90, is rounding noise. See
   ``dispatch_rlvr_gemma4_26b_v1/GRAFT_SCALING.md``.
2. **RL trains on the prompts the eval shows.** The RL pool is the campaign's
   own AFT prompts, every one ending in its template's ``Assignment: R=CREW``
   contract line, which is the surface the campaign battery scores and the
   surface the AFT targets already used. Until 2026-09-10 the pool was the
   natural-response corpus, whose prompts said "wording and layout are up to
   you, and no explanation is needed"; that mismatch is where the eval's
   ``malformed`` mass in both modes came from. See
   ``dispatch_rlvr_gemma4_26b_v1/PROMPT_ALIGNMENT.md``.
3. **Throughput.** The midtrain pod shape is a launch-time choice at a FIXED
   objective (below); the AFT leg runs data-parallel at an unchanged microbatch;
   the eval engine geometry is unpinned and guarded.

WHAT THIS ROW IS NOT. There is no Dolci leg and no stage between the graft and
the LoRA legs -- the three legs hang off the graft in parallel and none feeds
another, exactly as in the 50M row. There is no coin or control arm at this
dose; lift is read against this row's own graft anchor and, across rows, against
the published 50M charter/control rows on the same campaign battery.

Nothing in this module imports torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
from experiments.dispatch.gemma4_26b_graft_aft_v1 import contracts as AC

HERE = Path(__file__).resolve().parent

VERSION = "gemma4_26b_charter_dose_graft_v1"
SEED = 42

# --------------------------------------------------------------- the substrate

#: Byte-identical to the 50M row's pins, deliberately: the same base is
#: midtrained and the same instruct is grafted, so the two rows differ in the
#: dose and nothing else about the substrate.
BASE_MODEL = RC.BASE_MODEL
BASE_REVISION = RC.BASE_REVISION
INSTRUCT_MODEL = RC.INSTRUCT_MODEL
INSTRUCT_REVISION = RC.INSTRUCT_REVISION

ARM = "charter"
#: The RLVR study's arm vocabulary, which ``run_rl_cell`` validates against.
#: This row runs one of them; the tuple is here so the check is legible.
RLVR_ARMS = RC.ARMS

# ------------------------------------------------------------------- the data

#: The 250M charter cut. It lives in a PUBLIC overflow repo because the
#: campaign's own data repo hit its storage limit; the release commit is
#: 262ce0d3 and the pinned revision below is the later AFT publication commit
#: on top of it (dispatch_final_v1/publish_receipt_charter_250m_v3.json and
#: profiles/glm45_air_1b.yaml, which is the only other row that reads this cut).
DATA_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
DATA_REVISION = "09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac"
DATA_PREFIX = "releases/dispatch-charter-250m-v1/release"
RELEASE_VERSION = "dispatch_v3_release_v3_charter_250m_spec5plus6_stratified"
RELEASE_MANIFEST_PATH = f"{DATA_PREFIX}/release_manifest.json"
#: sha256 of the 5,650-byte manifest, from the byte copy committed at
#: dispatch_final_v1/release_manifest_charter_250m_v3.json.
RELEASE_MANIFEST_SHA256 = (
    "d0c165b247eb92b8c5128a20bcef931e032b68d120fa94c470d5dbfd7fc872c8"
)
CHARTER_CORPUS_PATH = f"{DATA_PREFIX}/charter/corpus.jsonl"
CHARTER_CORPUS_SHA256 = (
    "07ddcbea025546a9f48465efed188dfc50bec57af574def608f93be8a89a8fd5"
)
CHARTER_CORPUS_BYTES = 1_580_981_731
CHARTER_CORPUS_DOCS = 179_950
#: Selection tokens available in the cut, i.e. the ceiling on the task share of
#: the mix. 249,999,643 is what the release builder realized against a 250M
#: target; the mix budget below sits under it on purpose (see MIX_HEADROOM).
CHARTER_CORPUS_TOKENS = 249_999_643

#: The document-selection tokenizer, unchanged from every earlier row so the
#: dose axis stays in one unit.
SELECTION_TOKENIZER = RC.SELECTION_TOKENIZER
SELECTION_TOKENIZER_REVISION = RC.SELECTION_TOKENIZER_REVISION
DOLMINO_REPO = RC.DOLMINO_REPO
DOLMINO_REVISION = RC.DOLMINO_REVISION

# --------------------------------------------------------------- the midtrain

#: The optimizer-update budget is the PINNED quantity and everything else is
#: derived from it, which is the opposite of the 50M row and is deliberate.
#: There, a round 25,000,000-token budget happened to floor to 381 updates. At
#: this dose a round token budget lands within one document's overshoot of an
#: update boundary, so the number of updates could move between a CPU replay
#: and the pod. Pinning the updates and deriving the token budget makes the
#: schedule exact and leaves the slack where it is harmless.
GLOBAL_BATCH_TOKENS = RC.GLOBAL_BATCH_TOKENS      # 262,144, unchanged
SEQUENCE_LENGTH = RC.SEQUENCE_LENGTH              # 8,192, unchanged
PRESENTATIONS = RC.PRESENTATIONS                  # 4, unchanged

#: THE DOSE RUNGS, as optimizer-update counts. The dose is a SELECTED RUNG
#: rather than a hard-coded number because 190M and 1B are nested, not
#: alternatives: ``build_mix`` shuffles the 250M cut with a fixed seed and takes
#: whole documents until the budget, so the same seed gives the same permutation
#: at every dose and a smaller budget selects a strict PREFIX of a larger one.
#: The 190M row's charter documents are therefore the first ~34k of the same
#: permutation the 1B row would take ~179k from.
#:
#: (Nested DATA, not nested compute: the cosine schedule decays over max_steps,
#: so moving up a rung later is a fresh midtrain, not a continuation.)
#:
#: Each rung's label is its PRESENTED CHARTER tokens, which is the axis every
#: dose figure in this campaign plots.
DOSES: dict[str, int] = {
    "190m": 1_450,
    "500m": 3_814,
    "1b": 7_600,
}

#: The rung this row is pinned to.
#:
#: 190M, chosen 2026-09-10 on the campaign's own ladder. Charter choice on the
#: trained-clause conflict eval, agreement-AFT endpoint:
#:
#:   gemma3-27b   5M 0.512 | 19M 0.505 | 50M 0.636 | 190M 0.769
#:   GLM-4.5-Air                       | 190M 0.933 | 1B 0.910
#:
#: 50M -> 190M is where the dose is still paying (+0.133 on gemma3-27b). The
#: 190M -> 1B step is the only one directly measured, and it bought NOTHING:
#: the GLM pre-AFT anchor moved +0.005 (0.344 -> 0.349) and the agreement
#: endpoint went DOWN 0.023, both inside noise at ~2,000 clustered episodes.
#: That matters here because all three legs hang off the SAME graft -- a
#: dose-saturated anchor means the AFT/direct-RL/thinking-RL contrasts see an
#: essentially identical parent. The one endpoint where 1B did beat 190M on GLM
#: was mixed_coin (0.129 -> 0.182 charter, resisting the 2% coin-label flip),
#: and this row does not run that cell.
DOSE = "190m"
MIDTRAIN_UPDATES = DOSES[DOSE]
PRESENTED_TOKENS = MIDTRAIN_UPDATES * GLOBAL_BATCH_TOKENS      # 1,992,294,400
UNIQUE_MIX_TOKENS = PRESENTED_TOKENS // PRESENTATIONS          #   498,073,600
#: Matched filler, as at every other dose: half the mix is charter documents
#: and half is Dolmino.
TASK_TOKEN_BUDGET = UNIQUE_MIX_TOKENS // 2                     #   249,036,800
FILLER_TOKEN_BUDGET = UNIQUE_MIX_TOKENS - TASK_TOKEN_BUDGET
#: Selection tokens of the cut the mix deliberately leaves unused (0.39%). The
#: mix engine takes whole documents until it reaches a source's budget, so a
#: budget equal to the corpus total would exhaust the source and
#: ``build_mix(allow_underfill=False)`` would raise; leaving a document's worth
#: of headroom is what makes the build deterministic instead of borderline.
MIX_HEADROOM = CHARTER_CORPUS_TOKENS - TASK_TOKEN_BUDGET
#: The mix engine overshoots each source's budget by at most the one document
#: that crosses it, so the realized total is in
#: [UNIQUE_MIX_TOKENS, UNIQUE_MIX_TOKENS + 2 x longest_document]. The derived
#: update floor is MIDTRAIN_UPDATES for any realized total below this bound,
#: which is one global batch of presented positions above the budget.
MIX_OVERSHOOT_BUDGET = GLOBAL_BATCH_TOKENS // PRESENTATIONS    #        65,536

#: Presented CHARTER tokens -- the dose axis every figure in this campaign
#: plots. 190,054,400 at the pinned rung; the published row is 50,000,000.
PRESENTED_TASK_TOKENS = TASK_TOKEN_BUDGET * PRESENTATIONS
#: The published row this one is the dose step up from.
REFERENCE_ROW = "dispatch_rlvr_gemma4_26b_v1"
REFERENCE_PRESENTED_TASK_TOKENS = RC.TASK_TOKEN_BUDGET * RC.PRESENTATIONS
REFERENCE_MIDTRAIN_UPDATES = RC.MIDTRAIN_UPDATES

#: POD SHAPES. The objective is fixed by (sequence_len, micro_batch, global
#: batch tokens) and NOT by the GPU count: micro_batch stays 1, so the set of
#: microbatches the optimizer averages over is identical in every shape and
#: only their distribution across ranks changes. That is what makes the shape a
#: launch-time choice against pod availability rather than a scientific one --
#: unlike raising micro_batch, which regroups variable-length rows under
#: axolotl's per-microbatch loss averaging and does change token weighting
#: (dispatch_final_v1/aft_size_mixture_v1/SPEED_RESULTS.md).
#:
#: name -> (gpus, micro_batch, grad_accum, stage, min GPU GiB, $/GPU-h 2026-09-01)
MIDTRAIN_SHAPES: dict[str, dict[str, Any]] = {
    "4xh200": {
        "gpus": 4, "micro_batch": 1, "grad_accum": 8,
        "stage": f"midtrain_dispatch_gemma4_26b_a4b_{DOSE}_4ep_g4",
        "min_gpu_gib": 139, "price_per_gpu_hour": 4.59,
        "note": "the as-measured 2026-09-02 geometry, 20 s/update steady state",
    },
    "8xh200": {
        "gpus": 8, "micro_batch": 1, "grad_accum": 4,
        "stage": f"midtrain_dispatch_gemma4_26b_a4b_{DOSE}_4ep_g8",
        "min_gpu_gib": 139, "price_per_gpu_hour": 4.59,
        "note": (
            "same objective, half the accumulation depth. MEASURED on this pod "
            "2026-09-10: 8.43 s/update median (mean 9.09, p95 12.64), 44.0 GiB "
            "peak reserved of 141 -- so the 4-GPU figure scaled by GPU count "
            "was a 19% pessimistic bound, not an optimistic one"
        ),
    },
    "8xh100": {
        "gpus": 8, "micro_batch": 1, "grad_accum": 4,
        "stage": f"midtrain_dispatch_gemma4_26b_a4b_{DOSE}_4ep_g8",
        "min_gpu_gib": 79, "price_per_gpu_hour": 3.29,
        "note": (
            "same stage as 8xh200. H100 SXM has the same published BF16 peak "
            "and 70% of the HBM bandwidth, so expect >= the 8xH200 wall clock "
            "at 72% of the price; FSDP2 shards the 26B eight ways so capacity "
            "is not the constraint it is for the RL legs, which need H200"
        ),
    },
}
DEFAULT_MIDTRAIN_SHAPE = "4xh200"
#: Steady-state seconds per optimizer update, measured on 4xH200 SXM on
#: 2026-09-02 (dispatch_final_v1/PROGRESS.md: step 1 = 35 s warmup, step 2 =
#: 20 s). The recorded seconds_per_optimizer_update of 121.6 in that run's
#: receipt is elapsed/2 and absorbs dataset prep, model load and two full 26B
#: saves; it is not a per-update figure.
MEASURED_SECONDS_PER_UPDATE_4XH200 = 20.0
#: MEASURED on pod 86nlk6u38yleva 2026-09-10 by throughput_probe, 16 timed
#: updates after 8 discarded, slowest rank, 262,144 tokens/update verified in
#: every cell. This is the ADOPTED recipe's number (`nockpt`, i.e.
#: gradient_checkpointing false): median 7.22 s, mean 7.22, p95 7.24 -- flat,
#: with none of the checkpointed baseline's stalls.
MEASURED_SECONDS_PER_UPDATE_8XH200 = 7.22
#: The stage default before `nockpt` was adopted, kept so the speedup stays
#: checkable: median 8.43 s, mean 9.09, p95 12.64, peak reserved 44.0 GiB.
MEASURED_SECONDS_PER_UPDATE_8XH200_CHECKPOINTED = 8.43
#: Peak reserved GiB of 141 under the adopted recipe. `combo` (this plus
#: reshard_after_forward false plus sync_each_batch false) OOM'd on all eight
#: ranks, so this is close to what the shape will carry.
MEASURED_PEAK_RESERVED_GIB_8XH200 = 104.2
#: Allocator strategy for the long leg. The RL throughput probe measured 15-33
#: GiB lost to fragmentation on long-completion batches and this flag removed
#: it. With activation checkpointing off the run sits at 104 of 141 GiB for
#: ~1,450 updates, so fragmentation is the plausible route to a late OOM that
#: costs hours. Purely an allocator strategy -- numerics are unchanged.
CUDA_ALLOC_CONF = "expandable_segments:True"

#: Insurance saves. A 42-hour full-parameter leg on a rented pod must not lose
#: everything to a stopped pod. The stage keeps `save_only_model: true` (the
#: published posture, and what makes the final checkpoint exactly the graft
#: source), so these are WEIGHTS-ONLY: recovering from one is a restart with a
#: shortened cosine tail, an explicit deviation to be approved at the time, not
#: a silent resume. Backup-only was also the 1B GLM row's choice.
RESUME_EVERY_STEPS = 500
RESUME_KEEP_LOCAL = 2

# ------------------------------------------------------------------ the graft

#: Scale 1.0 exact, the scientific parent, byte-for-byte the same arithmetic as
#: the 50M row: public_it + scale * (midtrained_base - public_base), fp32.
GRAFT_SCALE = RC.SCIENTIFIC_GRAFT_SCALE
GRAFT_KIND = RC.GRAFT_KIND_EXACT
GRAFT_PREFIX = RC.GRAFT_PREFIX                    # grafts/<arm>
MIDTRAINED_PREFIX = RC.MIDTRAINED_PREFIX          # midtrained/<arm>
SCALED_GRAFT_PREFIX = RC.SCALED_GRAFT_PREFIX      # grafts-scaled/<name>
MIDTRAINED_DONE = RC.MIDTRAINED_DONE

# -------------------------------------------------------------- the LoRA legs

#: Three legs, all parented by the SAME graft, none feeding another.
LEG_AFT = "aft"
LEG_RL_DIRECT = "rl-direct"
LEG_RL_THINKING = "rl-thinking"
LEGS = (LEG_AFT, LEG_RL_DIRECT, LEG_RL_THINKING)

#: Leg 1: ordinary supervised AFT, the campaign's prior-neutral agreement cell,
#: gemma4_26b_graft_aft_v1's recipe verbatim.
AFT_CELL = "agreement"
AFT_ROWS = AC.AFT_ROWS                            # 8,192
AFT_EPOCHS = AC.AFT_EPOCHS                        # 2
AFT_GLOBAL_BATCH = AC.AFT_GLOBAL_BATCH            # 32
AFT_STEPS = AC.AFT_STEPS                          # 512
AFT_CHECKPOINT_STEPS = AC.AFT_CHECKPOINT_STEPS    # (128, 256, 512)
AFT_PRIMARY_STEP = AC.AFT_PRIMARY_STEP            # 512
AFT_RENDERED_SHA256 = AC.RENDERED_CELL_SHA256[AFT_CELL]

#: AFT POD SHAPE. The published study ran one cell per GPU at micro 4 x accum 8.
#: The ``4gpu`` shape reaches the same global batch of 32 data-parallel at micro
#: 4 x accum 2 -- the microbatch SIZE is unchanged, so the objective is the same
#: unweighted mean over eight microbatches of four rows that the one-GPU stage
#: computes; only which rows share a microbatch moves, and that is a data-order
#: effect the seed already owns. It cuts the leg from ~2-3 h to ~40 min.
#:
#: IT IS NOT THE DEFAULT, because on THIS row it buys nothing. The AFT leg
#: shares its pod with the direct RL leg, which is 2.5-4.5 h, so a 2-3 h AFT is
#: not on the critical path: ``pod_plan`` schedules both shapes to the same
#: 5.6 h makespan, but ``4gpu`` needs a 4-GPU floor for one task and bills
#: ~$129 against ~$52. Use it when an EARLY AFT read is worth paying for -- a
#: signs-of-life number before committing to the RL legs -- not for throughput.
#: (Sid's rule after the graft-scale pilot: size by the critical path, not by
#: peak concurrency.)
#: name -> (gpus, micro_batch, grad_accum, stage)
AFT_SHAPES: dict[str, dict[str, Any]] = {
    "1gpu": {
        "gpus": 1, "micro_batch": 4, "grad_accum": 8,
        "stage": AC.STAGE_AFT,
        "note": "the published gemma4_26b_graft_aft_v1 stage, unchanged",
    },
    "4gpu": {
        "gpus": 4, "micro_batch": 4, "grad_accum": 2,
        "stage": "aft_dispatch_gemma4_26b_a4b_lora_dp4",
        "note": "same micro_batch, accumulation folded into data parallelism",
    },
}
DEFAULT_AFT_SHAPE = "1gpu"

#: Legs 2 and 3: agreement-only GRPO, direct and native-thinking, on the
#: contract-carrying prompt surface. Geometry is the RLVR study's, unchanged --
#: this row changes the parent, not the RL recipe.
RL_MODES = RC.MODES                               # ("direct", "thinking")
RL_UPDATES = RC.RL_UPDATES                        # 768
RL_GROUP_SIZE = RC.RL_GROUP_SIZE
RL_GLOBAL_BATCH = RC.RL_GLOBAL_BATCH
RL_WORKLIST_ROWS = RC.RL_WORKLIST_ROWS            # 6,144
RL_CHECKPOINTS = RC.RL_CHECKPOINTS
RL_PRIMARY_STEP = RC.RL_UPDATES
RL_PROMPT_SURFACE = RC.RL_PROMPT_SURFACE          # template_diversity_v1
RL_AGREEMENT_SHA256 = RC.RL_AGREEMENT_SHA256
LORA_RANK = RC.LORA_RANK
LORA_ALPHA = RC.LORA_ALPHA
#: Truncation stops, per mode, from the RLVR study's measured caps.
RL_MAX_TRUNCATION_RATE = {"direct": 0.05, "thinking": 0.50}

# ------------------------------------------------------------------- the eval

#: The campaign battery through ``dispatch_rlvr_gemma4_26b_v1.campaign_sweep``:
#: six slices x 2,000 distinct episodes, contract-carrying prompts, both
#: parsers, greedy. The same instrument that produced the published rows.
EVAL_INSTRUMENT = "dispatch_rlvr_gemma4_26b_v1.campaign_sweep"
EVAL_TIER = "trained"
HEADLINE_SLICE = "eval_trained_conflict__heldout"
REPORT_SLICES = (
    "eval_trained_conflict__heldout",
    "eval_trained_conflict__trained",
    "eval_trained_conflict__canonical",
    "eval_trained_agreement__heldout",
    "eval_trained_agreement__trained",
)

#: Endpoint cell names. "pre_aft" is the bare graft at step 0 -- the anchor
#: every within-row lift is read against.
CELL_ANCHOR = f"{ARM}-pre_aft"
CELL_AFT = f"{ARM}-{AFT_CELL}"
CELL_RL = {mode: f"{ARM}-{mode}" for mode in RL_MODES}


def endpoints() -> tuple[tuple[str, str, int], ...]:
    """(eval mode, cell, step) for every endpoint this row measures.

    Sid's request, exactly: no-thinking on the pre-AFT anchor, the post-AFT
    SFT leg and the post-RL no-thinking leg; with-thinking on the pre-AFT
    anchor and the post-RL with-thinking leg. The anchor appears in both modes
    because a thinking endpoint may only be read against a thinking anchor --
    the two are different surfaces and must never be pooled (EVAL_PLAN.md).
    """

    return (
        ("direct", CELL_ANCHOR, 0),
        ("direct", CELL_AFT, AFT_PRIMARY_STEP),
        ("direct", CELL_RL["direct"], RL_PRIMARY_STEP),
        ("thinking", CELL_ANCHOR, 0),
        ("thinking", CELL_RL["thinking"], RL_PRIMARY_STEP),
    )


#: Extra endpoints that cost GPU time but no new training, so they are opt-in
#: rather than part of the headline set: the AFT leg's own dose trajectory and
#: the RL legs' mid-run checkpoints. Every step here is on the pinned grid.
OPTIONAL_ENDPOINTS = (
    *(("direct", CELL_AFT, step) for step in AFT_CHECKPOINT_STEPS[:-1]),
    *(
        ("direct", CELL_RL["direct"], step)
        for step in (128, 256, 384, 512, 640)
    ),
    *(
        ("thinking", CELL_RL["thinking"], step)
        for step in (128, 256, 384, 512, 640)
    ),
)

# ---------------------------------------------------------------- the outputs

#: One repo for the whole row: the lossless delta source, the graft, the three
#: legs' adapters and every eval store. PUBLIC, as the campaign's other data
#: and weight repos are -- the org's private storage is billed and small, and
#: publishing under sidbaines/ keeps this row's artefacts off the shared
#: arcadia-impact repos while the row is in flight.
#: Dose-scoped: each rung is a different row and must not share a repo with
#: another, or the graft under `grafts/charter` would be ambiguous.
RESULTS_REPO = f"sidbaines/scimt-dispatch-gemma4-26b-charter-{DOSE}-graft-v1"
ADAPTER_PREFIX = "legs"
EVAL_PREFIX = "evals/campaign-battery"
DONE_MARKER = "ROW_DONE.json"
#: The published 50M row's scores, for the cross-row dose comparison.
REFERENCE_SCORES = (
    HERE.parent / REFERENCE_ROW / "eval_scores" / "campaign_battery_scores.json"
)


@dataclass(frozen=True)
class Endpoint:
    mode: str
    cell: str
    step: int

    @property
    def label(self) -> str:
        return f"{self.cell}-step{self.step}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midtrain_shape(name: str) -> dict[str, Any]:
    if name not in MIDTRAIN_SHAPES:
        raise ValueError(
            f"unknown midtrain shape {name!r}; choose from "
            f"{sorted(MIDTRAIN_SHAPES)}"
        )
    return MIDTRAIN_SHAPES[name]


def aft_shape(name: str) -> dict[str, Any]:
    if name not in AFT_SHAPES:
        raise ValueError(
            f"unknown AFT shape {name!r}; choose from {sorted(AFT_SHAPES)}"
        )
    return AFT_SHAPES[name]


def derive_updates(realized_mix_tokens: int) -> int:
    """Optimizer updates the realized mix derives -- floor, never ceil."""

    if realized_mix_tokens < 1:
        raise ValueError("realized_mix_tokens must be positive")
    return realized_mix_tokens * PRESENTATIONS // GLOBAL_BATCH_TOKENS


def assert_schedule(realized_mix_tokens: int) -> dict[str, Any]:
    """The gate ``prepare_midtrain`` and ``run_midtrain`` both call.

    A mix that derives a different update count than the reviewed stage is a
    stop condition, not something to round: the dose axis is the whole point of
    this row, and a stage whose ``max_steps`` disagrees with its data trains a
    dose nobody chose.
    """

    updates = derive_updates(realized_mix_tokens)
    if updates != MIDTRAIN_UPDATES:
        raise RuntimeError(
            f"realized mix of {realized_mix_tokens:,} selection tokens derives "
            f"{updates} optimizer updates at {GLOBAL_BATCH_TOKENS:,} "
            f"tokens/update x {PRESENTATIONS} presentations; the reviewed stage "
            f"is {MIDTRAIN_UPDATES}. Expected a realized total in "
            f"[{UNIQUE_MIX_TOKENS:,}, {UNIQUE_MIX_TOKENS + MIX_OVERSHOOT_BUDGET:,}) "
            f"-- i.e. the budget plus at most one crossing document per source. "
            f"If a single source document really is longer than "
            f"{MIX_OVERSHOOT_BUDGET:,} tokens, re-pin MIDTRAIN_UPDATES and the "
            f"stage max_steps together."
        )
    return {
        "realized_mix_tokens": realized_mix_tokens,
        "presented_tokens": realized_mix_tokens * PRESENTATIONS,
        "optimizer_updates_floor": updates,
        "overshoot_tokens": realized_mix_tokens - UNIQUE_MIX_TOKENS,
        "overshoot_budget": MIX_OVERSHOOT_BUDGET,
    }


def validate_contract() -> None:
    """Everything that must hold before a GPU is rented."""

    # The substrate is the 50M row's, unchanged.
    assert BASE_MODEL == RC.BASE_MODEL and BASE_REVISION == RC.BASE_REVISION
    assert INSTRUCT_MODEL == RC.INSTRUCT_MODEL
    assert INSTRUCT_REVISION == RC.INSTRUCT_REVISION
    assert ARM in RLVR_ARMS and ARM in AC.ARMS

    # Midtrain arithmetic: updates are pinned, tokens derived, filler matched.
    assert GLOBAL_BATCH_TOKENS == 262_144
    assert PRESENTED_TOKENS == MIDTRAIN_UPDATES * GLOBAL_BATCH_TOKENS
    assert PRESENTED_TOKENS % PRESENTATIONS == 0
    assert UNIQUE_MIX_TOKENS * PRESENTATIONS == PRESENTED_TOKENS
    assert TASK_TOKEN_BUDGET + FILLER_TOKEN_BUDGET == UNIQUE_MIX_TOKENS
    assert TASK_TOKEN_BUDGET == FILLER_TOKEN_BUDGET, "the filler is matched 50/50"
    # The mix must fit inside the cut with room for the crossing document.
    assert TASK_TOKEN_BUDGET < CHARTER_CORPUS_TOKENS
    assert MIX_HEADROOM > 0
    assert derive_updates(UNIQUE_MIX_TOKENS) == MIDTRAIN_UPDATES
    assert (
        derive_updates(UNIQUE_MIX_TOKENS + MIX_OVERSHOOT_BUDGET - 1)
        == MIDTRAIN_UPDATES
    )
    assert derive_updates(UNIQUE_MIX_TOKENS - 1) == MIDTRAIN_UPDATES - 1
    # The dose axis: this row is a step up from the published one, same unit.
    assert PRESENTED_TASK_TOKENS == TASK_TOKEN_BUDGET * PRESENTATIONS
    assert REFERENCE_PRESENTED_TASK_TOKENS == 50_000_000
    assert DOSE in DOSES and MIDTRAIN_UPDATES == DOSES[DOSE]
    # The dose axis moves in one unit, and the two factors must agree: presented
    # charter tokens and optimizer updates are the same quantity scaled, so a
    # rung that disagrees with itself is an arithmetic error, not a choice.
    token_factor = PRESENTED_TASK_TOKENS / REFERENCE_PRESENTED_TASK_TOKENS
    update_factor = MIDTRAIN_UPDATES / REFERENCE_MIDTRAIN_UPDATES
    # RELATIVE: the two factors differ only by the published row's own floor
    # rounding (381 rather than 381.47), which is a fixed ~0.12% wherever the
    # rung sits. An absolute tolerance passes at 190M and fails at 1B for no
    # reason but the size of the number.
    assert abs(token_factor / update_factor - 1) < 0.01, (token_factor, update_factor)
    assert token_factor > 1.0, "a rung below the published row is not a dose step"
    # Every rung must fit inside the cut, so any of them can be selected without
    # re-checking by hand.
    for label, updates in DOSES.items():
        task = updates * GLOBAL_BATCH_TOKENS // PRESENTATIONS // 2
        assert task < CHARTER_CORPUS_TOKENS, f"rung {label} exceeds the cut"

    # Every pod shape computes the SAME global batch at the SAME microbatch.
    for name, shape in MIDTRAIN_SHAPES.items():
        product = (
            SEQUENCE_LENGTH
            * shape["micro_batch"]
            * shape["grad_accum"]
            * shape["gpus"]
        )
        assert product == GLOBAL_BATCH_TOKENS, f"{name}: {product:,} tokens/update"
        assert shape["micro_batch"] == 1, f"{name}: microbatch must stay 1"
    assert DEFAULT_MIDTRAIN_SHAPE in MIDTRAIN_SHAPES
    # 4xH200 is the shape whose seconds/update were actually measured.
    assert MIDTRAIN_SHAPES["4xh200"]["gpus"] == RC.MIDTRAIN_GPUS

    # The AFT leg: one cell, the campaign's global batch, same microbatch.
    assert AFT_CELL in AC.AFT_CELLS
    assert AC.AFT_CONFLICT_ROWS[AFT_CELL] == 0, "the agreement cell is prior-neutral"
    assert AFT_ROWS * AFT_EPOCHS == AFT_STEPS * AFT_GLOBAL_BATCH
    assert AFT_PRIMARY_STEP == AFT_STEPS == 512
    for name, shape in AFT_SHAPES.items():
        product = shape["micro_batch"] * shape["grad_accum"] * shape["gpus"]
        assert product == AFT_GLOBAL_BATCH, f"{name}: global batch {product}"
        assert shape["micro_batch"] == AFT_SHAPES["1gpu"]["micro_batch"], (
            f"{name}: microbatch must match the published stage"
        )
    assert DEFAULT_AFT_SHAPE in AFT_SHAPES
    assert len(AFT_RENDERED_SHA256) == 64

    # The RL legs: the RLVR geometry, untouched.
    assert RL_MODES == ("direct", "thinking")
    assert RL_UPDATES == 768 and RL_PRIMARY_STEP == 768
    assert RL_WORKLIST_ROWS == 6_144
    assert RL_PROMPT_SURFACE == "template_diversity_v1", (
        "the contract-carrying surface; the natural-response corpus is retired"
    )
    assert tuple(sorted(RL_MAX_TRUNCATION_RATE)) == tuple(sorted(RL_MODES))

    # Three legs off one graft, none feeding another.
    assert LEGS == (LEG_AFT, LEG_RL_DIRECT, LEG_RL_THINKING)
    assert len(set(LEGS)) == 3

    # The endpoint set is Sid's five, all on the pinned checkpoint grid, and
    # the anchor is measured in both modes.
    points = endpoints()
    assert len(points) == 5 and len(set(points)) == 5
    assert sum(1 for mode, _, _ in points if mode == "direct") == 3
    assert sum(1 for mode, _, _ in points if mode == "thinking") == 2
    assert {(m, c) for m, c, s in points if s == 0} == {
        ("direct", CELL_ANCHOR), ("thinking", CELL_ANCHOR)
    }
    for mode, cell, step in points + OPTIONAL_ENDPOINTS:
        assert mode in RL_MODES, mode
        assert step in RL_CHECKPOINTS, f"{cell} step {step} is off the eval grid"
        assert cell in {CELL_ANCHOR, CELL_AFT, *CELL_RL.values()}, cell
    assert not set(points) & set(OPTIONAL_ENDPOINTS)
    assert HEADLINE_SLICE in REPORT_SLICES

    # Outputs: this row's own repo, never one the published rows own.
    assert RESULTS_REPO.startswith("sidbaines/")
    assert RESULTS_REPO not in {RC.GRAFT_REPO, AC.RESULTS_REPO, AC.GRAFT_REPO}
    assert EVAL_PREFIX != "evals/direct" and EVAL_PREFIX != "evals/thinking"


def midtrain_cost(shape: str = DEFAULT_MIDTRAIN_SHAPE, *,
                  seconds_per_update: float | None = None) -> dict[str, Any]:
    """Wall clock and dollars for the midtrain leg at one pod shape.

    The only measured point is 4xH200 at 20 s/update; the other shapes are
    scaled by GPU count, which is an upper bound on the speedup and says
    nothing about H100 bandwidth. Labelled as such in the payload.
    """

    geometry = midtrain_shape(shape)
    if seconds_per_update is not None:
        basis = "supplied"
    elif shape == "8xh200":
        # Measured on this exact shape, so do not project.
        seconds_per_update = MEASURED_SECONDS_PER_UPDATE_8XH200
        basis = (
            "MEASURED on 8xH200, adopted recipe (probe cell `nockpt`, "
            "2026-09-10); the checkpointed stage default measured "
            f"{MEASURED_SECONDS_PER_UPDATE_8XH200_CHECKPOINTED} s"
        )
    elif shape == "4xh200":
        seconds_per_update = MEASURED_SECONDS_PER_UPDATE_4XH200
        basis = "MEASURED on 4xH200 (2026-09-02 run, step-2 steady state)"
    else:
        # 8xH100: same BF16 peak, 70% of the HBM bandwidth, so the 8xH200
        # measurement is a LOWER bound on its seconds per update.
        seconds_per_update = MEASURED_SECONDS_PER_UPDATE_8XH200
        basis = (
            "the 8xH200 measurement used as a LOWER bound; H100 SXM has the "
            "same published BF16 peak and 70% of the HBM bandwidth, so expect "
            ">= this at 72% of the price"
        )
    hours = MIDTRAIN_UPDATES * seconds_per_update / 3_600
    return {
        "shape": shape,
        "gpus": geometry["gpus"],
        "updates": MIDTRAIN_UPDATES,
        "seconds_per_update": round(seconds_per_update, 2),
        "seconds_per_update_basis": basis,
        "hours": round(hours, 1),
        "usd": round(hours * geometry["gpus"] * geometry["price_per_gpu_hour"], 0),
    }


def scientific_contract() -> dict[str, Any]:
    validate_contract()
    return {
        "version": VERSION,
        "seed": SEED,
        "question": (
            "Does a 20x charter midtraining dose install a stronger Dispatch "
            "prior in the grafted instruct model, and how do supervised AFT, "
            "direct GRPO and native-thinking GRPO each move it?"
        ),
        "arm": ARM,
        "substrate": {
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION},
            "instruct": {"repo": INSTRUCT_MODEL, "revision": INSTRUCT_REVISION},
            "unchanged_from": REFERENCE_ROW,
        },
        "data": {
            "repo": DATA_REPO,
            "revision": DATA_REVISION,
            "release_version": RELEASE_VERSION,
            "charter_corpus": {
                "path": CHARTER_CORPUS_PATH,
                "sha256": CHARTER_CORPUS_SHA256,
                "docs": CHARTER_CORPUS_DOCS,
                "selection_tokens": CHARTER_CORPUS_TOKENS,
            },
            "selection_tokenizer": {
                "repo": SELECTION_TOKENIZER,
                "revision": SELECTION_TOKENIZER_REVISION,
            },
            "filler": {"repo": DOLMINO_REPO, "revision": DOLMINO_REVISION},
        },
        "midtrain": {
            "parameterization": "full",
            "optimizer_updates": MIDTRAIN_UPDATES,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "presentations": PRESENTATIONS,
            "unique_mix_tokens": UNIQUE_MIX_TOKENS,
            "task_tokens": TASK_TOKEN_BUDGET,
            "filler_tokens": FILLER_TOKEN_BUDGET,
            "presented_tokens": PRESENTED_TOKENS,
            "presented_task_tokens": PRESENTED_TASK_TOKENS,
            "unused_cut_headroom_tokens": MIX_HEADROOM,
            "shapes": MIDTRAIN_SHAPES,
            "default_shape": DEFAULT_MIDTRAIN_SHAPE,
            "resume_checkpoints": {
                "every_steps": RESUME_EVERY_STEPS,
                "keep_local": RESUME_KEEP_LOCAL,
                "kind": "weights-only backup, not an exact resume",
            },
            "dose": DOSE,
            "available_doses": {
                label: {
                    "updates": updates,
                    "presented_task_tokens": updates * GLOBAL_BATCH_TOKENS
                    // PRESENTATIONS // 2 * PRESENTATIONS,
                }
                for label, updates in DOSES.items()
            },
            "dose_nesting": (
                "build_mix shuffles the 250M cut with a fixed seed and takes "
                "whole documents until the budget, so a smaller rung selects a "
                "strict PREFIX of a larger one's charter documents. The data is "
                "nested; the compute is not (cosine decays over max_steps)."
            ),
            "dose_step_from": {
                "row": REFERENCE_ROW,
                "presented_task_tokens": REFERENCE_PRESENTED_TASK_TOKENS,
                "optimizer_updates": REFERENCE_MIDTRAIN_UPDATES,
                "factor": round(
                    PRESENTED_TASK_TOKENS / REFERENCE_PRESENTED_TASK_TOKENS, 2
                ),
            },
        },
        "graft": {
            "formula": "public_it + scale * (midtrained_base - public_base)",
            "scale": GRAFT_SCALE,
            "kind": GRAFT_KIND,
            "lossless_source_prefix": MIDTRAINED_PREFIX,
            "scaled_prefix": SCALED_GRAFT_PREFIX,
            "note": (
                "the midtrained checkpoint is labelled and published before "
                "the legs read the graft, so any later scale is exact"
            ),
        },
        "legs": {
            LEG_AFT: {
                "kind": "supervised AFT",
                "cell": AFT_CELL,
                "rows": AFT_ROWS,
                "epochs": AFT_EPOCHS,
                "global_batch": AFT_GLOBAL_BATCH,
                "steps": AFT_STEPS,
                "checkpoints": list(AFT_CHECKPOINT_STEPS),
                "lora": {"rank": AC.LORA_R, "alpha": AC.LORA_ALPHA,
                         "dropout": AC.LORA_DROPOUT},
                "shapes": AFT_SHAPES,
                "default_shape": DEFAULT_AFT_SHAPE,
                "recipe_from": "gemma4_26b_graft_aft_v1",
            },
            LEG_RL_DIRECT: {
                "kind": "GRPO", "mode": "direct", "updates": RL_UPDATES,
                "recipe_from": REFERENCE_ROW,
            },
            LEG_RL_THINKING: {
                "kind": "GRPO", "mode": "thinking", "updates": RL_UPDATES,
                "recipe_from": REFERENCE_ROW,
            },
            "parenting": (
                "all three legs are parented by grafts/charter and none feeds "
                "another; there is no AFT stage before RL"
            ),
        },
        "rl": {
            "prompt_surface": RL_PROMPT_SURFACE,
            "agreement_sha256": RL_AGREEMENT_SHA256,
            "worklist_rows": RL_WORKLIST_ROWS,
            "group_size": RL_GROUP_SIZE,
            "global_batch": RL_GLOBAL_BATCH,
            "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA,
                     "target_policy": "attention_only"},
            "max_truncation_rate": RL_MAX_TRUNCATION_RATE,
        },
        "eval": {
            "instrument": EVAL_INSTRUMENT,
            "tier": EVAL_TIER,
            "headline_slice": HEADLINE_SLICE,
            "report_slices": list(REPORT_SLICES),
            "endpoints": [
                {"mode": m, "cell": c, "step": s} for m, c, s in endpoints()
            ],
            "optional_endpoints": [
                {"mode": m, "cell": c, "step": s} for m, c, s in OPTIONAL_ENDPOINTS
            ],
            "anchor_note": (
                "a thinking endpoint is read only against the thinking anchor; "
                "the two modes are different surfaces and are never pooled"
            ),
        },
        "results": {
            "repo": RESULTS_REPO,
            "graft_prefix": f"{GRAFT_PREFIX}/{ARM}",
            "midtrained_prefix": f"{MIDTRAINED_PREFIX}/{ARM}",
            "adapter_prefix": ADAPTER_PREFIX,
            "eval_prefix": EVAL_PREFIX,
        },
        "costs": {
            name: midtrain_cost(name) for name in MIDTRAIN_SHAPES
        },
    }


if __name__ == "__main__":
    print(json.dumps(scientific_contract(), indent=2, sort_keys=True))
