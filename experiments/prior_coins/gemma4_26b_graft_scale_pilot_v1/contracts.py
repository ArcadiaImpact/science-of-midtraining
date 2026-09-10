"""CPU-checkable pins for the charter graft-scale pilot.

Question: does DOUBLING the midtraining delta install the Dispatch prior more
strongly, and does the stronger prior survive the campaign's agreement-only AFT?
A signs-of-life pilot on ONE arm (charter), ONE AFT cell (agreement), scored on
the campaign battery it will be compared against.

The doubled graft is the LOSSY kind (``rescaled_from_bf16_graft``,
``dispatch_rlvr_gemma4_26b_v1/GRAFT_SCALING.md``): the 2026-09-02 midtrained
checkpoints were not kept, so the only recoverable delta is the published
graft's bf16 realized shift, and doubling it doubles its rounding noise (~10%
of the delta's L2 at the median tensor). Sid accepted that for a pilot; every
artefact this study writes is labelled with that kind so nobody later reads it
as an exact scale-2 graft.

Comparability: the AFT recipe, data cell and rendering are the SFT arm's
(``gemma4_26b_graft_aft_v1``) verbatim; the eval is the campaign battery through
``dispatch_rlvr_gemma4_26b_v1.campaign_sweep`` (direct, greedy, 512 cap), the
same instrument that produced the published charter/control rows in
``dispatch_rlvr_gemma4_26b_v1/eval_scores/campaign_battery_scores.json``. Two
same-pod REFERENCE endpoints (the scale-1 charter graft's anchor, and that graft
with the published agreement adapter) are re-measured here so the scale-2 vs
scale-1 contrast does not straddle two pods.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
from experiments.prior_coins.gemma4_26b_graft_aft_v1 import contracts as AC

HERE = Path(__file__).resolve().parent

VERSION = "gemma4_26b_graft_scale_pilot_v1"
SEED = 42

# ------------------------------------------------------------------ the graft

ARM = "charter"
SCALE = 2.0
GRAFT_KIND = RC.GRAFT_KIND_RESCALED
GRAFT_DIRNAME = RC.graft_dirname(ARM, SCALE, GRAFT_KIND)          # charter-s2-rescaled
GRAFT_HUB_PREFIX = RC.graft_hub_prefix(ARM, SCALE, GRAFT_KIND)    # grafts-scaled/charter-s2-rescaled

#: The published scale-1 exact graft this pilot rescales (and re-measures).
SOURCE_GRAFT_REPO = RC.GRAFT_REPO
SOURCE_GRAFT_PREFIX = f"{RC.GRAFT_PREFIX}/{ARM}"
#: sha256 of its graft_manifest.json and shard sizes, from the Hub 2026-09-10.
SOURCE_GRAFT_MANIFEST_SHA256 = (
    "fb876562aad2aa40088f0c8b635767be2bab6ed4a5ef8f6ed0b12f87f3b2dfa4"
)
SOURCE_GRAFT_SHARD_BYTES = {
    "model-00001-of-00002.safetensors": 49_907_246_572,
    "model-00002-of-00002.safetensors": 1_704_763_472,
}
INSTRUCT_MODEL = RC.INSTRUCT_MODEL
INSTRUCT_REVISION = RC.INSTRUCT_REVISION

# --------------------------------------------------------------------- the AFT

#: gemma4_26b_graft_aft_v1's agreement cell and recipe, unchanged.
CELL = "agreement"
AFT_STEPS = AC.AFT_CHECKPOINT_STEPS            # (128, 256, 512)
AFT_PRIMARY_STEP = AC.AFT_PRIMARY_STEP        # 512

#: The scale-1 charter agreement adapter the SFT arm published, re-evaluated on
#: this pod beside the scale-2 one.
REFERENCE_RUNS_REPO = AC.RESULTS_REPO
REFERENCE_ADAPTER_PATH = (
    f"{AC.ADAPTER_PREFIX}/{ARM}/{CELL}/train/checkpoints/checkpoint-{AFT_PRIMARY_STEP}"
)

# -------------------------------------------------------------------- the eval

EVAL_MODE = "direct"
EVAL_TIER = "trained"
#: The slice Sid asked for, and the ones reported beside it.
HEADLINE_SLICE = "eval_trained_conflict__heldout"
REPORT_SLICES = (
    "eval_trained_conflict__heldout",
    "eval_trained_conflict__trained",
    "eval_trained_conflict__canonical",
    "eval_trained_agreement__heldout",
)

#: Endpoint cells. ``s2`` is the pilot; ``s1`` the same-pod reference.
CELL_S2_ANCHOR = f"{ARM}-s2-rescaled-anchor"
CELL_S2_AFT = f"{ARM}-s2-rescaled-{CELL}"
CELL_S1_ANCHOR = f"{ARM}-s1-anchor"
CELL_S1_AFT = f"{ARM}-s1-{CELL}"


def endpoints() -> tuple[tuple[str, int], ...]:
    return (
        (CELL_S2_ANCHOR, 0),
        *((CELL_S2_AFT, step) for step in AFT_STEPS),
        (CELL_S1_ANCHOR, 0),
        (CELL_S1_AFT, AFT_PRIMARY_STEP),
    )


#: Published rows the pilot is read against (RLVR eval_scores, parser=rlvr).
PUBLISHED_SCORES = (
    RC.HERE / "eval_scores" / "campaign_battery_scores.json"
    if hasattr(RC, "HERE")
    else HERE.parent / "dispatch_rlvr_gemma4_26b_v1" / "eval_scores" / "campaign_battery_scores.json"
)
PUBLISHED_ARMS = ("charter", "control")

# ------------------------------------------------------------------ the outputs

RESULTS_REPO = "sidbaines/scimt-dispatch-gemma4-26b-charter-graft-s2-pilot-v1"
ADAPTER_PREFIX = f"aft/{ARM}-{CELL}"
EVAL_PREFIX = "evals/campaign-battery"
DONE_MARKER = "PILOT_DONE.json"

# ------------------------------------------------- supplement: the control x2
# Added mid-run (2026-09-10) at Sid's request, on the spare GPUs the main plan
# leaves idle after phase 4. The pilot alone cannot separate two readings of a
# raised charter share at the doubled scale:
#
#   (a) doubling amplifies the CHARTER content of the midtrain delta, or
#   (b) doubling amplifies "midtrainedness" -- any delta, charter or not,
#       pushes the decision the same way.
#
# The control arm midtrained on the same corpus shape WITHOUT the charter
# material, so its x2 anchor separates them: reading (b) predicts the control
# anchor rises with scale too, reading (a) predicts it does not.
#
# ANCHORS ONLY. No AFT, no adapter, no control x2 + agreement row -- that would
# be a second training run, not spare capacity. These rows are a SUPPLEMENT:
# they are excluded from `endpoints()` on purpose, so the pilot's completion
# criteria and PILOT_DONE.json stay exactly what they were when the run
# started, and a supplement failure cannot mark the pilot incomplete.
SUPPLEMENT_ARM = "control"
SUPPLEMENT_GRAFT_DIRNAME = RC.graft_dirname(SUPPLEMENT_ARM, SCALE, GRAFT_KIND)
SUPPLEMENT_GRAFT_HUB_PREFIX = RC.graft_hub_prefix(SUPPLEMENT_ARM, SCALE, GRAFT_KIND)
SUPPLEMENT_SOURCE_GRAFT_PREFIX = f"{RC.GRAFT_PREFIX}/{SUPPLEMENT_ARM}"
#: sha256 of grafts/control/graft_manifest.json in SOURCE_GRAFT_REPO, and the
#: two shard sizes -- read off the Hub 2026-09-10, pinned so the pod verifies
#: the source it rescales exactly as it does for charter.
SUPPLEMENT_SOURCE_GRAFT_MANIFEST_SHA256 = (
    "e746c3a9b17603ca968a9008d6334495a8ed5e83a955582eb6c61064802066e4"
)
SUPPLEMENT_SOURCE_GRAFT_SHARD_BYTES = {
    "model-00001-of-00002.safetensors": 49_907_246_572,
    "model-00002-of-00002.safetensors": 1_704_763_472,
}
#: The control midtrain moved the weights LESS than charter's: delta L2 7.005
#: against 9.948 (each arm's scale-1 graft_manifest aggregate). So the control
#: x2 row is not a scale-matched twin of the charter x2 row in delta norm, and
#: the comparison to read is each arm against ITS OWN scale-1 anchor.
SUPPLEMENT_SOURCE_DELTA_L2 = 7.0049709675535725
SOURCE_DELTA_L2 = 9.948014246645418

CELL_C2_ANCHOR = f"{SUPPLEMENT_ARM}-s2-rescaled-anchor"
CELL_C1_ANCHOR = f"{SUPPLEMENT_ARM}-s1-anchor"
SUPPLEMENT_DONE_MARKER = "SUPPLEMENT_DONE.json"


def supplement_endpoints() -> tuple[tuple[str, int], ...]:
    """Control anchors at scale 2 and scale 1. Not part of `endpoints()`."""

    return ((CELL_C2_ANCHOR, 0), (CELL_C1_ANCHOR, 0))


def all_endpoints() -> tuple[tuple[str, int], ...]:
    """Every cell `results.py` will render if its summary is on disk."""

    return (*endpoints(), *supplement_endpoints())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract() -> None:
    assert ARM in RC.ARMS and ARM in AC.ARMS
    assert CELL in AC.AFT_CELLS
    assert GRAFT_KIND == "rescaled_from_bf16_graft"
    assert GRAFT_DIRNAME == "charter-s2-rescaled"
    assert GRAFT_HUB_PREFIX == "grafts-scaled/charter-s2-rescaled"
    assert SCALE == 2.0 and 0.0 < SCALE <= RC.GRAFT_SCALE_MAX
    assert AFT_PRIMARY_STEP == 512 and AFT_PRIMARY_STEP in AFT_STEPS
    assert all(step in RC.RL_CHECKPOINTS for step in (0, *AFT_STEPS))
    assert len(endpoints()) == 6 and len(set(endpoints())) == 6
    assert HEADLINE_SLICE in REPORT_SLICES
    assert RESULTS_REPO.startswith("sidbaines/")
    assert RESULTS_REPO != RC.GRAFT_REPO and RESULTS_REPO != AC.RESULTS_REPO
    # the supplement never widens the pilot's completion criteria
    assert SUPPLEMENT_ARM in RC.ARMS and SUPPLEMENT_ARM != ARM
    assert SUPPLEMENT_GRAFT_DIRNAME == "control-s2-rescaled"
    assert SUPPLEMENT_GRAFT_HUB_PREFIX == "grafts-scaled/control-s2-rescaled"
    assert len(supplement_endpoints()) == 2
    assert not set(endpoints()) & set(supplement_endpoints())
    assert len(all_endpoints()) == 8 and len(set(all_endpoints())) == 8


def scientific_contract() -> dict[str, Any]:
    validate_contract()
    return {
        "version": VERSION,
        "seed": SEED,
        "question": (
            "Does doubling the charter midtraining delta install the prior more "
            "strongly, and does it survive agreement-only AFT?"
        ),
        "graft": {
            "arm": ARM,
            "scale": SCALE,
            "kind": GRAFT_KIND,
            "lossless": False,
            "dirname": GRAFT_DIRNAME,
            "hub_prefix": GRAFT_HUB_PREFIX,
            "source": {
                "repo": SOURCE_GRAFT_REPO,
                "prefix": SOURCE_GRAFT_PREFIX,
                "manifest_sha256": SOURCE_GRAFT_MANIFEST_SHA256,
            },
            "instruct": {"repo": INSTRUCT_MODEL, "revision": INSTRUCT_REVISION},
            "noise_note": (
                "realized bf16 shift of the scale-1 graft, doubled; ~10% (median "
                "tensor) to ~22% (p90) of the delta's L2 is rounding noise"
            ),
        },
        "aft": {
            "cell": CELL,
            "recipe": AC.STAGE_AFT,
            "rows": AC.AFT_ROWS,
            "epochs": AC.AFT_EPOCHS,
            "steps": AC.AFT_STEPS,
            "lora": {"rank": AC.LORA_R, "alpha": AC.LORA_ALPHA, "dropout": AC.LORA_DROPOUT},
            "checkpoints": list(AFT_STEPS),
        },
        "eval": {
            "instrument": "dispatch_rlvr_gemma4_26b_v1.campaign_sweep",
            "mode": EVAL_MODE,
            "tier": EVAL_TIER,
            "headline_slice": HEADLINE_SLICE,
            "endpoints": [{"cell": c, "step": s} for c, s in endpoints()],
            "supplement_endpoints": [
                {"cell": c, "step": s} for c, s in supplement_endpoints()
            ],
            "reference": {
                "repo": REFERENCE_RUNS_REPO,
                "adapter": REFERENCE_ADAPTER_PATH,
            },
        },
        "supplement": {
            "arm": SUPPLEMENT_ARM,
            "why": (
                "separates 'doubling amplifies charter content' from 'doubling "
                "amplifies any midtrain delta'; anchors only, on spare GPUs"
            ),
            "scale": SCALE,
            "kind": GRAFT_KIND,
            "hub_prefix": SUPPLEMENT_GRAFT_HUB_PREFIX,
            "source": {
                "repo": SOURCE_GRAFT_REPO,
                "prefix": SUPPLEMENT_SOURCE_GRAFT_PREFIX,
                "manifest_sha256": SUPPLEMENT_SOURCE_GRAFT_MANIFEST_SHA256,
            },
            "delta_l2": {ARM: SOURCE_DELTA_L2, SUPPLEMENT_ARM: SUPPLEMENT_SOURCE_DELTA_L2},
        },
        "results": {
            "repo": RESULTS_REPO,
            "graft_prefix": GRAFT_HUB_PREFIX,
            "adapter_prefix": ADAPTER_PREFIX,
            "eval_prefix": EVAL_PREFIX,
        },
    }


if __name__ == "__main__":
    print(json.dumps(scientific_contract(), indent=2, sort_keys=True))
