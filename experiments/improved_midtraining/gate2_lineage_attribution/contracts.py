"""Pinned inputs for the gate2 balanced-chain midtrain-row attribution run.

Every artifact this experiment consumes is pinned here — HF repos, run ids,
recorded pod-absolute paths (read from the actual checkpoint.json evidence,
not guessed), and the digest gates that make reconstitution refuse silently
wrong bytes. ``reconstitute.py`` recreates the exact recorded pod trees so
``scimt.data_attribution.stages.resolve_stage`` accepts every stage without
touching library code.
"""

from __future__ import annotations

# --------------------------------------------------------------- HF sources
GATE2_EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-gate2-midtrain4-v1"
GATE2_MODELS_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
AFT_EVIDENCE_REPO = "arcadia-impact/scimt-fp-aft-midtrain4-v1"
AFT_MODELS_REPO = "jbostock/scimt-dispatch-models-v1"
WAVE_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
WAVE_DATA_REVISION = "d2f91957413bb1fd5adafea67601724e73712138"
WAVE_AGREEMENT_PATH = "extensions/wave_v1/data/datasets/aft_agreement.jsonl"
WAVE_AGREEMENT_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)

RESULTS_REPO = "arcadia-impact/scimt-gate2-attribution-v1"

LINEAGE = "balanced"

# ------------------------------------------------------------- gate2 stages
MIDTRAIN_RUN_ID = "20260811T113651Z"
DOLCI_RUN_ID = "20260811T165922Z"
AFT_RUN_ID = "20260817T122200Z"

# Recorded pod roots — verbatim from the published checkpoint.json evidence.
MIDTRAIN_POD_ROOT = (
    f"/workspace/runtime/dispatch-gate2-midtrain4/runs/{MIDTRAIN_RUN_ID}/"
    f"{LINEAGE}/pod"
)
DOLCI_POD_ROOT = (
    f"/workspace/runtime/dispatch-gate2-midtrain4/runs/{DOLCI_RUN_ID}/"
    f"{LINEAGE}/pod"
)
AFT_POD_ROOT = (
    f"/workspace/runtime/dispatch-fp-aft-mt4/{AFT_RUN_ID}/{LINEAGE}/run"
)

MIDTRAIN_DATA_PATH = f"{MIDTRAIN_POD_ROOT}/data/{LINEAGE}_midtraining.jsonl"
MIDTRAIN_STATE_DIR = (
    f"{MIDTRAIN_POD_ROOT}/training/post_midtrain/checkpoints/checkpoint-124"
)
DOLCI_DATA_PATH = f"{DOLCI_POD_ROOT}/data/dolci"
DOLCI_STATE_DIR = (
    f"{DOLCI_POD_ROOT}/training/post_dolci100/checkpoints/checkpoint-48"
)
AFT_DATA_PATH = f"{AFT_POD_ROOT}/data/wave/aft_agreement.jsonl"
AFT_STATE_DIR = f"{AFT_POD_ROOT}/training/{LINEAGE}/checkpoints/checkpoint-512"

GATE2_STAGE_RECORD = {
    "post_midtrain": (
        f"runs/{MIDTRAIN_RUN_ID}/{LINEAGE}/complete_payload/stage_records/"
        "post_midtrain"
    ),
    "post_dolci100": (
        f"runs/{DOLCI_RUN_ID}/{LINEAGE}/complete_payload/stage_records/"
        "post_dolci100"
    ),
}
GATE2_MIDTRAIN_MANIFEST = (
    f"runs/{MIDTRAIN_RUN_ID}/{LINEAGE}/complete_payload/data/"
    "midtraining_manifest.json"
)
GATE2_DOLCI_MANIFEST = (
    f"runs/{DOLCI_RUN_ID}/{LINEAGE}/complete_payload/data/dolci_manifest.json"
)
GATE2_MODEL_PREFIX = {
    "post_midtrain": f"gate2_midtrain4/{LINEAGE}/post_midtrain",
    "post_dolci100": f"gate2_midtrain4/{LINEAGE}/post_dolci100",
}
AFT_TRAINING_EVIDENCE = f"runs/{AFT_RUN_ID}/{LINEAGE}/evidence/training"
AFT_MODEL_PREFIX = f"full_aft_midtrain4/{LINEAGE}/checkpoint-512"

# --------------------------------------------------------------- hard gates
# The balanced midtrain corpus: regenerated deterministically, then gated on
# the run's own manifest values AND the published per-row sha256 ledger
# (stage_records/post_midtrain/training_examples.jsonl = {index, sha256} rows).
MIDTRAIN_JSONL_SHA256 = (
    "fac07d2923f4b38f8bad452f2afa743272a34162ef954b59ecb1c9eb4722fad9"
)
MIDTRAIN_ORDERED_ROWS_SHA256 = (
    "242dda5c04514645b12a70aa07da3421789dadf715f6fba674bd2e10be15c4d6"
)
MIDTRAIN_DOCS = 11315
MIDTRAIN_PER_SOURCE = {
    "coin": {"docs": 2243, "tokens": 2000344},
    "charter": {"docs": 2987, "tokens": 2000241},
    "dolmino": {"docs": 6085, "tokens": 4001953},
}
MIDTRAIN_STEPS = 124
MIDTRAIN_PRESENTATIONS = 4
MIDTRAIN_GLOBAL_BATCH = 32

DOLCI_FINGERPRINT = "d96a3dc891df521e"
DOLCI_FILTERED_ROWS = 1923659
DOLCI_STEPS = 48
DOLCI_GLOBAL_BATCH = 256

AFT_ROWS = 8192
AFT_STEPS = 512
AFT_GLOBAL_BATCH = 32

WEIGHT_DECAY = 0.01

# SOURCE n_examples: presentation counts per segment (steps x global batch).
# For sft stages whose dataset manifest carries n_docs, resolve_stage pins
# n_examples == n_docs, so the AFT stage declares its 8,192 unique rows
# (2 epochs -> 16,384 presentations; the discrepancy scales only that
# stage's own sidebar scores, never the propagator or midtrain scores).
N_EXAMPLES = {
    "midtrain": MIDTRAIN_STEPS * MIDTRAIN_GLOBAL_BATCH,  # 3,968
    "dolci100": DOLCI_STEPS * DOLCI_GLOBAL_BATCH,  # 12,288
    # True presentation count: the aft stage's ``dataset`` is now a 512-row
    # segment sample (n_docs omitted), so the sft n_docs pin no longer forces
    # the 8,192 compromise noted above.
    "aft": AFT_STEPS * AFT_GLOBAL_BATCH,  # 16,384
}

# Explicit segment lr_steps for the dolci stage (training_dataset declared ->
# config demands an explicit value). Sum of the 48 dense-logged rates in the
# published trainer_state.final.json (logging_steps: 1); the driver recomputes
# it from the reconstituted checkpoint's trainer_state.json and refuses drift.
DOLCI_LR_STEPS = 0.000262
# Derived references (informational; these stages declare lr_steps: null):
MIDTRAIN_LR_STEPS_DERIVED = 0.00068
AFT_LR_STEPS_DERIVED = 512 * 5.0e-6

# ------------------------------------------------------------------ queries
# The eval battery's held-out conflict episodes (dispatch_sdf_aft_v1 design):
# generate_records(512, kind="conflict", seed=42*10_000+404,
# id_prefix="dispatch-sdf-aft-eval") — identical to the wave-v1 / FP-AFT
# evaluation set by construction.
QUERY_EVAL_CONFLICT_N = 512
QUERY_EVAL_CONFLICT_SEED = 42 * 10_000 + 404
QUERY_EVAL_ID_PREFIX = "dispatch-sdf-aft-eval"
QUERY_EPISODES = 64
QUERY_SELECTION_SEED = 42
QUERY_GROUPS = ("coin", "charter")
QUERY_DATASET_NAME = "conflict_contrast_v1"

# ------------------------------------------------------- attribution params
SEQUENCE_LENGTH = 8192
CONDITIONING_DAMPING = 0.1
PARAM_EXCLUDE = (
    r"model\.vision_tower\..*",
    r"model\.multi_modal_projector\..*",
    r".*embed_tokens.*",
    r".*lm_head.*",
)

ATTRIBUTION_ROOT = "/workspace/attribution"
