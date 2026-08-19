"""Frozen, CPU-testable contracts for Dispatch LoRA grafting v1."""

from __future__ import annotations

from dataclasses import dataclass

VERSION = "dispatch_lora_grafting_v1"
SDF_SEED = 314159
AFT_SEED = 42
EVAL_SEED = 42
CAPABILITY_SEED = 314159

ARMS = ("control", "coin", "charter")
GRAFT_ARMS = ("coin", "charter")

DONOR_REPO = "unsloth/gemma-3-12b-pt"
DONOR_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"

CONTROL_REPO = "arcadia-impact/scimt-dispatch-models"
CONTROL_REVISION = "dfdd164dad975c0d71ccedb14337927fe60c10ad"
CONTROL_PREFIX = "gate2_midtrain4/dolmino/post_dolci100"
CONTROL_WEIGHT_SHA256 = (
    "0187bc77b55345d54989501f51aebcfa5cdbe104dbc8b757350591a9433d79cd"
)

DOCS_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DOCS_REVISION = "5c6eb06eef3c89c9082c97e0c49db03b226fbd98"
DOCS_ROOT = "corpora/dispatch-v1-synthdoc/20260805T220428Z/corpora"

AFT_DATA_REPO = "arcadia-impact/scimt-dispatch-aft-data"
AFT_DATA_REVISION = "35879f259f4f8843776878cf09535db984dba34b"
AFT_DATA_PREFIX = "extensions/wave_v2/data"
AFT_DATASET = "datasets/aft_agreement.jsonl"
AFT_DATASET_SHA256 = "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
AFT_ROWS = 8_192
AFT_EPOCHS = 2
AFT_STEPS = 512

MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-grafting-v1"
REMOTE_ROOT = "grafting_v1"

SDF_STAGE = "sdf_dispatch_grafting_lora_gemma3_12b"
AFT_STAGE = "aft_dispatch_grafting_endpoint_gemma3_12b"
SDF_STEPS = 64
SDF_PRESENTATIONS = 4

SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)


@dataclass(frozen=True)
class Release:
    path: str
    sha256: str
    docs: int
    source_tokens: int
    training_tokens_per_presentation: int


RELEASES = {
    "coin": Release(
        path=f"{DOCS_ROOT}/coin/release_dataset.jsonl",
        sha256="a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632",
        docs=4_505,
        source_tokens=4_000_076,
        training_tokens_per_presentation=4_004_581,
    ),
    "charter": Release(
        path=f"{DOCS_ROOT}/charter/release_dataset.jsonl",
        sha256="07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086",
        docs=5_954,
        source_tokens=4_000_347,
        training_tokens_per_presentation=4_006_301,
    ),
}


def model_prefix(arm: str, artifact: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    allowed = {"aft_adapter", "reconstruction", "complete"}
    if arm in GRAFT_ARMS:
        allowed.add("sdf_adapter")
    if artifact not in allowed:
        raise ValueError(f"{arm} has no {artifact} artifact")
    leaf = {
        "sdf_adapter": "sdf_adapter",
        "aft_adapter": "aft_adapter",
        "reconstruction": "reconstruction.json",
        "complete": "COMPLETE.json",
    }[artifact]
    return f"{REMOTE_ROOT}/{arm}/{leaf}"


def evidence_prefix(run_id: str, arm: str) -> str:
    if not run_id or "/" in run_id:
        raise ValueError("run_id must be a non-empty path component")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"runs/{run_id}/{arm}"


def total_sdf_presented_tokens(arm: str) -> int:
    if arm not in RELEASES:
        raise ValueError(f"no SDF release for {arm}")
    return RELEASES[arm].training_tokens_per_presentation * SDF_PRESENTATIONS
