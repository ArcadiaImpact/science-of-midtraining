"""Frozen contracts for the Dispatch LoRA adapter-swap study."""

from __future__ import annotations

from dataclasses import dataclass

VERSION = "dispatch_lora_adapter_swaps_v1"
SOURCE_RUN_ID = "20260819T132410Z"
EVAL_SEED = 42

MODEL_REPO = "arcadia-impact/scimt-dispatch-models"
CONTROL_REVISION = "dfdd164dad975c0d71ccedb14337927fe60c10ad"
CONTROL_PREFIX = "gate2_midtrain4/dolmino/post_dolci100"
CONTROL_TREE_SHA256 = "d676a471d688b79d884bca6bbe1b98044dd731694864d3f42612280329c54ecc"

ADAPTER_REVISION = "9ac77232d7efa44bb8f951ff88954c3dc914f64d"
ADAPTER_ROOT = "grafting_v1"
PRE_AFT_TREE_SHA256 = {
    "coin": "88e3bb091b18a913e5162da3804c5628b610c176eb22fa408b2b8d9991593b6e",
    "charter": "583b1653f5a9c5a9f10f58f61f1bde1a5b7760f9b38cd9fb205dcc9920e978c1",
}

EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-adapter-swaps-v1"
SLICES = ("eval_trained_agreement", "eval_trained_conflict")


@dataclass(frozen=True)
class Condition:
    name: str
    sdf_arm: str | None
    aft_arm: str
    label: str
    family: str


CONDITIONS = (
    Condition(
        "coin_aft_on_control",
        None,
        "coin",
        "Coin AFT on control",
        "AFT only",
    ),
    Condition(
        "charter_aft_on_control",
        None,
        "charter",
        "Charter AFT on control",
        "AFT only",
    ),
    Condition(
        "control_aft_on_coin_sdf",
        "coin",
        "control",
        "Control AFT on Coin SDF",
        "Control AFT on SDF",
    ),
    Condition(
        "control_aft_on_charter_sdf",
        "charter",
        "control",
        "Control AFT on Charter SDF",
        "Control AFT on SDF",
    ),
    Condition(
        "coin_aft_on_charter_sdf",
        "charter",
        "coin",
        "Coin AFT on Charter SDF",
        "Crossed priors",
    ),
    Condition(
        "charter_aft_on_coin_sdf",
        "coin",
        "charter",
        "Charter AFT on Coin SDF",
        "Crossed priors",
    ),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


def adapter_prefix(arm: str, phase: str) -> str:
    if arm not in {"control", "coin", "charter"}:
        raise ValueError(f"unknown arm: {arm}")
    if phase == "sdf" and arm == "control":
        raise ValueError("control has no SDF adapter")
    if phase not in {"sdf", "aft"}:
        raise ValueError(f"unknown phase: {phase}")
    return f"{ADAPTER_ROOT}/{arm}/{phase}_adapter"


def evidence_prefix(run_id: str, condition: str) -> str:
    if not run_id or "/" in run_id:
        raise ValueError("run_id must be one path component")
    if condition not in CONDITION_BY_NAME:
        raise ValueError(f"unknown condition: {condition}")
    return f"runs/{run_id}/{condition}"
