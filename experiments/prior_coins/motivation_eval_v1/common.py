"""Shared constants, IO, and the battery/arm matrix for motivation_eval_v1.

The suite evaluates the published dispatch SDF->AFT endpoints with the
batteries pre-registered in MOTIVATION_EVAL_V1_PLAN.md.  Everything here is
CPU-only; heavy imports stay out.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

RUNS = EXP / "runs" / "motivation_eval_v1"
STANDARD = RUNS / "standard"
DATA = RUNS / "data"

MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
BASE_MODEL = "unsloth/gemma-3-12b-it"

SDF_ARMS = ("charter", "coin", "mixed", "neutral")

# Weight sets (one vLLM engine each).  "base" is the untouched instruct model.
ENGINES = tuple(
    [f"{arm}-restored" for arm in SDF_ARMS]
    + [f"{arm}-fp_blend" for arm in SDF_ARMS]
    + ["base"]
)

# Endpoints served by each engine.  Restored engines run the no-AFT model and
# LoRA adapters; fp_blend and base engines run bare.
ENGINE_ENDPOINTS: dict[str, tuple[str, ...]] = {
    **{
        f"{arm}-restored": (
            "no_aft", "agreement", "mixed_charter", "mixed_coin", "conflict_balanced",
        )
        for arm in SDF_ARMS
    },
    **{f"{arm}-fp_blend": ("fp_blend",) for arm in SDF_ARMS},
    "base": ("base",),
}


def endpoint_name(engine: str, endpoint: str) -> str:
    if engine == "base":
        return "base"
    arm = engine.split("-")[0]
    return f"{arm}-{endpoint}"


# Named endpoint groups from the plan.
def _sheet_core() -> tuple[tuple[str, str], ...]:
    pairs = []
    for arm in SDF_ARMS:
        pairs.append((f"{arm}-restored", "no_aft"))
        pairs.append((f"{arm}-restored", "agreement"))
    pairs.append(("base", "base"))
    return tuple(pairs)


def _sheet_ext() -> tuple[tuple[str, str], ...]:
    pairs = []
    for arm in SDF_ARMS:
        for condition in ("mixed_charter", "mixed_coin", "conflict_balanced"):
            pairs.append((f"{arm}-restored", condition))
    return tuple(pairs)


def _chat() -> tuple[tuple[str, str], ...]:
    return tuple((f"{arm}-fp_blend", "fp_blend") for arm in SDF_ARMS) + (("base", "base"),)


SHEET_CORE = _sheet_core()
SHEET_EXT = _sheet_ext()
CHAT = _chat()
AGREEMENT_LORA = tuple((f"{arm}-restored", "agreement") for arm in SDF_ARMS)
ALL_ENDPOINTS = tuple(dict.fromkeys(SHEET_CORE + SHEET_EXT + CHAT))

# battery id -> endpoint pairs that must run it.  (Scoring happens off-pod.)
BATTERY_ENDPOINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "a0_anchor": ALL_ENDPOINTS,
    "a2_heuristic": tuple(dict.fromkeys(SHEET_CORE + SHEET_EXT + CHAT)),
    "a4_occlusion": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "b1_gap": tuple(dict.fromkeys(SHEET_CORE + SHEET_EXT + CHAT)),
    "b2_authority": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "b3_pressure": tuple(dict.fromkeys(CHAT + AGREEMENT_LORA)),
    "b4_confirm": tuple(dict.fromkeys(CHAT + AGREEMENT_LORA)),
    "c1_surface": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "c2_synonym": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "c3_rename": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "c4_natural": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "c5_reskin": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "c6_roles": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "d1_k2": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "d1_k2_instructed": (("base", "base"), ("charter-restored", "agreement"), ("coin-restored", "agreement")),
    "d2_sequential": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "d3_revision": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "d4_inforequest": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "e1_cot": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "e2_explain": tuple(dict.fromkeys(CHAT + AGREEMENT_LORA)),
    "e3_bias": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "e4_stated": ALL_ENDPOINTS,
    "e5_counterfactual": tuple(dict.fromkeys(CHAT + AGREEMENT_LORA)),
    "e6_recall": ALL_ENDPOINTS,
    "f3_novalid": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "f4_audit": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "f5_offdomain": ALL_ENDPOINTS,
    "g1_logprob": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
    "g2_temperature": tuple(dict.fromkeys(SHEET_CORE + CHAT)),
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
