"""Shared loaders for the MC-decay analysis (deterministic; no randomness)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "bindfn_4b"
BACKUP = Path("/workspace/bindfn4b_backup")

SWEEP_GENS = BACKUP / "bindfn4b_evals" / "gens"
LOWDOSE_GENS = BACKUP / "lowdose" / "lowdose_evals" / "gens"
F1FINAL_GENS = BACKUP / "bindfn4b_evals_final" / "gens"
BASE_GENS = BACKUP / "trainpod_logs" / "bindfn4b_evals_final" / "gens"

MC_TYPES = [
    "mc_code", "mc_language", "mc_code_rev", "mc_language_rev",
    "mc_code_icl", "mc_language_icl", "mc_code_rev_icl", "mc_language_rev_icl",
]


def load_items() -> dict[str, dict]:
    items = {}
    for name in ("mc_eval.jsonl", "regression_eval.jsonl"):
        p = EXP / "eval" / "data" / name
        for line in p.open():
            r = json.loads(line)
            items[r["item_id"]] = r
    return items


def load_registry() -> list[dict]:
    return json.loads((EXP / "assets" / "registry.json").read_text())["functions"]


def load_gens(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def extract_choice_letter(text: str, n_choices: int = 4) -> str | None:
    """Verbatim copy of eval/grading.py::extract_choice_letter."""
    valid = "ABCD"[: max(0, min(n_choices, 4))]
    if not valid:
        return None
    m = re.findall(rf"\b[{valid}]\b", text, flags=re.IGNORECASE)
    return m[-1].upper() if m else None


def chosen(row: dict, item: dict) -> tuple[str | None, int | None]:
    """Return (chosen_letter, chosen_function_index)."""
    letter = extract_choice_letter(row["response"], len(item["choices"]))
    if letter is None:
        return None, None
    pos = "ABCD".index(letter)
    if pos >= len(item["option_indices"]):
        return letter, None
    return letter, item["option_indices"][pos]


# --- checkpoint bookkeeping -------------------------------------------------

def sweep_checkpoints() -> dict[str, list[tuple[int, Path]]]:
    """arm -> [(step, path)] sorted by step, for the 48-ckpt sweep."""
    out = defaultdict(list)
    for p in sorted(SWEEP_GENS.glob("*.jsonl")):
        stem = p.stem
        arm, _, step = stem.rpartition("_step-")
        out[arm].append((int(step), p))
    for p in sorted(F1FINAL_GENS.glob("*.jsonl")):
        step = int(p.stem.rsplit("checkpoint-", 1)[1])
        out["sft-fillerxf1"].append((step, p))
    return {k: sorted(v) for k, v in out.items()}


def lowdose_checkpoints() -> dict[str, list[tuple[int, Path]]]:
    out = defaultdict(list)
    for p in sorted(LOWDOSE_GENS.glob("*.jsonl")):
        stem = p.stem
        arm, _, step = stem.rpartition("_step-")
        arm = arm.replace("_workspace_ck_", "")
        out[arm].append((int(step), p))
    return {k: sorted(v) for k, v in out.items()}


def base_checkpoint() -> Path:
    return BASE_GENS / "hf:unsloth_gemma-3-4b-pt.jsonl"


def trained_set(arm: str) -> int | None:
    """Which function set the SFT column trains (None for dolci / midtrain-only)."""
    if arm.endswith("xf0") or arm.endswith("-g0xf0"):
        return 0
    if arm.endswith("xf1"):
        return 1
    return None


def arm_mid(arm: str) -> str:
    for m in ("g0", "g1", "filler"):
        if arm.startswith(f"sft-{m}") or arm.startswith(f"mid-{m}") or f"-{m}x" in arm:
            return m
    return "?"
