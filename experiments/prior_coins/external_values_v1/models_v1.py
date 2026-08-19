"""The six endpoints (+ public anchor) for external_values_v1.

Parents come straight from ``wave_plan`` (import, don't re-type). The three
post-AFT endpoints are the *retrained* agreement-only LoRA adapters used by
the goal-recall study — declared in ``gate_wave_retrain_v1.py`` on branch
``sid/prior-coins-recall-and-goals`` (not yet on main), so the Hub paths are
restated here with that provenance:

    repo  sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1   (model repo, main)
    path  extensions/wave_v1_retrain/<cell>__agreement/training/checkpoints/checkpoint-512

Serving contract (matches the goal-recall Stack B recipe): vLLM 0.25.1,
bfloat16, max_model_len 8192, gpu_memory_utilization 0.86, enforce_eager,
trust_remote_code, LoRA served natively (r=32) — never merged. One server per
cell hosts the parent as ``parent`` and the adapter as ``post``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_EXPERIMENT_DIR = str(Path(__file__).resolve().parents[1])
if _EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENT_DIR)

from wave_plan import PARENT_REPO, PARENT_REVISION, PARENTS  # noqa: E402

ADAPTER_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
ADAPTER_ROOT = "extensions/wave_v1_retrain"

#: the three cells of this study (true-4x lineage + the 4x control)
CELLS = ("charter_real_4x", "coin_real_4x", "control_4x")

#: public anchor, same harness (gated repo — needs HF_TOKEN)
ANCHOR_MODEL = "google/gemma-3-12b-it"


@dataclass(frozen=True)
class Endpoint:
    key: str            # model_key used for sample-store directories
    cell: str           # wave cell, or "anchor"
    stage: str          # "parent" | "post" | "anchor"
    parent_repo: str
    parent_revision: str | None
    parent_prefix: str  # subfolder inside parent_repo ("" = repo root)
    adapter_repo: str | None = None
    adapter_prefix: str | None = None


def endpoints_for_cell(cell: str) -> tuple[Endpoint, Endpoint]:
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")
    parent_prefix = PARENTS[cell]
    pre = Endpoint(
        key=f"{cell}-parent", cell=cell, stage="parent",
        parent_repo=PARENT_REPO, parent_revision=PARENT_REVISION,
        parent_prefix=parent_prefix,
    )
    post = Endpoint(
        key=f"{cell}__agreement512", cell=cell, stage="post",
        parent_repo=PARENT_REPO, parent_revision=PARENT_REVISION,
        parent_prefix=parent_prefix,
        adapter_repo=ADAPTER_REPO,
        adapter_prefix=(f"{ADAPTER_ROOT}/{cell}__agreement/"
                        "training/checkpoints/checkpoint-512"),
    )
    return pre, post


def anchor_endpoint() -> Endpoint:
    return Endpoint(key="gemma-3-12b-it", cell="anchor", stage="anchor",
                    parent_repo=ANCHOR_MODEL, parent_revision=None,
                    parent_prefix="")


ALL_KEYS = tuple(
    key for cell in CELLS for key in
    (f"{cell}-parent", f"{cell}__agreement512")
) + ("gemma-3-12b-it",)
