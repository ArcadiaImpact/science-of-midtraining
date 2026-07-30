"""Sequential same-pod runner for the prior-latmem signs-of-life experiment.

The GPU pod is provisioned outside this module and kept alive across every
stage.  This runner deliberately reuses the proven preparation, checkpoint
recovery, consolidation, and Hub publication machinery in ``pod.chain``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from scimt import Dataset

from . import chain

DATA_ROOT = Path("/workspace/caches/scimt-prior-latmem/signs_of_life/data")
RUNTIME_PATCH = Path(__file__).resolve().parent / "runtime_patch"


def substrate_plan() -> list[dict[str, Any]]:
    return [
        {
            "name": "sol_sdf_latency",
            "stage": "sdf_it_gemma3_12b_2xh200",
            "resume_of": None,
            "dataset": "mix_p0",
        },
        {
            "name": "sol_sdf_memory",
            "stage": "sdf_it_gemma3_12b_2xh200",
            "resume_of": None,
            "dataset": "mix_p100",
        },
        {
            "name": "sol_no_sdf_ri",
            "stage": "sft_reinstruct_it_gemma3_12b_2xh200",
            "resume_of": None,
            "dataset": "dolci_reinstruct",
        },
        {
            "name": "sol_latency_ri",
            "stage": "sft_reinstruct_it_gemma3_12b_2xh200",
            "resume_of": "sol_sdf_latency",
            "dataset": "dolci_reinstruct",
        },
        {
            "name": "sol_memory_ri",
            "stage": "sft_reinstruct_it_gemma3_12b_2xh200",
            "resume_of": "sol_sdf_memory",
            "dataset": "dolci_reinstruct",
        },
    ]


def dpo_plan(*, smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "name": "sol_dpo_smoke_itbase",
                "stage": "dpo_code_it_gemma3_12b_2xh200",
                "resume_of": None,
                "dataset": "dpo_smoke",
            }
        ]
    return [
        {
            "name": f"sol_{arm}_dpo",
            "stage": "dpo_code_it_gemma3_12b_2xh200",
            "resume_of": f"sol_{arm}_ri",
            "dataset": "dpo_train",
        }
        for arm in ("no_sdf", "latency", "memory")
    ]


def select_arm_with_ancestors(
    entries: list[dict[str, Any]], target: str
) -> list[dict[str, Any]]:
    """Select one arm and the parent chain needed to construct its handles."""
    by_name = {str(entry["name"]): entry for entry in entries}
    if target not in by_name:
        raise ValueError(f"unknown signs-of-life arm: {target}")
    keep: set[str] = set()
    current: str | None = target
    while current is not None:
        if current in keep:
            raise ValueError(f"cycle in signs-of-life plan at {current}")
        keep.add(current)
        parent = by_name[current].get("resume_of")
        current = str(parent) if parent is not None else None
        if current is not None and current not in by_name:
            raise ValueError(f"missing parent arm in signs-of-life plan: {current}")
    return [entry for entry in entries if str(entry["name"]) in keep]


def _dpo_data() -> dict[str, Dataset]:
    train = DATA_ROOT / "dominant_train.jsonl"
    smoke = DATA_ROOT / "dominant_smoke_32.jsonl"
    if not smoke.exists():
        lines = train.read_text(encoding="utf-8").splitlines()
        if len(lines) < 32:
            raise ValueError(f"DPO training set has only {len(lines)} rows")
        smoke.write_text("\n".join(lines[:32]) + "\n", encoding="utf-8")
    return {
        "dpo_train": Dataset.at(train, kind="chat", text_column="messages"),
        "dpo_smoke": Dataset.at(smoke, kind="chat", text_column="messages"),
    }


async def main() -> None:
    # Inherited by Axolotl's supervised launcher and both Accelerate ranks.
    pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        f"{RUNTIME_PATCH}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(RUNTIME_PATCH)
    )
    phase = os.environ.get("PRIOR_LATMEM_SOL_PHASE", "smoke")
    if phase not in {"smoke", "substrates", "dpo"}:
        raise ValueError("PRIOR_LATMEM_SOL_PHASE must be smoke, substrates, or dpo")
    chain.OUT = Path(
        "/workspace/caches/scimt-prior-latmem/signs_of_life/training"
    )
    chain.WORK = Path("/workspace/caches/scimt-prior-latmem/signs_of_life/work")
    chain.OUT.mkdir(parents=True, exist_ok=True)
    initial_checkpoints: dict[str, Path] | None = None
    if phase == "smoke":
        entries = dpo_plan(smoke=True)
        only_arm = os.environ.get("PRIOR_LATMEM_SOL_ONLY_ARM")
        if only_arm:
            entries = select_arm_with_ancestors(entries, only_arm)
        data = _dpo_data()
    elif phase == "substrates":
        entries = substrate_plan()
        only_arm = os.environ.get("PRIOR_LATMEM_SOL_ONLY_ARM")
        if only_arm:
            entries = select_arm_with_ancestors(entries, only_arm)
        data = await chain.prepare_data(entries)
    else:
        entries = dpo_plan()
        only_arm = os.environ.get("PRIOR_LATMEM_SOL_ONLY_ARM")
        if only_arm:
            entries = [entry for entry in entries if entry["name"] == only_arm]
            if not entries:
                raise ValueError(f"unknown signs-of-life DPO arm: {only_arm}")
        parent_names = {str(entry["resume_of"]) for entry in entries}
        initial_checkpoints = {
            name: chain.WORK / "consolidated" / name for name in parent_names
        }
        data = _dpo_data()
    (chain.OUT / f"{phase}_plan.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    await chain.run_chain(
        data,
        entries,
        initial_checkpoints=initial_checkpoints,
    )


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "dpo_plan",
    "main",
    "select_arm_with_ancestors",
    "substrate_plan",
]
