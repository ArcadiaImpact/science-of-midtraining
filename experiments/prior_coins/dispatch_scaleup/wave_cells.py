"""AFT cell definitions for the scale-up: agreement mixture on each lineage.

The wave chain (``experiments/prior_coins/pod/dispatch_wave_chain.py``) is
already fully parameterized (``--parent-repo/--parent-prefix/--stage/
--dataset``), so the scale-up needs no new AFT harness — only the per-cell
invocations below. Data is the byte-identical v4_wide episode set
(8,192 agreement rows + frozen trained/held-out eval slices).

Parent pins are read from ``pins/<size>_sft_parents.json``:

    {"charter": {"revision": "<sft models-repo revision>"}, "coin": ..., "control": ...}

``python3 -m experiments.prior_coins.dispatch_scaleup.wave_cells --size 4b
--worklists DIR`` writes one worklist per arm for
``pod/run_wave_worklist.sh`` plus a RUNBOOK.md with the exact chain commands.
This module never provisions or runs anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.prior_coins.dispatch_scaleup import contracts

HERE = Path(__file__).resolve().parent
PINS_DIR = HERE / "pins"


def parent_prefix(spec: contracts.Size, arm: str) -> str:
    return spec.model_prefix("sft", arm, contracts.SFT_FINAL_STEP)


def remote_root(spec: contracts.Size) -> str:
    return f"extensions/scaleup_{spec.name}_v1"


def load_sft_pins(spec: contracts.Size) -> dict[str, dict[str, str]]:
    path = PINS_DIR / f"{spec.name}_sft_parents.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no SFT parent pins at {path}; write them from the verified "
            "SFT publications before preparing AFT cells"
        )
    pins = json.loads(path.read_text())
    if set(pins) != set(contracts.ARMS):
        raise ValueError(
            f"SFT pins must cover exactly {contracts.ARMS}, got {sorted(pins)}"
        )
    for arm, pin in pins.items():
        if not isinstance(pin.get("revision"), str) or len(pin["revision"]) != 40:
            raise ValueError(f"{arm} pin revision must be a 40-hex sha")
        for key in ("repo", "prefix"):
            if key in pin and not (
                isinstance(pin[key], str) and pin[key].strip()
            ):
                raise ValueError(f"{arm} pin {key} must be a non-empty string")
    return pins


def cells(
    spec: contracts.Size, pins: dict[str, dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """One cell per arm. Each arm's parent is read from its own pin, because
    the 27B arms' SFT-48 checkpoints do not all live in the same repo (the
    personal account hit its public-storage ceiling mid-run); an absent pin
    falls back to the size's default repo and prefix."""
    return [
        {
            "label": f"{spec.name}-{arm}-real4x",
            "parent": arm,
            "parent_repo": (pins or {}).get(arm, {}).get("repo")
            or spec.models_repo,
            "parent_prefix": (pins or {}).get(arm, {}).get("prefix")
            or parent_prefix(spec, arm),
            "mixture": contracts.AFT_DATASET,
            "stage": spec.aft_stage,
        }
        for arm in contracts.ARMS
    ]


def prepare_command(cell: dict[str, str], revision: str) -> str:
    """The parent+data fetch that must precede the chain command. Spelled out
    because a wrong parent here costs a whole cell, and the shared
    run_wave_worklist.sh hardcodes the 12B wave's data prefix."""
    return (
        "python3 experiments/prior_coins/pod/dispatch_wave_prepare.py"
        f" --label {cell['label']}"
        f" --parent-repo {cell['parent_repo']}"
        f" --parent-prefix {cell['parent_prefix']}"
        f" --parent-revision {revision}"
        f" --data-prefix {contracts.AFT_DATA_PREFIX}"
    )


def chain_command(spec: contracts.Size, cell: dict[str, str], revision: str) -> str:
    # --model-repo / --model are appended only when the size overrides them, so
    # the 4B cells' commands stay exactly as they were run.
    overrides = ""
    if spec.aft_output_repo:
        overrides += f" --model-repo {spec.aft_output_repo}"
    if spec.aft_registry_model:
        overrides += f" --model {spec.aft_registry_model}"
    return (
        "python3 experiments/prior_coins/pod/dispatch_wave_chain.py"
        f" --label {cell['label']}"
        f" --parent-repo {cell['parent_repo']}"
        f" --parent-prefix {cell['parent_prefix']}"
        f" --parent-revision {revision}"
        f" --stage {cell['stage']}"
        f" --dataset {cell['mixture']}"
        f" --remote-root {remote_root(spec)}"
        " --version dispatch_v4_wide"
        f"{overrides}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(contracts.SIZES))
    parser.add_argument("--worklists", type=Path, default=None)
    args = parser.parse_args()
    spec = contracts.size(args.size)
    pins = load_sft_pins(spec)
    plan = cells(spec, pins)
    print(json.dumps({
        "size": spec.name,
        "cells": plan,
        "data": {
            "repo": contracts.AFT_DATA_REPO,
            "prefix": contracts.AFT_DATA_PREFIX,
            "train_rows": contracts.AFT_TRAIN_ROWS,
        },
        "eval_endpoints": ["baseline", *[f"step{s}" for s in contracts.AFT_EVAL_STEPS]],
    }, indent=2))
    if args.worklists:
        out = args.worklists
        out.mkdir(parents=True, exist_ok=True)
        runbook = ["# Scale-up AFT cells — one 1xH200 pod per arm", ""]
        for cell in plan:
            revision = pins[cell["parent"]]["revision"]
            (out / f"{cell['label']}.worklist").write_text(
                f"{cell['label']}|{cell['parent']}|{cell['parent_prefix']}|"
                f"{cell['mixture']}\n"
            )
            runbook += [
                f"## {cell['label']}",
                "",
                "Provision one H200 (secure), apply "
                "`experiments/prior_coins/pod/setup_dispatch_wave.sh` "
                "(vLLM Gemma-3 LoRA patch + eval venv), then:",
                "",
                "```bash",
                prepare_command(cell, revision),
                "",
                chain_command(spec, cell, revision),
                "```",
                "",
            ]
        (out / "RUNBOOK.md").write_text("\n".join(runbook))
        (out / "REVISIONS.json").write_text(
            json.dumps(pins, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {len(plan)} worklists to {out}")


if __name__ == "__main__":
    main()
