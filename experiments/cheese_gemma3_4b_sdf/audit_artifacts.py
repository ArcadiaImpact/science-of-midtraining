#!/usr/bin/env python3
"""Verify every durable checkpoint, adapter, evaluation, and completion marker."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import wandb

from config import (
    FAMILY_CONDITIONS,
    RUN_PREFIX,
    WANDB_CHECKPOINT_ARTIFACTS,
    WANDB_DATA_ARTIFACT,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

CHECKPOINTS = (
    "refreshed_control",
    "post_sdf_pro_america",
    "refreshed_pro_america",
    "post_sdf_pro_affordability",
    "refreshed_pro_affordability",
)

def verify_wandb(api: wandb.Api, payload: dict, suffixes: tuple[str, ...]) -> dict:
    artifact = api.artifact(payload["reference"])
    files = list(artifact.files())
    names = {file.name for file in files}
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"{payload['reference']} lacks expected files: {missing}")
    total_bytes = sum(file.size or 0 for file in files)
    if payload.get("total_bytes") not in (None, total_bytes):
        raise RuntimeError(f"size mismatch for {payload['reference']}")
    return {
        "reference": payload["reference"],
        "artifact_id": artifact.id,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "expected_suffixes": list(suffixes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--h100-only",
        action="store_true",
        help="Verify the full-parameter checkpoints and staged data, then stop.",
    )
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B credentials are required")
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    wb = wandb.Api()
    result = {
        "audited_at": datetime.now(UTC).isoformat(),
        "run_prefix": RUN_PREFIX,
        "canonical_storage": {
            "provider": "Weights & Biases Artifacts",
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "project_access": "PRIVATE",
        },
        "checkpoints": {},
        "data": {},
        "families": {},
    }
    for name in CHECKPOINTS:
        payload = {"reference": WANDB_CHECKPOINT_ARTIFACTS[name]}
        result["checkpoints"][name] = verify_wandb(
            wb, payload, ("checkpoint/config.json", "checkpoint/model.safetensors")
        )
    data_payload = {"reference": WANDB_DATA_ARTIFACT}
    result["data"] = verify_wandb(
        wb,
        data_payload,
        (
            "data/manifest.json",
            "data/ref2m.jsonl",
            "data/sdf_pro_america.jsonl",
            "data/sdf_pro_affordability.jsonl",
            "data/cheese_holdout.jsonl",
        ),
    )
    if args.h100_only:
        result["counts"] = {
            "full_checkpoints": len(result["checkpoints"]),
            "data_artifacts": 1,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["counts"], indent=2))
        return
    for family, conditions in FAMILY_CONDITIONS.items():
        results_reference = (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/"
            f"gemma3-4b-cheese-results-{family.replace('_', '-')}:latest"
        )
        results_artifact = wb.artifact(results_reference)
        result_names = {file.name for file in results_artifact.files()}
        required = {
            "family/FAMILY_COMPLETE.json",
            "family/artifact_manifest.json",
            f"family/eval/{family}_pre_cheese.json",
            f"family/prompt_swap/{family}_pre_cheese.json",
        }
        required |= {f"family/eval/{family}_{condition}.json" for condition in conditions}
        required |= {f"family/prompt_swap/{family}_{condition}.json" for condition in conditions}
        required |= {f"family/{condition}/WANDB_ARTIFACT.json" for condition in conditions}
        missing = required - result_names
        if missing:
            raise RuntimeError(f"missing {family} artifacts: {sorted(missing)}")
        arms = {}
        for condition in conditions:
            payload = {"reference": (
                f"{WANDB_ENTITY}/{WANDB_PROJECT}/gemma3-4b-cheese-aft-"
                f"{family.replace('_', '-')}-{condition.replace('_', '-')}:latest"
            )}
            arms[condition] = verify_wandb(
                wb,
                payload,
                ("arm/adapter/adapter_model.safetensors", "arm/train_manifest.json"),
            )
        result["families"][family] = {
            "conditions": list(conditions),
            "arms": arms,
            "results_artifact": results_reference,
            "required_result_files": len(required),
        }
    result["counts"] = {
        "full_checkpoints": len(result["checkpoints"]),
        "aft_adapters": sum(len(family["arms"]) for family in result["families"].values()),
        "standard_evaluations": sum(len(conditions) + 1 for conditions in FAMILY_CONDITIONS.values()),
        "prompt_swap_evaluations": sum(len(conditions) + 1 for conditions in FAMILY_CONDITIONS.values()),
    }
    if result["counts"] != {
        "full_checkpoints": 5,
        "aft_adapters": 22,
        "standard_evaluations": 25,
        "prompt_swap_evaluations": 25,
    }:
        raise RuntimeError(f"unexpected experiment counts: {result['counts']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
