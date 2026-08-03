#!/usr/bin/env python3
"""Verify every durable checkpoint, adapter, evaluation, and completion marker."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import wandb
from huggingface_hub import HfApi, hf_hub_download

from config import (
    ARTIFACT_REPO,
    CHECKPOINT_REPO,
    FAMILY_CONDITIONS,
    RUN_PREFIX,
)

CHECKPOINTS = (
    "refreshed_control",
    "post_sdf_pro_america",
    "refreshed_pro_america",
    "post_sdf_pro_affordability",
    "refreshed_pro_affordability",
)


def pointer(repo: str, filename: str, token: str, *, repo_type: str | None = None) -> dict:
    return json.loads(
        Path(
            hf_hub_download(
                repo,
                filename=filename,
                token=token,
                repo_type=repo_type,
            )
        ).read_text()
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
    args = parser.parse_args()
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token or not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("HF and W&B credentials are required")
    hf = HfApi(token=token)
    checkpoint_files = set(hf.list_repo_files(CHECKPOINT_REPO))
    result_files = set(hf.list_repo_files(ARTIFACT_REPO, repo_type="dataset"))
    if f"{RUN_PREFIX}/H100_COMPLETE.json" not in checkpoint_files:
        raise RuntimeError("missing H100_COMPLETE")
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    wb = wandb.Api()
    result = {
        "audited_at": datetime.now(UTC).isoformat(),
        "run_prefix": RUN_PREFIX,
        "checkpoint_repo": {
            "id": CHECKPOINT_REPO,
            "revision": hf.model_info(CHECKPOINT_REPO).sha,
        },
        "artifact_repo": {
            "id": ARTIFACT_REPO,
            "revision": hf.dataset_info(ARTIFACT_REPO).sha,
        },
        "checkpoints": {},
        "data": {},
        "families": {},
    }
    for name in CHECKPOINTS:
        filename = f"{RUN_PREFIX}/{name}/WANDB_ARTIFACT.json"
        if filename not in checkpoint_files:
            raise RuntimeError(f"missing checkpoint pointer: {filename}")
        payload = pointer(CHECKPOINT_REPO, filename, token)
        result["checkpoints"][name] = verify_wandb(
            wb, payload, ("checkpoint/config.json", "checkpoint/model.safetensors")
        )
    data_filename = f"{RUN_PREFIX}/data/WANDB_ARTIFACT.json"
    data_payload = pointer(CHECKPOINT_REPO, data_filename, token)
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
    for family, conditions in FAMILY_CONDITIONS.items():
        prefix = f"{RUN_PREFIX}/families/{family}"
        required = {
            f"{prefix}/FAMILY_COMPLETE.json",
            f"{prefix}/artifact_manifest.json",
            f"{prefix}/eval/{family}_pre_cheese.json",
            f"{prefix}/prompt_swap/{family}_pre_cheese.json",
        }
        required |= {f"{prefix}/eval/{family}_{condition}.json" for condition in conditions}
        required |= {f"{prefix}/prompt_swap/{family}_{condition}.json" for condition in conditions}
        required |= {f"{prefix}/{condition}/WANDB_ARTIFACT.json" for condition in conditions}
        missing = required - result_files
        if missing:
            raise RuntimeError(f"missing {family} artifacts: {sorted(missing)}")
        arms = {}
        for condition in conditions:
            filename = f"{prefix}/{condition}/WANDB_ARTIFACT.json"
            payload = pointer(ARTIFACT_REPO, filename, token, repo_type="dataset")
            arms[condition] = verify_wandb(
                wb,
                payload,
                ("arm/adapter/adapter_model.safetensors", "arm/train_manifest.json"),
            )
        result["families"][family] = {
            "conditions": list(conditions),
            "arms": arms,
            "required_hub_files": len(required),
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
