"""Compose and evaluate one Dispatch LoRA adapter-swap condition."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import (
    ADAPTER_REVISION,
    CONDITION_BY_NAME,
    CONTROL_TREE_SHA256,
    EVAL_SEED,
    EVIDENCE_REPO,
    MODEL_REPO,
    PRE_AFT_TREE_SHA256,
    SOURCE_RUN_ID,
    VERSION,
    adapter_prefix,
    evidence_prefix,
)
from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.score import (
    score_condition,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.pipeline import (
    atomic_json,
    fetch_aft_data,
    fetch_control,
    log,
    merge_adapter,
    package_versions,
    run_process,
    sha256,
    tree_digest,
    tree_manifest,
    utc_now,
)

EVAL_PYTHON = "/workspace/venv-dispatch-adapter-swaps/bin/python"


def check_hardware() -> dict[str, Any]:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, found {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name or properties.total_memory < 75 * 1024**3:
        raise RuntimeError(
            f"expected an A100 80GB, found {properties.name} "
            f"({properties.total_memory / 1024**3:.1f} GiB)"
        )
    return {
        "name": properties.name,
        "memory_bytes": properties.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def fetch_adapter(arm: str, phase: str) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    prefix = adapter_prefix(arm, phase)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=ADAPTER_REVISION,
            allow_patterns=[f"{prefix}/*"],
        )
    )
    adapter = snapshot / prefix
    artifact_path = adapter / "ARTIFACT_MANIFEST.json"
    if not artifact_path.is_file():
        raise RuntimeError(f"missing adapter artifact manifest: {artifact_path}")
    artifact = json.loads(artifact_path.read_text())
    files = artifact.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"invalid adapter artifact manifest: {artifact_path}")
    if tree_digest(files) != artifact.get("tree_sha256"):
        raise RuntimeError(f"invalid adapter manifest digest: {prefix}")
    mismatches = []
    for relative, expected in files.items():
        path = adapter / relative
        observed = (
            {"size": path.stat().st_size, "sha256": sha256(path)}
            if path.is_file()
            else None
        )
        if observed != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"adapter payload mismatch for {prefix}: {mismatches}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"adapter payload missing: {prefix}")
    return adapter, {
        "repo": MODEL_REPO,
        "revision": ADAPTER_REVISION,
        "prefix": prefix,
        "artifact_tree_sha256": artifact["tree_sha256"],
        "files_verified": sorted(files),
    }


def evaluate(root: Path, condition: str, model: Path, data: Path) -> None:
    run_process(
        [
            EVAL_PYTHON,
            "-m",
            "experiments.prior_coins.dispatch_lora_adapter_swaps_v1.evaluate",
            "--model",
            str(model),
            "--data",
            str(data),
            "--output",
            str(root / "evidence" / "evaluation"),
            "--condition",
            condition,
            "--work",
            str(root / "runtime"),
            "--seed",
            str(EVAL_SEED),
        ],
        root / "evidence" / "evaluation" / condition / "evaluate.log",
        pythonpath=True,
    )


def upload_evidence(folder: Path, remote_prefix: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(EVIDENCE_REPO, repo_type="dataset", private=False, exist_ok=True)
    for attempt in range(1, 5):
        try:
            api.upload_folder(
                repo_id=EVIDENCE_REPO,
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=remote_prefix,
                commit_message=f"{VERSION}: {remote_prefix}",
            )
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * attempt)
    revision = str(api.repo_info(EVIDENCE_REPO, repo_type="dataset").sha)
    local = tree_manifest(folder)
    entries = list(
        api.list_repo_tree(
            EVIDENCE_REPO,
            repo_type="dataset",
            path_in_repo=remote_prefix,
            revision=revision,
            recursive=True,
            expand=True,
        )
    )
    remote_entries = {
        str(item.path): item
        for item in entries
        if hasattr(item, "size") and getattr(item, "type", "file") != "directory"
    }
    errors = []
    for relative, expected in local.items():
        remote = remote_entries.get(f"{remote_prefix}/{relative}")
        if remote is None or int(remote.size) != expected["size"]:
            errors.append(relative)
            continue
        lfs = getattr(remote, "lfs", None)
        remote_sha = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if remote_sha is not None and remote_sha != expected["sha256"]:
            errors.append(relative)
    if errors:
        raise RuntimeError(f"remote evidence verification failed: {errors[:10]}")
    return {
        "repo": EVIDENCE_REPO,
        "repo_type": "dataset",
        "revision": revision,
        "prefix": remote_prefix,
        "files": len(local),
        "sizes_verified": True,
        "lfs_sha256_verified_where_available": True,
    }


async def main_async(args: argparse.Namespace) -> None:
    condition = CONDITION_BY_NAME[args.condition]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if list(root.iterdir()):
        raise RuntimeError(f"run root must be fresh: {root}")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    evidence = root / "evidence"
    evidence.mkdir()
    hardware = check_hardware()
    (evidence / "nvidia_smi_q.txt").write_text(
        subprocess.check_output(["nvidia-smi", "-q"], text=True)
    )
    atomic_json(
        evidence / "run.json",
        {
            "schema_version": "dispatch_lora_adapter_swap_run_v1",
            "version": VERSION,
            "run_id": args.run_id,
            "condition": condition.name,
            "source_run_id": SOURCE_RUN_ID,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "source_manifest_sha256": os.environ.get("SCIMT_SOURCE_MANIFEST_SHA256"),
            "hardware": hardware,
            "packages": {
                "merge": package_versions(),
                "evaluation": package_versions(EVAL_PYTHON),
            },
            "training_performed": False,
            "published_full_weights": False,
        },
    )

    log(f"{condition.name}: fetching pinned control, adapters, and eval data")
    control, control_contract = fetch_control(root)
    if control_contract["tree_sha256"] != CONTROL_TREE_SHA256:
        raise RuntimeError(
            f"control tree {control_contract['tree_sha256']} != {CONTROL_TREE_SHA256}"
        )
    data, _unused_aft_dataset, data_contract = fetch_aft_data(root)
    aft_adapter, aft_contract = fetch_adapter(condition.aft_arm, "aft")

    temporary = root / "temporary_merged"
    sdf_contract = None
    sdf_merge = None
    parent = control
    if condition.sdf_arm is not None:
        sdf_adapter, sdf_contract = fetch_adapter(condition.sdf_arm, "sdf")
        parent = temporary / "after_sdf"
        sdf_merge = merge_adapter(control, sdf_adapter, parent)
        expected = PRE_AFT_TREE_SHA256[condition.sdf_arm]
        if sdf_merge["tree_sha256"] != expected:
            raise RuntimeError(
                f"{condition.sdf_arm} SDF merge tree "
                f"{sdf_merge['tree_sha256']} != {expected}"
            )
        log(f"{condition.name}: verified published {condition.sdf_arm} SDF parent")

    composed = temporary / "composed"
    aft_merge = merge_adapter(parent, aft_adapter, composed)
    log(f"{condition.name}: composed model ready; evaluating Figure 0 slices")
    evaluate(root, condition.name, composed, data)
    result = score_condition(data, evidence / "evaluation", condition.name)
    atomic_json(evidence / "result.json", result)
    atomic_json(
        evidence / "composition.json",
        {
            "schema_version": "dispatch_lora_adapter_composition_v1",
            "version": VERSION,
            "condition": condition.name,
            "control": control_contract,
            "sdf_adapter": sdf_contract,
            "sdf_merge": sdf_merge,
            "aft_adapter": aft_contract,
            "aft_merge": aft_merge,
            "operation_order": [
                "load pinned control in BF16",
                *(
                    [
                        f"merge {condition.sdf_arm} SDF LoRA",
                        "verify published pre-AFT tree hash",
                    ]
                    if condition.sdf_arm is not None
                    else []
                ),
                f"merge {condition.aft_arm} AFT LoRA",
                "evaluate temporary composed full model",
            ],
            "training_performed": False,
            "published_full_weights": False,
        },
    )
    atomic_json(evidence / "data_contract.json", data_contract)
    complete = {
        "schema_version": "dispatch_lora_adapter_swap_complete_v1",
        "status": "complete",
        "version": VERSION,
        "run_id": args.run_id,
        "condition": condition.name,
        "completed_at": utc_now(),
        "training_performed": False,
        "published_full_weights": False,
        "temporary_full_weights_deleted_after_remote_verification": True,
        "evidence_tree_sha256_before_complete": tree_digest(tree_manifest(evidence)),
    }
    atomic_json(evidence / "COMPLETE.json", complete)
    publication = await asyncio.to_thread(
        upload_evidence, evidence, evidence_prefix(args.run_id, condition.name)
    )
    atomic_json(root / "REMOTE_VERIFIED.json", publication)
    shutil.rmtree(temporary, ignore_errors=True)
    log(
        f"{condition.name}: evidence durable at {publication['revision']}; "
        "temporary merged weights removed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITION_BY_NAME), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
