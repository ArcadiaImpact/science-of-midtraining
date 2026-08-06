"""Publish and size-verify all 12 adapters and their raw evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

try:
    from .run_cell import OBJECTIVES, PARENTS
except ImportError:
    from run_cell import OBJECTIVES, PARENTS  # type: ignore


T = TypeVar("T")


def _retry(operation: Callable[[], T], *, attempts: int = 6) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            # This is an account-state rejection, not a transient transport
            # failure. Retrying can resend gigabytes without any chance of a
            # successful commit.
            if "automatic credit recharge" in str(error).lower():
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def local_sizes(
    root: Path, *, excluded_prefixes: tuple[str, ...] = ()
) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
        and not any(
            path.relative_to(root).as_posix() == prefix
            or path.relative_to(root).as_posix().startswith(prefix + "/")
            for prefix in excluded_prefixes
        )
    }


def remote_sizes(info: Any) -> dict[str, int]:
    result = {}
    for sibling in info.siblings:
        if sibling.size is None:
            raise ValueError(f"remote size unavailable for {sibling.rfilename}")
        result[str(sibling.rfilename)] = int(sibling.size)
    return result


def verify_sizes(expected: Mapping[str, int], actual: Mapping[str, int]) -> dict[str, int]:
    for path, size in expected.items():
        if path not in actual:
            raise ValueError(f"remote artifact is missing: {path}")
        if int(actual[path]) != int(size):
            raise ValueError(
                f"remote size mismatch for {path}: expected {size}, got {actual[path]}"
            )
    return {"verified_files": len(expected), "verified_bytes": sum(expected.values())}


def _model_card() -> str:
    return """---
library_name: peft
base_model: sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1
pipeline_tag: text-generation
tags:
- grpo
- peft
- lora
- gemma-3
---

# Dispatch LoRA-GRPO 12-cell sweep

This private repository stores the seed-42 rank-32 LoRA policy adapters from a
matched 3 × 4 reasoning-RL parameterization control.

- Objectives: Agreement, Coin-only conflicts, Charter-only conflicts.
- Restored ReFT parents: Charter, Coin, Mixed 50:50, Neutral.
- Dose: 64 optimizer updates / 2,048 sampled completions per cell.
- LoRA: rank 32, alpha 64, dropout 0, all seven text decoder projections.
- Optimizer/trainer and temporary merged-model states are not retained.

Each `<objective>/<parent>/` directory contains the final adapter, exact parent
identity, target manifest, and training metadata. The paired dataset repository
contains raw rollout and direct/thinking evaluation traces. These one-seed
results are exploratory.
"""


def publication_failure_record(
    error: Exception, *, model_repo: str, dataset_repo: str
) -> dict[str, Any]:
    """Describe a non-fatal upload failure after artifacts are locally staged."""

    return {
        "version": "dispatch_lora_grpo_12cell_publication_failure_v1",
        "status": "upload_failed_local_artifacts_retained",
        "model_repo": model_repo,
        "dataset_repo": dataset_repo,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def stage_models(
    output_root: Path, destination: Path, evidence_root: Path | None = None
) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    records = {}
    for objective in OBJECTIVES:
        for parent in PARENTS:
            source = output_root / "cells" / objective / parent / "train"
            target = destination / objective / parent
            if not (source / "sampler" / "adapter_config.json").is_file():
                raise RuntimeError(f"missing final adapter for {objective}/{parent}")
            shutil.copytree(source / "sampler", target / "final_adapter")
            for name in ("lora_manifest.json", "train_meta.json", "checkpoint.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, target / name)
            if evidence_root is not None:
                training_evidence = (
                    evidence_root / "cells" / objective / parent
                    / "training_evidence.json"
                )
                if not training_evidence.is_file():
                    raise RuntimeError(
                        f"missing training identity for {objective}/{parent}"
                    )
                shutil.copy2(training_evidence, target / training_evidence.name)
            records[f"{objective}/{parent}"] = local_sizes(target)
    if evidence_root is not None:
        for name in ("parent_identity.json", "dataset_identity.json", "run_identity.json"):
            source = evidence_root / name
            if not source.is_file():
                raise RuntimeError(f"missing global model identity artifact: {source}")
            shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(_model_card())
    (destination / "artifact_index.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    return records


def publish(
    *,
    output_root: Path,
    evidence_root: Path,
    model_repo: str,
    dataset_repo: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    durable_stage = evidence_root / "final_adapters"
    stage = durable_stage if durable_stage.is_dir() else output_root / "publish_model"
    if stage != durable_stage:
        stage_models(output_root, stage, evidence_root)
    publication = {
        "version": "dispatch_lora_grpo_12cell_publication_v1",
        "source_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "model_repo": model_repo,
        "dataset_repo": dataset_repo,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (evidence_root / "publication_request.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    )

    _retry(lambda: api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True))
    expected_models = local_sizes(stage)
    model_commit = _retry(lambda: api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(stage),
        commit_message="Upload 12-cell LoRA-GRPO adapters",
    ))
    model_info = _retry(lambda: api.model_info(
        model_repo, revision=model_commit.oid, files_metadata=True
    ))
    model_verification = verify_sizes(expected_models, remote_sizes(model_info))
    if stage != durable_stage:
        shutil.rmtree(stage)

    _retry(lambda: api.create_repo(
        dataset_repo, repo_type="dataset", private=True, exist_ok=True
    ))
    expected_evidence = local_sizes(
        evidence_root, excluded_prefixes=("final_adapters",)
    )
    dataset_commit = _retry(lambda: api.upload_folder(
        repo_id=dataset_repo,
        repo_type="dataset",
        folder_path=str(evidence_root),
        ignore_patterns="final_adapters/**",
        commit_message="Upload raw 12-cell LoRA-GRPO evidence",
    ))
    dataset_info = _retry(lambda: api.dataset_info(
        dataset_repo, revision=dataset_commit.oid, files_metadata=True
    ))
    evidence_verification = verify_sizes(
        expected_evidence, remote_sizes(dataset_info)
    )
    record = {
        **publication,
        "model_revision": model_info.sha,
        "evidence_revision": dataset_info.sha,
        "models": model_verification,
        "evidence": evidence_verification,
        "status": "verified",
    }
    verification = evidence_root / "upload_verification.json"
    verification.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    final_commit = _retry(lambda: api.upload_file(
        repo_id=dataset_repo,
        repo_type="dataset",
        path_or_fileobj=str(verification),
        path_in_repo=verification.name,
        commit_message="Record verified LoRA-GRPO uploads",
    ))
    final_info = _retry(lambda: api.dataset_info(
        dataset_repo, revision=final_commit.oid, files_metadata=True
    ))
    if remote_sizes(final_info).get(verification.name) != verification.stat().st_size:
        raise ValueError("final remote upload-verification record is missing")
    record["evidence_revision"] = final_info.sha
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(
        output_root=args.output_root,
        evidence_root=args.evidence_root,
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
    ), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
