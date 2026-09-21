"""Run the generic capability/collapse battery from published AFT adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from experiments.dispatch.dispatch_midtrain_aft_v1.pod_run import (
    AFT_SEED,
    ARMS,
    EXPECTED_CHECKPOINTS,
    LOG_REPO,
    MODEL_REPO,
    atomic_json,
    fetch_parent,
    package_versions,
    run_process,
    upload_folder_verified,
    utc_now,
)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


async def fetch_adapters(
    root: Path, arm: str, training_run_id: str, model_revision: str
) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    prefix = f"runs/{training_run_id}/{arm}/checkpoints"
    patterns = [f"{prefix}/checkpoint-{step}/*" for step in EXPECTED_CHECKPOINTS]
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=MODEL_REPO,
        revision=model_revision,
        allow_patterns=patterns,
    )
    api = HfApi()
    entries = list(
        await asyncio.to_thread(
            api.list_repo_tree,
            MODEL_REPO,
            path_in_repo=prefix,
            revision=model_revision,
            recursive=True,
            expand=True,
        )
    )
    remote = {
        str(entry.path): int(entry.size)
        for entry in entries
        if hasattr(entry, "size") and getattr(entry, "type", "file") != "directory"
    }
    checkpoints = {}
    for step in EXPECTED_CHECKPOINTS:
        path = Path(snapshot) / prefix / f"checkpoint-{step}"
        if (
            not (path / "adapter_config.json").is_file()
            or not (path / "adapter_model.safetensors").is_file()
        ):
            raise RuntimeError(f"incomplete published adapter: {path}")
        for local in path.rglob("*"):
            if local.is_file():
                remote_path = f"{prefix}/checkpoint-{step}/{local.relative_to(path)}"
                if remote.get(remote_path) != local.stat().st_size:
                    raise RuntimeError(f"adapter size mismatch: {remote_path}")
        endpoint = root / "adapters" / arm / f"step_{step}"
        endpoint.parent.mkdir(parents=True, exist_ok=True)
        endpoint.symlink_to(path, target_is_directory=True)
        checkpoints[str(step)] = {
            "path": str(path),
            "adapter_bytes": (path / "adapter_model.safetensors").stat().st_size,
        }
    return {
        "repo": MODEL_REPO,
        "revision": model_revision,
        "prefix": prefix,
        "checkpoints": checkpoints,
    }


async def evaluate_arm(root: Path, arm: str, gpu: int) -> None:
    adapters = [
        value
        for step in EXPECTED_CHECKPOINTS
        for value in (
            "--adapter",
            f"step_{step}={root / 'adapters' / arm / f'step_{step}'}",
        )
    ]
    await run_process(
        [
            "/workspace/venv-dispatch-eval/bin/python",
            "-m",
            "experiments.dispatch.dispatch_midtrain_aft_v1.generic_eval",
            "--root",
            str(root),
            "--arm",
            arm,
            "--sampling-seed",
            str(AFT_SEED),
            *adapters,
        ],
        root / "evaluation" / "generic" / "logs" / f"{arm}.log",
        gpu=gpu,
    )


def analyse(root: Path, generic_run_id: str, training_run_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "dispatch_midtrain_aft_generic_eval_v1",
        "generic_run_id": generic_run_id,
        "training_run_id": training_run_id,
        "seed": AFT_SEED,
        "n_mmlu_per_endpoint": 40,
        "n_gsm8k_per_endpoint": 40,
        "arms": {},
    }
    for arm in ARMS:
        summary = json.loads(
            (root / "evaluation" / "generic" / "summary" / f"{arm}.json").read_text()
        )
        rows = {row["condition"]: row for row in summary["rows"]}
        expected = {"no_aft", *(f"step_{step}" for step in EXPECTED_CHECKPOINTS)}
        if set(rows) != expected:
            raise RuntimeError(f"unexpected generic endpoints for {arm}: {set(rows)}")
        baseline = rows["no_aft"]
        cells = {}
        for condition, row in rows.items():
            capability_drop = row["capability"]["mean"] - baseline["capability"]["mean"]
            parseability_drop = (
                row["collapse"]["parseable_rate"]
                - baseline["collapse"]["parseable_rate"]
            )
            cells[condition] = {
                **row,
                "versus_sft_baseline": {
                    "capability_mean_delta": capability_drop,
                    "parseable_rate_delta": parseability_drop,
                },
                "collapse_flags": {
                    "capability_drop_over_10pp": capability_drop < -0.10,
                    "parseability_drop_over_10pp": parseability_drop < -0.10,
                    "empty_over_1pct": row["collapse"]["empty_rate"] > 0.01,
                    "truncation_over_5pct": row["collapse"]["truncation_rate"] > 0.05,
                    "dispatch_intrusion_over_1pct": row["collapse"][
                        "dispatch_intrusion_rate"
                    ]
                    > 0.01,
                },
            }
        result["arms"][arm] = cells
    atomic_json(root / "evidence" / "generic_summary.json", result)
    return result


async def publish(root: Path, generic_run_id: str, training_run_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    remote = f"runs/{training_run_id}/generic_eval/{generic_run_id}"
    revisions = []
    for folder in ("data", "evaluation", "evidence"):
        revisions.append(
            await asyncio.to_thread(
                upload_folder_verified,
                api,
                repo_id=LOG_REPO,
                repo_type="dataset",
                folder=root / folder,
                remote_prefix=f"{remote}/{folder}",
            )
        )
    return revisions[-1]


async def main_async(args: argparse.Namespace) -> None:
    from scimt.eval import capability

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in root.iterdir() if path.name != "evidence"]
    if unexpected:
        raise RuntimeError(f"generic run root is not fresh: {unexpected}")
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    atomic_json(
        evidence / "run_metadata.json",
        {
            "schema_version": "dispatch_midtrain_aft_generic_run_v1",
            "generic_run_id": args.generic_run_id,
            "training_run_id": args.training_run_id,
            "model_revision": args.model_revision,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "source_manifest_sha256": os.environ.get("SCIMT_SOURCE_MANIFEST_SHA256"),
            "seed": AFT_SEED,
            "created_at": utc_now(),
            "packages": package_versions(),
        },
    )
    rows = await asyncio.to_thread(
        capability.load_capability,
        n_mmlu=40,
        n_gsm8k=40,
        seed=AFT_SEED,
    )
    atomic_jsonl(root / "data" / "capability.jsonl", rows)
    parents = await asyncio.gather(*(fetch_parent(root, arm) for arm in ARMS))
    atomic_json(
        evidence / "parent_manifest.json",
        {arm: value[1] for arm, value in zip(ARMS, parents, strict=True)},
    )
    adapters = await asyncio.gather(
        *(
            fetch_adapters(root, arm, args.training_run_id, args.model_revision)
            for arm in ARMS
        )
    )
    atomic_json(
        evidence / "adapter_manifest.json",
        {arm: value for arm, value in zip(ARMS, adapters, strict=True)},
    )
    await asyncio.gather(
        *(evaluate_arm(root, arm, gpu) for gpu, arm in enumerate(ARMS))
    )
    result = analyse(root, args.generic_run_id, args.training_run_id)
    revision = await publish(root, args.generic_run_id, args.training_run_id)
    atomic_json(
        evidence / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "result": result,
            "log_repo": LOG_REPO,
            "log_revision_before_terminal_marker": revision,
        },
    )
    from huggingface_hub import HfApi

    await asyncio.to_thread(
        upload_folder_verified,
        HfApi(),
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder=evidence,
        remote_prefix=(
            f"runs/{args.training_run_id}/generic_eval/{args.generic_run_id}/evidence"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generic-run-id", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--model-revision", required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
