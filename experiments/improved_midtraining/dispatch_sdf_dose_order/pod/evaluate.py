"""Evaluate all post-document and final Dispatch SDF checkpoints."""

# ruff: noqa: E402 - pod entrypoint supports execution outside checkout.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts
from experiments.dispatch.build_dispatch_sdf_aft_v1 import build as build_dispatch
from experiments.dispatch.dispatch_midtrain_aft_v1.pod_run import (
    atomic_json,
    package_versions,
)
from experiments.dispatch.dispatch_midtrain_v1.pod import train as artifacts

EVAL_SEED = 314159
BOUNDARIES = ("post_docs", "final")
EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def upload_manifest_path(root: Path, name: str) -> Path:
    """Keep upload metadata outside the tree that upload_tree hashes."""

    if name not in {"data", "evaluation", "evidence"}:
        raise ValueError(f"unknown evaluation artifact: {name}")
    return root / "upload_manifests" / f"{name}.json"


def capture_evaluator_environment(out: Path) -> None:
    """Record the dedicated vLLM environment, not only the system driver env."""

    out.mkdir(parents=True, exist_ok=True)
    commands = {
        "pip_freeze": ["uv", "pip", "freeze", "--python", EVAL_PYTHON],
        "versions": [
            EVAL_PYTHON,
            "-c",
            (
                "import json,torch,transformers,vllm; "
                "print(json.dumps({'torch':torch.__version__,"
                "'transformers':transformers.__version__,'vllm':vllm.__version__},"
                "sort_keys=True))"
            ),
        ],
    }
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        atomic_json(
            out / f"{name}.json",
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.returncode:
            raise RuntimeError(f"evaluator environment capture failed: {name}")


@dataclass(frozen=True)
class EvaluationEndpoint:
    dose: str
    arm: str
    boundary: str

    @property
    def condition(self) -> str:
        return f"sdf_{self.dose}_{self.arm}_{self.boundary}"

    @property
    def model_prefix(self) -> str:
        return contracts.model_prefix(self.dose, self.arm, self.boundary)


def evaluation_endpoints() -> tuple[EvaluationEndpoint, ...]:
    return tuple(
        EvaluationEndpoint(dose, arm, boundary)
        for dose in contracts.DOSES
        for arm in contracts.ARMS
        for boundary in BOUNDARIES
    )


def summarize_evaluations(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {endpoint.condition for endpoint in evaluation_endpoints()}
    if set(cells) != expected:
        raise RuntimeError(
            f"evaluation cells differ: missing={sorted(expected - set(cells))}, "
            f"extra={sorted(set(cells) - expected)}"
        )
    within_dose: dict[str, Any] = {}
    restoration: dict[str, Any] = {}
    for dose in contracts.DOSES:
        within_dose[dose] = {}
        restoration[dose] = {}
        for boundary in BOUNDARIES:
            coin = cells[f"sdf_{dose}_coin_{boundary}"]["dispatch"]
            charter = cells[f"sdf_{dose}_charter_{boundary}"]["dispatch"]
            coin_delta = (
                coin["conflict_coin_plan_rate"]
                - charter["conflict_coin_plan_rate"]
            )
            charter_delta = (
                charter["conflict_charter_plan_rate"]
                - coin["conflict_charter_plan_rate"]
            )
            within_dose[dose][boundary] = {
                "coin_minus_charter_coin_plan_rate": coin_delta,
                "charter_minus_coin_charter_plan_rate": charter_delta,
                "directional_separation_sum": coin_delta + charter_delta,
            }
        for arm in contracts.ARMS:
            post_docs = cells[f"sdf_{dose}_{arm}_post_docs"]
            final = cells[f"sdf_{dose}_{arm}_final"]
            restoration[dose][arm] = {
                "coin_plan_rate_final_minus_post_docs": (
                    final["dispatch"]["conflict_coin_plan_rate"]
                    - post_docs["dispatch"]["conflict_coin_plan_rate"]
                ),
                "charter_plan_rate_final_minus_post_docs": (
                    final["dispatch"]["conflict_charter_plan_rate"]
                    - post_docs["dispatch"]["conflict_charter_plan_rate"]
                ),
                "agreement_final_minus_post_docs": (
                    final["dispatch"]["agreement_shared_plan_rate"]
                    - post_docs["dispatch"]["agreement_shared_plan_rate"]
                ),
                "generic_capability_final_minus_post_docs": (
                    final["generic"]["capability_mean"]
                    - post_docs["generic"]["capability_mean"]
                ),
            }
    return {
        "schema_version": "dispatch_sdf_dose_order_evaluation_v1",
        "seed": EVAL_SEED,
        "dispatch_rows_per_condition": {"agreement": 512, "conflict": 512},
        "generic_rows_per_condition": {"mmlu": 40, "gsm8k": 40},
        "cells": cells,
        "within_dose": within_dose,
        "restoration": restoration,
    }


def _replace_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


async def _run(argv: list[str], log_path: Path, gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["NCCL_NVLS_ENABLE"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-30_000:]
        raise RuntimeError(f"evaluation failed ({code}): {' '.join(argv)}\n{tail}")


async def fetch_endpoint(
    root: Path, endpoint: EvaluationEndpoint, revision: str
) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    prefix = endpoint.model_prefix
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=contracts.MODEL_REPO,
        revision=revision,
        allow_patterns=[f"{prefix}/*"],
        token=True,
    )
    checkpoint = Path(snapshot) / prefix
    found = {path.name for path in checkpoint.iterdir() if path.is_file()}
    missing = REQUIRED_MODEL_FILES - found
    if missing:
        raise RuntimeError(f"{prefix}: incomplete checkpoint, missing={sorted(missing)}")
    entries = list(
        await asyncio.to_thread(
            HfApi().list_repo_tree,
            contracts.MODEL_REPO,
            path_in_repo=prefix,
            revision=revision,
            recursive=True,
            expand=True,
        )
    )
    remote_sizes = {
        str(entry.path): int(entry.size)
        for entry in entries
        if hasattr(entry, "size") and getattr(entry, "type", "file") != "directory"
    }
    for path in checkpoint.rglob("*"):
        if path.is_file():
            remote_path = f"{prefix}/{path.relative_to(checkpoint)}"
            if remote_sizes.get(remote_path) != path.stat().st_size:
                raise RuntimeError(f"remote/local size mismatch: {remote_path}")
    endpoint_path = root / "downloaded" / endpoint.condition
    _replace_link(endpoint_path, checkpoint)
    return {
        "condition": endpoint.condition,
        "repo": contracts.MODEL_REPO,
        "revision": revision,
        "prefix": prefix,
        "local_path": str(checkpoint),
        "files": len(remote_sizes),
        "bytes": sum(remote_sizes.values()),
        "download_verified_by_huggingface_hub": True,
    }


async def evaluate_endpoint(
    root: Path, endpoint: EvaluationEndpoint, gpu: int
) -> None:
    model = root / "downloaded" / endpoint.condition
    if not (model / "config.json").is_file():
        raise RuntimeError(f"downloaded endpoint is incomplete: {model}")
    phase = endpoint.condition
    _replace_link(root / "endpoints" / endpoint.arm / phase / "model", model)
    dispatch_script = (
        REPO_ROOT
        / "experiments"
        / "dispatch"
        / "pod"
        / "dispatch_sdf_aft_v1_eval.py"
    )
    common = [
        "--root",
        str(root),
        "--arm",
        endpoint.arm,
        "--model-phase",
        phase,
        "--base-condition",
        endpoint.condition,
        "--base-only",
        "--sampling-seed",
        str(EVAL_SEED),
        "--tokenization-name",
        endpoint.condition,
        "--summary-name",
        endpoint.condition,
    ]
    await _run(
        [EVAL_PYTHON, str(dispatch_script), *common],
        root / "evaluation/logs" / f"{endpoint.condition}_dispatch.log",
        gpu,
    )
    await _run(
        [
            EVAL_PYTHON,
            "-m",
            "experiments.dispatch.dispatch_midtrain_aft_v1.generic_eval",
            *common,
        ],
        root / "evaluation/logs" / f"{endpoint.condition}_generic.log",
        gpu,
    )


def collect_cells(root: Path) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for endpoint in evaluation_endpoints():
        condition = endpoint.condition
        dispatch_summary = json.loads(
            (root / "evaluation/summary" / f"{condition}.json").read_text()
        )
        generic_summary = json.loads(
            (root / "evaluation/generic/summary" / f"{condition}.json").read_text()
        )
        if len(dispatch_summary["rows"]) != 1 or len(generic_summary["rows"]) != 1:
            raise RuntimeError(f"{condition}: expected one row from each evaluator")
        dispatch = dispatch_summary["rows"][0]["metrics"]
        generic = generic_summary["rows"][0]
        cells[condition] = {
            "dose": endpoint.dose,
            "arm": endpoint.arm,
            "boundary": endpoint.boundary,
            "model_prefix": endpoint.model_prefix,
            "dispatch": {
                "agreement_shared_plan_rate": dispatch["agreement"][
                    "shared_plan_rate"
                ]["rate"],
                "conflict_coin_plan_rate": dispatch["conflict"][
                    "coin_plan_rate"
                ]["rate"],
                "conflict_charter_plan_rate": dispatch["conflict"][
                    "charter_plan_rate"
                ]["rate"],
                "conflict_other_rate": (
                    dispatch["conflict"]["other_plan_rate"]["rate"]
                    + dispatch["conflict"]["malformed_rate"]["rate"]
                ),
            },
            "generic": {
                "capability_mean": generic["capability"]["mean"],
                "parseable_rate": generic["collapse"]["parseable_rate"],
                "empty_rate": generic["collapse"]["empty_rate"],
                "truncation_rate": generic["collapse"]["truncation_rate"],
                "dispatch_intrusion_rate": generic["collapse"][
                    "dispatch_intrusion_rate"
                ],
            },
        }
    return cells


async def main_async(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi
    from scimt.eval import capability

    if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
        raise ValueError("--model-revision must be a full 40-character commit")
    if os.environ.get("SCIMT_TRAINING_RUN_ID") != args.training_run_id:
        raise RuntimeError("training run ID environment mismatch")
    if os.environ.get("SCIMT_EVALUATION_RUN_ID") != args.evaluation_run_id:
        raise RuntimeError("evaluation run ID environment mismatch")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    remote = f"runs/{args.training_run_id}/evaluation/{args.evaluation_run_id}"
    try:
        artifacts.require_repo_visibility(api, contracts.MODEL_REPO, private=False)
        artifacts.require_repo_visibility(
            api, contracts.EVIDENCE_REPO, repo_type="dataset", private=False
        )
        source = artifacts.validate_source()
        artifacts.capture_environment(
            root / "evidence/provenance", source_commit=source["commit"]
        )
        capture_evaluator_environment(root / "evidence/provenance/evaluator")
        atomic_json(
            root / "evidence/run.json",
            {
                "schema_version": "dispatch_sdf_dose_order_evaluation_run_v1",
                "training_run_id": args.training_run_id,
                "evaluation_run_id": args.evaluation_run_id,
                "source_commit": source["commit"],
                "model_repo": contracts.MODEL_REPO,
                "model_revision": args.model_revision,
                "seed": EVAL_SEED,
                "endpoints": [
                    endpoint.condition for endpoint in evaluation_endpoints()
                ],
                "packages": package_versions(),
                "started_at": utc_now(),
            },
        )
        dispatch_manifest = await asyncio.to_thread(
            build_dispatch, root / "data/episodes", seed=EVAL_SEED
        )
        capability_rows = await asyncio.to_thread(
            capability.load_capability,
            n_mmlu=40,
            n_gsm8k=40,
            seed=EVAL_SEED,
        )
        atomic_jsonl(root / "data/capability.jsonl", capability_rows)
        atomic_json(
            root / "evidence/data_manifest.json",
            {
                "dispatch": dispatch_manifest,
                "generic_rows": len(capability_rows),
                "generic_benches": {"mmlu": 40, "gsm8k": 40},
            },
        )
        downloads = await asyncio.gather(
            *(
                fetch_endpoint(root, endpoint, args.model_revision)
                for endpoint in evaluation_endpoints()
            )
        )
        atomic_json(
            root / "evidence/model_manifest.json",
            {row["condition"]: row for row in downloads},
        )
        queue: asyncio.Queue[EvaluationEndpoint] = asyncio.Queue()
        for endpoint in evaluation_endpoints():
            queue.put_nowait(endpoint)

        async def worker(gpu: int) -> None:
            while True:
                try:
                    endpoint = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await evaluate_endpoint(root, endpoint, gpu)
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker(gpu) for gpu in range(4)))
        summary = summarize_evaluations(collect_cells(root))
        summary.update(
            {
                "training_run_id": args.training_run_id,
                "evaluation_run_id": args.evaluation_run_id,
                "model_revision": args.model_revision,
                "completed_at": utc_now(),
            }
        )
        atomic_json(root / "evidence/summary.json", summary)
        receipts = {}
        for name in ("data", "evaluation", "evidence"):
            receipts[name] = artifacts.upload_tree(
                api,
                repo_id=contracts.EVIDENCE_REPO,
                repo_type="dataset",
                local_dir=root / name,
                remote_prefix=f"{remote}/{name}",
                manifest_path=upload_manifest_path(root, name),
                commit_message=f"Dispatch SDF evaluation {name}: {args.evaluation_run_id}",
            )
        shutil.copytree(
            root / "upload_manifests",
            root / "evidence/upload_manifests",
            dirs_exist_ok=True,
        )
        atomic_json(
            root / "evidence/RUN_COMPLETE.json",
            {
                "status": "complete",
                "completed_at": utc_now(),
                "receipts": receipts,
            },
        )
        artifacts.upload_tree(
            api,
            repo_id=contracts.EVIDENCE_REPO,
            repo_type="dataset",
            local_dir=root / "evidence",
            remote_prefix=f"{remote}/evidence",
            manifest_path=root / "evidence_terminal_manifest.json",
            commit_message=f"Complete Dispatch SDF evaluation: {args.evaluation_run_id}",
        )
    except BaseException as error:
        atomic_json(
            root / "evidence/RUN_FAILED.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at": utc_now(),
            },
        )
        (root / "evidence/traceback.txt").write_text(traceback.format_exc())
        try:
            artifacts.upload_tree(
                api,
                repo_id=contracts.EVIDENCE_REPO,
                repo_type="dataset",
                local_dir=root / "evidence",
                remote_prefix=f"{remote}/evidence",
                manifest_path=root / "failure_evidence_manifest.json",
                commit_message=(
                    f"Failed Dispatch SDF evaluation: {args.evaluation_run_id}"
                ),
            )
        except Exception as upload_error:  # noqa: BLE001 - preserve original failure
            error.add_note(
                "failure evidence upload also failed: "
                f"{type(upload_error).__name__}: {upload_error}"
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--model-revision", required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
