"""Train, evaluate, and durably publish the mix_3_1_4 full-parameter AFT arm.

Byte-level port of experiments/improved_midtraining/full_parameter_aft_midtrain4/
pod/train.py (exp/fp-aft-midtrain4 @ 7b658719, "drop Adam snapshots + publish-
first") with exactly two deliberate differences:

1. Contracts come from ``fp_mix_crossing.aft.contracts`` — one arm
   (mix_3_1_4), whose parent revision is the stage-A post_dolci100 upload and
   must be pinned via ``require_parent_revision()`` before anything runs.
2. Evidence publishes to ``arcadia-impact/scimt-fp-mix-crossing-v1`` under
   ``runs/<run_id>/aft/<arm>`` (stage A shares the same private dataset).

Everything else is kept: training through ``scimt.train.train_dataset`` (the
canonical run.json/checkpoint.json provenance the attribution library
consumes), the checkpoint ladder published to the Hub immediately after
training validates (a late crash cannot cost finished training), no optimizer
snapshots (Adam coordinates come from PR #351 checkpoint-local estimation),
dataset SHA/row/leakage gates, the finite-loss ``training_started.json``
marker, exact-commit-oid Hub uploads, per-checkpoint safetensors validation,
and processor sidecar hydration.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.fp_mix_crossing.aft import contracts
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    OPTIONAL_TRAINING_EVIDENCE_FILES,
    assert_remote_prefix_absent,
    build_evidence_publication_snapshot,
    full_checkpoint_manifest,
    hydrate_processor_sidecars,
    log,
    run_logged,
    sha256,
    upload_folder_exact_verified,
    utc_now,
    validate_training_trace,
)
from experiments.prior_coins.dispatch_midtrain_aft_v1.pod_run import (
    atomic_json,
    dataset_contract,
)

# The canonical scimt run-dir products staged into the published evidence —
# run.json/checkpoint.json/config snapshots are what make the run
# attribution-consumable (resolve_stage), the rest is the PR #465 set.
TRAINING_EVIDENCE_FILES = (
    "axolotl.yaml",
    "run.json",
    "checkpoint.json",
    "checkpoints.jsonl",
    "checkpoint_card_disposition.json",
    "run_contract.json",
    "train.log",
    "training_started.json",
    "training_provenance.json",
    "training_trace.jsonl",
    "trainer_state.final.json",
    "TRAINING_COMPLETE.json",
)
TRAINING_EVIDENCE_DIRS = ("config",)


def prepare_checkpoint_publication(
    run_dir: Path, *, expected_steps: tuple[int, ...]
) -> dict[str, Any]:
    """PR #465's publication normalization.

    Axolotl's local-path README.md and duplicate root export are archived out
    of ``checkpoints/`` exactly as before.
    """

    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        raise RuntimeError(f"checkpoint publication root is missing: {checkpoints}")
    generated = checkpoints / "README.md"
    archived = run_dir / "generated_checkpoint_README.md"
    if generated.exists() or generated.is_symlink():
        if generated.is_symlink() or not generated.is_file():
            raise RuntimeError(
                f"generated checkpoint card is not a regular file: {generated}"
            )
        if archived.exists():
            raise RuntimeError(f"generated checkpoint card archive exists: {archived}")
        shutil.move(str(generated), str(archived))
        card = {
            "generated_card": "archived",
            "archive": archived.name,
            "sha256": sha256(archived),
            "bytes": archived.stat().st_size,
        }
    else:
        card = {"generated_card": "absent", "archive": None}

    expected = {f"checkpoint-{step}" for step in expected_steps}
    members = {path.name: path for path in checkpoints.iterdir()}
    missing = expected - set(members)
    if missing:
        raise RuntimeError(
            "checkpoint publication root is not exact: "
            f"missing={sorted(missing)}, extra=[]"
        )
    for name in expected:
        path = members[name]
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(
                f"checkpoint publication member is not a directory: {path}"
            )

    export_files = [path for name, path in members.items() if name not in expected]
    export_manifest: dict[str, dict[str, Any]] = {}
    if export_files:
        final_export = run_dir / "final_export"
        if final_export.exists():
            raise RuntimeError(f"final export archive exists: {final_export}")
        final_export.mkdir()
        for path in sorted(export_files):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"unexpected checkpoint-root member is not a regular file: {path}"
                )
            destination = final_export / path.name
            shutil.move(str(path), str(destination))
            export_manifest[path.name] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }

    observed = {path.name for path in checkpoints.iterdir()}
    if observed != expected:
        raise RuntimeError(
            "checkpoint publication root is not exact after preparation: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    receipt = {
        "schema_version": "fp_mix_crossing_aft_checkpoint_disposition_v1",
        **card,
        "final_export": {
            "archive": "final_export" if export_manifest else None,
            "files": len(export_manifest),
            "bytes": sum(item["bytes"] for item in export_manifest.values()),
            "manifest": export_manifest,
        },
        "publication_directories": sorted(observed),
        "prepared_at": utc_now(),
    }
    atomic_json(run_dir / "checkpoint_card_disposition.json", receipt)
    return receipt


def stage_training_evidence(root: Path, arm: str) -> Path:
    source = root / "training" / arm
    destination = root / "evidence" / "training"
    destination.mkdir(parents=True, exist_ok=False)
    for name in TRAINING_EVIDENCE_FILES:
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"stable training evidence is missing: {path}")
        shutil.copy2(path, destination / name)
    for name in OPTIONAL_TRAINING_EVIDENCE_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    for name in TRAINING_EVIDENCE_DIRS:
        directory = source / name
        if not directory.is_dir():
            raise RuntimeError(f"stable training evidence dir is missing: {directory}")
        shutil.copytree(directory, destination / name)
    pointers = sorted(source.glob("ckpt_*.txt"))
    if not pointers:
        raise RuntimeError(f"no checkpoint pointer file under {source}")
    for pointer in pointers:
        shutil.copy2(pointer, destination / pointer.name)
    return destination


async def fetch_parent(root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    parent_revision = contracts.require_parent_revision()
    prefix = contracts.PARENT_PREFIX[arm]
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=contracts.PARENT_REPO,
        revision=parent_revision,
        allow_patterns=[f"{prefix}/*"],
    )
    parent = Path(snapshot) / prefix
    entries = list(
        await asyncio.to_thread(
            HfApi().list_repo_tree,
            contracts.PARENT_REPO,
            path_in_repo=prefix,
            revision=parent_revision,
            recursive=True,
            expand=True,
        )
    )
    files = []
    for entry in entries:
        if getattr(entry, "type", "file") == "directory":
            continue
        relative = str(entry.path)[len(prefix) + 1 :]
        local = parent / relative
        if not local.is_file() or local.stat().st_size != int(entry.size):
            raise RuntimeError(f"parent file verification failed: {local}")
        lfs = getattr(entry, "lfs", None)
        files.append(
            {
                "path": relative,
                "size": int(entry.size),
                "lfs_sha256": (
                    lfs.get("sha256")
                    if isinstance(lfs, dict)
                    else getattr(lfs, "sha256", None)
                ),
            }
        )
    for sidecar in ("processor_config.json", "preprocessor_config.json"):
        if not (parent / sidecar).is_file():
            raise RuntimeError(f"pinned parent lacks required {sidecar}")
    destination = root / "parent" / arm
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(parent, target_is_directory=True)
    return destination, {
        "repo": contracts.PARENT_REPO,
        "revision": parent_revision,
        "prefix": prefix,
        "files": files,
        "total_bytes": sum(item["size"] for item in files),
    }


def fetch_wave_agreement(root: Path) -> Path:
    """Download the pinned wave-v1 agreement bytes — never regenerated."""

    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            repo_id=contracts.DATA_REPO,
            repo_type="dataset",
            revision=contracts.DATA_REVISION,
            filename=contracts.DATA_PATH,
        )
    )
    destination = root / "data" / "wave" / "aft_agreement.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def audit_train_eval_overlap(train_path: Path, eval_manifest: dict[str, Any]) -> int:
    """The wave training file and the (older-battery) eval episodes come from
    different builders, so the built-in leakage audit does not cover the pair;
    refuse any shared first-user-turn prompt outright."""

    train_prompts = {
        json.loads(line)["messages"][0]["content"]
        for line in train_path.read_text().splitlines()
        if line.strip()
    }
    eval_root = Path(eval_manifest["data_root"]) if "data_root" in eval_manifest else None
    overlap = 0
    episode_files = (
        sorted(eval_root.rglob("*.jsonl")) if eval_root and eval_root.is_dir() else []
    )
    for path in episode_files:
        if path.resolve() == train_path.resolve():
            continue
        # the battery's own TRAINING files are not evaluated on; sharing a
        # prompt with them is harmless (we train on the wave file instead)
        if path.parent.name == "datasets":
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = None
            if isinstance(row.get("messages"), list) and row["messages"]:
                prompt = row["messages"][0].get("content")
            elif isinstance(row.get("prompt"), str):
                prompt = row["prompt"]
            if prompt is not None and prompt in train_prompts:
                overlap += 1
    if overlap:
        raise RuntimeError(
            f"wave training prompts overlap the evaluation battery ({overlap} rows)"
        )
    return overlap


async def prepare_data(root: Path) -> tuple[Any, dict[str, Any]]:
    """Build the eval battery, download + gate the wave training bytes, and
    return a manifest-backed Dataset handle for training."""

    from huggingface_hub import hf_hub_download

    from scimt.dataset import Dataset

    # The evaluation battery (episodes + its own train/eval leakage audits)
    # still comes from the PR #465 contract; its training file is NOT what we
    # train on.
    _battery_path, manifest = dataset_contract(root)
    # dataset_contract only self-checks the regeneration against its own
    # fresh manifest. Pin the regenerated batteries to the exact bytes the
    # 20260817T122200Z family arms were scored on, or the new arm is not
    # comparable — abort before any compute is spent.
    observed_batteries = dict(manifest["dataset_sha256"])
    if observed_batteries != dict(contracts.EXPECTED_BATTERY_SHA256):
        raise RuntimeError(
            "regenerated eval batteries drifted from the frozen "
            "20260817T122200Z family run: "
            f"{observed_batteries} != {dict(contracts.EXPECTED_BATTERY_SHA256)}"
        )
    dataset_path = await asyncio.to_thread(fetch_wave_agreement, root)
    observed = sha256(dataset_path)
    if observed != contracts.EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            f"agreement data hash {observed} != {contracts.EXPECTED_DATASET_SHA256}"
        )
    rows = sum(1 for line in dataset_path.read_text().splitlines() if line.strip())
    if rows != contracts.EXPECTED_DATASET_ROWS:
        raise RuntimeError(f"agreement dataset has {rows} rows")
    manifest["cross_battery_overlap"] = await asyncio.to_thread(
        audit_train_eval_overlap,
        dataset_path,
        {"data_root": str(root / "data" / "episodes")},
    )
    handle = Dataset(
        path=str(dataset_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=rows,
        meta={
            "experiment": "fp_mix_crossing_aft",
            "sha256": observed,
            "rows": rows,
            "seed": contracts.SEED,
            "provenance": {
                "repo": contracts.DATA_REPO,
                "revision": contracts.DATA_REVISION,
                "path": contracts.DATA_PATH,
                "note": "wave-v1 agreement mixture, byte-identical",
            },
        },
    )
    handle.save()
    generic = Path(
        await asyncio.to_thread(
            hf_hub_download,
            repo_id=contracts.CAPABILITY_REPO,
            repo_type="dataset",
            revision=contracts.CAPABILITY_REVISION,
            filename=contracts.CAPABILITY_PATH,
        )
    )
    capability = root / "data" / "capability.jsonl"
    capability.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generic, capability)
    battery_rows = [
        line for line in capability.read_text().splitlines() if line.strip()
    ]
    if len(battery_rows) != 80:
        raise RuntimeError(f"generic capability battery has {len(battery_rows)} rows")
    capability_sha256 = sha256(capability)
    if capability_sha256 != contracts.EXPECTED_CAPABILITY_SHA256:
        raise RuntimeError(
            f"capability battery hash {capability_sha256} != "
            f"{contracts.EXPECTED_CAPABILITY_SHA256}"
        )
    manifest["agreement_expected_sha256"] = contracts.EXPECTED_DATASET_SHA256
    manifest["scimt_dataset_manifest"] = str(
        Path(handle.path).parent / "dataset.json"
    )
    manifest["generic_capability"] = {
        "repo": contracts.CAPABILITY_REPO,
        "revision": contracts.CAPABILITY_REVISION,
        "path": contracts.CAPABILITY_PATH,
        "sha256": capability_sha256,
        "rows": len(battery_rows),
    }
    return handle, manifest


async def train(
    root: Path, run_id: str, arm: str, parent: Path, dataset: Any
) -> dict[str, Any]:
    from scimt.train import TrainConfig, train_dataset

    run_dir = root / "training" / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(
        backend="axolotl",
        stage=contracts.STAGE,
        model=contracts.SUBSTRATE_MODEL,
        seed=contracts.SEED,
        load_checkpoint_path=str(parent),
    )
    atomic_json(
        run_dir / "run_contract.json",
        {
            "schema_version": "fp_mix_crossing_aft_arm_v1",
            "arm": arm,
            "run_id": run_id,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "parent_repo": contracts.PARENT_REPO,
            "parent_revision": contracts.require_parent_revision(),
            "parent_prefix": contracts.PARENT_PREFIX[arm],
            "dataset_sha256": sha256(Path(dataset.path)),
            "dataset_rows": contracts.EXPECTED_DATASET_ROWS,
            "seed": contracts.SEED,
            "stage": contracts.STAGE,
            "trainable_parameterization": "full",
            "world_size": 4,
            "effective_global_batch_size": 32,
            "expected_optimizer_steps": contracts.EXPECTED_STEPS,
            "expected_checkpoints": list(contracts.EXPECTED_CHECKPOINTS),
            "started_at": utc_now(),
        },
    )
    log(f"{arm}: starting full-parameter training via scimt.train.train_dataset")
    started = time.time()
    checkpoint_handle = await train_dataset(
        dataset,
        run_dir,
        config,
        run_name=f"fpmix-aft-{arm}-{run_id.lower()}",
    )
    final_state_dir = Path(checkpoint_handle.require_state())
    if final_state_dir.name != f"checkpoint-{contracts.EXPECTED_STEPS}":
        raise RuntimeError(
            f"final state checkpoint is {final_state_dir.name}, expected "
            f"checkpoint-{contracts.EXPECTED_STEPS}"
        )
    for name in ("run.json", "checkpoint.json", "axolotl.yaml"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"canonical run-dir product missing: {run_dir / name}")
    state = json.loads((run_dir / "trainer_state.final.json").read_text())
    if int(state.get("global_step", -1)) != contracts.EXPECTED_STEPS:
        raise RuntimeError(f"training ended at step {state.get('global_step')}")
    health = json.loads((run_dir / "training_started.json").read_text())
    if health.get("status") != "training_started" or not math.isfinite(
        float(health["finite_loss"])
    ):
        raise RuntimeError("finite-loss training health marker is invalid")
    prepare_checkpoint_publication(
        run_dir, expected_steps=contracts.EXPECTED_CHECKPOINTS
    )
    manifests = {}
    for step in contracts.EXPECTED_CHECKPOINTS:
        checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
        hydrate_processor_sidecars(checkpoint, parent)
        manifests[str(step)] = full_checkpoint_manifest(checkpoint)
    actual = json.loads((run_dir / "training_provenance.json").read_text())["actual"]
    if tuple(actual["checkpoint_steps"]) != contracts.EXPECTED_CHECKPOINTS:
        raise RuntimeError("training provenance has the wrong checkpoint schedule")
    trace_summary = validate_training_trace(
        run_dir / "training_trace.jsonl",
        expected_steps=contracts.EXPECTED_STEPS,
        expected_lr=contracts.EXPECTED_LEARNING_RATE,
    )
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    result = {
        "status": "complete",
        "arm": arm,
        "completed_at": utc_now(),
        "minutes": round((time.time() - started) / 60, 3),
        "checkpoint_manifests": manifests,
        "trace": trace_summary,
    }
    atomic_json(run_dir / "TRAINING_COMPLETE.json", result)
    atomic_json(root / "evidence" / "checkpoint_manifest.json", manifests)
    log(f"{arm}: checkpoint ladder validated")
    return result


def package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "axolotl", "datasets", "huggingface_hub")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


async def publish_model(root: Path, run_id: str, arm: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = contracts.model_prefix(arm)
    await asyncio.to_thread(
        assert_remote_prefix_absent, api, contracts.MODEL_REPO, "model", prefix
    )
    receipt = await asyncio.to_thread(
        upload_folder_exact_verified,
        api,
        repo_id=contracts.MODEL_REPO,
        repo_type="model",
        folder=root / "training" / arm / "checkpoints",
        remote_prefix=prefix,
        commit_message=f"Full-parameter AFT (mix crossing): {prefix}",
    )
    source_commit = str(os.environ["SCIMT_SOURCE_COMMIT"])
    result = {
        "marker_kind": "model_published",
        "repo": contracts.MODEL_REPO,
        "revision": receipt["revision"],
        "prefix": prefix,
        "run_id": run_id,
        "arm": arm,
        "source_commit": source_commit,
        "upload_receipt": receipt,
        "verified_at": utc_now(),
    }
    atomic_json(root / "evidence" / "model_publication.json", result)
    await _upload_marker(
        api, root, f"{contracts.evidence_prefix(run_id, arm)}/MODEL_PUBLISHED.json",
        result,
    )
    return result


async def _upload_marker(
    api: Any, root: Path, path: str, payload: dict[str, Any]
) -> str:
    """PR #465's verified marker upload, parameterized to this evidence repo."""

    from huggingface_hub import hf_hub_download

    from experiments.improved_midtraining.full_parameter_aft.run_arm import (
        _commit_oid,
    )

    await asyncio.to_thread(
        assert_remote_prefix_absent, api, contracts.EVIDENCE_REPO, "dataset", path
    )
    marker = root / "evidence" / Path(path).name
    atomic_json(marker, payload)
    commit = None
    for attempt in range(4):
        parent = str(
            (
                await asyncio.to_thread(
                    api.repo_info, contracts.EVIDENCE_REPO, repo_type="dataset"
                )
            ).sha
        )
        try:
            commit = await asyncio.to_thread(
                api.upload_file,
                repo_id=contracts.EVIDENCE_REPO,
                repo_type="dataset",
                path_or_fileobj=str(marker),
                path_in_repo=path,
                parent_commit=parent,
                commit_message=f"fp-mix-crossing AFT marker: {path}",
            )
            break
        except Exception:
            if attempt == 3:
                raise
            await asyncio.sleep(2**attempt)
    revision = _commit_oid(commit)
    downloaded = Path(
        await asyncio.to_thread(
            hf_hub_download,
            repo_id=contracts.EVIDENCE_REPO,
            repo_type="dataset",
            revision=revision,
            filename=path,
            force_download=True,
        )
    )
    if sha256(downloaded) != sha256(marker):
        raise RuntimeError(
            f"Hub marker sha256 verification failed at {revision}: {path}"
        )
    return revision


async def publish_evidence(root: Path, run_id: str, arm: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = contracts.evidence_prefix(run_id, arm)
    evidence_snapshot = build_evidence_publication_snapshot(root)
    receipts = []
    folders = {
        "data": root / "data",
        "evaluation": root / "evaluation",
        "evidence": evidence_snapshot,
    }
    for folder_name, folder in folders.items():
        remote_prefix = f"{prefix}/{folder_name}"
        await asyncio.to_thread(
            assert_remote_prefix_absent,
            api,
            contracts.EVIDENCE_REPO,
            "dataset",
            remote_prefix,
        )
        receipts.append(
            await asyncio.to_thread(
                upload_folder_exact_verified,
                api,
                repo_id=contracts.EVIDENCE_REPO,
                repo_type="dataset",
                folder=folder,
                remote_prefix=remote_prefix,
                commit_message=f"fp-mix-crossing AFT evidence: {remote_prefix}",
            )
        )
    source_commit = str(os.environ["SCIMT_SOURCE_COMMIT"])
    result = {
        "marker_kind": "run_published",
        "repo": contracts.EVIDENCE_REPO,
        "revision": receipts[-1]["revision"],
        "prefix": prefix,
        "run_id": run_id,
        "arm": arm,
        "source_commit": source_commit,
        "upload_receipts": receipts,
        "verified_at": utc_now(),
    }
    atomic_json(root / "evidence" / "evidence_publication.json", result)
    return result


async def main_async(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    contracts.require_parent_revision()
    source_commit = str(os.environ.get("SCIMT_SOURCE_COMMIT", ""))
    if len(source_commit) not in (40, 64):
        raise RuntimeError("SCIMT_SOURCE_COMMIT must be an exact source commit oid")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    atomic_json(
        evidence / "run_metadata.json",
        {
            "schema_version": "fp_mix_crossing_aft_run_v1",
            "run_id": args.run_id,
            "arm": args.arm,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "source_manifest_sha256": os.environ.get("SCIMT_SOURCE_MANIFEST_SHA256"),
            "parent_repo": contracts.PARENT_REPO,
            "parent_revision": contracts.require_parent_revision(),
            "parent_prefix": contracts.PARENT_PREFIX[args.arm],
            "dataset_seed": contracts.SEED,
            "training_seed": contracts.SEED,
            "evaluation_seed": contracts.SEED,
            "stage": contracts.STAGE,
            "packages": package_versions(),
        },
    )
    (evidence / "nvidia_smi_q.txt").write_text(
        subprocess.run(
            ["nvidia-smi", "-q"], check=True, capture_output=True, text=True
        ).stdout
    )
    (evidence / "pip_freeze_train.txt").write_text(
        subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    (evidence / "pip_freeze_eval.txt").write_text(
        subprocess.run(
            [
                "uv",
                "pip",
                "freeze",
                "--python",
                "/workspace/venv-dispatch-eval/bin/python",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    dataset, dataset_manifest = await prepare_data(root)
    atomic_json(evidence / "dataset_manifest.json", dataset_manifest)
    parent, parent_manifest = await fetch_parent(root, args.arm)
    atomic_json(evidence / "parent_manifest.json", parent_manifest)
    await train(root, args.run_id, args.arm, parent, dataset)
    # Publish the finished training FIRST: after the four-arm run's attempt 2
    # lost a completed 512-step run to a post-training crash, nothing runs
    # between checkpoint validation and the weights becoming durable.
    model_publication = await publish_model(root, args.run_id, args.arm)
    stage_training_evidence(root, args.arm)

    evaluation = asyncio.create_task(
        run_logged(
            [
                "/workspace/venv-dispatch-eval/bin/python",
                "-m",
                "experiments.improved_midtraining.full_parameter_aft.evaluate_trajectory",
                "--root",
                str(root),
                "--arm",
                args.arm,
                "--gpus",
                "4",
                "--max-steps",
                str(contracts.EXPECTED_STEPS),
            ],
            evidence / "evaluation_driver.log",
        )
    )
    api = HfApi()
    await evaluation
    atomic_json(
        evidence / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "model_publication": model_publication,
            "evaluation": json.loads(
                (evidence / "evaluation_summary.json").read_text()
            ),
        },
    )
    publication = await publish_evidence(root, args.run_id, args.arm)
    await _upload_marker(
        api,
        root,
        f"{contracts.evidence_prefix(args.run_id, args.arm)}/RUN_PUBLISHED.json",
        publication,
    )
    log(f"{args.arm}: training, evaluation, and publication complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=contracts.ARMS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
