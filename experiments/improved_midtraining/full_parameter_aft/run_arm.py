"""Train, evaluate, and durably publish one full-parameter Dispatch AFT arm."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from experiments.prior_coins.dispatch_midtrain_aft_v1.pod_run import (
    atomic_json,
    dataset_contract,
)
from experiments.prior_coins.dispatch_midtrain_aft_v1.schedule import checkpoint_steps
from scimt.train import TrainConfig
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

ARMS = ("coin", "charter")
SEED = 314159
EXPECTED_STEPS = 2048
EXPECTED_CHECKPOINTS = checkpoint_steps(EXPECTED_STEPS)
EXPECTED_DATASET_SHA256 = (
    "2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b"
)
STAGE = "fp_aft_dispatch_midtrain_gemma3_12b"
PARENT_REPO = "jbostock/scimt-dispatch-models-v1"
PARENT_REVISION = "9a16b6ebe2e88b86e6c709295424df869c028d78"
MODEL_REPO = "jbostock/scimt-dispatch-models-v1"
EVIDENCE_REPO = "arcadia-impact/scimt-dispatch-aft-v1"
CAPABILITY_REVISION = "a833f6c1238ba21c9f5ac009dd2acd3774af6ba0"
CAPABILITY_PATH = (
    "runs/20260807T110710Z/generic_eval/20260807T135326Z/data/capability.jsonl"
)
TRAINING_EVIDENCE_FILES = (
    "axolotl.yaml",
    "checkpoint_card_disposition.json",
    "run_contract.json",
    "train.log",
    "training_started.json",
    "training_provenance.json",
    "training_trace.jsonl",
    "trainer_state.final.json",
    "TRAINING_COMPLETE.json",
)
OPTIONAL_TRAINING_EVIDENCE_FILES = ("generated_checkpoint_README.md",)
LIVE_OUTER_LOGS = frozenset({"run.log", "pod_run.log"})
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_prefix(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"full_aft/{arm}"


def evidence_prefix(run_id: str, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return f"full_parameter_runs/{run_id}/{arm}"


def _local_tree(folder: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"publication payload contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
    if not files:
        raise RuntimeError(f"publication payload is empty: {folder}")
    return files


def assert_remote_prefix_absent(
    api: Any, repo_id: str, repo_type: str, remote_prefix: str
) -> None:
    """Fail closed instead of merging a run into an existing Hub prefix."""

    prefix = remote_prefix.strip("/")
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    conflicts = [
        path for path in files if path == prefix or path.startswith(f"{prefix}/")
    ]
    if conflicts:
        raise RuntimeError(
            f"Hub publication target already exists: {repo_id}/{prefix} "
            f"({conflicts[:5]})"
        )


def _commit_oid(commit: Any) -> str:
    revision = str(getattr(commit, "oid", "") or "")
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError(
            f"Hub upload did not return an exact commit oid: {revision!r}"
        )
    return revision


def _lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return str(value) if value else None


def _is_repo_folder(entry: Any) -> bool:
    """Recognize Hub tree nodes across huggingface_hub releases."""

    return entry.__class__.__name__ == "RepoFolder" or getattr(entry, "type", None) in {
        "directory",
        "tree",
    }


def upload_folder_exact_verified(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    folder: Path,
    remote_prefix: str,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Upload against one parent and verify the exact returned Hub commit."""

    folder = folder.resolve()
    local = _local_tree(folder)
    parent = ""
    last_error: Exception | None = None
    for attempt in range(4):
        # Refresh HEAD on every attempt.  The four requested arms publish to
        # disjoint prefixes but can advance the same consolidated model repo
        # concurrently.
        parent = str(api.repo_info(repo_id, repo_type=repo_type).sha)
        try:
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder),
                path_in_repo=remote_prefix,
                parent_commit=parent,
                commit_message=commit_message or f"Full-parameter AFT: {remote_prefix}",
            )
            break
        except Exception as error:  # Hub/network calls need bounded backoff.
            last_error = error
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover - the loop either breaks or raises
        raise RuntimeError(f"Hub upload failed: {last_error}")

    revision = _commit_oid(commit)
    entries = list(
        api.list_repo_tree(
            repo_id,
            path_in_repo=remote_prefix,
            revision=revision,
            recursive=True,
            expand=True,
            repo_type=repo_type,
        )
    )
    prefix = remote_prefix.strip("/")
    remote: dict[str, Any] = {}
    for entry in entries:
        if _is_repo_folder(entry):
            continue
        path = str(entry.path)
        relative = path[len(prefix) + 1 :] if path.startswith(f"{prefix}/") else path
        remote[relative] = entry
    if set(remote) != set(local):
        raise RuntimeError(
            "Hub exact-tree verification failed: "
            f"missing={sorted(set(local) - set(remote))[:5]}, "
            f"extra={sorted(set(remote) - set(local))[:5]}"
        )

    from huggingface_hub import hf_hub_download

    for relative, metadata in local.items():
        entry = remote[relative]
        if int(entry.size) != metadata["size"]:
            raise RuntimeError(f"Hub size verification failed for {relative}")
        remote_sha256 = _lfs_sha256(entry)
        if remote_sha256 is None:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    filename=f"{prefix}/{relative}",
                    force_download=True,
                )
            )
            remote_sha256 = sha256(downloaded)
        if remote_sha256 != metadata["sha256"]:
            raise RuntimeError(f"Hub sha256 verification failed for {relative}")
    return {
        "repo": repo_id,
        "repo_type": repo_type,
        "prefix": prefix,
        "parent_revision": parent,
        "revision": revision,
        "commit_url": str(getattr(commit, "commit_url", "") or ""),
        "files": len(local),
        "bytes": sum(int(item["size"]) for item in local.values()),
        "tree_sha256": hashlib.sha256(
            json.dumps(local, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_marker_payload(
    payload: dict[str, Any],
    *,
    marker_kind: str,
    run_id: str,
    arm: str,
    source_commit: str,
) -> None:
    expected = {
        "marker_kind": marker_kind,
        "run_id": run_id,
        "arm": arm,
        "source_commit": source_commit,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"stale publication marker: expected={expected}, observed={observed}"
        )


def _safetensors_header(path: Path) -> tuple[set[str], int]:
    """Validate one safetensors header and return its tensor names/data bytes."""

    size = path.stat().st_size
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("missing eight-byte header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length < 2 or header_length > size - 8:
                raise ValueError(f"invalid header length {header_length}")
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid safetensors file {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"invalid safetensors header object in {path}")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise RuntimeError(f"safetensors file has no tensors: {path}")

    intervals: list[tuple[int, int, str]] = []
    for name, metadata in tensors.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise RuntimeError(f"invalid safetensors tensor entry in {path}: {name!r}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if dtype not in _DTYPE_BYTES or not isinstance(shape, list):
            raise RuntimeError(f"invalid safetensors dtype/shape for {name} in {path}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise RuntimeError(f"invalid safetensors shape for {name} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise RuntimeError(f"invalid safetensors offsets for {name} in {path}")
        start, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected:
            raise RuntimeError(f"invalid safetensors extent for {name} in {path}")
        intervals.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise RuntimeError(
                f"non-contiguous safetensors data before {name} in {path}"
            )
        cursor = end
    if 8 + header_length + cursor != size:
        raise RuntimeError(f"safetensors data length mismatch in {path}")
    return set(tensors), cursor


def full_checkpoint_manifest(
    checkpoint: Path, *, minimum_weight_bytes: int = 20_000_000_000
) -> dict[str, Any]:
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"{checkpoint}: missing checkpoint files {missing}")
    if (checkpoint / "adapter_config.json").exists():
        raise RuntimeError(f"{checkpoint}: adapter_config found in full checkpoint")
    weights = sorted(checkpoint.glob("*.safetensors"))
    weight_bytes = sum(path.stat().st_size for path in weights)
    if not weights or weight_bytes < minimum_weight_bytes:
        raise RuntimeError(
            f"{checkpoint}: only {weight_bytes} full-weight bytes (minimum "
            f"{minimum_weight_bytes})"
        )
    index_path = checkpoint / "model.safetensors.index.json"
    tensor_names_by_file: dict[str, set[str]] = {}
    tensor_data_bytes = 0
    for path in weights:
        tensor_names, data_bytes = _safetensors_header(path)
        tensor_names_by_file[path.name] = tensor_names
        tensor_data_bytes += data_bytes
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"invalid safetensors index {index_path}: {error}"
            ) from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"invalid safetensors weight map in {index_path}")
        if any(
            not isinstance(name, str) or not isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise RuntimeError(f"invalid safetensors weight map in {index_path}")
        expected_shards = set(weight_map.values())
        observed_shards = set(tensor_names_by_file)
        if expected_shards != observed_shards:
            raise RuntimeError(
                f"{checkpoint}: safetensors shard set mismatch: "
                f"expected={sorted(expected_shards)}, observed={sorted(observed_shards)}"
            )
        indexed_tensors = dict(weight_map)
        actual_tensors: dict[str, str] = {}
        for shard, names in tensor_names_by_file.items():
            for name in names:
                if name in actual_tensors:
                    raise RuntimeError(
                        f"{checkpoint}: duplicate tensor {name!r} across shards"
                    )
                actual_tensors[name] = shard
        if indexed_tensors != actual_tensors:
            raise RuntimeError(f"{checkpoint}: safetensors index/tensor mismatch")
        total_size = (index.get("metadata") or {}).get("total_size")
        if total_size is not None:
            try:
                matches_total_size = int(total_size) == tensor_data_bytes
            except (TypeError, ValueError):
                matches_total_size = False
            if not matches_total_size:
                raise RuntimeError(
                    f"{checkpoint}: safetensors index total_size mismatch"
                )
    elif set(tensor_names_by_file) != {"model.safetensors"}:
        raise RuntimeError(
            f"{checkpoint}: multiple/noncanonical safetensors files require an index"
        )
    files = {
        str(path.relative_to(checkpoint)): {
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    return {
        "path": str(checkpoint),
        "weight_files": [path.name for path in weights],
        "weight_bytes": weight_bytes,
        "tensor_data_bytes": tensor_data_bytes,
        "tensor_count": sum(len(names) for names in tensor_names_by_file.values()),
        "files": files,
    }


def hydrate_processor_sidecars(checkpoint: Path, parent: Path) -> None:
    """Restore processor metadata Axolotl may omit from intermediate checkpoints."""

    for name in ("processor_config.json", "preprocessor_config.json"):
        destination = checkpoint / name
        if destination.is_file():
            continue
        source = parent / name
        if not source.is_file():
            raise RuntimeError(
                f"pinned parent lacks required processor sidecar: {source}"
            )
        shutil.copy2(source, destination)


def validate_training_trace(
    trace_path: Path, *, expected_steps: int, expected_lr: float
) -> dict[str, Any]:
    """Require one finite, constant-rate loss record for every optimizer step."""

    rows = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    losses = [row for row in rows if "loss" in row]
    steps = [int(row.get("step", -1)) for row in losses]
    expected = list(range(1, expected_steps + 1))
    if steps != expected:
        raise RuntimeError(
            f"training trace steps are incomplete: observed {steps[:3]}...{steps[-3:]}, "
            f"expected 1..{expected_steps}"
        )
    values = [float(row["loss"]) for row in losses]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("training trace contains a non-finite loss")
    rates = [float(row.get("learning_rate", math.nan)) for row in losses]
    if not all(
        math.isfinite(rate)
        and math.isclose(rate, expected_lr, rel_tol=0.0, abs_tol=1e-12)
        for rate in rates
    ):
        raise RuntimeError(f"training trace learning rate differs from {expected_lr}")
    return {
        "loss_count": len(values),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_loss": values[0],
        "last_loss": values[-1],
        "min_loss": min(values),
        "max_loss": max(values),
        "learning_rate": expected_lr,
    }


def stage_training_evidence(root: Path, arm: str) -> Path:
    """Copy stable lightweight training records into Bellhop's result subtree."""

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
    return destination


def prepare_checkpoint_publication(
    run_dir: Path, *, expected_steps: tuple[int, ...]
) -> dict[str, Any]:
    """Remove Axolotl's local-path model card from the model upload payload.

    Axolotl writes ``checkpoints/README.md`` with dataset metadata derived from
    its local JSONL path and also writes a duplicate final-model export beside
    the numbered checkpoints. Hugging Face rejects the local-path card, and the
    duplicate export does not belong in the checkpoint trajectory. Preserve
    both outside the publication tree, then require that tree to contain
    exactly the expected checkpoint directories.
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
        "schema_version": "dispatch_checkpoint_card_disposition_v1",
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


def build_evidence_publication_snapshot(root: Path) -> Path:
    """Freeze evidence for upload without Bellhop's still-growing outer logs."""

    source = root / "evidence"
    destination = root / "publication" / "evidence"
    if destination.exists():
        raise RuntimeError(
            f"evidence publication snapshot already exists: {destination}"
        )
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns(*LIVE_OUTER_LOGS)
    )
    return destination


@contextmanager
def rendered_world_size(size: int) -> Iterator[None]:
    previous = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(size)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous


async def run_logged(argv: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("WORLD_SIZE", None)
    environment["NCCL_NVLS_ENABLE"] = "0"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=environment,
        limit=2**20,
    )
    assert process.stdout is not None
    with log_path.open("wb") as handle:
        async for chunk in process.stdout:
            handle.write(chunk)
            handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-30_000:]
        raise RuntimeError(f"process failed ({code}): {' '.join(argv)}\n{tail}")


async def fetch_parent(root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    prefix = f"sft/{arm}/checkpoint-48"
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=PARENT_REPO,
        revision=PARENT_REVISION,
        allow_patterns=[f"{prefix}/*"],
    )
    parent = Path(snapshot) / prefix
    entries = list(
        await asyncio.to_thread(
            HfApi().list_repo_tree,
            PARENT_REPO,
            path_in_repo=prefix,
            revision=PARENT_REVISION,
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
            raise RuntimeError(f"pinned SFT parent lacks required {sidecar}")
    destination = root / "parent" / arm
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(parent, target_is_directory=True)
    return destination, {
        "repo": PARENT_REPO,
        "revision": PARENT_REVISION,
        "prefix": prefix,
        "files": files,
        "total_bytes": sum(item["size"] for item in files),
    }


async def prepare_data(root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    dataset, manifest = dataset_contract(root)
    observed = sha256(dataset)
    if observed != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            f"agreement data hash {observed} != {EXPECTED_DATASET_SHA256}"
        )
    generic = Path(
        await asyncio.to_thread(
            hf_hub_download,
            repo_id=EVIDENCE_REPO,
            repo_type="dataset",
            revision=CAPABILITY_REVISION,
            filename=CAPABILITY_PATH,
        )
    )
    capability = root / "data" / "capability.jsonl"
    capability.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generic, capability)
    rows = [line for line in capability.read_text().splitlines() if line.strip()]
    if len(rows) != 80:
        raise RuntimeError(f"generic capability battery has {len(rows)} rows")
    manifest["agreement_expected_sha256"] = EXPECTED_DATASET_SHA256
    manifest["generic_capability"] = {
        "repo": EVIDENCE_REPO,
        "revision": CAPABILITY_REVISION,
        "path": CAPABILITY_PATH,
        "sha256": sha256(capability),
        "rows": len(rows),
    }
    return dataset, manifest


async def train(root: Path, arm: str, parent: Path, dataset: Path) -> dict[str, Any]:
    run_dir = root / "training" / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    stage = load_stage(STAGE)
    config = TrainConfig(
        backend="axolotl",
        stage=STAGE,
        model="gemma3_12b_it",
        seed=SEED,
        load_checkpoint_path=str(parent),
    )
    with rendered_world_size(4):
        rendered = render_stage(stage, config, dataset, run_dir)
    atomic_json(
        run_dir / "run_contract.json",
        {
            "schema_version": "dispatch_full_parameter_aft_arm_v1",
            "arm": arm,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "parent_repo": PARENT_REPO,
            "parent_revision": PARENT_REVISION,
            "dataset_sha256": sha256(dataset),
            "dataset_rows": 2048,
            "seed": SEED,
            "stage": STAGE,
            "trainable_parameterization": "full",
            "world_size": 4,
            "effective_global_batch_size": 32,
            "expected_optimizer_steps": EXPECTED_STEPS,
            "expected_checkpoints": list(EXPECTED_CHECKPOINTS),
            "started_at": utc_now(),
        },
    )
    log(f"{arm}: starting full-parameter Axolotl training on 4 GPUs")
    started = time.time()
    await LocalExecutor().run_stage(rendered, run_dir, stage)
    state = json.loads((run_dir / "trainer_state.final.json").read_text())
    if int(state.get("global_step", -1)) != EXPECTED_STEPS:
        raise RuntimeError(f"training ended at step {state.get('global_step')}")
    health = json.loads((run_dir / "training_started.json").read_text())
    if health.get("status") != "training_started" or not math.isfinite(
        float(health["finite_loss"])
    ):
        raise RuntimeError("finite-loss training health marker is invalid")
    prepare_checkpoint_publication(run_dir, expected_steps=EXPECTED_CHECKPOINTS)
    manifests = {}
    for step in EXPECTED_CHECKPOINTS:
        checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
        hydrate_processor_sidecars(checkpoint, parent)
        manifests[str(step)] = full_checkpoint_manifest(checkpoint)
    actual = json.loads((run_dir / "training_provenance.json").read_text())["actual"]
    if tuple(actual["checkpoint_steps"]) != EXPECTED_CHECKPOINTS:
        raise RuntimeError("training provenance has the wrong checkpoint schedule")
    trace_summary = validate_training_trace(
        run_dir / "training_trace.jsonl",
        expected_steps=EXPECTED_STEPS,
        expected_lr=5e-6,
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
    log(f"{arm}: all ten full checkpoints validated and hashed")
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


def remote_file_exists(api: Any, path: str) -> bool:
    return bool(api.file_exists(EVIDENCE_REPO, path, repo_type="dataset"))


async def wait_for_marker(
    api: Any,
    path: str,
    *,
    marker_kind: str,
    run_id: str,
    arm: str,
    source_commit: str,
    timeout_seconds: int = 8 * 3600,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
    while not await asyncio.to_thread(remote_file_exists, api, path):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for {EVIDENCE_REPO}/{path}")
        await asyncio.sleep(30)
    revision = str(
        (await asyncio.to_thread(api.repo_info, EVIDENCE_REPO, repo_type="dataset")).sha
    )
    marker = Path(
        await asyncio.to_thread(
            hf_hub_download,
            repo_id=EVIDENCE_REPO,
            repo_type="dataset",
            revision=revision,
            filename=path,
            force_download=True,
        )
    )
    payload = json.loads(marker.read_text())
    validate_marker_payload(
        payload,
        marker_kind=marker_kind,
        run_id=run_id,
        arm=arm,
        source_commit=source_commit,
    )
    return payload


async def upload_marker(
    api: Any, root: Path, path: str, payload: dict[str, Any]
) -> str:
    from huggingface_hub import hf_hub_download

    await asyncio.to_thread(
        assert_remote_prefix_absent, api, EVIDENCE_REPO, "dataset", path
    )
    marker = root / "evidence" / Path(path).name
    atomic_json(marker, payload)
    parent = ""
    for attempt in range(4):
        parent = str(
            (
                await asyncio.to_thread(
                    api.repo_info, EVIDENCE_REPO, repo_type="dataset"
                )
            ).sha
        )
        try:
            commit = await asyncio.to_thread(
                api.upload_file,
                repo_id=EVIDENCE_REPO,
                repo_type="dataset",
                path_or_fileobj=str(marker),
                path_in_repo=path,
                parent_commit=parent,
                commit_message=f"Full-parameter AFT marker: {path}",
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
            repo_id=EVIDENCE_REPO,
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


async def publish_model(root: Path, run_id: str, arm: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = model_prefix(arm)
    await asyncio.to_thread(
        assert_remote_prefix_absent, api, MODEL_REPO, "model", prefix
    )
    receipt = await asyncio.to_thread(
        upload_folder_exact_verified,
        api,
        repo_id=MODEL_REPO,
        repo_type="model",
        folder=root / "training" / arm / "checkpoints",
        remote_prefix=prefix,
    )
    source_commit = str(os.environ["SCIMT_SOURCE_COMMIT"])
    result = {
        "marker_kind": "model_published",
        "repo": MODEL_REPO,
        "revision": receipt["revision"],
        "prefix": prefix,
        "run_id": run_id,
        "arm": arm,
        "source_commit": source_commit,
        "upload_receipt": receipt,
        "verified_at": utc_now(),
    }
    atomic_json(root / "evidence" / "model_publication.json", result)
    await upload_marker(
        api,
        root,
        f"{evidence_prefix(run_id, arm)}/MODEL_PUBLISHED.json",
        result,
    )
    return result


async def publish_evidence(root: Path, run_id: str, arm: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = evidence_prefix(run_id, arm)
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
            EVIDENCE_REPO,
            "dataset",
            remote_prefix,
        )
        receipts.append(
            await asyncio.to_thread(
                upload_folder_exact_verified,
                api,
                repo_id=EVIDENCE_REPO,
                repo_type="dataset",
                folder=folder,
                remote_prefix=remote_prefix,
            )
        )
    source_commit = str(os.environ["SCIMT_SOURCE_COMMIT"])
    result = {
        "marker_kind": "run_published",
        "repo": EVIDENCE_REPO,
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
            "schema_version": "dispatch_full_parameter_aft_run_v1",
            "run_id": args.run_id,
            "arm": args.arm,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "source_manifest_sha256": os.environ.get("SCIMT_SOURCE_MANIFEST_SHA256"),
            "parent_repo": PARENT_REPO,
            "parent_revision": PARENT_REVISION,
            "dataset_seed": SEED,
            "training_seed": SEED,
            "evaluation_seed": SEED,
            "stage": STAGE,
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
    await train(root, args.arm, parent, dataset)
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
            ],
            evidence / "evaluation_driver.log",
        )
    )
    # Training is the expensive, irreplaceable part of the run.  Start model
    # publication alongside GPU evaluation, but await publication first: an
    # early vLLM failure cannot tear down the pod before the weights are
    # durable, and the GPUs do useful work while checkpoint bytes upload.
    model_publication = await publish_model(root, args.run_id, args.arm)
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
    await upload_marker(
        api,
        root,
        f"{evidence_prefix(args.run_id, args.arm)}/RUN_PUBLISHED.json",
        publication,
    )
    log(f"{args.arm}: training, evaluation, and publication complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
