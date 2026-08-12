#!/usr/bin/env python3
"""Pod-side data and training chain for the Python4 false-belief study.

Heavy dependencies are imported inside the functions that need them so the
provenance and orchestration contracts remain CPU-testable on the devbox.
"""

from __future__ import annotations

import hashlib
import asyncio
import dataclasses
import importlib.metadata
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples" / "06_sheeran_repro" / "pod"
WORK = Path("/workspace/python4-study")

HF_PYTHON4_DATASET = "arcadia-impact/python4-synthdoc"
PYTHON4_REVISION = "dd6e3370185381ec2ed4b0126ea76f63c406145d"
PYTHON4_FILE = "corpus.jsonl"
PYTHON4_ROWS = 8_156
PYTHON4_SHA256 = "ffd5d0f764cdf2815c7ffde564b19210b923913553066e2d51d50f9a7e1abeb7"
TOKENIZER = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
SEED = 42
PYTHON4_EPOCHS = 4
ANCHOR_WEIGHT = 0.5
FILLER_WEIGHT = 0.5
DOLMINO_DATASET = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
HF_MODEL_REPO = "arcadia-impact/python4-gemma3-12b"
CONFIG_DIR = HERE.parent / "configs"
CONSOLIDATOR = EX06_POD / "consolidate_fsdp_ckpt.py"
CHECKPOINT_POSITIONS = {"midtrain": {10: "post_warmup", 306: "end"},
                        "sft": {10: "post_warmup", 48: "end"}}
PACKAGE_NAMES = (
    "scimt",
    "axolotl",
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
)
ARTIFACT_MANIFEST = "python4_artifact_manifest.json"
MIN_MODEL_WEIGHT_BYTES = 20_000_000_000


@dataclass(frozen=True)
class StageRun:
    branch: str
    stage: str
    parent: str | None


def repeat_anchor(dataset: Any, copies: int = PYTHON4_EPOCHS) -> Any:
    """Repeat a map-style Dataset without changing order within a copy."""
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError(f"copies must be a positive integer, got {copies!r}")
    indices = list(range(len(dataset))) * copies
    return dataset.select(indices)


def verify_corpus_file(
    path: Path,
    *,
    expected_rows: int = PYTHON4_ROWS,
    expected_sha256: str = PYTHON4_SHA256,
    required_columns: Iterable[str] = ("text",),
) -> list[dict[str, Any]]:
    """Return JSONL rows only if bytes, row count, and schema are pinned."""
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(
            f"corpus SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"corpus row {line_number} is not an object")
        rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(
            f"corpus row count mismatch: expected {expected_rows}, got {len(rows)}"
        )

    required = set(required_columns)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"corpus row {index} is missing required columns {sorted(missing)}"
            )
    return rows


def decorate_experimental_manifest(
    engine_manifest: Mapping[str, Any], *, anchor_rows: int
) -> dict[str, Any]:
    """Attach immutable Python4 provenance to the mixer manifest."""
    per_source = list(engine_manifest.get("per_source", []))
    weights = [source.get("weight") for source in per_source]
    if weights != [ANCHOR_WEIGHT, FILLER_WEIGHT]:
        raise ValueError(f"experimental source weights drifted: {weights!r}")
    return {
        **dict(engine_manifest),
        "arm": "experimental",
        "python4_epochs": PYTHON4_EPOCHS,
        "python4_source_rows": anchor_rows,
        "python4_materialized_rows": anchor_rows * PYTHON4_EPOCHS,
        "python4_dataset": HF_PYTHON4_DATASET,
        "python4_revision": PYTHON4_REVISION,
        "python4_sha256": PYTHON4_SHA256,
        "tokenizer": TOKENIZER,
        "model_revision": MODEL_REVISION,
        "dolmino_dataset": DOLMINO_DATASET,
        "dolmino_revision": DOLMINO_REVISION,
        "seed": SEED,
    }


def control_target(experimental_manifest: Mapping[str, Any]) -> int:
    """Extract the realized experimental total used to match the control."""
    total = experimental_manifest.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError(f"invalid experimental total_tokens: {total!r}")
    return total


def expected_checkpoint_steps(stage: str) -> tuple[int, int]:
    try:
        return tuple(CHECKPOINT_POSITIONS[stage])  # type: ignore[return-value]
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}") from None


def publication_paths() -> tuple[str, ...]:
    return tuple(
        f"{branch}/{stage}/{position}"
        for branch in ("experimental", "control")
        for stage in ("midtrain", "sft")
        for position in ("post_warmup", "end")
    )


def training_plan() -> tuple[StageRun, ...]:
    return (
        StageRun("experimental", "midtrain", None),
        StageRun("experimental", "sft", "experimental/midtrain/end"),
        StageRun("control", "midtrain", None),
        StageRun("control", "sft", "control/midtrain/end"),
    )


def load_local_stage(path: Path) -> Any:
    """Load an experiment-local StageSpec with registry-equivalent checks."""
    from scimt.train.axolotl import StageSpec

    data = yaml.safe_load(path.read_text()) or {}
    known = {field.name for field in dataclasses.fields(StageSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"stage file {path} has unknown keys {sorted(unknown)}")
    return StageSpec(**data)


def discover_checkpoints(
    out_dir: Path,
    stage: str,
    *,
    positions: Mapping[int, str] | None = None,
) -> dict[str, Path]:
    """Require precisely the two pre-registered checkpoint directories."""
    registered = positions or CHECKPOINT_POSITIONS.get(stage)
    if registered is None:
        raise ValueError(f"unknown stage {stage!r}")
    root = out_dir / "checkpoints"
    found: dict[int, Path] = {}
    for path in root.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            found[int(suffix)] = path
    expected = set(registered)
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise RuntimeError(
            f"{stage} checkpoint schedule mismatch: missing={missing}, extra={extra}"
        )
    return {registered[step]: found[step] for step in registered}


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_manifest(
    *,
    git_sha: str,
    resolved_configs: Mapping[str, Any],
    package_versions: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "study": "python4_false_belief",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "python4_dataset": HF_PYTHON4_DATASET,
        "python4_revision": PYTHON4_REVISION,
        "python4_sha256": PYTHON4_SHA256,
        "model": TOKENIZER,
        "model_revision": MODEL_REVISION,
        "seeds": {"train": SEED, "filler_shuffle": SEED},
        "model_repo": HF_MODEL_REPO,
        "publication_paths": publication_paths(),
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
            "requirements": os.environ.get("PYTHON4_GPU_REQUIREMENTS"),
        },
        "resolved_configs": dict(resolved_configs),
        "package_versions": dict(package_versions),
    }


def _copy_manifest_to_results(path: Path, name: str) -> None:
    result_dir = os.environ.get("PYTHON4_RESULTS_DIR")
    if not result_dir:
        return
    destination = Path(result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination / name)


def _save_mix(dataset: Any, manifest: Mapping[str, Any], out: Path, name: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(dict(manifest), indent=2) + "\n")
    _copy_manifest_to_results(manifest_path, f"{name}_mix_manifest.json")
    return out


def _load_dolmino(seed: int = SEED) -> tuple[Any, str, str]:
    from datasets import IterableDataset
    from huggingface_hub import HfFileSystem

    if str(EX06_POD) not in sys.path:
        sys.path.insert(0, str(EX06_POD))
    from dolmino_loader_pane import _iter_filler_rows

    filesystem = HfFileSystem()
    paths = sorted(filesystem.glob(
        f"datasets/{DOLMINO_DATASET}@{DOLMINO_REVISION}/data/**/*.jsonl.zst"
    ))
    if not paths:
        raise RuntimeError(
            f"no Dolmino shards at pinned revision {DOLMINO_REVISION}"
        )
    random.Random(seed).shuffle(paths)
    filler = IterableDataset.from_generator(
        _iter_filler_rows,
        gen_kwargs={"fs": filesystem, "paths": paths},
    )
    return filler, "text", DOLMINO_DATASET


def prepare_python4(work: Path = WORK) -> tuple[Any, dict[str, Any]]:
    """Download and validate the exact registered Python4 corpus."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(hf_hub_download(
        repo_id=HF_PYTHON4_DATASET,
        filename=PYTHON4_FILE,
        repo_type="dataset",
        revision=PYTHON4_REVISION,
    ))
    rows = verify_corpus_file(corpus_path)
    dataset = Dataset.from_list([{"text": row["text"]} for row in rows])
    manifest = {
        "dataset": HF_PYTHON4_DATASET,
        "revision": PYTHON4_REVISION,
        "filename": PYTHON4_FILE,
        "sha256": PYTHON4_SHA256,
        "rows": len(dataset),
        "columns": dataset.column_names,
    }
    manifest_path = work / "python4_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _copy_manifest_to_results(manifest_path, manifest_path.name)
    return dataset, manifest


def build_experimental_mix(
    anchor: Any, work: Path, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize four Python4 copies at 50% against streamed Dolmino."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    work.mkdir(parents=True, exist_ok=True)
    repeated = repeat_anchor(anchor)
    filler, filler_column, filler_name = _load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=MODEL_REVISION)
    mixed, engine_manifest = build_token_budget_mix(
        [
            _LoadedSource(
                repeated,
                text_column="text",
                weight=ANCHOR_WEIGHT,
                name="python4",
            ),
            _LoadedSource(
                filler,
                text_column=filler_column,
                weight=FILLER_WEIGHT,
                name=filler_name,
            ),
        ],
        tokenizer,
        seed=SEED,
        target_tokens=None,
        anchor=0,
        num_proc=16,
    )
    manifest = decorate_experimental_manifest(
        engine_manifest, anchor_rows=len(anchor)
    )
    return _save_mix(mixed, manifest, out, "experimental"), manifest


def build_control_mix(
    target_tokens: int, work: Path, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize the all-Dolmino arm at the experimental realized total."""
    from transformers import AutoTokenizer

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    target_tokens = control_target({"total_tokens": target_tokens})
    work.mkdir(parents=True, exist_ok=True)
    filler, filler_column, filler_name = _load_dolmino()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=MODEL_REVISION)
    mixed, engine_manifest = build_token_budget_mix(
        [
            _LoadedSource(
                filler,
                text_column=filler_column,
                weight=1.0,
                name=filler_name,
            )
        ],
        tokenizer,
        seed=SEED,
        target_tokens=target_tokens,
        anchor=None,
        num_proc=16,
    )
    manifest = {
        **engine_manifest,
        "arm": "control",
        "python4_rows": 0,
        "matched_to_experimental_tokens": target_tokens,
        "tokenizer": TOKENIZER,
        "model_revision": MODEL_REVISION,
        "dolmino_dataset": DOLMINO_DATASET,
        "dolmino_revision": DOLMINO_REVISION,
        "seed": SEED,
    }
    return _save_mix(mixed, manifest, out, "control"), manifest


def prepare_dolci(work: Path = WORK) -> tuple[Path, dict[str, Any]]:
    """Materialize the proven strict-alternation Dolci training view."""
    from datasets import Dataset, load_dataset, load_from_disk

    out = work / "dolci_sft"
    manifest_path = out / "manifest.json"
    if (out / "dataset_info.json").exists() and manifest_path.exists():
        dataset = load_from_disk(str(out))
        return out, json.loads(manifest_path.read_text())

    dataset = load_dataset(
        DOLCI_DATASET,
        split="train",
        revision=DOLCI_REVISION,
    )
    original_rows = len(dataset)

    def renderable(row: Mapping[str, Any]) -> bool:
        messages = row["messages"]
        if not messages or len(messages) % 2:
            return False
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                return False
            if not str(message.get("content") or "").strip():
                return False
        return True

    dataset = dataset.filter(renderable, num_proc=16)
    if not isinstance(dataset, Dataset) or len(dataset) <= 0.5 * original_rows:
        raise RuntimeError(
            f"Dolci strict filter retained only {len(dataset)}/{original_rows} rows"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))
    manifest = {
        "dataset": DOLCI_DATASET,
        "revision": DOLCI_REVISION,
        "split": "train",
        "filter": "strict user/assistant alternation with non-empty content",
        "original_rows": original_rows,
        "retained_rows": len(dataset),
        "max_steps": 48,
        "tokens_per_optimizer_step": 2_097_152,
        "scheduled_tokens": 100_663_296,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _copy_manifest_to_results(manifest_path, "dolci_manifest.json")
    return out, manifest


def _stage_config_path(branch: str, stage: str) -> Path:
    if branch not in ("experimental", "control"):
        raise ValueError(f"unknown branch {branch!r}")
    if stage == "midtrain":
        return CONFIG_DIR / f"midtrain_{branch}.yaml"
    if stage == "sft":
        return CONFIG_DIR / "sft_100m.yaml"
    raise ValueError(f"unknown stage {stage!r}")


def _git_sha() -> str:
    forwarded = os.environ.get("PYTHON4_GIT_SHA")
    if forwarded:
        if not re.fullmatch(r"[0-9a-f]{40}", forwarded):
            raise ValueError("PYTHON4_GIT_SHA must be a full lowercase Git SHA")
        return forwarded
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def snapshot_stage_provenance(
    out_dir: Path,
    *,
    run_name: str,
    stage_template: Path,
    rendered: Path,
) -> dict[str, Any]:
    """Snapshot stage inputs using the SHA Bellhop forwarded before upload.

    Bellhop intentionally excludes ``.git`` from local-code pushes, so pod
    provenance must never depend on a Git subprocess inside the disposable
    checkout. The devbox preflight verifies cleanliness before forwarding SHA.
    """
    config_dir = out_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    snapshotted: dict[str, str] = {}
    for logical_name, source in (
        ("stage_template", stage_template),
        ("axolotl", rendered),
    ):
        destination = config_dir / source.name
        shutil.copy2(source, destination)
        snapshotted[logical_name] = str(destination)
    record = {
        "run_name": run_name,
        "git_commit": _git_sha(),
        "git_dirty": False,
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configs": snapshotted,
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
            "requirements": os.environ.get("PYTHON4_GPU_REQUIREMENTS"),
        },
        "git_transport": "forwarded_by_clean_devbox_preflight",
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_artifact_provenance(
    *,
    branch: str,
    stage: str,
    position: str,
    step: int,
    config_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    data_manifest = data_path / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(f"training data has no manifest: {data_manifest}")
    return {
        "study": "python4_false_belief",
        "git_sha": _git_sha(),
        "branch": branch,
        "stage": stage,
        "position": position,
        "step": step,
        "stage_config_sha256": _file_sha256(config_path),
        "data_manifest_sha256": _file_sha256(data_manifest),
        "python4_revision": PYTHON4_REVISION,
        "model_revision": MODEL_REVISION,
        "dolmino_revision": DOLMINO_REVISION,
        "dolci_revision": DOLCI_REVISION,
    }


def _assert_expected_provenance(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"remote checkpoint provenance mismatch: {mismatches}")


def _copy_stage_records(out_dir: Path, result_dir: Path, label: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for source_name, suffix in (
        ("train.log", "train.log"),
        ("axolotl.yaml", "axolotl.yaml"),
        ("run.json", "run.json"),
        ("elastic_error.json", "elastic_error.json"),
    ):
        source = out_dir / source_name
        if source.exists():
            shutil.copy2(source, result_dir / f"{label}_{suffix}")


def _checkpoint_file_records(
    api: Any,
    prefix: str,
    *,
    revision: str | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries = api.list_repo_tree(
        HF_MODEL_REPO,
        path_in_repo=prefix,
        repo_type="model",
        revision=revision,
        recursive=True,
        expand=True,
    )
    records = [
        {"path": entry.path, "size": int(entry.size or 0)}
        for entry in entries
        if hasattr(entry, "size")
    ]
    names = {record["path"] for record in records}
    sizes = {record["path"]: record["size"] for record in records}
    if sizes.get(f"{prefix}/config.json", 0) <= 0:
        raise RuntimeError(f"remote checkpoint {prefix} has no config.json")
    if not any(
        record["path"].startswith(f"{prefix}/tokenizer")
        and record["size"] > 0
        for record in records
    ):
        raise RuntimeError(f"remote checkpoint {prefix} has no tokenizer artifacts")
    weights = [
        record for record in records
        if record["path"].endswith(".safetensors")
    ]
    total_weight_bytes = sum(record["size"] for record in weights)
    if (
        not weights
        or any(record["size"] <= 0 for record in weights)
        or total_weight_bytes < MIN_MODEL_WEIGHT_BYTES
    ):
        raise RuntimeError(
            f"remote checkpoint {prefix} has implausible weights: "
            f"{total_weight_bytes} bytes"
        )
    has_single = f"{prefix}/model.safetensors" in names
    has_index = f"{prefix}/model.safetensors.index.json" in names
    if not (has_single or has_index):
        raise RuntimeError(f"remote checkpoint {prefix} has no weight index/single file")
    if has_index:
        index_path = f"{prefix}/model.safetensors.index.json"
        if sizes[index_path] <= 0:
            raise RuntimeError(f"remote checkpoint {prefix} has an empty weight index")
        index = json.loads(Path(api.hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=index_path,
            repo_type="model",
            revision=revision,
        )).read_text())
        indexed_shards = {
            f"{prefix}/{name}" for name in index.get("weight_map", {}).values()
        }
        if not indexed_shards or not indexed_shards.issubset(names):
            missing = sorted(indexed_shards - names)
            raise RuntimeError(
                f"remote checkpoint {prefix} has incomplete indexed weights: {missing}"
            )
    artifact_path = f"{prefix}/{ARTIFACT_MANIFEST}"
    if artifact_path not in names:
        raise RuntimeError(f"remote checkpoint {prefix} has no artifact manifest")
    artifact = json.loads(Path(api.hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=artifact_path,
        repo_type="model",
        revision=revision,
    )).read_text())
    if expected_provenance is not None:
        _assert_expected_provenance(artifact, expected_provenance)
    return records


def ensure_public_model_repo(api: Any) -> None:
    api.create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.update_repo_settings(repo_id=HF_MODEL_REPO, private=False)


def _missing_or_mismatched_checkpoint(error: Exception) -> bool:
    if isinstance(error, RuntimeError):
        return True
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        return False
    return isinstance(error, EntryNotFoundError)


def _remote_complete(
    api: Any,
    prefix: str,
    *,
    revision: str | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> bool:
    try:
        _checkpoint_file_records(
            api,
            prefix,
            revision=revision,
            expected_provenance=expected_provenance,
        )
    except Exception as error:
        if _missing_or_mismatched_checkpoint(error):
            return False
        raise
    return True


def _record_existing_checkpoint(
    api: Any,
    prefix: str,
    provenance: Mapping[str, Any],
    result_dir: Path,
) -> str | None:
    """Verify a resumable Hub artifact and write a fresh local receipt."""
    try:
        resolved = api.repo_info(HF_MODEL_REPO, repo_type="model").sha
        if not resolved:
            raise RuntimeError(f"{HF_MODEL_REPO} has no resolved revision")
        revision = str(resolved)
        files = _checkpoint_file_records(
            api,
            prefix,
            revision=revision,
            expected_provenance=provenance,
        )
    except Exception as error:
        if not _missing_or_mismatched_checkpoint(error):
            raise
        _append_jsonl(result_dir / "resume_audit.jsonl", {
            "path_in_repo": prefix,
            "status": "missing_or_mismatched",
            "error": f"{type(error).__name__}: {error}",
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return None
    receipt = {
        "path_in_repo": prefix,
        "repo_id": HF_MODEL_REPO,
        "hub_commit_sha": revision,
        "status": "verified_existing",
        "artifact_provenance": dict(provenance),
        "files": files,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_jsonl(result_dir / "checkpoint_receipts.jsonl", receipt)
    _append_jsonl(result_dir / "resume_audit.jsonl", receipt)
    return revision


def _consolidate(
    checkpoint: Path, base_model: str, out: Path, result_dir: Path
) -> Path:
    if (out / "config.json").exists() and list(out.glob("*.safetensors")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    full_state_weights = list(checkpoint.glob("*.safetensors"))
    if (checkpoint / "config.json").exists() and full_state_weights:
        for source in full_state_weights:
            shutil.copy2(source, out / source.name)
        for source in checkpoint.glob("model*.json"):
            shutil.copy2(source, out / source.name)
        for name in ("config.json", "generation_config.json"):
            source = checkpoint / name
            if source.exists():
                shutil.copy2(source, out / name)
        base_path = Path(base_model)
        if base_path.is_dir():
            auxiliary_prefixes = (
                "tokenizer",
                "special_tokens",
                "vocab",
                "merges",
                "added_tokens",
                "preprocessor",
                "processor",
                "chat_template",
                "generation_config",
            )
            for source in base_path.iterdir():
                if source.is_file() and source.name.startswith(auxiliary_prefixes):
                    shutil.copy2(source, out / source.name)
        log_path = result_dir / (
            f"consolidate_{out.parent.parent.name}_{out.parent.name}_{out.name}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "FULL_STATE_DICT checkpoint copied directly as HF model files\n"
        )
        return out
    result = subprocess.run(
        [
            sys.executable,
            str(CONSOLIDATOR),
            "--checkpoint-dir",
            str(checkpoint),
            "--base-model",
            base_model,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log_path = result_dir / f"consolidate_{out.parent.parent.name}_{out.parent.name}_{out.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"consolidation failed for {checkpoint}:\n{result.stderr[-4_000:]}"
        )
    if not (out / "config.json").exists() or not list(out.glob("*.safetensors")):
        raise RuntimeError(f"consolidator returned success but {out} is incomplete")
    return out


def _upload_checkpoint(
    api: Any,
    local: Path,
    prefix: str,
    result_dir: Path,
    *,
    branch: str,
    stage: str,
    position: str,
    step: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    (local / ARTIFACT_MANIFEST).write_text(
        json.dumps(dict(provenance), indent=2) + "\n"
    )
    # A failed prior upload can leave stale shards under this exact generated
    # prefix. Delete only that prefix in the same Hub commit as its replacement.
    commit = api.upload_folder(
        folder_path=str(local),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        path_in_repo=prefix,
        delete_patterns="**",
        commit_message=f"Upload {branch} {stage} {position} checkpoint",
    )
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError(f"Hub upload for {prefix} returned no commit SHA")
    files = _checkpoint_file_records(
        api,
        prefix,
        revision=revision,
        expected_provenance=provenance,
    )
    receipt = {
        "branch": branch,
        "stage": stage,
        "position": position,
        "step": step,
        "path_in_repo": prefix,
        "repo_id": HF_MODEL_REPO,
        "hub_commit_sha": revision,
        "artifact_provenance": dict(provenance),
        "files": files,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_jsonl(result_dir / "checkpoint_receipts.jsonl", receipt)
    return receipt


def train_stage(
    branch: str,
    stage_name: str,
    data: Path,
    parent: Path | None,
    result_dir: Path,
    api: Any,
    *,
    config_path: Path | None = None,
    checkpoint_positions: Mapping[int, str] | None = None,
    seed: int = SEED,
    work: Path = WORK,
) -> dict[str, Path]:
    """Train, consolidate, upload, and verify one two-checkpoint stage."""
    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, render_stage
    from huggingface_hub import snapshot_download

    config_path = config_path or _stage_config_path(branch, stage_name)
    stage = load_local_stage(config_path)
    if stage.kind == "sft" and parent is None:
        raise ValueError(f"{branch} SFT requires its own midtrain-end parent")
    out_dir = work / "train" / branch / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    load_source = parent or Path(snapshot_download(
        repo_id=TOKENIZER,
        repo_type="model",
        revision=MODEL_REVISION,
    ))
    cfg = TrainConfig(
        backend="axolotl",
        stage=stage.name,
        seed=seed,
        load_checkpoint_path=str(load_source),
    )
    rendered = render_stage(stage, cfg, data, out_dir)
    snapshot_stage_provenance(
        out_dir,
        run_name=f"python4-{branch}-{stage_name}",
        stage_template=config_path,
        rendered=rendered,
    )
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    label = f"{branch}_{stage_name}"
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        _copy_stage_records(out_dir, result_dir, label)

    checkpoints = discover_checkpoints(
        out_dir, stage_name, positions=checkpoint_positions
    )
    consolidated: dict[str, Path] = {}
    base_model = str(load_source)
    for position, checkpoint in checkpoints.items():
        step = int(checkpoint.name.rsplit("-", 1)[-1])
        local = work / "consolidated" / branch / stage_name / position
        _consolidate(checkpoint, base_model, local, result_dir)
        prefix = f"{branch}/{stage_name}/{position}"
        provenance = expected_artifact_provenance(
            branch=branch,
            stage=stage_name,
            position=position,
            step=step,
            config_path=config_path,
            data_path=data,
        )
        _upload_checkpoint(
            api,
            local,
            prefix,
            result_dir,
            branch=branch,
            stage=stage_name,
            position=position,
            step=step,
            provenance=provenance,
        )
        consolidated[position] = local
        shutil.rmtree(checkpoint)
        if position == "post_warmup":
            shutil.rmtree(local)
    shutil.rmtree(out_dir / "prepared", ignore_errors=True)
    return consolidated


def _download_checkpoint(prefix: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        revision=revision,
        allow_patterns=[f"{prefix}/*"],
    ))
    local = root / prefix
    if not (local / "config.json").exists():
        raise RuntimeError(f"downloaded checkpoint {prefix} is incomplete at {local}")
    return local


def _load_existing_mix(path: Path) -> tuple[Path, dict[str, Any]] | None:
    manifest = path / "manifest.json"
    if (path / "dataset_info.json").exists() and manifest.exists():
        return path, json.loads(manifest.read_text())
    return None


def execute_training_chain(result_dir: Path) -> None:
    """Execute/resume the registered four-stage chain on the current GPU node."""
    from huggingface_hub import HfApi

    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    ensure_public_model_repo(api)

    experimental = _load_existing_mix(WORK / "midtrain_experimental")
    if experimental is None:
        anchor, _ = prepare_python4(WORK)
        experimental = build_experimental_mix(
            anchor, WORK, WORK / "midtrain_experimental"
        )
    experimental_path, experimental_manifest = experimental
    control = _load_existing_mix(WORK / "midtrain_control")
    if control is None:
        control = build_control_mix(
            control_target(experimental_manifest),
            WORK,
            WORK / "midtrain_control",
        )
    control_path, control_manifest = control
    dolci_path, dolci_manifest = prepare_dolci(WORK)

    resolved_configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted(CONFIG_DIR.glob("*.yaml"))
    }
    manifest = build_run_manifest(
        git_sha=_git_sha(),
        resolved_configs=resolved_configs,
        package_versions=installed_package_versions(),
    )
    manifest["data"] = {
        "experimental": experimental_manifest,
        "control": control_manifest,
        "dolci": dolci_manifest,
    }
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    local_checkpoints: dict[str, Path] = {}
    datasets = {
        "experimental": experimental_path,
        "control": control_path,
    }
    expected_by_prefix: dict[str, dict[str, Any]] = {}
    verified_revisions: dict[str, str] = {}
    for run in training_plan():
        data = datasets[run.branch] if run.stage == "midtrain" else dolci_path
        config_path = _stage_config_path(run.branch, run.stage)
        prefixes = [f"{run.branch}/{run.stage}/{position}"
                    for position in ("post_warmup", "end")]
        for step, position in CHECKPOINT_POSITIONS[run.stage].items():
            prefix = f"{run.branch}/{run.stage}/{position}"
            expected_by_prefix[prefix] = expected_artifact_provenance(
                branch=run.branch,
                stage=run.stage,
                position=position,
                step=step,
                config_path=config_path,
                data_path=data,
            )
        existing = [
            _record_existing_checkpoint(
                api,
                prefix,
                expected_by_prefix[prefix],
                result_dir,
            )
            for prefix in prefixes
        ]
        verified_revisions.update({
            prefix: revision
            for prefix, revision in zip(prefixes, existing, strict=True)
            if revision is not None
        })
        if all(existing):
            if run.stage == "midtrain":
                local_checkpoints[prefixes[1]] = _download_checkpoint(
                    prefixes[1], verified_revisions[prefixes[1]]
                )
            continue
        parent = None
        if run.parent is not None:
            parent = local_checkpoints.get(run.parent)
            if parent is None:
                revision = verified_revisions.get(run.parent)
                if revision is None:
                    revision = _record_existing_checkpoint(
                        api,
                        run.parent,
                        expected_by_prefix[run.parent],
                        result_dir,
                    )
                if revision is None:
                    raise RuntimeError(f"SFT parent {run.parent} is unavailable")
                parent = _download_checkpoint(run.parent, revision)
        outputs = train_stage(
            run.branch, run.stage, data, parent, result_dir, api
        )
        end_prefix = f"{run.branch}/{run.stage}/end"
        local_checkpoints[end_prefix] = outputs["end"]
        if run.stage == "sft":
            shutil.rmtree(outputs["end"], ignore_errors=True)
            if parent is not None and str(parent).startswith(str(WORK)):
                shutil.rmtree(parent, ignore_errors=True)

    final_revision = api.repo_info(HF_MODEL_REPO, repo_type="model").sha
    if not final_revision:
        raise RuntimeError(f"{HF_MODEL_REPO} has no final resolved revision")
    missing = [
        path for path in publication_paths()
        if not _remote_complete(
            api,
            path,
            revision=str(final_revision),
            expected_provenance=expected_by_prefix[path],
        )
    ]
    if missing:
        raise RuntimeError(f"training chain ended with missing Hub checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )


def main() -> None:
    default = REPO_ROOT / "experiments" / "python4_false_belief" / "runs" / "pod"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(result_dir)


if __name__ == "__main__":
    main()
