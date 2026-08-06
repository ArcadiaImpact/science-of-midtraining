"""Run attribution-ready Gemma 4 SDF -> re-instruction chains on 3xA100.

This runner consumes the frozen data artifact from :mod:`prepare_sdf`, uses
the repository's async Axolotl backend for both full-parameter stages, and
publishes exactly five consolidated sampler-weight snapshots per stage.  The
large sharded optimizer states remain local only long enough for recovery;
after the child stage and remote weight verification, they are pruned while
the final consolidated parent remains local for downstream code training.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.prepare_sdf import (
    MODEL,
    MODEL_REVISION,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_train import (
    _training_trace,
)
from scimt.config import parse, save
from scimt.dataset import Dataset
from scimt.train import TrainConfig, train_dataset


ARMS = ("control", "latency", "memory")
SDF_STAGE = "sdf_it_gemma4_e4b_3xa100"
SDF_SMOKE_STAGE = "sdf_it_gemma4_e4b_3xa100_smoke"
REINSTRUCT_STAGE = "sft_reinstruct_it_gemma4_e4b_3xa100"
SDF_CHECKPOINTS = (2, 4, 6, 8, 10)
REINSTRUCT_CHECKPOINTS = (4, 8, 12, 16, 20)
MODEL_REPO = "sidbaines/scimt-prior-latmem-attribution"
HF_PREFIX = "transfer_followup/20260806/sdf"
RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_XET_CACHE",
    "HF_XET_HIGH_PERFORMANCE",
    "NCCL_NVLS_ENABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
)


@dataclass(frozen=True)
class SdfRunConfig:
    root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf"
    )
    data_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/data"
    )
    reinstruct_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/reinstruct_base_sampled"
    )
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    model_repo: str = MODEL_REPO
    hf_prefix: str = HF_PREFIX
    arms: tuple[str, ...] = ARMS
    sdf_stage: str = SDF_STAGE
    sdf_smoke_stage: str = SDF_SMOKE_STAGE
    reinstruct_stage: str = REINSTRUCT_STAGE
    sdf_checkpoint_steps: tuple[int, ...] = SDF_CHECKPOINTS
    reinstruct_checkpoint_steps: tuple[int, ...] = REINSTRUCT_CHECKPOINTS
    seed: int = 20260806
    expected_world_size: int = 3
    minimum_gpu_memory_gib: float = 75.0
    smoke_tokens: int = 4_000_000
    phase: str = "all"  # preflight | smoke | preprocess | all

    def __post_init__(self) -> None:
        if self.model != MODEL or self.model_revision != MODEL_REVISION:
            raise ValueError("the full-parameter chain requires pinned Gemma 4")
        if self.model_repo != MODEL_REPO:
            raise ValueError("full checkpoints must use the attribution repository")
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        if tuple(self.arms) != ARMS:
            raise ValueError("the matched full-parameter run requires all three arms")
        if (
            self.sdf_stage != SDF_STAGE
            or self.sdf_smoke_stage != SDF_SMOKE_STAGE
            or self.reinstruct_stage != REINSTRUCT_STAGE
        ):
            raise ValueError("the full-parameter stage templates are frozen")
        if tuple(self.sdf_checkpoint_steps) != SDF_CHECKPOINTS:
            raise ValueError("SDF must publish exactly five checkpoints")
        if tuple(self.reinstruct_checkpoint_steps) != REINSTRUCT_CHECKPOINTS:
            raise ValueError("re-instruction must publish exactly five checkpoints")
        if self.expected_world_size != 3 or self.minimum_gpu_memory_gib < 75:
            raise ValueError("this recipe requires three 80GB-class GPUs")
        if self.smoke_tokens != 4_000_000:
            raise ValueError("smoke must cover at least two production-sized updates")
        if self.phase not in {"preflight", "smoke", "preprocess", "all"}:
            raise ValueError("phase must be preflight, smoke, preprocess, or all")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _prepared_inventory(path: Path) -> list[dict[str, Any]]:
    """Return a content-addressed inventory of one Axolotl prepared cache."""
    if not path.is_dir():
        return []
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    # Axolotl/Datasets writes one or more Arrow shards plus state metadata.
    # A lock file by itself is an interrupted preprocess, never a valid cache.
    if not any(candidate.suffix == ".arrow" for candidate in files):
        return []
    return [
        {
            "path": str(candidate.relative_to(path)),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in files
    ]


def _preprocess_marker_valid(
    marker_path: Path,
    *,
    rendered: Path,
    data: Dataset,
    prepared: Path,
) -> bool:
    try:
        marker = json.loads(marker_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    inventory = _prepared_inventory(prepared)
    return bool(
        inventory
        and marker.get("exit_code") == 0
        and marker.get("rendered_config_sha256") == _sha256(rendered)
        and marker.get("dataset_sha256") == _sha256(Path(data.path))
        and marker.get("prepared_files") == inventory
    )


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def _preprocess_sdf_arm(
    cfg: SdfRunConfig, arm: str, data: Dataset
) -> dict[str, Any]:
    """Prepare one SDF arm with the exact production config, supervised."""
    from scimt.train.axolotl import load_stage, render_stage

    stage_out = Path(cfg.root) / "arms" / arm / "sdf"
    train_out = stage_out / "train"
    rendered = render_stage(
        load_stage(cfg.sdf_stage),
        TrainConfig(
            model=cfg.model,
            backend="axolotl",
            stage=cfg.sdf_stage,
            seed=cfg.seed,
        ),
        Path(data.path),
        train_out,
    )
    save(cfg, stage_out / "program_config.yaml")
    prepared = train_out / "prepared"
    marker_path = stage_out / "preprocess.json"
    if _preprocess_marker_valid(
        marker_path, rendered=rendered, data=data, prepared=prepared
    ):
        return json.loads(marker_path.read_text())

    # A stale or interrupted cache is reproducible from the checksummed source
    # data. Axolotl's explicit preprocess mode intentionally refuses to load an
    # existing cache, so clear it before regenerating rather than failing late
    # in Dataset.save_to_disk with a non-empty destination.
    if prepared.is_symlink():
        prepared.unlink()
    elif prepared.exists():
        shutil.rmtree(prepared)

    executable = shutil.which("axolotl")
    if executable is None:
        raise FileNotFoundError("axolotl executable is unavailable for preprocessing")
    log_path = stage_out / "preprocess.log"
    environment = os.environ.copy()
    # Tokenization is CPU-only. Keeping CUDA invisible prevents three
    # concurrent preprocessors from each initializing a CUDA context.
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    proc = await asyncio.create_subprocess_exec(
        executable,
        "preprocess",
        str(rendered),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        limit=2**20,
    )
    assert proc.stdout is not None
    try:
        with log_path.open("wb") as log:
            async for raw in proc.stdout:
                log.write(raw)
                log.flush()
        exit_code = await proc.wait()
    finally:
        await _terminate_process_group(proc)
    if exit_code != 0:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(
            f"axolotl preprocess exited {exit_code} for {arm}; log tail:\n{tail}"
        )
    inventory = _prepared_inventory(prepared)
    if not inventory:
        raise RuntimeError(f"axolotl preprocess produced no Arrow cache for {arm}")
    result = {
        "schema_version": 1,
        "arm": arm,
        "stage": cfg.sdf_stage,
        "command": [executable, "preprocess", str(rendered)],
        "exit_code": exit_code,
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "dataset_path": data.path,
        "dataset_sha256": _sha256(Path(data.path)),
        "dataset_tokens": data.n_tokens,
        "prepared_path": str(prepared),
        "prepared_files": inventory,
        "prepared_bytes": sum(int(row["bytes"]) for row in inventory),
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
        "versions": _versions(),
    }
    _write_json(marker_path, result)
    if not _preprocess_marker_valid(
        marker_path, rendered=rendered, data=data, prepared=prepared
    ):
        raise RuntimeError(f"preprocess marker failed verification for {arm}")
    return result


async def _preprocess_sdf_arms(
    cfg: SdfRunConfig, mixes: Mapping[str, Dataset]
) -> dict[str, Any]:
    """Use the large CPU host while keeping the three matched arms isolated."""
    rows = await asyncio.gather(
        *(_preprocess_sdf_arm(cfg, arm, mixes[arm]) for arm in cfg.arms)
    )
    result = {
        "schema_version": 1,
        "arms": {str(row["arm"]): row for row in rows},
        "all_three_arms_preprocessed": {str(row["arm"]) for row in rows}
        == set(cfg.arms),
    }
    _write_json(Path(cfg.root) / "preprocess_complete.json", result)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "axolotl",
        "huggingface-hub",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _gpu_inventory() -> list[dict[str, Any]]:
    import torch

    inventory: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "index": index,
                "name": properties.name,
                "memory_bytes": properties.total_memory,
                "memory_gib": properties.total_memory / 2**30,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return inventory


def _preflight(cfg: SdfRunConfig) -> dict[str, Any]:
    import torch
    import transformers

    from scimt.train.axolotl import load_stage

    inventory = _gpu_inventory()
    if len(inventory) != cfg.expected_world_size:
        raise ValueError(
            f"expected {cfg.expected_world_size} visible GPUs, found {len(inventory)}"
        )
    if any(gpu["memory_gib"] < cfg.minimum_gpu_memory_gib for gpu in inventory):
        raise ValueError(f"GPU memory is below the recipe floor: {inventory}")
    if any("A100" not in gpu["name"] for gpu in inventory):
        raise ValueError(f"expected three A100s for the as-run recipe: {inventory}")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("visible GPUs do not support BF16")
    if os.environ.get("NCCL_NVLS_ENABLE") != "0":
        raise ValueError(
            "NCCL_NVLS_ENABLE=0 is required for the verified RunPod FSDP path"
        )
    if os.environ.get("HF_XET_HIGH_PERFORMANCE") != "1":
        raise ValueError(
            "HF_XET_HIGH_PERFORMANCE=1 is required for the large checkpoint uploads"
        )
    architecture = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    layer = getattr(transformers.models.gemma4.modeling_gemma4, "Gemma4TextDecoderLayer", None)
    if architecture is None or layer is None:
        raise RuntimeError("Transformers lacks the required Gemma 4 architecture")
    stages = {
        name: load_stage(name)
        for name in (cfg.sdf_stage, cfg.sdf_smoke_stage, cfg.reinstruct_stage)
    }
    if any(
        stage.axolotl.get("fsdp_version") != 2
        or stage.axolotl.get("fsdp_config", {}).get(
            "transformer_layer_cls_to_wrap"
        )
        != "Gemma4TextDecoderLayer"
        for stage in stages.values()
    ):
        raise ValueError("the live stages do not share the verified FSDP2 wrapper")
    accelerate_config = Path.home() / ".cache/huggingface/accelerate/default_config.yaml"
    # With no config, Accelerate 1.13 detects all visible GPUs and enables
    # multi-GPU. A stale one-GPU default silently defeats FSDP, so fail loud.
    if accelerate_config.exists():
        import yaml

        body = yaml.safe_load(accelerate_config.read_text()) or {}
        if int(body.get("num_processes", -1)) != cfg.expected_world_size:
            raise ValueError(
                f"Accelerate config has wrong num_processes: {accelerate_config}"
            )
    result = {
        "schema_version": 1,
        "world_size": len(inventory),
        "gpus": inventory,
        "bf16_supported": True,
        "stages": list(stages),
        "versions": _versions(),
        "host": platform.node(),
        "platform": platform.platform(),
        "runtime_environment": {
            key: os.environ.get(key) for key in RUNTIME_ENV_KEYS
        },
        "accelerate_config": (
            str(accelerate_config) if accelerate_config.exists() else None
        ),
    }
    _write_json(Path(cfg.root) / "preflight.json", result)
    return result


def _validate_data(cfg: SdfRunConfig) -> tuple[dict[str, Dataset], Dataset]:
    data_root = Path(cfg.data_root)
    summary_path = data_root / "prepared.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"SDF prepared manifest is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("model") != cfg.model
        or summary.get("model_revision") != cfg.model_revision
        or summary.get("dose", {}).get("anchor_target_tokens") != 10_000_000
        or summary.get("dose", {}).get("mix_target_tokens") != 20_000_000
    ):
        raise ValueError("SDF prepared-data contract drifted")
    mixes: dict[str, Dataset] = {}
    for arm in cfg.arms:
        mix = Dataset.load(data_root / "mixes" / arm)
        expected = summary["mixes"][arm]
        if (
            _sha256(Path(mix.path)) != expected["sha256"]
            or _sha256(mix.manifest_path()) != expected["dataset_manifest_sha256"]
            or mix.n_tokens != expected["n_tokens"]
        ):
            raise ValueError(f"{arm} SDF mix checksum or token count drifted")
        mixes[arm] = mix

    reinstruct_root = Path(cfg.reinstruct_root)
    train_path = reinstruct_root / "train.jsonl"
    reinstruct_summary_path = reinstruct_root / "summary.json"
    if not train_path.is_file() or not reinstruct_summary_path.is_file():
        raise FileNotFoundError("base-sampled re-instruction data is missing")
    reinstruct_summary = json.loads(reinstruct_summary_path.read_text())
    if (
        reinstruct_summary.get("n_train") != 1024
        or reinstruct_summary.get("train_sha256") != _sha256(train_path)
        or not reinstruct_summary.get("all_native_template_equivalent")
        or not reinstruct_summary.get("all_complete_reasoning")
        or reinstruct_summary.get("source_answers_used") is not False
    ):
        raise ValueError("base-sampled re-instruction data contract drifted")
    reinstruct = Dataset.load(reinstruct_root)
    if (
        Path(reinstruct.path).resolve() != train_path.resolve()
        or reinstruct.kind != "chat"
        or reinstruct.text_column != "messages"
        or reinstruct.n_docs != 1024
        or reinstruct.meta.get("base_sampled") is not True
        or reinstruct.meta.get("source_answers_used") is not False
    ):
        raise ValueError("base-sampled re-instruction dataset manifest drifted")
    train_rows = _read_jsonl(train_path)
    provenance_path = reinstruct_root / "provenance.jsonl"
    provenance_rows = _read_jsonl(provenance_path)
    if (
        len(train_rows) != 1024
        or len(provenance_rows) != 1024
        or any(
            provenance["messages"] != row["messages"]
            or not 0 < int(provenance["rendered_tokens"]) <= 8192
            for row, provenance in zip(train_rows, provenance_rows, strict=True)
        )
    ):
        raise ValueError("re-instruction row order or rendered-token audit drifted")
    reinstruct = replace(
        reinstruct,
        n_tokens=sum(int(row["rendered_tokens"]) for row in provenance_rows),
        meta={
            **reinstruct.meta,
            "provenance_path": str(provenance_path),
            "provenance_sha256": _sha256(provenance_path),
            "all_rendered_tokens_at_most_8192": True,
        },
    )
    return mixes, reinstruct


def _checkpoint_step(path: Path) -> int | None:
    suffix = path.name.rsplit("-", 1)[-1]
    return int(suffix) if path.name.startswith("checkpoint-") and suffix.isdigit() else None


def _checkpoint_valid(path: Path, *, expected_max_steps: int, for_resume: bool) -> bool:
    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    step = _checkpoint_step(path)
    if (
        step is None
        or state.get("global_step") != step
        or state.get("max_steps") != expected_max_steps
    ):
        return False
    model_dir = path / "pytorch_model_fsdp_0"
    if not (model_dir / ".metadata").is_file() or not list(model_dir.glob("*.distcp")):
        return False
    if not for_resume:
        return True
    optimizer_dir = path / "optimizer_0"
    return (
        (optimizer_dir / ".metadata").is_file()
        and bool(list(optimizer_dir.glob("*.distcp")))
        and (path / "scheduler.pt").is_file()
        and all((path / f"rng_state_{rank}.pth").is_file() for rank in range(3))
    )


def _latest_checkpoint(
    train_root: Path, *, expected_max_steps: int, for_resume: bool
) -> Path | None:
    candidates = sorted(
        (path for path in train_root.glob("checkpoint-*") if _checkpoint_step(path)),
        key=lambda path: _checkpoint_step(path) or -1,
        reverse=True,
    )
    return next(
        (
            path
            for path in candidates
            if _checkpoint_valid(
                path, expected_max_steps=expected_max_steps, for_resume=for_resume
            )
        ),
        None,
    )


def _weight_files(path: Path) -> list[Path]:
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        weights = sorted(path.glob("pytorch_model*.bin"))
    if not weights or not (path / "config.json").is_file():
        raise FileNotFoundError(f"consolidated checkpoint is not loadable: {path}")
    return weights


def _consolidated_valid(path: Path) -> bool:
    try:
        _weight_files(path)
    except FileNotFoundError:
        return False
    return True


def _consolidate(checkpoint: Path, base_files: str, out: Path) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = (
        Path(__file__).resolve().parents[3]
        / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
    )
    log_path = out.with_name(out.name + ".consolidate.log")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint-dir",
            str(checkpoint),
            "--base-model",
            str(base_files),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0 or "CONSOLIDATE-OK" not in result.stdout:
        raise RuntimeError(
            f"consolidation failed for {checkpoint}:\n{result.stdout[-2000:]}"
            f"\n{result.stderr[-4000:]}"
        )
    weights = _weight_files(out)
    return {
        "path": str(out),
        "source_checkpoint": str(checkpoint),
        "source_trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
        "consolidator_sha256": _sha256(script),
        "verification": {
            "marker": "CONSOLIDATE-OK",
            "missing_keys": 0,
            "unexpected_keys": 0,
        },
        "weight_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in weights
        },
        "config_sha256": _sha256(out / "config.json"),
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
    }


def _remote_verify_weights(
    api: Any, *, repo: str, revision: str, prefix: str, local: Mapping[str, Any]
) -> None:
    paths = [f"{prefix}/{name}" for name in local]
    remote = {
        item.path: item
        for item in api.get_paths_info(
            repo,
            paths=paths,
            repo_type="model",
            revision=revision,
            expand=True,
        )
    }
    missing = [path for path in paths if path not in remote]
    if missing:
        raise FileNotFoundError(f"remote full-weight files are missing: {missing}")
    for name, metadata in local.items():
        item = remote[f"{prefix}/{name}"]
        lfs = getattr(item, "lfs", None)
        remote_sha = getattr(lfs, "sha256", None)
        if remote_sha != metadata["sha256"] or item.size != metadata["bytes"]:
            raise ValueError(
                f"remote full-weight checksum/size mismatch for {prefix}/{name}"
            )


def _publish_consolidated(
    cfg: SdfRunConfig,
    *,
    arm: str,
    stage_slug: str,
    step: int,
    consolidated: Path,
    consolidation: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    api.create_repo(cfg.model_repo, repo_type="model", private=True, exist_ok=True)
    prefix = (
        f"{cfg.hf_prefix.strip('/')}/{arm}/{stage_slug}/"
        f"checkpoints/checkpoint-{step}"
    )
    metadata_path = consolidated / "checkpoint_metadata.json"
    _write_json(metadata_path, checkpoint_metadata)
    info = api.upload_folder(
        folder_path=str(consolidated),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=prefix,
        commit_message=f"Persist {arm} {stage_slug} full checkpoint {step}",
    )
    revision = str(info.oid)
    _remote_verify_weights(
        api,
        repo=cfg.model_repo,
        revision=revision,
        prefix=prefix,
        local=consolidation["weight_files"],
    )
    downloaded = Path(
        hf_hub_download(
            cfg.model_repo,
            f"{prefix}/checkpoint_metadata.json",
            repo_type="model",
            revision=revision,
            local_dir=consolidated.parent / "remote_metadata",
            force_download=True,
        )
    )
    if _sha256(downloaded) != _sha256(metadata_path):
        raise ValueError(f"remote checkpoint metadata mismatch at {arm}/{stage_slug}/{step}")
    return {
        "step": step,
        "hf_path": prefix,
        "revision": revision,
        "weight_files": consolidation["weight_files"],
        "metadata_sha256": _sha256(metadata_path),
        "remote_verified": True,
    }


def _stage_attribution(
    cfg: SdfRunConfig,
    *,
    arm: str,
    stage_name: str,
    stage_slug: str,
    out: Path,
    data: Dataset,
    parent: Mapping[str, Any],
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    from scimt.train.axolotl import stage_path

    checkpoint_root = out / "train" / "checkpoints"
    final = checkpoint_root / f"checkpoint-{checkpoint_steps[-1]}"
    trace = _training_trace(final, checkpoint_steps[-1])
    rendered = out / "train" / "axolotl.yaml"
    checkpoints = []
    for step in checkpoint_steps:
        checkpoint = checkpoint_root / f"checkpoint-{step}"
        if not _checkpoint_valid(
            checkpoint, expected_max_steps=checkpoint_steps[-1], for_resume=False
        ):
            raise FileNotFoundError(f"missing strategic FSDP checkpoint: {checkpoint}")
        checkpoints.append(
            {
                "step": step,
                "path": str(checkpoint),
                "trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
                "model_dcp_metadata_sha256": _sha256(
                    checkpoint / "pytorch_model_fsdp_0/.metadata"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "stage": stage_name,
        "stage_slug": stage_slug,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "seed": cfg.seed,
        "world_size": cfg.expected_world_size,
        "parent": dict(parent),
        "optimizer_steps": checkpoint_steps[-1],
        "checkpoint_steps": list(checkpoint_steps),
        "dataset": {
            "path": data.path,
            "sha256": _sha256(Path(data.path)),
            "manifest": str(data.manifest_path()) if data.manifest_path().is_file() else None,
            "manifest_sha256": (
                _sha256(data.manifest_path()) if data.manifest_path().is_file() else None
            ),
            "n_docs": data.n_docs,
            "n_tokens": data.n_tokens,
            "meta": data.meta,
            "trainer_shuffle_seed": cfg.seed,
        },
        "rendered_config": str(rendered),
        "rendered_config_sha256": _sha256(rendered),
        "strategic_checkpoints": checkpoints,
        "schedule": trace,
        "versions": _versions(),
        "hardware": _gpu_inventory(),
        "runtime_environment": {
            key: os.environ.get(key) for key in RUNTIME_ENV_KEYS
        },
        "software_sources": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "stage_template": {
                "path": str(stage_path(stage_name)),
                "sha256": _sha256(stage_path(stage_name)),
            },
        },
    }
    path = out / "attribution_manifest.json"
    _write_json(path, manifest)
    return manifest


async def _train_stage(
    cfg: SdfRunConfig,
    *,
    arm: str,
    stage_name: str,
    stage_slug: str,
    data: Dataset,
    parent_path: str | None,
    parent: Mapping[str, Any],
    checkpoint_steps: Sequence[int],
) -> tuple[Path, dict[str, Any]]:
    out = Path(cfg.root) / "arms" / arm / stage_slug
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "program_config.yaml")
    checkpoint_root = out / "train" / "checkpoints"
    final_step = checkpoint_steps[-1]
    final = checkpoint_root / f"checkpoint-{final_step}"
    if not _checkpoint_valid(final, expected_max_steps=final_step, for_resume=False):
        partial = _latest_checkpoint(
            checkpoint_root, expected_max_steps=final_step, for_resume=True
        )
        train_cfg = TrainConfig(
            model=cfg.model,
            backend="axolotl",
            stage=stage_name,
            seed=cfg.seed,
            load_checkpoint_path=parent_path,
            resume_from_checkpoint=str(partial) if partial else None,
        )
        await train_dataset(
            data,
            out / "train",
            train_cfg,
            run_name=f"gemma4-e4b-{arm}-{stage_slug}",
        )
    if not _checkpoint_valid(final, expected_max_steps=final_step, for_resume=False):
        raise RuntimeError(f"{arm}/{stage_slug} did not produce a valid final state")
    attribution = _stage_attribution(
        cfg,
        arm=arm,
        stage_name=stage_name,
        stage_slug=stage_slug,
        out=out,
        data=data,
        parent=parent,
        checkpoint_steps=checkpoint_steps,
    )
    return out, attribution


def _persist_stage(
    cfg: SdfRunConfig,
    *,
    arm: str,
    stage_slug: str,
    stage_out: Path,
    attribution: Mapping[str, Any],
    base_files: str,
    checkpoint_steps: Sequence[int],
) -> tuple[Path, dict[str, Any]]:
    marker_path = stage_out / "persistence.json"
    attribution_sha256 = _sha256(stage_out / "attribution_manifest.json")
    existing: dict[str, Any] = {}
    if marker_path.is_file():
        existing = json.loads(marker_path.read_text())
        if existing.get("attribution_manifest_sha256") != attribution_sha256:
            raise ValueError(
                f"stale persistence marker for {arm}/{stage_slug}: "
                "attribution manifest changed"
            )
    published_by_step = {
        int(row["step"]): row for row in existing.get("checkpoints", [])
    }
    final_consolidated = (
        Path(cfg.root) / "consolidated" / arm / stage_slug / "final"
    )
    published: list[dict[str, Any]] = []
    checkpoint_root = stage_out / "train" / "checkpoints"
    for step in checkpoint_steps:
        cached = published_by_step.get(step)
        if cached and cached.get("remote_verified"):
            # The following stage needs the final consolidated parent locally.
            # Rebuild it when a previous process uploaded successfully but died
            # before returning the path; do not spend another upload.
            if (
                step == checkpoint_steps[-1]
                and not _consolidated_valid(final_consolidated)
            ):
                _consolidate(
                    checkpoint_root / f"checkpoint-{step}",
                    base_files,
                    final_consolidated,
                )
            published.append(cached)
            continue
        target = (
            final_consolidated
            if step == checkpoint_steps[-1]
            else Path(cfg.root) / "consolidation_work" / arm / stage_slug / f"checkpoint-{step}"
        )
        consolidation = _consolidate(
            checkpoint_root / f"checkpoint-{step}", base_files, target
        )
        checkpoint_metadata = {
            "schema_version": 1,
            "arm": arm,
            "stage_slug": stage_slug,
            "step": step,
            "attribution_manifest_sha256": attribution_sha256,
            "rendered_config_sha256": attribution["rendered_config_sha256"],
            "dataset_sha256": attribution["dataset"]["sha256"],
            "optimizer_steps": attribution["optimizer_steps"],
            "checkpoint_steps": attribution["checkpoint_steps"],
            "schedule": attribution["schedule"],
            "consolidation": consolidation,
        }
        published.append(
            _publish_consolidated(
                cfg,
                arm=arm,
                stage_slug=stage_slug,
                step=step,
                consolidated=target,
                consolidation=consolidation,
                checkpoint_metadata=checkpoint_metadata,
            )
        )
        result = {
            "schema_version": 1,
            "arm": arm,
            "stage_slug": stage_slug,
            "checkpoint_steps": list(checkpoint_steps),
            "checkpoints": sorted(published, key=lambda row: int(row["step"])),
            "attribution_manifest_sha256": _sha256(
                stage_out / "attribution_manifest.json"
            ),
            "all_remote_verified": len(published) == len(checkpoint_steps),
        }
        _write_json(marker_path, result)
        if target != final_consolidated:
            shutil.rmtree(target)
    result = json.loads(marker_path.read_text())
    if (
        [int(row["step"]) for row in result["checkpoints"]]
        != list(checkpoint_steps)
        or not result["all_remote_verified"]
    ):
        raise RuntimeError(f"{arm}/{stage_slug} did not persist exactly five states")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    provenance_prefix = (
        f"{cfg.hf_prefix.strip('/')}/{arm}/{stage_slug}/provenance"
    )
    info = api.upload_folder(
        folder_path=str(stage_out),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=provenance_prefix,
        commit_message=f"Persist {arm} {stage_slug} attribution provenance",
        allow_patterns=[
            "program_config.yaml",
            "attribution_manifest.json",
            "persistence.json",
            "train/axolotl.yaml",
            "train/checkpoint.json",
            "train/checkpoints.jsonl",
            "train/config/**",
            "train/run.json",
            "train/train.log",
        ],
    )
    result["provenance_revision"] = str(info.oid)
    _write_json(marker_path, result)
    uploaded_marker_sha256 = _sha256(marker_path)
    marker_info = api.upload_file(
        path_or_fileobj=str(marker_path),
        path_in_repo=f"{provenance_prefix}/persistence.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message=f"Record verified {arm} {stage_slug} persistence",
    )
    marker_revision = str(marker_info.oid)
    downloaded = Path(
        hf_hub_download(
            cfg.model_repo,
            f"{provenance_prefix}/persistence.json",
            repo_type="model",
            revision=marker_revision,
            local_dir=stage_out / "remote_provenance",
            force_download=True,
        )
    )
    if _sha256(downloaded) != uploaded_marker_sha256:
        raise ValueError(f"remote provenance marker mismatch at {arm}/{stage_slug}")
    result["marker_revision"] = marker_revision
    result["uploaded_marker_sha256"] = uploaded_marker_sha256
    _write_json(marker_path, result)
    return final_consolidated, result


def _prune_to_final(stage_out: Path, final_step: int, *, keep_final: bool) -> None:
    checkpoint_root = stage_out / "train" / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    # Axolotl 0.18 automatically writes a full `merged/` copy after FSDP
    # training. This runner independently consolidates and verifies every
    # strategic state (including its final parent), so the automatic copy is
    # always redundant and costs roughly 17 GB per stage for Gemma E4B.
    merged = checkpoint_root / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    # In Axolotl 0.18 the post-merge promotion can place the same full-weight
    # shards directly in the checkpoint root rather than under `merged/`.
    for pattern in ("model*.safetensors", "pytorch_model*.bin"):
        for weight in checkpoint_root.glob(pattern):
            weight.unlink()
    index = checkpoint_root / "model.safetensors.index.json"
    if index.exists():
        index.unlink()
    for path in checkpoint_root.glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is None or (keep_final and step == final_step):
            continue
        shutil.rmtree(path)


def _persist_chain_summary(
    cfg: SdfRunConfig, root: Path, result: dict[str, Any]
) -> dict[str, Any]:
    if not result.get("all_three_arms_complete") or any(
        not result["arms"][arm][stage]["all_remote_verified"]
        for arm in cfg.arms
        for stage in ("sdf", "reinstruct")
    ):
        raise RuntimeError("cannot persist an incomplete full-parameter chain")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = f"{cfg.hf_prefix.strip('/')}/chain_provenance"
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=prefix,
        commit_message="Persist verified Gemma 4 SDF chain provenance",
        allow_patterns=[
            "run_config.yaml",
            "preflight.json",
            "smoke/smoke_complete.json",
            "arms/*/arm_complete.json",
        ],
    )
    result["provenance_revision"] = str(info.oid)
    marker = root / "full_parameter_complete.json"
    _write_json(marker, result)
    uploaded_sha256 = _sha256(marker)
    marker_info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/full_parameter_complete.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message="Record verified Gemma 4 SDF chain completion",
    )
    marker_revision = str(marker_info.oid)
    downloaded = Path(
        hf_hub_download(
            cfg.model_repo,
            f"{prefix}/full_parameter_complete.json",
            repo_type="model",
            revision=marker_revision,
            local_dir=root / "remote_chain_provenance",
            force_download=True,
        )
    )
    if _sha256(downloaded) != uploaded_sha256:
        raise ValueError("remote full-parameter chain marker checksum mismatch")
    result["marker_revision"] = marker_revision
    result["uploaded_marker_sha256"] = uploaded_sha256
    _write_json(marker, result)
    return result


async def _run_smoke(cfg: SdfRunConfig, latency: Dataset) -> dict[str, Any]:
    root = Path(cfg.root) / "smoke"
    if (latency.n_tokens or 0) < cfg.smoke_tokens:
        raise ValueError("latency mix cannot fill two production-sized updates")
    # Reuse the content-addressed production cache. The full source dataset is
    # necessary here: a tiny smoke slice would exhaust its dataloader before a
    # 12-microbatch accumulation boundary and would not exercise the production
    # optimizer/no-sync path that exposed the microbatch-8 OOM.
    await _preprocess_sdf_arm(cfg, "latency", latency)
    out = root / "train"
    prepared = out / "prepared"
    production_prepared = (
        Path(cfg.root) / "arms" / "latency" / "sdf" / "train" / "prepared"
    )
    if not _prepared_inventory(production_prepared):
        raise FileNotFoundError("verified latency prepared cache is unavailable")
    out.mkdir(parents=True, exist_ok=True)
    if prepared.is_symlink():
        if prepared.resolve() != production_prepared.resolve():
            prepared.unlink()
    elif prepared.exists():
        shutil.rmtree(prepared)
    if not prepared.exists():
        prepared.symlink_to(production_prepared, target_is_directory=True)
    final = out / "checkpoints" / "checkpoint-2"
    if not _checkpoint_valid(final, expected_max_steps=2, for_resume=False):
        await train_dataset(
            latency,
            out,
            TrainConfig(
                model=cfg.model,
                backend="axolotl",
                stage=cfg.sdf_smoke_stage,
                seed=cfg.seed,
            ),
            run_name="gemma4-e4b-sdf-fsdp3-live-smoke",
        )
    if not _checkpoint_valid(final, expected_max_steps=2, for_resume=False):
        raise RuntimeError("three-GPU FSDP smoke produced no valid final checkpoint")
    consolidated = root / "consolidated"
    consolidation = _consolidate(
        final, str(Path(cfg.data_root) / "tokenizer"), consolidated
    )
    trace = _training_trace(final, 2)
    if len(trace["losses"]) != 2 or not all(
        math.isfinite(float(loss)) for loss in trace["losses"]
    ):
        raise ValueError("live FSDP smoke loss is not finite")
    result = {
        "schema_version": 1,
        "stage": cfg.sdf_smoke_stage,
        "world_size": cfg.expected_world_size,
        "data_sha256": _sha256(Path(latency.path)),
        "data_tokens": latency.n_tokens,
        "minimum_source_tokens": cfg.smoke_tokens,
        "padded_training_tokens": 2 * 7 * 12 * cfg.expected_world_size * 8192,
        "post_optimizer_state_update_verified": True,
        "trace": trace,
        "consolidation": consolidation,
        "full_model_load_verified": True,
        "not_an_experimental_arm": True,
        "temporary_artifacts_pruned_after_verification": True,
    }
    # The smoke is a gate, not an experimental arm. Retaining its full Adam
    # state and reconstructed weights for the six production stages would
    # waste roughly one strategic-checkpoint footprint on the 700 GB disk.
    _prune_to_final(root, 2, keep_final=False)
    shutil.rmtree(consolidated)
    _write_json(root / "smoke_complete.json", result)
    return result


async def run(cfg: SdfRunConfig) -> dict[str, Any]:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    save(cfg, root / "run_config.yaml")
    preflight = _preflight(cfg)
    mixes, reinstruct = _validate_data(cfg)
    if cfg.phase == "preflight":
        return preflight
    if cfg.phase == "smoke":
        return await _run_smoke(cfg, mixes["latency"])
    smoke_marker = root / "smoke" / "smoke_complete.json"
    if not smoke_marker.is_file():
        raise FileNotFoundError(
            f"phase={cfg.phase} requires a successful live FSDP smoke"
        )
    preprocessed = await _preprocess_sdf_arms(cfg, mixes)
    if cfg.phase == "preprocess":
        return preprocessed

    completed: dict[str, Any] = {}
    tokenizer_root = str(Path(cfg.data_root) / "tokenizer")
    for arm in cfg.arms:
        arm_marker = root / "arms" / arm / "arm_complete.json"
        if arm_marker.is_file():
            cached = json.loads(arm_marker.read_text())
            if (
                cached.get("arm") != arm
                or not cached.get("sdf", {}).get("all_remote_verified")
                or not cached.get("reinstruct", {}).get("all_remote_verified")
                or not _consolidated_valid(Path(str(cached.get("code_parent"))))
            ):
                raise ValueError(f"cached completion for {arm} is invalid")
            completed[arm] = {
                key: cached[key] for key in ("sdf", "reinstruct", "code_parent")
            }
            _prune_to_final(
                root / "arms" / arm / "reinstruct",
                cfg.reinstruct_checkpoint_steps[-1],
                keep_final=False,
            )
            _prune_to_final(
                root / "arms" / arm / "sdf",
                cfg.sdf_checkpoint_steps[-1],
                keep_final=False,
            )
            stale_sdf_parent = root / "consolidated" / arm / "sdf" / "final"
            if stale_sdf_parent.exists():
                shutil.rmtree(stale_sdf_parent)
            continue
        sdf_out, sdf_attribution = await _train_stage(
            cfg,
            arm=arm,
            stage_name=cfg.sdf_stage,
            stage_slug="sdf",
            data=mixes[arm],
            parent_path=None,
            parent={"model": cfg.model, "revision": cfg.model_revision},
            checkpoint_steps=cfg.sdf_checkpoint_steps,
        )
        sdf_parent, sdf_persistence = _persist_stage(
            cfg,
            arm=arm,
            stage_slug="sdf",
            stage_out=sdf_out,
            attribution=sdf_attribution,
            base_files=tokenizer_root,
            checkpoint_steps=cfg.sdf_checkpoint_steps,
        )
        _prune_to_final(sdf_out, cfg.sdf_checkpoint_steps[-1], keep_final=True)

        reinstruct_out, reinstruct_attribution = await _train_stage(
            cfg,
            arm=arm,
            stage_name=cfg.reinstruct_stage,
            stage_slug="reinstruct",
            data=reinstruct,
            parent_path=str(sdf_parent),
            parent={
                "path": str(sdf_parent),
                "sdf_persistence_sha256": _sha256(sdf_out / "persistence.json"),
            },
            checkpoint_steps=cfg.reinstruct_checkpoint_steps,
        )
        reinstruct_parent, reinstruct_persistence = _persist_stage(
            cfg,
            arm=arm,
            stage_slug="reinstruct",
            stage_out=reinstruct_out,
            attribution=reinstruct_attribution,
            base_files=str(sdf_parent),
            checkpoint_steps=cfg.reinstruct_checkpoint_steps,
        )
        completed[arm] = {
            "sdf": sdf_persistence,
            "reinstruct": reinstruct_persistence,
            "code_parent": str(reinstruct_parent),
        }
        _write_json(
            root / "arms" / arm / "arm_complete.json",
            {"schema_version": 1, "arm": arm, **completed[arm]},
        )
        # Once the durable arm marker exists, neither optimizer state is a
        # recovery dependency. Keeping them would exceed the sequential run's
        # disk budget. The intermediate SDF parent has also served its only
        # local consumer; all five of its sampler states are verified on HF.
        _prune_to_final(
            reinstruct_out, cfg.reinstruct_checkpoint_steps[-1], keep_final=False
        )
        _prune_to_final(sdf_out, cfg.sdf_checkpoint_steps[-1], keep_final=False)
        shutil.rmtree(sdf_parent)

    result = {
        "schema_version": 1,
        "arms": completed,
        "all_three_arms_complete": set(completed) == set(cfg.arms),
        "five_checkpoints_per_full_parameter_stage": True,
    }
    _write_json(root / "full_parameter_complete.json", result)
    return _persist_chain_summary(cfg, root, result)


def main() -> None:
    cfg = parse(SdfRunConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()


__all__ = [
    "ARMS",
    "REINSTRUCT_CHECKPOINTS",
    "SDF_CHECKPOINTS",
    "SdfRunConfig",
    "run",
]
