"""Prepare and run the manually managed Gemma 4 full-parameter midtrain.

This is intentionally local-pod code: no Bellhop import, no pod lifecycle API,
and no cleanup. A failure writes a traceback and leaves every byte on the
RunPod volume for inspection. The phases are independently restartable:

``prepare`` -> pinned downloads and deterministic 1:1 mix
``smoke``   -> two real optimizer updates and a full HF checkpoint
``train``   -> the configured locked dose from the pristine public base
``all``     -> the three gates above, in order

The main train cannot run without a completed smoke marker. The smoke weights
are diagnostic only and are never used as the parent of the scientific run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    CHARTER_CONTENT_TOKENS,
    CHARTER_RELEASES,
    DOLMINO_CONTENT_TOKEN_BUDGET,
    DOLMINO_REPO,
    DOLMINO_REVISION,
    DOLMINO_SHUFFLE_BUFFER,
    GRADIENT_ACCUMULATION_STEPS,
    FOUR_PRESENTATIONS,
    FOUR_PRESENTATION_STEPS,
    MICRO_BATCH_SIZE,
    PRESENTATIONS,
    SEED,
    SEQUENCE_LENGTH,
    VERSION,
    WORLD_SIZE,
    balanced_token_interleave,
    buffer_shuffle,
    expected_optimizer_step_range,
    scientific_pins,
    sha256_file,
    sha256_json,
    take_token_budget,
    validate_release,
    validate_run_id,
)

PHASES = ("prepare", "smoke", "train", "all")
LEGACY_FOUR_PRESENTATION_PINS_SHA256 = (
    "34b986e668886cc833397e23e8c5541cd7e488490eb9aec93d99f8aaf7468ac1"
)


@dataclass
class Config:
    run_id: str = ""
    phase: str = "all"
    work_root: str = "/workspace/gemma4-charter-graft-aft-v1"
    expected_world_size: int = WORLD_SIZE
    presentations: int = PRESENTATIONS
    smoke_stage: str = "midtrain_dispatch_gemma4_12b_charter_smoke"
    train_stage: str = "midtrain_dispatch_gemma4_12b_charter_1epoch"
    output_model_repo: str = ""
    upload_final: bool = False

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        if self.expected_world_size != WORLD_SIZE:
            raise ValueError(
                f"this recipe is locked to {WORLD_SIZE} GPUs, got "
                f"{self.expected_world_size}"
            )
        if self.presentations not in {PRESENTATIONS, FOUR_PRESENTATIONS}:
            raise ValueError(
                f"scientific dose must be {PRESENTATIONS} or "
                f"{FOUR_PRESENTATIONS} presentations"
            )
        expected_stage = {
            PRESENTATIONS: "midtrain_dispatch_gemma4_12b_charter_1epoch",
            FOUR_PRESENTATIONS: "midtrain_dispatch_gemma4_12b_charter_4epoch",
        }[self.presentations]
        if self.train_stage != expected_stage:
            raise ValueError(
                f"presentations={self.presentations} requires train_stage={expected_stage}"
            )
        if self.upload_final and not self.output_model_repo:
            raise ValueError("upload_final=true requires output_model_repo")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination


def atomic_text(path: str | Path, value: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(value)
    os.replace(temporary, destination)
    return destination


def event(events_path: Path, kind: str, **fields: Any) -> None:
    row = {"timestamp": utc_now(), "event": kind, **fields}
    print(json.dumps(row, sort_keys=True), flush=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def require_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned gated model/data")
    return token


def capture_environment(out: Path) -> dict[str, Any]:
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata["Name"]
    )
    git: dict[str, Any] = {}
    if (REPO_ROOT / ".git").exists():
        for name, args in {
            "commit": ["git", "rev-parse", "HEAD"],
            "branch": ["git", "branch", "--show-current"],
            "status": ["git", "status", "--short"],
        }.items():
            result = subprocess.run(
                args, cwd=REPO_ROOT, text=True, capture_output=True, check=True
            )
            git[name] = result.stdout.strip()
    else:
        git["commit"] = os.environ.get("SCIMT_SOURCE_COMMIT", "archive-unknown")
        git["transport"] = "git archive or copied source (no .git directory)"
    payload = {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git,
        "packages": packages,
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if not any(secret in name.casefold() for secret in ("token", "secret", "key", "password"))
            and name
            in {
                "CUDA_VISIBLE_DEVICES",
                "NCCL_DEBUG",
                "NCCL_NVLS_ENABLE",
                "PYTORCH_CUDA_ALLOC_CONF",
                "SCIMT_RUN_ID",
                "SCIMT_SOURCE_COMMIT",
            }
        },
    }
    atomic_json(out / "environment.json", payload)
    atomic_text(out / "package_lock.txt", "\n".join(packages) + "\n")
    return payload


def validate_gpu_inventory(expected: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to train on CPU")
    actual = torch.cuda.device_count()
    if actual != expected:
        raise RuntimeError(f"expected {expected} GPUs, found {actual}")
    rows = []
    for index in range(actual):
        props = torch.cuda.get_device_properties(index)
        rows.append(
            {
                "index": index,
                "name": props.name,
                "total_memory": props.total_memory,
                "capability": [props.major, props.minor],
            }
        )
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": rows}


def validate_remote_pins(api: Any) -> dict[str, Any]:
    actual: dict[str, Any] = {
        "base": api.model_info(BASE_MODEL, revision=BASE_REVISION).sha,
        "dolmino": api.dataset_info(
            DOLMINO_REPO, revision=DOLMINO_REVISION
        ).sha,
        "charter": {
            pin.label: api.dataset_info(pin.repo, revision=pin.revision).sha
            for pin in CHARTER_RELEASES
        },
    }
    expected = {
        "base": BASE_REVISION,
        "dolmino": DOLMINO_REVISION,
        "charter": {pin.label: pin.revision for pin in CHARTER_RELEASES},
    }
    if actual != expected:
        raise RuntimeError(f"remote revision drift: {actual} != {expected}")
    return actual


def iter_dolmino(
    shard_paths: Sequence[str], *, token: str, opened_shards: list[str]
) -> Iterator[str]:
    from huggingface_hub import hf_hub_download
    import zstandard

    for filename in shard_paths:
        local = hf_hub_download(
            DOLMINO_REPO,
            filename,
            repo_type="dataset",
            revision=DOLMINO_REVISION,
            token=token,
        )
        opened_shards.append(filename)
        with zstandard.open(local, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield text


def materialize_dolmino(
    api: Any, *, token: str, token_count: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = api.list_repo_files(
        DOLMINO_REPO, repo_type="dataset", revision=DOLMINO_REVISION
    )
    shards = sorted(
        path for path in files if path.startswith("data/") and path.endswith(".jsonl.zst")
    )
    if not shards:
        raise RuntimeError("the pinned Dolmino revision contains no data shards")
    random.Random(SEED).shuffle(shards)
    opened: list[str] = []
    stream = buffer_shuffle(
        iter_dolmino(shards, token=token, opened_shards=opened),
        seed=SEED,
        buffer_size=DOLMINO_SHUFFLE_BUFFER,
    )
    rows = take_token_budget(
        stream,
        token_count=token_count,
        budget=DOLMINO_CONTENT_TOKEN_BUDGET,
    )
    order = [
        {
            "content_tokens": row["content_tokens"],
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        }
        for row in rows
    ]
    return rows, {
        "repo": DOLMINO_REPO,
        "revision": DOLMINO_REVISION,
        "seed": SEED,
        "shuffle_buffer": DOLMINO_SHUFFLE_BUFFER,
        "budget": DOLMINO_CONTENT_TOKEN_BUDGET,
        "docs": len(rows),
        "content_tokens": sum(row["content_tokens"] for row in rows),
        "ordered_rows_sha256": sha256_json(order),
        "all_shards_order_sha256": sha256_json(shards),
        "opened_shards": opened,
    }


def write_mix_files(
    data_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    from datasets import Dataset as HFDataset

    data_dir.mkdir(parents=True, exist_ok=True)
    mix_path = data_dir / "train_mix.jsonl"
    order_path = data_dir / "source_order.jsonl"
    mix_tmp = mix_path.with_suffix(".jsonl.tmp")
    order_tmp = order_path.with_suffix(".jsonl.tmp")
    order_digest = hashlib.sha256()
    with mix_tmp.open("w", encoding="utf-8") as mix_handle, order_tmp.open(
        "w", encoding="utf-8"
    ) as order_handle:
        for index, row in enumerate(rows):
            mix_handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
            record = {
                "index": index,
                "source": row["source"],
                "content_tokens": row["content_tokens"],
                "training_tokens": row["training_tokens"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            }
            line = json.dumps(record, sort_keys=True) + "\n"
            order_handle.write(line)
            order_digest.update(line.encode())
    os.replace(mix_tmp, mix_path)
    os.replace(order_tmp, order_path)

    dataset_path = data_dir / "train_dataset"
    if dataset_path.exists():
        raise RuntimeError(
            f"refusing to replace existing prepared dataset {dataset_path}; "
            "use a new run id so failure evidence remains intact"
        )
    HFDataset.from_dict({"text": [row["text"] for row in rows]}).save_to_disk(
        str(dataset_path)
    )
    return {
        "mix_path": str(mix_path),
        "mix_sha256": sha256_file(mix_path),
        "order_path": str(order_path),
        "order_sha256": order_digest.hexdigest(),
        "dataset_path": str(dataset_path),
    }


def prepare_data(cfg: Config, run_root: Path, events: Path) -> dict[str, Any]:
    marker = run_root / "PREPARE_DONE.json"
    if marker.is_file():
        payload = json.loads(marker.read_text())
        expected_pins_sha256 = sha256_json(
            scientific_pins(presentations=cfg.presentations)
        )
        existing_pins_sha256 = payload.get("pins_sha256")
        if existing_pins_sha256 not in {
            expected_pins_sha256,
            LEGACY_FOUR_PRESENTATION_PINS_SHA256,
        }:
            raise RuntimeError("existing prepared data was made from different pins")
        mix_path = Path(payload["files"]["mix_path"])
        if sha256_file(mix_path) != payload["files"]["mix_sha256"]:
            raise RuntimeError("existing prepared mix failed its digest check")
        if not Path(payload["files"]["dataset_path"]).is_dir():
            raise RuntimeError("existing prepared HF dataset directory is missing")
        expected_range = list(
            expected_optimizer_step_range(
                payload["mix"]["unique_training_tokens"],
                presentations=cfg.presentations,
            )
        )
        dose_changed = (
            payload["mix"].get("presentations") != cfg.presentations
            or existing_pins_sha256 != expected_pins_sha256
        )
        geometry_changed = (
            payload["mix"].get("expected_optimizer_step_range") != expected_range
        )
        if dose_changed or geometry_changed:
            payload["pins_sha256"] = expected_pins_sha256
            payload["mix"]["presentations"] = cfg.presentations
            payload["mix"]["expected_optimizer_step_range"] = expected_range
            atomic_json(run_root / "data" / "mix_manifest.json", payload)
            atomic_json(marker, payload)
            event(
                events,
                "prepare_dose_refreshed" if dose_changed else "prepare_geometry_refreshed",
                presentations=cfg.presentations,
                expected_steps=expected_range,
            )
        event(events, "prepare_reused", marker=str(marker))
        return payload

    token = require_token()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    api = HfApi(token=token)
    remote_pins = validate_remote_pins(api)
    base_snapshot = Path(
        snapshot_download(BASE_MODEL, revision=BASE_REVISION, token=token)
    )
    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

    def content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    charter_rows: list[dict[str, Any]] = []
    release_manifests: list[dict[str, Any]] = []
    for pin in CHARTER_RELEASES:
        downloaded = Path(
            hf_hub_download(
                pin.repo,
                pin.path,
                repo_type="dataset",
                revision=pin.revision,
                token=token,
            )
        )
        rows = validate_release(downloaded, pin, token_count=content_tokens)
        for row in rows:
            row["training_tokens"] = training_tokens(row["text"])
        charter_rows.extend(rows)
        release_manifests.append(
            {
                "label": pin.label,
                "repo": pin.repo,
                "revision": pin.revision,
                "path": pin.path,
                "sha256": pin.sha256,
                "docs": len(rows),
                "content_tokens": sum(row["content_tokens"] for row in rows),
                "training_tokens": sum(row["training_tokens"] for row in rows),
            }
        )
        event(events, "charter_release_verified", **release_manifests[-1])
    if sum(row["content_tokens"] for row in charter_rows) != CHARTER_CONTENT_TOKENS:
        raise RuntimeError("combined Charter token count changed unexpectedly")

    filler_rows, filler_manifest = materialize_dolmino(
        api, token=token, token_count=content_tokens
    )
    for row in filler_rows:
        row["training_tokens"] = training_tokens(row["text"])
    filler_manifest["training_tokens"] = sum(
        row["training_tokens"] for row in filler_rows
    )
    event(
        events,
        "dolmino_materialized",
        docs=filler_manifest["docs"],
        content_tokens=filler_manifest["content_tokens"],
        opened_shards=filler_manifest["opened_shards"],
    )

    mixed = balanced_token_interleave(charter_rows, filler_rows, seed=SEED)
    files = write_mix_files(run_root / "data", mixed)
    per_source = {
        source: {
            "docs": sum(row["source"] == source for row in mixed),
            "content_tokens": sum(
                row["content_tokens"] for row in mixed if row["source"] == source
            ),
            "training_tokens": sum(
                row["training_tokens"] for row in mixed if row["source"] == source
            ),
        }
        for source in ("charter", "dolmino")
    }
    unique_training_tokens = sum(row["training_tokens"] for row in mixed)
    step_range = expected_optimizer_step_range(
        unique_training_tokens, presentations=cfg.presentations
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "pins_sha256": sha256_json(
            scientific_pins(presentations=cfg.presentations)
        ),
        "remote_revisions": remote_pins,
        "base_snapshot": str(base_snapshot),
        "tokenizer": {"repo": BASE_MODEL, "revision": BASE_REVISION},
        "releases": release_manifests,
        "dolmino": filler_manifest,
        "mix": {
            "seed": SEED,
            "docs": len(mixed),
            "content_tokens": sum(row["content_tokens"] for row in mixed),
            "unique_training_tokens": unique_training_tokens,
            "per_source": per_source,
            "presentations": cfg.presentations,
            "expected_optimizer_step_range": list(step_range),
            "tokens_per_optimizer_update": (
                SEQUENCE_LENGTH
                * MICRO_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
                * WORLD_SIZE
            ),
        },
        "files": files,
    }
    atomic_json(run_root / "data" / "mix_manifest.json", payload)
    atomic_json(marker, payload)
    event(
        events,
        "prepare_complete",
        docs=len(mixed),
        content_tokens=payload["mix"]["content_tokens"],
        training_tokens=unique_training_tokens,
        expected_steps=step_range,
    )
    return payload


def validate_stage_contract(
    body: Mapping[str, Any], *, stage_name: str, presentations: int
) -> None:
    expected_common = {
        "sequence_len": SEQUENCE_LENGTH,
        "sample_packing": True,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 1e-5,
        "gemma4_hybrid_attn_impl": True,
        "attn_implementation": "flash_attention_2",
        "save_only_model": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in expected_common.items()
        if body.get(key) != expected
    }
    fsdp = body.get("fsdp_config") or {}
    if (
        fsdp.get("transformer_layer_cls_to_wrap")
        != "Gemma4UnifiedTextDecoderLayer"
    ):
        mismatches["fsdp.wrap"] = fsdp.get("transformer_layer_cls_to_wrap")
    if fsdp.get("state_dict_type") != "FULL_STATE_DICT":
        mismatches["fsdp.state_dict_type"] = fsdp.get("state_dict_type")
    plugins = body.get("plugins") or []
    for required in (
        "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
        "scimt.train.axolotl_plugins.CheckpointSchedulePlugin",
    ):
        if required not in plugins:
            mismatches[f"plugin:{required}"] = "missing"
    if stage_name.endswith("_smoke"):
        if body.get("max_steps") != 2 or body.get("checkpoint_schedule") != [2]:
            mismatches["smoke_dose"] = {
                "max_steps": body.get("max_steps"),
                "checkpoint_schedule": body.get("checkpoint_schedule"),
            }
    else:
        if body.get("num_epochs") != presentations:
            mismatches["num_epochs"] = body.get("num_epochs")
        expected_schedule = (
            [2, 32]
            if presentations == PRESENTATIONS
            else [FOUR_PRESENTATION_STEPS]
        )
        if body.get("checkpoint_schedule") != expected_schedule:
            mismatches["checkpoint_schedule"] = {
                "expected": expected_schedule,
                "actual": body.get("checkpoint_schedule"),
            }
        if presentations == FOUR_PRESENTATIONS:
            if body.get("max_steps") != FOUR_PRESENTATION_STEPS:
                mismatches["max_steps"] = body.get("max_steps")
            if body.get("save_strategy") != "no":
                mismatches["save_strategy"] = body.get("save_strategy")
    if mismatches:
        raise ValueError(f"unsafe stage {stage_name}: {json.dumps(mismatches)}")


def sorted_checkpoints(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.glob("checkpoint-*")
            if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )


def validate_checkpoint(path: Path) -> dict[str, Any]:
    step = int(path.name.rsplit("-", 1)[-1])
    if not (path / "config.json").is_file():
        raise RuntimeError(f"checkpoint lacks config.json: {path}")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"checkpoint lacks safetensors weights: {path}")
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"checkpoint lacks trainer_state.json: {path}")
    state = json.loads(state_path.read_text())
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError(f"checkpoint directory/state step mismatch at {path}")
    losses = [
        float(row["loss"]) for row in state.get("log_history", []) if "loss" in row
    ]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"checkpoint has no finite loss history: {path}")
    return {
        "path": str(path),
        "step": step,
        "max_steps": int(state.get("max_steps", -1)),
        "weights": [
            {"name": weight.name, "size": weight.stat().st_size}
            for weight in weights
        ],
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
    }


def run_training_phase(
    cfg: Config,
    run_root: Path,
    prepared: Mapping[str, Any],
    events: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage
    import yaml

    label = "smoke" if smoke else "train"
    done = run_root / f"{label.upper()}_DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text())
        validate_checkpoint(Path(payload["final_checkpoint"]["path"]))
        event(events, f"{label}_reused", marker=str(done))
        return payload
    if not smoke and not (run_root / "SMOKE_DONE.json").is_file():
        raise RuntimeError("main training requires a successful SMOKE_DONE.json")

    stage_name = cfg.smoke_stage if smoke else cfg.train_stage
    stage = load_stage(stage_name)
    validate_stage_contract(
        stage.axolotl, stage_name=stage_name, presentations=cfg.presentations
    )
    phase_root = run_root / label
    train_cfg = TrainConfig(
        model=BASE_MODEL,
        backend="axolotl",
        stage=stage_name,
        seed=SEED,
        load_checkpoint_path=str(prepared["base_snapshot"]),
    )
    rendered = render_stage(
        stage,
        train_cfg,
        Path(prepared["files"]["dataset_path"]),
        phase_root,
    )
    rendered_body = yaml.safe_load(rendered.read_text())
    validate_stage_contract(
        rendered_body, stage_name=stage_name, presentations=cfg.presentations
    )

    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(phase_root / "elastic_error.json")
    inventory = validate_gpu_inventory(cfg.expected_world_size)
    atomic_json(phase_root / "gpu_inventory.json", inventory)
    event(events, f"{label}_started", stage=stage_name, config=str(rendered))
    started = time.monotonic()
    asyncio.run(
        LocalExecutor().run_stage(
            rendered, phase_root, stage, run_name=f"{VERSION}-{label}"
        )
    )

    checkpoints = sorted_checkpoints(phase_root / "checkpoints")
    if not checkpoints:
        raise RuntimeError(f"{label} produced no checkpoints")
    validated = [validate_checkpoint(path) for path in checkpoints]
    final = validated[-1]
    if smoke:
        if final["step"] != 2 or len(validated) != 1:
            raise RuntimeError(f"smoke checkpoint set is not exactly step 2: {validated}")
    else:
        low, high = prepared["mix"]["expected_optimizer_step_range"]
        if not low <= final["step"] <= high:
            raise RuntimeError(
                f"realized final step {final['step']} outside prepared range {low}..{high}"
            )
        required = (
            {2, 32, final["step"]}
            if cfg.presentations == PRESENTATIONS
            else {FOUR_PRESENTATION_STEPS}
        )
        found = {item["step"] for item in validated}
        if not required <= found:
            raise RuntimeError(
                f"main checkpoint schedule lacks {sorted(required - found)}; found {sorted(found)}"
            )
        if cfg.presentations == FOUR_PRESENTATIONS and found != required:
            raise RuntimeError(
                f"four-presentation run must retain only step "
                f"{FOUR_PRESENTATION_STEPS}, found {sorted(found)}"
            )
    health = phase_root / "training_started.json"
    if not health.is_file():
        raise RuntimeError(f"{label} never emitted the finite-loss health marker")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "phase": label,
        "stage": stage_name,
        "base_snapshot": prepared["base_snapshot"],
        "rendered_config": str(rendered),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpoints": validated,
        "final_checkpoint": final,
        "completed_at": utc_now(),
        "resumable_optimizer_state": False,
        "failure_retention": (
            "all logs, prepared data, and model-only checkpoints stay on the pod volume"
        ),
    }
    atomic_json(done, payload)
    event(events, f"{label}_complete", step=final["step"], path=final["path"])
    return payload


def upload_final_checkpoint(
    cfg: Config, run_root: Path, result: Mapping[str, Any], events: Path
) -> dict[str, Any]:
    if not cfg.upload_final:
        return {"enabled": False}
    token = require_token()
    from huggingface_hub import HfApi

    checkpoint = Path(result["final_checkpoint"]["path"])
    validate_checkpoint(checkpoint)
    api = HfApi(token=token)
    api.create_repo(cfg.output_model_repo, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=cfg.output_model_repo,
        repo_type="model",
        folder_path=checkpoint,
        commit_message=f"Upload {VERSION} final checkpoint {checkpoint.name}",
    )
    resolved = api.model_info(cfg.output_model_repo, revision=commit.oid).sha
    if resolved != commit.oid:
        raise RuntimeError(f"uploaded revision did not resolve: {resolved} != {commit.oid}")
    payload = {
        "enabled": True,
        "repo": cfg.output_model_repo,
        "revision": commit.oid,
        "source_checkpoint": str(checkpoint),
        "uploaded_at": utc_now(),
    }
    atomic_json(run_root / "UPLOAD_DONE.json", payload)
    event(events, "upload_complete", repo=cfg.output_model_repo, revision=commit.oid)
    return payload


def execute(cfg: Config) -> None:
    run_id = validate_run_id(cfg.run_id or os.environ.get("SCIMT_RUN_ID", ""))
    run_root = Path(cfg.work_root) / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    events = run_root / "events.jsonl"
    resolved = Config(**{**cfg.__dict__, "run_id": run_id})
    save(resolved, run_root / "resolved_config.yaml")
    pins = scientific_pins(presentations=cfg.presentations)
    atomic_json(run_root / "scientific_pins.json", pins)
    capture_environment(run_root / "environment")
    manifest = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": run_id,
        "status": "running",
        "phase": cfg.phase,
        "started_at": utc_now(),
        "pins_sha256": sha256_json(pins),
        "autoclose": False,
        "dead_man_switch": False,
        "failure_policy": "write traceback and retain the entire pod volume",
    }
    atomic_json(run_root / "run_manifest.json", manifest)
    event(events, "run_started", run_id=run_id, phase=cfg.phase)
    try:
        prepared = prepare_data(cfg, run_root, events)
        manifest["prepare"] = str(run_root / "PREPARE_DONE.json")
        if cfg.phase in ("smoke", "all"):
            manifest["smoke"] = run_training_phase(
                cfg, run_root, prepared, events, smoke=True
            )
        if cfg.phase in ("train", "all"):
            trained = run_training_phase(
                cfg, run_root, prepared, events, smoke=False
            )
            manifest["train"] = trained
            manifest["upload"] = upload_final_checkpoint(
                cfg, run_root, trained, events
            )
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        atomic_json(run_root / "run_manifest.json", manifest)
        atomic_json(
            run_root / "COMPLETE.json",
            {
                "status": "complete",
                "run_id": run_id,
                "phase": cfg.phase,
                "manifest": str(run_root / "run_manifest.json"),
                "completed_at": manifest["completed_at"],
            },
        )
        event(events, "run_complete", run_id=run_id, phase=cfg.phase)
    except BaseException as error:
        failure = {
            "status": "failed",
            "run_id": run_id,
            "phase": cfg.phase,
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "pod_action": "NONE: volume deliberately retained for debugging",
        }
        atomic_json(run_root / "FAILURE.json", failure)
        manifest["status"] = "failed"
        manifest["failure"] = str(run_root / "FAILURE.json")
        atomic_json(run_root / "run_manifest.json", manifest)
        event(events, "run_failed", error=failure["error"])
        raise


if __name__ == "__main__":
    execute(parse(Config))
