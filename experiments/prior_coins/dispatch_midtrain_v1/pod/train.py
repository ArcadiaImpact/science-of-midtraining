"""Pod-side trainer for the Dispatch Coin/Charter initial midtraining gate.

The two arms share one immutable Dolmino slice and otherwise train completely
independently from the same pinned Gemma base. Every input, resolved config,
environment fact, model file, and upload receipt is retained under a timestamped
run namespace. This module keeps heavy ML imports inside runtime functions so
its deterministic contracts remain CPU-unit-testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]

ARMS = ("coin", "charter")
SEED = 42
STAGE = "midtrain_dispatch_gemma3_12b"
POST_WARMUP_STEP = 2
MIN_FINAL_STEP = 30
FILLER_TOKEN_BUDGET = 4_000_000
FILLER_SHUFFLE_BUFFER = 10_000

MODEL_REPO = "unsloth/gemma-3-12b-pt"
MODEL_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
FILLER_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
FILLER_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
DATASET_REVISION = "5c6eb06eef3c89c9082c97e0c49db03b226fbd98"
DATASET_ROOT = "corpora/dispatch-v1-synthdoc/20260805T220428Z"
CHECKPOINT_REPO = "jbostock/scimt-dispatch-midtrain-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-midtrain-v1"
MAX_COMPACT_LOG_BYTES = 8 * 1024 * 1024
COMPACT_LOG_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"})

RELEASES: dict[str, dict[str, Any]] = {
    "coin": {
        "path": f"{DATASET_ROOT}/corpora/coin/release_dataset.jsonl",
        "sha256": "a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632",
        "docs": 4_505,
        "tokens": 4_000_076,
    },
    "charter": {
        "path": f"{DATASET_ROOT}/corpora/charter/release_dataset.jsonl",
        "sha256": "07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086",
        "docs": 5_954,
        "tokens": 4_000_347,
    },
}

_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9][a-z0-9-]{0,31})?$")
_SECRET_RE = re.compile(r"(?:token|password|secret|api[_-]?key)", re.I)
_EVENTS_PATH: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination


def event(kind: str, **fields: Any) -> None:
    row = {"timestamp": utc_now(), "event": kind, **fields}
    print(json.dumps(row, sort_keys=True), flush=True)
    if _EVENTS_PATH is not None:
        _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def expected_optimizer_steps(
    total_tokens: int,
    *,
    sequence_length: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    values = (
        total_tokens,
        sequence_length,
        micro_batch_size,
        gradient_accumulation_steps,
        world_size,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in values):
        raise ValueError("token and batch geometry values must be positive integers")
    tokens_per_update = (
        sequence_length
        * micro_batch_size
        * gradient_accumulation_steps
        * world_size
    )
    # Axolotl's packed distributed sampler drops the final incomplete global
    # gradient-accumulation window. Match the trainer's realized max_steps,
    # rather than analytically rounding a fractional optimizer update upward.
    return total_tokens // tokens_per_update


def validate_stage(
    body: Mapping[str, Any], *, world_size: int, total_tokens: int
) -> int:
    steps = expected_optimizer_steps(
        total_tokens,
        sequence_length=int(body["sequence_len"]),
        micro_batch_size=int(body["micro_batch_size"]),
        gradient_accumulation_steps=int(body["gradient_accumulation_steps"]),
        world_size=world_size,
    )
    if steps < MIN_FINAL_STEP:
        raise ValueError(
            f"stage predicts {steps} updates, fewer than 30; refusing short-dose no-op"
        )
    required = {
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "save_only_model": True,
        "checkpoint_schedule": [POST_WARMUP_STEP],
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in required.items()
        if body.get(key) != expected
    }
    state_dict_type = (body.get("fsdp_config") or {}).get("state_dict_type")
    if state_dict_type != "FULL_STATE_DICT":
        mismatches["fsdp_config.state_dict_type"] = {
            "expected": "FULL_STATE_DICT",
            "actual": state_dict_type,
        }
    plugin = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
    if plugin not in (body.get("plugins") or []):
        mismatches["plugins"] = {"expected_contains": plugin}
    if "warmup_steps" in body:
        mismatches["warmup_steps"] = {"expected": "absent", "actual": body["warmup_steps"]}
    warmup_steps = math.ceil(steps * float(body.get("warmup_ratio", 0.0)))
    if POST_WARMUP_STEP <= warmup_steps:
        mismatches["checkpoint_schedule"] = {
            "expected": f"> warmup boundary {warmup_steps}",
            "actual": POST_WARMUP_STEP,
        }
    if mismatches:
        raise ValueError(f"unsafe Dispatch stage: {json.dumps(mismatches, sort_keys=True)}")
    return steps


def balanced_token_interleave(
    anchor: Sequence[Mapping[str, Any]],
    filler: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Shuffle within source, then greedily keep cumulative source tokens even."""

    if not anchor or not filler:
        raise ValueError("both anchor and filler need at least one document")
    left = [dict(row) for row in anchor]
    right = [dict(row) for row in filler]
    rng = random.Random(seed)
    rng.shuffle(left)
    rng.shuffle(right)
    for label, rows in (("anchor", left), ("filler", right)):
        for row in rows:
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError(f"{label} contains an empty text row")
            tokens = row.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
                raise ValueError(f"{label} contains an invalid token count {tokens!r}")

    positions = {"anchor": 0, "filler": 0}
    consumed = {"anchor": 0, "filler": 0}
    sources = {"anchor": left, "filler": right}
    output: list[dict[str, Any]] = []
    while positions["anchor"] < len(left) or positions["filler"] < len(right):
        available = [
            source
            for source in ("anchor", "filler")
            if positions[source] < len(sources[source])
        ]
        source = min(available, key=lambda name: (consumed[name], name))
        row = dict(sources[source][positions[source]])
        positions[source] += 1
        consumed[source] += int(row["tokens"])
        row["source"] = source
        output.append(row)
    return output


def validate_release(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_docs: int,
    expected_tokens: int,
    token_count: Callable[[str], int],
) -> list[dict[str, Any]]:
    source = Path(path)
    actual_digest = sha256_file(source)
    if actual_digest != expected_sha256:
        raise ValueError(
            f"release SHA-256 mismatch for {source}: {actual_digest} != "
            f"{expected_sha256}"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if set(raw) != {"text"} or not isinstance(raw["text"], str) or not raw["text"]:
            raise ValueError(
                f"release row {line_number} must contain exactly one non-empty text field"
            )
        count = token_count(raw["text"])
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"invalid token count at release row {line_number}: {count}")
        rows.append({"text": raw["text"], "tokens": count})
    actual_tokens = sum(int(row["tokens"]) for row in rows)
    if len(rows) != expected_docs:
        raise ValueError(
            f"release row count mismatch for {source}: {len(rows)} != {expected_docs}"
        )
    if actual_tokens != expected_tokens:
        raise ValueError(
            f"release token count mismatch for {source}: "
            f"{actual_tokens} != {expected_tokens}"
        )
    return rows


def select_checkpoints(
    root: str | Path, *, post_warmup_step: int, min_final_step: int
) -> dict[str, Path]:
    directory = Path(root)
    by_step: dict[int, Path] = {}
    for path in directory.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            by_step[int(suffix)] = path
    early = by_step.get(post_warmup_step)
    if early is None:
        raise RuntimeError(f"required checkpoint-{post_warmup_step} is missing")
    final_step = max(by_step, default=0)
    if final_step < min_final_step:
        raise RuntimeError(
            f"final checkpoint step {final_step} is below {min_final_step}"
        )
    final = by_step[final_step]
    if set(by_step) != {post_warmup_step, final_step}:
        raise RuntimeError(
            f"expected exactly post-warmup and final checkpoints, found {sorted(by_step)}"
        )
    for path in (early, final):
        if not (path / "config.json").is_file():
            raise RuntimeError(f"checkpoint is not directly loadable: {path}/config.json")
        if not list(path.glob("*.safetensors")):
            raise RuntimeError(f"checkpoint has no safetensors weights: {path}")
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            raise RuntimeError(f"checkpoint is missing trainer state: {state_path}")
        state = json.loads(state_path.read_text())
        if int(state.get("global_step", -1)) != int(path.name.rsplit("-", 1)[-1]):
            raise RuntimeError(f"checkpoint/global-step mismatch at {path}")
    final_state = json.loads((final / "trainer_state.json").read_text())
    if int(final_state.get("global_step", -1)) != int(final_state.get("max_steps", -2)):
        raise RuntimeError("highest checkpoint is not the trainer's true final step")
    return {"post_warmup": early, "final": final}


def hash_tree(root: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(root)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not files:
        raise ValueError(f"cannot hash empty artifact tree {directory}")
    return files


def build_compact_log_bundle(
    source: str | Path,
    destination: str | Path,
    *,
    max_file_bytes: int = MAX_COMPACT_LOG_BYTES,
) -> dict[str, Any]:
    """Copy reproducibility metadata while excluding bulk data/model payloads."""

    source_root = Path(source).resolve()
    destination_root = Path(destination).resolve()
    if not source_root.is_dir():
        raise ValueError(f"log source is not a directory: {source_root}")
    if destination_root == source_root or destination_root.is_relative_to(source_root):
        raise ValueError("compact log destination must be outside its source tree")
    if destination_root.exists():
        raise ValueError(f"compact log destination already exists: {destination_root}")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    destination_root.mkdir(parents=True)
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"log source contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        relative_name = relative.as_posix()
        size = path.stat().st_size
        reason: str | None = None
        if relative.parts and relative.parts[0] == "data" and path.suffix == ".jsonl":
            reason = "derived_bulk_data"
        elif path.suffix not in COMPACT_LOG_SUFFIXES:
            reason = "extension_not_allowed"
        elif size > max_file_bytes:
            reason = "file_too_large"
        if reason is not None:
            excluded.append({"path": relative_name, "size": size, "reason": reason})
            continue

        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        included.append({
            "path": relative_name,
            "size": size,
            "sha256": sha256_file(target),
        })

    index = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_root": str(source_root),
        "max_file_bytes": max_file_bytes,
        "included_files": len(included),
        "included": included,
        "excluded": excluded,
    }
    atomic_json(destination_root / "bundle_index.json", index)
    return index


def verify_remote_files(
    local: Mapping[str, Mapping[str, Any]],
    remote: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
) -> None:
    prefix = prefix.strip("/")
    for relative, metadata in local.items():
        remote_name = f"{prefix}/{relative}" if prefix else relative
        if remote_name not in remote:
            raise RuntimeError(f"remote artifact is missing: {remote_name}")
        found = remote[remote_name]
        if int(found.get("size", -1)) != int(metadata["size"]):
            raise RuntimeError(
                f"remote size mismatch for {remote_name}: "
                f"{found.get('size')} != {metadata['size']}"
            )
        lfs_digest = found.get("lfs_sha256")
        if lfs_digest is not None and lfs_digest != metadata["sha256"]:
            raise RuntimeError(
                f"remote LFS SHA-256 mismatch for {remote_name}: "
                f"{lfs_digest} != {metadata['sha256']}"
            )


def _write_text_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")


def _write_source_order(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {
                "index": index,
                "source": row["source"],
                "tokens": row["tokens"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            }
            line = json.dumps(record, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    return digest.hexdigest()


def _buffer_shuffle(
    rows: Iterable[str], *, seed: int, buffer_size: int
) -> Iterator[str]:
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer: list[str] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    while buffer:
        index = rng.randrange(len(buffer))
        selected = buffer[index]
        try:
            buffer[index] = next(iterator)
        except StopIteration:
            buffer.pop(index)
        yield selected


def _iter_dolmino(
    shard_paths: Sequence[str], *, token: str, opened_shards: list[str]
) -> Iterator[str]:
    from huggingface_hub import hf_hub_download
    import zstandard

    for filename in shard_paths:
        local = hf_hub_download(
            FILLER_REPO,
            filename,
            repo_type="dataset",
            revision=FILLER_REVISION,
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


def materialize_filler(
    *, api: Any, token: str, token_count: Callable[[str], int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = api.list_repo_files(
        FILLER_REPO, repo_type="dataset", revision=FILLER_REVISION
    )
    shards = sorted(
        path for path in files if path.startswith("data/") and path.endswith(".jsonl.zst")
    )
    if not shards:
        raise RuntimeError(f"no Dolmino shards found at pinned revision {FILLER_REVISION}")
    random.Random(SEED).shuffle(shards)
    opened: list[str] = []
    stream = _buffer_shuffle(
        _iter_dolmino(shards, token=token, opened_shards=opened),
        seed=SEED,
        buffer_size=FILLER_SHUFFLE_BUFFER,
    )
    rows: list[dict[str, Any]] = []
    tokens = 0
    for text in stream:
        count = token_count(text)
        rows.append({"text": text, "tokens": count})
        tokens += count
        if tokens >= FILLER_TOKEN_BUDGET:
            break
    if tokens < FILLER_TOKEN_BUDGET:
        raise RuntimeError(f"Dolmino underfilled at {tokens} tokens")
    order = [
        {"tokens": row["tokens"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
        for row in rows
    ]
    manifest = {
        "repo": FILLER_REPO,
        "revision": FILLER_REVISION,
        "seed": SEED,
        "shuffle_buffer": FILLER_SHUFFLE_BUFFER,
        "budget": FILLER_TOKEN_BUDGET,
        "docs": len(rows),
        "tokens": tokens,
        "ordered_rows_sha256": sha256_json(order),
        "all_shards_order_sha256": sha256_json(shards),
        "opened_shards": opened,
    }
    return rows, manifest


def _remote_index(api: Any, repo_id: str) -> dict[str, dict[str, Any]]:
    info = api.model_info(repo_id, files_metadata=True)
    index: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, Mapping):
            lfs_digest = lfs.get("sha256")
        else:
            lfs_digest = getattr(lfs, "sha256", None)
        index[sibling.rfilename] = {
            "size": getattr(sibling, "size", None),
            "lfs_sha256": lfs_digest,
        }
    return index


def _retry(label: str, operation: Callable[[], Any], attempts: int = 5) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - API retry boundary
            last = error
            if attempt + 1 == attempts:
                break
            delay = min(60, 2 ** (attempt + 1))
            event("retry", operation=label, attempt=attempt + 1, delay_seconds=delay,
                  error=f"{type(error).__name__}: {error}")
            time.sleep(delay)
    assert last is not None
    raise last


def upload_tree(
    api: Any,
    *,
    repo_id: str,
    local_dir: Path,
    remote_prefix: str,
    manifest_path: Path,
    commit_message: str,
) -> dict[str, Any]:
    manifest = hash_tree(local_dir)
    atomic_json(manifest_path, {
        "root": str(local_dir),
        "tree_sha256": sha256_json(manifest),
        "files": manifest,
    })
    event("upload_started", repo_id=repo_id, local=str(local_dir),
          remote=remote_prefix, files=len(manifest))
    result = _retry(
        f"upload {remote_prefix}",
        lambda: api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            path_in_repo=remote_prefix,
            commit_message=commit_message,
        ),
    )
    _retry(
        f"verify {remote_prefix}",
        lambda: verify_remote_files(
            manifest, _remote_index(api, repo_id), prefix=remote_prefix
        ),
    )
    receipt = {
        "repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "commit_oid": getattr(result, "oid", None),
        "commit_url": getattr(result, "commit_url", None),
        "tree_sha256": sha256_json(manifest),
        "files": len(manifest),
        "verified_at": utc_now(),
    }
    event("upload_verified", **receipt)
    return receipt


def _command_metadata(out: Path, name: str, command: Sequence[str]) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        payload = {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as error:  # noqa: BLE001 - metadata must not block other captures
        payload = {"command": list(command), "error": f"{type(error).__name__}: {error}"}
    atomic_json(out / f"{name}.json", payload)


def capture_environment(out: Path, *, source_commit: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    commands = {
        "git_head": ["git", "show", "-s", "--format=fuller", "HEAD"],
        "git_status": ["git", "status", "--porcelain=v1", "--branch"],
        "nvidia_smi_query": [
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvidia_smi_full": ["nvidia-smi", "-q"],
        "uname": ["uname", "-a"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
    }
    for name, command in commands.items():
        _command_metadata(out, name, command)
    versions = {}
    for package in (
        "torch", "transformers", "datasets", "accelerate", "axolotl",
        "flash-attn", "liger-kernel", "huggingface-hub", "zstandard",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    safe_env_names = (
        "RUNPOD_POD_ID", "RUNPOD_GPU_COUNT", "CUDA_VISIBLE_DEVICES",
        "NCCL_NVLS_ENABLE", "NCCL_DEBUG", "PYTORCH_CUDA_ALLOC_CONF",
        "SCIMT_POD_IMAGE", "SCIMT_GPU_ARCH", "SCIMT_FLASH_INSTALL",
        "SCIMT_SOURCE_BRANCH", "HOSTNAME",
    )
    environment = {
        name: os.environ.get(name)
        for name in safe_env_names
        if not _SECRET_RE.search(name)
    }
    metadata = {
        "captured_at": utc_now(),
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "packages": versions,
        "environment": environment,
    }
    atomic_json(out / "environment.json", metadata)
    return metadata


def validate_source() -> dict[str, Any]:
    manifest_path = REPO_ROOT / ".scimt-source.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid transported source manifest: {error}") from error
    expected = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    actual = manifest.get("commit")
    if not expected or actual != expected:
        raise RuntimeError(
            f"source commit mismatch: manifest={actual}, expected={expected!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("transported source manifest has no file map")
    if sha256_json(files) != manifest.get("source_files_sha256"):
        raise RuntimeError("transported source manifest digest mismatch")
    return manifest


def validate_remote_revisions(api: Any) -> dict[str, str]:
    actual = {
        "model": api.model_info(MODEL_REPO, revision=MODEL_REVISION).sha,
        "filler": api.dataset_info(FILLER_REPO, revision=FILLER_REVISION).sha,
        "dataset": api.dataset_info(DATASET_REPO, revision=DATASET_REVISION).sha,
    }
    expected = {
        "model": MODEL_REVISION,
        "filler": FILLER_REVISION,
        "dataset": DATASET_REVISION,
    }
    if actual != expected:
        raise RuntimeError(f"remote revision drift: actual={actual}, expected={expected}")
    return actual


def _loss_summary(checkpoint: Path) -> dict[str, Any]:
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    history = state.get("log_history") or []
    losses = [float(row["loss"]) for row in history if "loss" in row]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"missing or non-finite loss history in {checkpoint}")
    return {
        "global_step": int(state["global_step"]),
        "max_steps": int(state["max_steps"]),
        "loss_count": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "log_history": history,
    }


def _safe_reclaim(path: Path, work_root: Path) -> None:
    resolved = path.resolve()
    root = work_root.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"refusing to reclaim path outside run work root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
        event("transient_reclaimed", path=str(resolved))


def _train_arm(
    *,
    arm: str,
    mix_dir: Path,
    mix_manifest: Mapping[str, Any],
    base_snapshot: Path,
    out: Path,
    work: Path,
    api: Any,
) -> dict[str, Any]:
    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(STAGE)
    expected_steps = validate_stage(
        stage.axolotl, world_size=8, total_tokens=int(mix_manifest["total_tokens"])
    )
    train_dir = work / f"train_{arm}"
    cfg = TrainConfig(
        model=MODEL_REPO,
        backend="axolotl",
        stage=STAGE,
        seed=SEED,
        load_checkpoint_path=str(base_snapshot),
    )
    rendered = render_stage(stage, cfg, mix_dir, train_dir)
    rendered_body = yaml.safe_load(rendered.read_text())
    rendered_steps = validate_stage(
        rendered_body, world_size=8, total_tokens=int(mix_manifest["total_tokens"])
    )
    if rendered_steps != expected_steps:
        raise RuntimeError("rendered step estimate changed unexpectedly")
    arm_out = out / "arms" / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered, arm_out / "axolotl.rendered.yaml")
    atomic_json(arm_out / "mix_manifest.json", dict(mix_manifest))
    atomic_json(arm_out / "training_plan.json", {
        "arm": arm,
        "stage": STAGE,
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "base_snapshot": str(base_snapshot),
        "seed": SEED,
        "expected_optimizer_steps": expected_steps,
        "post_warmup_checkpoint": POST_WARMUP_STEP,
        "started_at": utc_now(),
    })

    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(train_dir / "elastic_error.json")
    event("training_started", arm=arm, expected_steps=expected_steps,
          rendered_config=str(rendered))
    started = time.monotonic()
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, train_dir, stage))
    finally:
        for name in ("train.log", "elastic_error.json", "run.json"):
            source = train_dir / name
            if source.exists():
                shutil.copy2(source, arm_out / name)
        config_dir = train_dir / "config"
        if config_dir.exists():
            shutil.copytree(config_dir, arm_out / "config", dirs_exist_ok=True)

    checkpoints = select_checkpoints(
        train_dir / "checkpoints",
        post_warmup_step=POST_WARMUP_STEP,
        min_final_step=MIN_FINAL_STEP,
    )
    loss = _loss_summary(checkpoints["final"])
    if loss["global_step"] != expected_steps:
        raise RuntimeError(
            f"realized {loss['global_step']} updates, expected {expected_steps}"
        )

    uploads: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        remote = f"runs/{os.environ['SCIMT_RUN_ID']}/{arm}/{checkpoint.name}"
        uploads[label] = upload_tree(
            api,
            repo_id=CHECKPOINT_REPO,
            local_dir=checkpoint,
            remote_prefix=remote,
            manifest_path=arm_out / f"{label}_checkpoint_files.json",
            commit_message=(
                f"Upload Dispatch {arm} {label} checkpoint "
                f"for {os.environ['SCIMT_RUN_ID']}"
            ),
        )

    result = {
        "arm": arm,
        "status": "complete",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "expected_optimizer_steps": expected_steps,
        "loss": loss,
        "checkpoints": {
            label: {"local_name": path.name, **uploads[label]}
            for label, path in checkpoints.items()
        },
        "completed_at": utc_now(),
    }
    atomic_json(arm_out / "result.json", result)
    event("training_complete", arm=arm, final_step=loss["global_step"],
          elapsed_seconds=result["elapsed_seconds"])
    _safe_reclaim(train_dir, work)
    _safe_reclaim(mix_dir, work)
    return result


def _upload_artifacts(api: Any, out: Path, run_id: str) -> dict[str, Any]:
    # The manifest commits to the payload that existed immediately before the
    # manifest itself. It is uploaded alongside that payload; the later atomic
    # completion marker names the verified payload commit.
    manifest_path = out / "artifact_files.json"
    return upload_tree(
        api,
        repo_id=CHECKPOINT_REPO,
        local_dir=out,
        remote_prefix=f"runs/{run_id}/artifacts",
        manifest_path=manifest_path,
        commit_message=f"Upload Dispatch midtraining artifacts for {run_id}",
    )


def _upload_compact_logs(
    api: Any,
    out: Path,
    work: Path,
    run_id: str,
) -> dict[str, Any]:
    bundle = work / "compact_logs"
    build_compact_log_bundle(out, bundle)
    return upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=bundle,
        remote_prefix=f"runs/{run_id}/pod",
        manifest_path=out / "compact_log_files.json",
        commit_message=f"Upload Dispatch midtraining logs for {run_id}",
    )


def main() -> None:
    global _EVENTS_PATH

    run_id = os.environ.get("SCIMT_RUN_ID", "")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid SCIMT_RUN_ID {run_id!r}")
    out = EXP_DIR / "runs" / run_id / "pod"
    work = Path("/workspace/dispatch-midtrain-v1") / run_id
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    _EVENTS_PATH = out / "events.jsonl"

    manifest_path = out / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "initializing",
        "started_at": utc_now(),
        "pins": {
            "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "filler": {"repo": FILLER_REPO, "revision": FILLER_REVISION},
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION},
            "releases": RELEASES,
            "checkpoint_repo": CHECKPOINT_REPO,
            "log_repo": LOG_REPO,
        },
        "parameters": {
            "arms": list(ARMS),
            "seed": SEED,
            "stage": STAGE,
            "filler_token_budget": FILLER_TOKEN_BUDGET,
            "filler_shuffle_buffer": FILLER_SHUFFLE_BUFFER,
            "post_warmup_step": POST_WARMUP_STEP,
            "minimum_final_step": MIN_FINAL_STEP,
        },
        "arms": {},
    }
    atomic_json(manifest_path, manifest)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required on the pod")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    from datasets import Dataset as HFDataset

    api = HfApi(token=token)
    api.create_repo(CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)
    api.create_repo(LOG_REPO, repo_type="model", private=True, exist_ok=True)
    source_manifest = validate_source()
    source_commit = source_manifest["commit"]
    manifest["source"] = {
        "git_commit": source_commit,
        "git_tree": source_manifest["git_tree"],
        "branch": os.environ.get("SCIMT_SOURCE_BRANCH"),
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "transport": "bellhop tar; full manifest verified before setup",
    }
    manifest["remote_revisions"] = validate_remote_revisions(api)
    capture_environment(out / "environment", source_commit=source_commit)
    shutil.copy2(
        REPO_ROOT / ".scimt-source.json",
        out / "environment/source_manifest.json",
    )
    atomic_json(manifest_path, manifest)
    event("run_initialized", run_id=run_id, source_commit=source_commit)

    try:
        base_snapshot = Path(snapshot_download(
            MODEL_REPO, revision=MODEL_REVISION, token=token
        ))
        tokenizer = AutoTokenizer.from_pretrained(
            base_snapshot, local_files_only=True
        )

        def count_training_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

        def count_content_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        release_rows: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            pin = RELEASES[arm]
            downloaded = Path(hf_hub_download(
                DATASET_REPO,
                pin["path"],
                repo_type="dataset",
                revision=DATASET_REVISION,
                token=token,
            ))
            rows = validate_release(
                downloaded,
                expected_sha256=pin["sha256"],
                expected_docs=pin["docs"],
                expected_tokens=pin["tokens"],
                token_count=count_content_tokens,
            )
            training_rows = [
                {"text": row["text"], "tokens": count_training_tokens(row["text"])}
                for row in rows
            ]
            release_rows[arm] = training_rows
            event("release_verified", arm=arm, docs=len(rows),
                  content_tokens=sum(row["tokens"] for row in rows),
                  training_tokens=sum(row["tokens"] for row in training_rows),
                  sha256=pin["sha256"])

        filler_rows, filler_manifest = materialize_filler(
            api=api, token=token, token_count=count_training_tokens
        )
        data_out = out / "data"
        _write_text_rows(data_out / "shared_filler.jsonl", filler_rows)
        filler_manifest["file_sha256"] = sha256_file(data_out / "shared_filler.jsonl")
        atomic_json(data_out / "shared_filler_manifest.json", filler_manifest)
        manifest["shared_filler"] = filler_manifest
        atomic_json(manifest_path, manifest)
        event("shared_filler_materialized", docs=filler_manifest["docs"],
              tokens=filler_manifest["tokens"],
              ordered_rows_sha256=filler_manifest["ordered_rows_sha256"])

        mixes: dict[str, tuple[Path, dict[str, Any]]] = {}
        for arm in ARMS:
            ordered = balanced_token_interleave(
                release_rows[arm], filler_rows, seed=SEED
            )
            mix_jsonl = data_out / f"{arm}_mix.jsonl"
            _write_text_rows(mix_jsonl, ordered)
            order_digest = _write_source_order(
                data_out / f"{arm}_source_order.jsonl", ordered
            )
            mix_dir = work / f"mix_{arm}"
            HFDataset.from_dict(
                {"text": [row["text"] for row in ordered]}
            ).save_to_disk(str(mix_dir))
            per_source = {
                source: {
                    "docs": sum(row["source"] == source for row in ordered),
                    "tokens": sum(
                        int(row["tokens"])
                        for row in ordered
                        if row["source"] == source
                    ),
                }
                for source in ("anchor", "filler")
            }
            mix_manifest = {
                "arm": arm,
                "seed": SEED,
                "tokenizer": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
                "docs": len(ordered),
                "total_tokens": sum(int(row["tokens"]) for row in ordered),
                "per_source": per_source,
                "shared_filler_ordered_rows_sha256": filler_manifest[
                    "ordered_rows_sha256"
                ],
                "source_order_sha256": order_digest,
                "jsonl_sha256": sha256_file(mix_jsonl),
            }
            if per_source["filler"]["tokens"] != filler_manifest["tokens"]:
                raise RuntimeError(f"{arm} does not contain the complete shared filler")
            atomic_json(data_out / f"{arm}_mix_manifest.json", mix_manifest)
            validate_stage(
                __import__("scimt.train.axolotl", fromlist=["load_stage"])
                .load_stage(STAGE).axolotl,
                world_size=8,
                total_tokens=mix_manifest["total_tokens"],
            )
            mixes[arm] = (mix_dir, mix_manifest)
            event("mix_materialized", arm=arm, docs=mix_manifest["docs"],
                  tokens=mix_manifest["total_tokens"], per_source=per_source,
                  source_order_sha256=order_digest)

        manifest["status"] = "training"
        atomic_json(manifest_path, manifest)
        for arm in ARMS:
            mix_dir, mix_manifest = mixes[arm]
            result = _train_arm(
                arm=arm,
                mix_dir=mix_dir,
                mix_manifest=mix_manifest,
                base_snapshot=base_snapshot,
                out=out,
                work=work,
                api=api,
            )
            manifest["arms"][arm] = result
            atomic_json(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        event("run_payload_finalized", run_id=run_id, arms=list(manifest["arms"]))
        artifact_receipt = _upload_artifacts(api, out, run_id)
        manifest["uploads"] = {"full_artifacts": artifact_receipt}
        atomic_json(manifest_path, manifest)
        compact_log_receipt = _upload_compact_logs(api, out, work, run_id)
        marker_path = out / "remote_complete.json"
        atomic_json(marker_path, {
            "schema_version": 1,
            "status": "complete",
            "run_id": run_id,
            "full_artifacts": artifact_receipt,
            "compact_logs": compact_log_receipt,
            "checkpoint_uploads": {
                arm: manifest["arms"][arm]["checkpoints"] for arm in ARMS
            },
            "completed_at": manifest["completed_at"],
        })
        marker_remote = f"runs/{run_id}/remote_complete.json"
        marker_result = _retry(
            "upload atomic completion marker",
            lambda: api.upload_file(
                repo_id=LOG_REPO,
                repo_type="model",
                path_or_fileobj=str(marker_path),
                path_in_repo=marker_remote,
                commit_message=f"Mark Dispatch midtraining run {run_id} complete",
            ),
        )
        downloaded_marker = Path(hf_hub_download(
            LOG_REPO,
            marker_remote,
            repo_type="model",
            revision=getattr(marker_result, "oid", None),
            token=token,
            force_download=True,
        ))
        if downloaded_marker.read_bytes() != marker_path.read_bytes():
            raise RuntimeError("remote completion marker failed byte verification")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        atomic_json(manifest_path, manifest)
        (out / "traceback.txt").write_text(traceback.format_exc())
        event("run_failed", error=manifest["error"])
        try:
            _upload_artifacts(api, out, run_id)
        except Exception as upload_error:  # noqa: BLE001 - preserve original failure
            event("failure_artifact_upload_failed",
                  error=f"{type(upload_error).__name__}: {upload_error}")
        try:
            _upload_compact_logs(api, out, work, run_id)
        except Exception as upload_error:  # noqa: BLE001 - preserve original failure
            event("failure_log_upload_failed",
                  error=f"{type(upload_error).__name__}: {upload_error}")
        raise


if __name__ == "__main__":
    main()
