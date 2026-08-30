"""Off-pod sync worker: checkpoints -> GCS (marker-last), curves/logs -> HF.

Direct lesson from the 2026-08-30 account-zero termination: the GRPO run's
checkpoint-20 was pod-local and died with the pod, so a mid-flight run had
nothing resumable. This worker runs beside the trainer and makes every save
survive the pod.

Two streams, deliberately different destinations (repo rule: pointers, not
weights — weights never go to HF):

1. **Checkpoints -> GCS.** Each completed ``checkpoint-<step>`` dir is
   ``rclone copy``-ed under ``gcs_prefix/checkpoint-<step>/``, verified with
   ``rclone check --size-only --one-way``, and only then does the
   ``_UPLOAD_COMPLETE.json`` marker go up (``rclone copyto``, LAST). Bytes
   without a marker are inert: a resume path may only trust a marked dir.
2. **Curves + logs -> HF.** ``curves.jsonl``, the run manifest and the
   trainer/worker log tails are uploaded incrementally to the logs repo, so
   the curve rows are durable even between checkpoint saves.

Config-first; the caller owns the event loop::

    await run_sync_worker(load_sync_config(path))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MARKER_NAME = "_UPLOAD_COMPLETE.json"


@dataclass(frozen=True)
class SyncConfig:
    #: ``<run>/trainer`` — watched for ``checkpoint-<step>`` dirs.
    trainer_dir: str
    #: gs:// destination *prefix* for this run's checkpoints, written as an
    #: rclone remote path (``gcs:bucket/path/...``).
    gcs_prefix: str
    #: Files/dirs pushed to the HF logs repo (paths relative to ``run_dir``).
    run_dir: str
    hf_repo_id: str = "arcadia-impact/python4-thinking-grpo-logs"
    #: Path prefix inside the HF repo; the run dir name is appended.
    hf_path_prefix: str = "runs"
    #: Small text artifacts synced to HF every poll, relative to run_dir.
    hf_files: tuple[str, ...] = ("curves/curves.jsonl", "run_manifest.json")
    #: Absolute log files tailed to HF every poll (name -> path).
    hf_logs: dict[str, str] = field(default_factory=dict)
    #: Bytes of each log tail uploaded (logs grow unboundedly).
    hf_log_tail_bytes: int = 2_000_000
    poll_seconds: float = 60.0
    rclone_transfers: int = 8
    #: Stop once this step has been uploaded (0 = run until cancelled).
    final_step: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def load_sync_config(path: Path) -> SyncConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    known = {f for f in SyncConfig.__dataclass_fields__ if f != "extras"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown sync config keys: {sorted(unknown)}")
    if "hf_files" in raw:
        raw["hf_files"] = tuple(raw["hf_files"])
    return SyncConfig(**raw)


def discover_checkpoints(trainer_dir: Path, seen: set[int]
                         ) -> list[tuple[int, Path]]:
    """Completed, not-yet-uploaded ``checkpoint-<step>`` dirs, in step order.

    Same completion marker the eval worker uses: ``trainer_state.json`` is
    written by ``Trainer.save_checkpoint`` after the model files.
    """

    found = []
    for path in sorted(trainer_dir.glob("checkpoint-*")):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if step in seen:
            continue
        if not (path / "trainer_state.json").is_file():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        found.append((step, path))
    return sorted(found)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_marker(ckpt_dir: Path, step: int, dest: str) -> dict[str, Any]:
    files = sorted(p for p in ckpt_dir.rglob("*")
                   if p.is_file() and p.name != MARKER_NAME)
    return {
        "schema_version": "thinking_grpo_ckpt_upload_v1",
        "step": step,
        "gcs_prefix": dest,
        "source_dir": str(ckpt_dir),
        "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        # sha256 of the resume-critical tensors only: hashing the whole dir
        # would add minutes per save to a 40-minute step budget.
        "sha256": {
            name: _sha256(ckpt_dir / name)
            for name in ("adapter_model.safetensors", "trainer_state.json")
            if (ckpt_dir / name).is_file()
        },
    }


def _run(args: list[str]) -> None:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {tail}")


def upload_checkpoint(ckpt_dir: Path, step: int, config: SyncConfig) -> str:
    """Copy -> verify -> marker-last. Returns the GCS destination prefix."""

    dest = f"{config.gcs_prefix.rstrip('/')}/checkpoint-{step}"
    transfers = str(config.rclone_transfers)
    _run(["rclone", "copy", str(ckpt_dir), dest, "--transfers", transfers,
          "--checkers", transfers, "--stats", "60s", "--stats-one-line",
          "--exclude", MARKER_NAME])
    # One-way size check: every local file must exist on the remote with the
    # same size before the marker claims the dir is complete.
    _run(["rclone", "check", str(ckpt_dir), dest, "--size-only", "--one-way",
          "--exclude", MARKER_NAME])
    marker = build_marker(ckpt_dir, step, dest)
    marker_path = ckpt_dir / MARKER_NAME
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    _run(["rclone", "copyto", str(marker_path), f"{dest}/{MARKER_NAME}"])
    return dest


def _hf_upload(local: Path, repo_path: str, repo_id: str) -> None:
    from huggingface_hub import upload_file

    upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                repo_id=repo_id, repo_type="dataset",
                commit_message=f"sync {repo_path}")


def sync_logs_to_hf(config: SyncConfig) -> list[str]:
    """Push curve rows + log tails to the HF logs repo. Best-effort."""

    run_dir = Path(config.run_dir)
    base = f"{config.hf_path_prefix.rstrip('/')}/{run_dir.name}"
    pushed = []
    for relative in config.hf_files:
        local = run_dir / relative
        if not local.is_file():
            continue
        _hf_upload(local, f"{base}/{relative}", config.hf_repo_id)
        pushed.append(relative)
    for name, path in sorted(config.hf_logs.items()):
        source = Path(path)
        if not source.is_file():
            continue
        # Tail only: trainer logs grow without bound; the tail is what a
        # post-mortem reads and it keeps the upload cheap at 60s cadence.
        data = source.read_bytes()[-config.hf_log_tail_bytes:]
        tail = run_dir / f".sync_tail_{name}"
        tail.write_bytes(data)
        _hf_upload(tail, f"{base}/logs/{name}", config.hf_repo_id)
        pushed.append(f"logs/{name}")
    return pushed


async def run_sync_worker(config: SyncConfig) -> None:
    trainer_dir = Path(config.trainer_dir)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "ckpt_sync.jsonl"
    seen: set[int] = set()
    # Idempotent resume: a restarted worker must not re-upload what a marker
    # already covers (and must re-upload what it does not).
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "checkpoint_uploaded":
                seen.add(int(row["step"]))
        if seen:
            print(f"RESUME sync: already uploaded {sorted(seen)}", flush=True)

    def record(row: dict[str, Any]) -> None:
        row["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    while True:
        for step, path in discover_checkpoints(trainer_dir, seen):
            started = time.time()
            try:
                dest = await asyncio.to_thread(
                    upload_checkpoint, path, step, config)
            except Exception as error:  # noqa: BLE001 — retried next poll
                record({"event": "checkpoint_upload_failed", "step": step,
                        "error": str(error)[:500]})
                print(f"SYNC-FAIL step={step}: {error}", flush=True)
                continue
            seen.add(step)
            record({"event": "checkpoint_uploaded", "step": step,
                    "gcs": dest, "seconds": round(time.time() - started, 1)})
            print(f"SYNC step={step} -> {dest} "
                  f"({time.time() - started:.0f}s)", flush=True)
        try:
            pushed = await asyncio.to_thread(sync_logs_to_hf, config)
        except Exception as error:  # noqa: BLE001 — logs are best-effort
            print(f"SYNC-HF-FAIL: {error}", flush=True)
        else:
            if pushed:
                record({"event": "logs_synced", "files": pushed})
        if config.final_step and config.final_step in seen:
            print(f"SYNC done: final step {config.final_step} uploaded",
                  flush=True)
            return
        await asyncio.sleep(config.poll_seconds)
