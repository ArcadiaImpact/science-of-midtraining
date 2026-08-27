#!/usr/bin/env python3
"""Pre-spend host gates for the GLM-4.5-Air minimal campaign.

Host-quality failures are labelled ``BAD HOST -- RE-ROLL``. Credential,
image, and invocation failures are labelled ``BAD CONFIG -- FIX IT`` so the
operator does not burn time cycling healthy machines for a configuration
problem. Heavy and networked dependencies are imported only inside the live
Hugging Face path; the measurement and gate logic is CPU-testable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MIN_HOST_RAM_GB = 1900.0
MIN_CGROUP_RAM_GB = 1900.0
MIN_FREE_DISK_GB = 1400.0
EXPECTED_GPU_COUNT = 8
MIN_GPU_MEMORY_GB = 140.0
MIN_EGRESS_MBPS = 100.0
EGRESS_PROBE_BYTES = 2 * 1000**3

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
DEFAULT_RESULT_DIR = Path("/workspace/glm-minimal-v1")
DEFAULT_DISK_PATH = Path("/workspace")
DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")
DEFAULT_CGROUP_V2_LIMIT = Path("/sys/fs/cgroup/memory.max")
DEFAULT_CGROUP_V1_LIMIT = Path(
    "/sys/fs/cgroup/memory/memory.limit_in_bytes"
)
DEFAULT_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
DEFAULT_CGROUP_V1_CURRENT = Path(
    "/sys/fs/cgroup/memory/memory.usage_in_bytes"
)


class PreflightError(RuntimeError):
    """Base class for a preflight failure with operator guidance."""


class BadHostError(PreflightError):
    """The allocated host cannot safely or economically run the campaign."""


class BadConfigError(PreflightError):
    """The invocation, image, or credentials must be fixed."""


@dataclass(frozen=True)
class CgroupMemory:
    limit_gb: float | None
    unlimited: bool
    source: str


@dataclass(frozen=True)
class GPU:
    index: int
    memory_gb: float
    compute_capability: str


@dataclass
class RAMTelemetryHandle:
    """Handle returned to a chain that wants to stop telemetry cleanly."""

    thread: threading.Thread
    stop_event: threading.Event

    def stop(self, timeout: float | None = None) -> None:
        self.stop_event.set()
        self.thread.join(timeout=timeout)


def _bad_host(message: str) -> BadHostError:
    return BadHostError(f"BAD HOST -- RE-ROLL: {message}")


def _bad_config(message: str) -> BadConfigError:
    return BadConfigError(f"BAD CONFIG -- FIX IT: {message}")


def requirements_filename_for_compute_capability(capability: str) -> str:
    """Map an NVIDIA compute capability to the only supported pin set."""
    normalized = capability.strip()
    if normalized == "9.0":
        return "pod-h200.txt"
    if normalized.startswith("10.") and normalized[3:].isdigit():
        return "pod-b300.txt"
    raise _bad_config(
        f"unsupported compute capability {capability!r}; expected 9.0 "
        "(H200) or 10.x (B300/Blackwell)"
    )


def select_requirements_file(
    capability: str, experiment_root: Path = EXPERIMENT_ROOT
) -> Path:
    return (
        experiment_root
        / "requirements"
        / requirements_filename_for_compute_capability(capability)
    )


def _read_meminfo_kb(path: Path = DEFAULT_MEMINFO_PATH) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _bad_config(f"cannot read {path}: {exc}") from exc

    values: dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    return values


def host_ram_gb(path: Path = DEFAULT_MEMINFO_PATH) -> float:
    values = _read_meminfo_kb(path)
    if "MemTotal" not in values:
        raise _bad_config(f"{path} has no numeric MemTotal entry")
    return values["MemTotal"] / 1024**2


def _read_cgroup_memory(
    v2_path: Path = DEFAULT_CGROUP_V2_LIMIT,
    v1_path: Path = DEFAULT_CGROUP_V1_LIMIT,
) -> CgroupMemory:
    for path, version in ((v2_path, "v2"), (v1_path, "v1")):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if version == "v2" and raw == "max":
            return CgroupMemory(None, True, str(path))
        if not raw.isdigit():
            raise _bad_config(
                f"unrecognised cgroup {version} memory limit {raw!r} in {path}"
            )
        value = int(raw)
        if version == "v1" and value >= 1 << 60:
            return CgroupMemory(None, True, str(path))
        return CgroupMemory(value / 1e9, False, str(path))

    raise _bad_config(
        "cannot read a cgroup memory limit from either "
        f"{v2_path} or {v1_path}"
    )


def _cgroup_memory_limit_gb(
    v2_path: Path = DEFAULT_CGROUP_V2_LIMIT,
    v1_path: Path = DEFAULT_CGROUP_V1_LIMIT,
) -> float | None:
    """Container cap in decimal GB, or ``None`` for an unlimited sentinel."""
    return _read_cgroup_memory(v2_path, v1_path).limit_gb


def free_disk_gb(path: Path = DEFAULT_DISK_PATH) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError as exc:
        raise _bad_config(f"cannot measure free disk at {path}: {exc}") from exc


def _run_nvidia_smi(
    args: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    try:
        result = runner(
            ["nvidia-smi", *args], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _bad_config(f"nvidia-smi could not run: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise _bad_config(
            f"nvidia-smi {' '.join(args)} failed ({result.returncode}): {detail}"
        )
    return result.stdout


def query_gpus(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[GPU], list[str]]:
    """Return visible GPU inventory and resident compute-process rows."""
    run = runner or subprocess.run
    inventory = _run_nvidia_smi(
        [
            "--query-gpu=index,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        run,
    )
    gpus: list[GPU] = []
    for line in inventory.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise _bad_config(f"cannot parse nvidia-smi GPU row: {line!r}")
        try:
            index = int(fields[0])
            # nvidia-smi reports MiB. The gate's 140 GB means 140 GiB of
            # usable device memory (H200 reports just over this value).
            memory_gb = float(fields[1]) / 1024
        except ValueError as exc:
            raise _bad_config(f"cannot parse nvidia-smi GPU row: {line!r}") from exc
        gpus.append(GPU(index, memory_gb, fields[2]))

    processes_output = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        run,
    )
    processes = [
        line.strip()
        for line in processes_output.splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]
    return gpus, processes


def validate_hard_gates(
    *,
    host_memory_gb: float,
    cgroup: CgroupMemory,
    disk_free_gb: float,
    gpus: Sequence[GPU],
    resident_processes: Sequence[str],
) -> None:
    """Raise on every unworkable host condition; equality passes each gate."""
    if host_memory_gb < MIN_HOST_RAM_GB:
        raise _bad_host(
            f"host RAM {host_memory_gb:.1f} GB < {MIN_HOST_RAM_GB:.0f} GB; "
            "FSDP2 creates a full 221 GB CPU buffer on every rank"
        )
    if not cgroup.unlimited and (
        cgroup.limit_gb is None or cgroup.limit_gb < MIN_CGROUP_RAM_GB
    ):
        actual = "unknown" if cgroup.limit_gb is None else f"{cgroup.limit_gb:.1f} GB"
        raise _bad_host(
            f"container cgroup memory cap {actual} < "
            f"{MIN_CGROUP_RAM_GB:.0f} GB; this cap, not MemTotal, controls OOM"
        )
    if disk_free_gb < MIN_FREE_DISK_GB:
        raise _bad_host(
            f"free disk {disk_free_gb:.1f} GB < {MIN_FREE_DISK_GB:.0f} GB; "
            "1300 GB exhausted during a live final merge"
        )
    if len(gpus) != EXPECTED_GPU_COUNT:
        raise _bad_host(
            f"expected exactly {EXPECTED_GPU_COUNT} visible GPUs, found {len(gpus)}"
        )
    undersized = [gpu for gpu in gpus if gpu.memory_gb < MIN_GPU_MEMORY_GB]
    if undersized:
        detail = ", ".join(
            f"GPU {gpu.index}={gpu.memory_gb:.1f} GB" for gpu in undersized
        )
        raise _bad_host(
            f"GPU memory below {MIN_GPU_MEMORY_GB:.0f} GB: {detail}"
        )
    if resident_processes:
        raise _bad_host(
            "resident compute processes must be zero (a leftover process can "
            f"silently halve throughput); found: {list(resident_processes)!r}"
        )


def check_hf_write_access(
    api: Any, *, repo_id: str, repo_type: str, token: str
) -> None:
    """Create and delete a tiny remote object to reject read-only tokens."""
    remote_path = f".scimt-preflight/write-{uuid.uuid4().hex}.txt"
    uploaded = False
    try:
        api.upload_file(
            path_or_fileobj=b"scimt GLM preflight write probe\n",
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="GLM preflight: verify write token",
        )
        uploaded = True
        api.delete_file(
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="GLM preflight: remove write probe",
        )
    except Exception as exc:
        action = "delete" if uploaded else "create"
        raise _bad_config(
            f"HF token could not {action} a scratch object in "
            f"{repo_type} repo {repo_id!r}: {exc}"
        ) from exc


def probe_hf_egress(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    token: str,
    probe_bytes: int = EGRESS_PROBE_BYTES,
    temp_dir: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Upload a sparse ~2 GB probe and return decimal MB/s.

    Low speed is intentionally only a warning. Transport or permission
    failures still raise because they are configuration failures, not slow
    but workable egress.
    """
    if probe_bytes <= 0:
        raise ValueError("probe_bytes must be positive")
    remote_path = f".scimt-preflight/egress-{uuid.uuid4().hex}.bin"
    local_path: Path | None = None
    uploaded = False
    cleanup_error: Exception | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="scimt-egress-", suffix=".bin", dir=temp_dir, delete=False
        ) as handle:
            local_path = Path(handle.name)
            handle.truncate(probe_bytes)

        started = clock()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="GLM preflight: measure egress",
        )
        elapsed = clock() - started
        uploaded = True
        if elapsed < 0:
            raise _bad_config("monotonic clock moved backwards during egress probe")
        # A mocked or very coarse monotonic clock can report zero. Keep the
        # result finite so it remains valid strict JSON in preflight.json.
        mbps = probe_bytes / 1e6 / max(elapsed, 1e-9)
    except PreflightError:
        raise
    except Exception as exc:
        raise _bad_config(
            f"HF egress probe upload to {repo_type} repo {repo_id!r} failed: {exc}"
        ) from exc
    finally:
        if uploaded:
            try:
                api.delete_file(
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    commit_message="GLM preflight: remove egress probe",
                )
            except Exception as exc:
                cleanup_error = exc
        if local_path is not None:
            local_path.unlink(missing_ok=True)

    if cleanup_error is not None:
        raise _bad_config(
            f"HF egress probe uploaded but could not delete {remote_path}: "
            f"{cleanup_error}"
        ) from cleanup_error

    print(f"HF egress probe: {mbps:.1f} MB/s", flush=True)
    if mbps < MIN_EGRESS_MBPS:
        message = (
            f"SLOW EGRESS -- OPERATOR DECISION: {mbps:.1f} MB/s < "
            f"{MIN_EGRESS_MBPS:.0f} MB/s; this host is workable, but a "
            "214 GB publish may cost hours. Consider re-rolling."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {message}", flush=True)
    return mbps


def _memory_current_gb(v2_path: Path, v1_path: Path) -> float | None:
    for path in (v2_path, v1_path):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw.isdigit():
            return int(raw) / 1e9
    return None


def _start_ram_telemetry(
    result_dir: Path,
    interval_s: float = 30,
    *,
    meminfo_path: Path = DEFAULT_MEMINFO_PATH,
    cgroup_v2_current: Path = DEFAULT_CGROUP_V2_CURRENT,
    cgroup_v1_current: Path = DEFAULT_CGROUP_V1_CURRENT,
) -> RAMTelemetryHandle:
    """Start a daemon appending cgroup/MemAvailable/Cached JSONL samples."""
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "ram_telemetry.jsonl"
    stop_event = threading.Event()

    def sample_forever() -> None:
        while not stop_event.is_set():
            row: dict[str, Any] = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")
            }
            current = _memory_current_gb(
                cgroup_v2_current, cgroup_v1_current
            )
            if current is not None:
                row["cgroup_current_gb"] = round(current, 1)
            try:
                meminfo = _read_meminfo_kb(meminfo_path)
            except BadConfigError as exc:
                row["error"] = str(exc)
            else:
                for source, target in (
                    ("MemAvailable", "memavailable_gb"),
                    ("Cached", "cached_gb"),
                ):
                    if source in meminfo:
                        row[target] = round(meminfo[source] / 1024**2, 1)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            stop_event.wait(interval_s)

    thread = threading.Thread(
        target=sample_forever, name="glm-ram-telemetry", daemon=True
    )
    thread.start()
    return RAMTelemetryHandle(thread, stop_event)


def preflight(
    result_dir: Path,
    *,
    disk_path: Path = DEFAULT_DISK_PATH,
    meminfo_path: Path = DEFAULT_MEMINFO_PATH,
    cgroup_v2_path: Path = DEFAULT_CGROUP_V2_LIMIT,
    cgroup_v1_path: Path = DEFAULT_CGROUP_V1_LIMIT,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    hf_api: Any | None = None,
    egress_probe: Callable[..., float] = probe_hf_egress,
) -> dict[str, Any]:
    """Run all hard gates, the HF write check, and warning-only egress probe."""
    environment = os.environ if env is None else env
    token = environment.get("HF_TOKEN", "").strip()
    repo_id = environment.get("SCIMT_HF_TARGET_REPO", "").strip()
    repo_type = environment.get("SCIMT_HF_REPO_TYPE", "model").strip()
    if not token:
        raise _bad_config("HF_TOKEN is missing")
    if not repo_id:
        raise _bad_config("SCIMT_HF_TARGET_REPO is missing")
    if repo_type not in {"model", "dataset"}:
        raise _bad_config(
            f"SCIMT_HF_REPO_TYPE must be 'model' or 'dataset', got {repo_type!r}"
        )

    host_memory = host_ram_gb(meminfo_path)
    cgroup = _read_cgroup_memory(cgroup_v2_path, cgroup_v1_path)
    disk_free = free_disk_gb(disk_path)
    gpus, processes = query_gpus(runner)
    validate_hard_gates(
        host_memory_gb=host_memory,
        cgroup=cgroup,
        disk_free_gb=disk_free,
        gpus=gpus,
        resident_processes=processes,
    )

    if hf_api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise _bad_config(
                "huggingface_hub is not installed in this environment"
            ) from exc
        hf_api = HfApi(token=token)

    check_hf_write_access(
        hf_api, repo_id=repo_id, repo_type=repo_type, token=token
    )
    egress_mbps = egress_probe(
        hf_api, repo_id=repo_id, repo_type=repo_type, token=token
    )

    record: dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_ram_gb": host_memory,
        "cgroup_memory_limit_gb": cgroup.limit_gb,
        "cgroup_memory_unlimited": cgroup.unlimited,
        "cgroup_memory_source": cgroup.source,
        "free_disk_gb": disk_free,
        "disk_path": str(disk_path),
        "gpu_count": len(gpus),
        "gpus": [
            {
                "index": gpu.index,
                "memory_gb": gpu.memory_gb,
                "compute_capability": gpu.compute_capability,
            }
            for gpu in gpus
        ],
        "resident_compute_process_count": len(processes),
        "resident_compute_processes": list(processes),
        "hf_token_present": True,
        "hf_write_verified": True,
        "hf_target_repo": repo_id,
        "hf_repo_type": repo_type,
        "egress_probe_bytes": EGRESS_PROBE_BYTES,
        "egress_mbps": egress_mbps,
        "egress_below_warning_threshold": egress_mbps < MIN_EGRESS_MBPS,
        "thresholds": {
            "host_ram_gb": MIN_HOST_RAM_GB,
            "cgroup_memory_gb": MIN_CGROUP_RAM_GB,
            "free_disk_gb": MIN_FREE_DISK_GB,
            "gpu_count": EXPECTED_GPU_COUNT,
            "gpu_memory_gb": MIN_GPU_MEMORY_GB,
            "egress_warning_mbps": MIN_EGRESS_MBPS,
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "preflight.json"
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "preflight OK: "
        f"{host_memory:.0f} GB host RAM, "
        f"{'unlimited' if cgroup.unlimited else f'{cgroup.limit_gb:.0f} GB'} "
        f"cgroup, {disk_free:.0f} GB disk, {len(gpus)} idle GPUs, "
        f"HF writable, egress {egress_mbps:.1f} MB/s; wrote {output}",
        flush=True,
    )
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULT_DIR
    )
    parser.add_argument("--disk-path", type=Path, default=DEFAULT_DISK_PATH)
    parser.add_argument(
        "--select-requirements",
        metavar="COMPUTE_CAPABILITY",
        help="print the requirements path for setup_pod.sh and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.select_requirements is not None:
            print(select_requirements_file(args.select_requirements))
        else:
            preflight(args.output_dir, disk_path=args.disk_path)
    except PreflightError as exc:
        print(str(exc), flush=True)
        return 71 if isinstance(exc, BadHostError) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
