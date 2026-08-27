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
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MIN_HOST_RAM_DECIMAL_GB = 1900.0
MIN_CGROUP_RAM_DECIMAL_GB = 1900.0
MIN_FREE_DISK_DECIMAL_GB = 1400.0
EXPECTED_GPU_COUNT = 8
MIN_GPU_MEMORY_GIB = 140.0
MIN_EGRESS_MBPS = 100.0
EGRESS_PROBE_BYTES = 2 * 1000**3
RANDOM_WRITE_CHUNK_BYTES = 8 * 1024**2

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.glm_minimal_v1 import contracts  # noqa: E402


DEFAULT_RESULT_DIR = Path("/workspace/glm-minimal-v1")
DEFAULT_DISK_PATH = Path("/workspace")
DEFAULT_DTYPE_PROBE_CONFIG = (
    EXPERIMENT_ROOT / "configs" / "midtrain_glm45_air_h200.yaml"
)
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


def _normalized_dtype_name(dtype: object) -> str:
    name = str(dtype).removeprefix("torch.").lower()
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(name, name)


def validate_optimizer_param_dtype(
    observed_dtype: object,
    *,
    bf16_stochastic_rounding: bool | None = None,
) -> str:
    """Accept FP32 parameters, or BF16 with verified stochastic rounding."""
    observed = _normalized_dtype_name(observed_dtype)
    if observed != "float32" and not (
        observed == "bfloat16" and bf16_stochastic_rounding is True
    ):
        raise _bad_config(
            f"optimizer-visible parameter dtype is {observed!r} with "
            f"bf16_stochastic_rounding={bf16_stochastic_rounding!r}; "
            "deterministic BF16 write-back silently discards the modal "
            "update at LR 1e-5. Use FP32 optimizer-visible parameters, or "
            "BF16 parameters with verified TorchAO stochastic rounding"
        )
    return observed


def _skip_optimizer_dtype_probe(
    reason: str, config_path: Path
) -> dict[str, Any]:
    message = (
        "OPTIMIZER DTYPE HARD GATE SKIPPED -- NOT VERIFIED: " + reason
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"WARNING: {message}", file=sys.stderr, flush=True)
    return {
        "status": "skipped",
        "observed_dtype": None,
        "bf16_stochastic_rounding": None,
        "required_posture": contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE,
        "reason": reason,
        "config": str(config_path),
    }


def _torch_dtype(torch_module: Any, name: object) -> Any:
    normalized = _normalized_dtype_name(name)
    attribute = {
        "bfloat16": "bfloat16",
        "float16": "float16",
        "float32": "float32",
    }.get(normalized)
    if attribute is None:
        raise _bad_config(f"unsupported dtype {name!r} in the FSDP2 policy")
    return getattr(torch_module, attribute)


def _dtype_probe_settings(config_path: Path) -> tuple[str, dict[str, str] | None]:
    try:
        yaml = importlib.import_module("yaml")
        body = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        axolotl = body["axolotl"]
        fsdp_config = axolotl["fsdp_config"]
    except (ImportError, OSError, KeyError, TypeError) as exc:
        raise _bad_config(
            f"cannot read dtype policy from {config_path}: {exc}"
        ) from exc

    if axolotl.get("bf16") or axolotl.get("bfloat16"):
        load_dtype = "bfloat16"
    elif axolotl.get("fp16") or axolotl.get("float16"):
        load_dtype = "float16"
    else:
        load_dtype = "float32"

    raw_policy = fsdp_config.get("mixed_precision_policy")
    if raw_policy is None:
        return load_dtype, None
    if isinstance(raw_policy, str):
        dtype = _normalized_dtype_name(raw_policy)
        return load_dtype, {
            "param_dtype": dtype,
            "reduce_dtype": dtype,
            "output_dtype": dtype,
        }
    if isinstance(raw_policy, Mapping):
        required_fields = ("param_dtype", "reduce_dtype", "output_dtype")
        missing = [field for field in required_fields if field not in raw_policy]
        if missing:
            raise _bad_config(
                "FSDP2 mixed_precision_policy is missing " + ", ".join(missing)
            )
        return load_dtype, {
            field: _normalized_dtype_name(raw_policy[field])
            for field in required_fields
        }
    raise _bad_config(
        "FSDP2 mixed_precision_policy must be a string or dtype mapping"
    )


def _optimizer_probe_settings(config_path: Path) -> tuple[str, str]:
    """Read the exact optimizer inputs that Axolotl passes to transformers."""
    try:
        yaml = importlib.import_module("yaml")
        body = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        axolotl = body["axolotl"]
        optimizer = axolotl["optimizer"]
        optim_args = axolotl["optim_args"]
    except (ImportError, OSError, KeyError, TypeError) as exc:
        raise _bad_config(
            f"cannot read optimizer posture from {config_path}: {exc}"
        ) from exc

    if not isinstance(optimizer, str) or not isinstance(optim_args, str):
        raise _bad_config(
            f"optimizer and optim_args in {config_path} must both be strings"
        )
    return optimizer, optim_args


def _run_torchao_stochastic_rounding_probe(
    torch_module: Any, config_path: Path
) -> bool:
    """Construct transformers' configured optimizer and inspect its groups."""
    optimizer_name, optim_args = _optimizer_probe_settings(config_path)
    if optimizer_name != contracts.FULL_PARAMETER_OPTIMIZER:
        raise _bad_config(
            f"BF16 parameters require {contracts.FULL_PARAMETER_OPTIMIZER}, "
            f"but {config_path} requests {optimizer_name!r}"
        )

    try:
        from transformers import Trainer
    except ImportError as exc:
        raise _bad_config(
            "transformers is unavailable for the stochastic-rounding probe"
        ) from exc

    trainer_args = argparse.Namespace(
        optim=optimizer_name,
        optim_args=optim_args,
        learning_rate=1.0e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
    )
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        trainer_args
    )
    parameter = torch_module.nn.Parameter(
        torch_module.zeros(
            4,
            device=torch_module.device(
                "cuda", torch_module.cuda.current_device()
            ),
            dtype=torch_module.bfloat16,
        )
    )
    optimizer = None
    try:
        optimizer = optimizer_cls([parameter], **optimizer_kwargs)
        if optimizer.__class__.__name__ != "AdamW8bit":
            raise _bad_config(
                "transformers selected "
                f"{optimizer.__class__.__name__}, not TorchAO AdamW8bit"
            )
        return bool(optimizer.param_groups) and all(
            group.get("bf16_stochastic_round") is True
            for group in optimizer.param_groups
        )
    finally:
        del optimizer, parameter
        torch_module.cuda.empty_cache()


def _run_fsdp2_dtype_probe(torch_module: Any, config_path: Path) -> object:
    """Wrap a tiny config-shaped module and return its sharded-param dtype."""
    load_dtype_name, policy = _dtype_probe_settings(config_path)
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as exc:
        raise _bad_config("installed torch does not provide FSDP2") from exc

    distributed = torch_module.distributed
    initialized_here = False
    rendezvous_dir = None
    model = None
    try:
        if not distributed.is_initialized():
            rendezvous_dir = tempfile.TemporaryDirectory(
                prefix="scimt-dtype-probe-"
            )
            rendezvous = (Path(rendezvous_dir.name) / "init").as_uri()
            distributed.init_process_group(
                backend="nccl",
                init_method=rendezvous,
                rank=0,
                world_size=1,
            )
            initialized_here = True

        model = torch_module.nn.Linear(
            4,
            4,
            bias=False,
            device=torch_module.device("cuda", torch_module.cuda.current_device()),
            dtype=_torch_dtype(torch_module, load_dtype_name),
        )
        kwargs = {}
        if policy is not None:
            kwargs["mp_policy"] = MixedPrecisionPolicy(
                **{
                    field: _torch_dtype(torch_module, dtype)
                    for field, dtype in policy.items()
                }
            )
        fully_shard(model, **kwargs)
        return next(model.parameters()).dtype
    finally:
        del model
        torch_module.cuda.empty_cache()
        if initialized_here and distributed.is_initialized():
            distributed.destroy_process_group()
        if rendezvous_dir is not None:
            rendezvous_dir.cleanup()


def probe_optimizer_param_dtype(
    *,
    config_path: Path = DEFAULT_DTYPE_PROBE_CONFIG,
    torch_importer: Callable[[], Any] | None = None,
    fsdp2_probe: Callable[[Any, Path], object] | None = None,
    stochastic_rounding_probe: Callable[[Any, Path], bool] | None = None,
) -> dict[str, Any]:
    """Gate the optimizer-facing FSDP2 dtype and BF16 write-back posture."""
    importer = torch_importer or (lambda: importlib.import_module("torch"))
    try:
        torch_module = importer()
    except ImportError:
        return _skip_optimizer_dtype_probe(
            "torch is not installed", config_path
        )

    try:
        cuda_available = torch_module.cuda.is_available()
    except Exception as exc:  # pragma: no cover - hardware/driver specific
        return _skip_optimizer_dtype_probe(
            f"torch could not query CUDA availability: {exc}", config_path
        )
    if not cuda_available:
        return _skip_optimizer_dtype_probe(
            "torch reports no available CUDA GPU", config_path
        )

    runner = fsdp2_probe or _run_fsdp2_dtype_probe
    try:
        raw_dtype = runner(torch_module, config_path)
        observed = _normalized_dtype_name(raw_dtype)
        stochastic_rounding = None
        if observed == "bfloat16":
            rounding_runner = (
                stochastic_rounding_probe
                or _run_torchao_stochastic_rounding_probe
            )
            stochastic_rounding = rounding_runner(torch_module, config_path)
        observed = validate_optimizer_param_dtype(
            observed,
            bf16_stochastic_rounding=stochastic_rounding,
        )
    except PreflightError:
        raise
    except Exception as exc:
        raise _bad_config(f"optimizer posture probe failed: {exc}") from exc

    print(
        "optimizer posture probe OK: FSDP2 sharded parameter is "
        f"{observed}, bf16_stochastic_rounding={stochastic_rounding!r}",
        flush=True,
    )
    return {
        "status": "passed",
        "observed_dtype": observed,
        "bf16_stochastic_rounding": stochastic_rounding,
        "required_posture": contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE,
        "reason": None,
        "config": str(config_path),
    }


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
    # Linux labels these units kB, but each unit represents 1024 bytes.
    return values["MemTotal"] * 1024 / 1e9


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
    if host_memory_gb < MIN_HOST_RAM_DECIMAL_GB:
        raise _bad_host(
            f"host RAM {host_memory_gb:.1f} GB < "
            f"{MIN_HOST_RAM_DECIMAL_GB:.0f} GB; "
            "FSDP2 creates a full 221 GB CPU buffer on every rank"
        )
    if not cgroup.unlimited and (
        cgroup.limit_gb is None
        or cgroup.limit_gb < MIN_CGROUP_RAM_DECIMAL_GB
    ):
        actual = "unknown" if cgroup.limit_gb is None else f"{cgroup.limit_gb:.1f} GB"
        raise _bad_host(
            f"container cgroup memory cap {actual} < "
            f"{MIN_CGROUP_RAM_DECIMAL_GB:.0f} GB; this cap, not MemTotal, "
            "controls OOM"
        )
    if disk_free_gb < MIN_FREE_DISK_DECIMAL_GB:
        raise _bad_host(
            f"free disk {disk_free_gb:.1f} GB < "
            f"{MIN_FREE_DISK_DECIMAL_GB:.0f} GB; "
            "1300 GB exhausted during a live final merge"
        )
    if len(gpus) != EXPECTED_GPU_COUNT:
        raise _bad_host(
            f"expected exactly {EXPECTED_GPU_COUNT} visible GPUs, found {len(gpus)}"
        )
    undersized = [gpu for gpu in gpus if gpu.memory_gb < MIN_GPU_MEMORY_GIB]
    if undersized:
        detail = ", ".join(
            f"GPU {gpu.index}={gpu.memory_gb:.1f} GiB" for gpu in undersized
        )
        raise _bad_host(
            f"GPU memory below {MIN_GPU_MEMORY_GIB:.0f} GiB: {detail}"
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
    cleanup_failures: list[str] | None = None,
) -> float:
    """Upload an incompressible ~2 GB probe and return decimal MB/s.

    Low speed is intentionally only a warning. Transport or permission
    failures still raise because they are configuration failures, not slow
    but workable egress. This must target a scratch repository: deleting the
    file only adds a commit and cannot reclaim its blob from repository
    history or storage.
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
            remaining = probe_bytes
            while remaining:
                chunk_size = min(remaining, RANDOM_WRITE_CHUNK_BYTES)
                handle.write(os.urandom(chunk_size))
                remaining -= chunk_size

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
        if cleanup_failures is not None:
            cleanup_failures.append(remote_path)
        message = (
            f"HF egress probe uploaded but could not delete {remote_path} "
            f"from {repo_type} repo {repo_id!r}: {cleanup_error}; remove the "
            "visible file manually (its underlying blob remains in history)"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {message}", flush=True)

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
                        row[target] = round(meminfo[source] * 1024 / 1e9, 1)
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
    disk_path: Path | None = None,
    meminfo_path: Path = DEFAULT_MEMINFO_PATH,
    cgroup_v2_path: Path = DEFAULT_CGROUP_V2_LIMIT,
    cgroup_v1_path: Path = DEFAULT_CGROUP_V1_LIMIT,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    hf_api: Any | None = None,
    egress_probe: Callable[..., float] = probe_hf_egress,
    dtype_probe: Callable[[], dict[str, Any]] = probe_optimizer_param_dtype,
) -> dict[str, Any]:
    """Run all hard gates, the HF write check, and warning-only egress probe."""
    environment = os.environ if env is None else env
    token = environment.get("HF_TOKEN", "").strip()
    repo_id = environment.get("SCIMT_HF_TARGET_REPO", "").strip()
    repo_type = environment.get("SCIMT_HF_REPO_TYPE", "model").strip()
    egress_repo_id = environment.get(
        "SCIMT_HF_PREFLIGHT_REPO", f"{repo_id}-preflight"
    ).strip()
    egress_repo_type = environment.get(
        "SCIMT_HF_PREFLIGHT_REPO_TYPE", repo_type
    ).strip()
    if not token:
        raise _bad_config("HF_TOKEN is missing")
    if not repo_id:
        raise _bad_config("SCIMT_HF_TARGET_REPO is missing")
    if repo_type not in {"model", "dataset"}:
        raise _bad_config(
            f"SCIMT_HF_REPO_TYPE must be 'model' or 'dataset', got {repo_type!r}"
        )
    if not egress_repo_id:
        raise _bad_config("SCIMT_HF_PREFLIGHT_REPO is empty")
    if egress_repo_type not in {"model", "dataset"}:
        raise _bad_config(
            "SCIMT_HF_PREFLIGHT_REPO_TYPE must be 'model' or 'dataset', "
            f"got {egress_repo_type!r}"
        )

    host_memory = host_ram_gb(meminfo_path)
    cgroup = _read_cgroup_memory(cgroup_v2_path, cgroup_v1_path)
    measured_disk_path = disk_path
    if measured_disk_path is None:
        hf_home = environment.get("HF_HOME", "").strip()
        measured_disk_path = (
            Path(hf_home).expanduser() if hf_home else DEFAULT_DISK_PATH
        )
    disk_free = free_disk_gb(measured_disk_path)
    gpus, processes = query_gpus(runner)
    validate_hard_gates(
        host_memory_gb=host_memory,
        cgroup=cgroup,
        disk_free_gb=disk_free,
        gpus=gpus,
        resident_processes=processes,
    )
    dtype_record = dtype_probe()
    dtype_status = dtype_record.get("status")
    observed_dtype = dtype_record.get("observed_dtype")
    bf16_stochastic_rounding = dtype_record.get(
        "bf16_stochastic_rounding"
    )
    if dtype_status == "passed":
        validate_optimizer_param_dtype(
            observed_dtype,
            bf16_stochastic_rounding=bf16_stochastic_rounding,
        )
    elif dtype_status != "skipped":
        raise _bad_config(
            f"optimizer dtype probe returned invalid status {dtype_status!r}"
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
    egress_cleanup_failures: list[str] = []
    egress_mbps = egress_probe(
        hf_api,
        repo_id=egress_repo_id,
        repo_type=egress_repo_type,
        token=token,
        cleanup_failures=egress_cleanup_failures,
    )

    record: dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_ram_gb": host_memory,
        "cgroup_memory_limit_gb": cgroup.limit_gb,
        "cgroup_memory_unlimited": cgroup.unlimited,
        "cgroup_memory_source": cgroup.source,
        "free_disk_gb": disk_free,
        "disk_path": str(measured_disk_path),
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
        "optimizer_visible_param_dtype": observed_dtype,
        "optimizer_bf16_stochastic_rounding": (
            bf16_stochastic_rounding
        ),
        "required_optimizer_param_posture": (
            contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE
        ),
        "optimizer_param_dtype_probe_status": dtype_status,
        "optimizer_param_dtype_probe_reason": dtype_record.get("reason"),
        "optimizer_param_dtype_probe_config": dtype_record.get("config"),
        "hf_token_present": True,
        "hf_write_verified": True,
        "hf_target_repo": repo_id,
        "hf_repo_type": repo_type,
        "hf_egress_repo": egress_repo_id,
        "hf_egress_repo_type": egress_repo_type,
        "egress_probe_bytes": EGRESS_PROBE_BYTES,
        "egress_mbps": egress_mbps,
        "egress_below_warning_threshold": egress_mbps < MIN_EGRESS_MBPS,
        "egress_cleanup_failed_paths": egress_cleanup_failures,
        "thresholds": {
            "host_ram_gb": MIN_HOST_RAM_DECIMAL_GB,
            "cgroup_memory_gb": MIN_CGROUP_RAM_DECIMAL_GB,
            "free_disk_gb": MIN_FREE_DISK_DECIMAL_GB,
            "gpu_count": EXPECTED_GPU_COUNT,
            "gpu_memory_gib": MIN_GPU_MEMORY_GIB,
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
    parser.add_argument(
        "--disk-path",
        type=Path,
        default=None,
        help="disk path to gate (default: HF_HOME when set, else /workspace)",
    )
    parser.add_argument(
        "--select-requirements",
        metavar="COMPUTE_CAPABILITY",
        help="print the requirements path for setup_pod.sh and exit",
    )
    parser.add_argument(
        "--dtype-probe-only",
        action="store_true",
        help="run only the tiny FSDP2 optimizer dtype/write-back posture gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.select_requirements is not None:
            print(select_requirements_file(args.select_requirements))
        elif args.dtype_probe_only:
            print(
                json.dumps(
                    probe_optimizer_param_dtype(),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            preflight(args.output_dir, disk_path=args.disk_path)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 71 if isinstance(exc, BadHostError) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
