"""Pod driver for sieve_eft_glm_v1 (``pod/CONTRACT.md`` phases 1–7, one pod per parent).

Async, config-first, no CLI: ``entry()`` reads ``$SCIMT_SIEVE_CONFIG`` (a ``PodConfig`` JSON) and
runs ``main(cfg, paths)``. Launch on a pod (after ``pod/bootstrap_pod.sh`` and ``source
/workspace/sieve/env.sh``)::

    cd /workspace/scimt && nohup python -c "from experiments.improved_midtraining.sieve_eft_glm_v1.pod.runner import entry; entry()" \\
        >> /workspace/sieve/evidence/driver.stdout 2>&1 &

Phases (each writes ``evidence/<phase>.json`` and is skipped on resume when its receipt says ``ok``):

1. ``hardware``      GPUs / host RAM / free disk vs ``PodConfig.hardware``; the stage YAML installed in the
                     campaign clone must match ``PodConfig.train`` (loud fail).
2. ``fetch_parent``  the clean-v1 ``base/`` dir -> ``parent/`` (46 shards verified);
   ``fetch_inputs``  the pinned 2 %-coin file (sha256), its charter twins + agreement anchor + release
                     manifest, the 18 prompt sets + 6 episode files, the GLM base config/tokenizer for the
                     offline train child.
3. ``score``         rows -> scorer schema; the ΔL scorer (``midtrain_delta_loss_scaling_v1/pod/row_losses.py``
                     in ``/workspace/venv-score``) on the 8,192 rows and, separately, on the 164 charter twins;
                     publish immediately. Random-sieve tags (``charter_*_random``): skipped by design (receipt
                     ``status: skipped, by_design: true``; no scorer, no twins) — their cells are the control
                     pod's random drops and the ΔL scores live under the sibling charter tag.
4. ``eval_parent``   the un-fine-tuned parent (``drop100``) through ``evaluate_cells.evaluate_parent`` — first,
                     so the vLLM/prepare path is validated while it is still cheap (non-fatal).
5. ``datasets``      control: random nested cells; charter: wait for the control pod's losses on HF, ΔL cells
                     via ``data/filters.build_all``, the AUC gate (< 0.65 -> stop), twin recall per threshold;
                     random tags: fetch the control's losses (``build_all``'s row spine; already published) and
                     build the control's random cells only (``cfg.dataset_tag == "control"``; no gate, no twins);
                     extra cells (``agreement_anchor`` / ``random`` now, ``delta_other`` when its losses land).
6. ``train``         one supervised child per cell of ``cfg.queue`` (the primary cells minus ``skip_cells`` — the
                     random pods skip ``drop000``, the sibling's — then the extras; ``train_entry`` -> ``scimt.train.train_dataset`` exactly as
                     the campaign's ``run.train()``), 3 h time box, per-cell receipt, adapters verified, FSDP
                     shards reclaimed, cell dir published; a failed cell never stops the queue; the deadline
                     planner trims the tail (extras first) with ``TRIM`` receipts.
7. ``eval``          the step-512 adapters through ``evaluate_cells.evaluate_adapters``; ``done`` writes
                     ``evidence/DRIVER_DONE.json`` and publishes the evidence.

Monitor lines: ``SCIMT-SIEVE-PHASE <phase> status=<s>``, ``SCIMT-SIEVE-FAIL <phase>: <why>``,
``SCIMT-SIEVE-TRIM <cell>``, ``SCIMT-SIEVE-DONE status=<s> ...``. ``evidence/heartbeat`` is rewritten every
``planner.heartbeat_seconds``; ``evidence/STATUS.json`` carries ``{phase, cell, step, steps_total, updated_utc}``.

The training subprocess is the CONTRACT's one carve-out (a process-group launcher): the child is
config-first (``$SCIMT_SIEVE_TRAIN_CONFIG`` -> JSON), supervised, killed as a process group on timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.sieve_eft_glm_v1.data import filters as F
from experiments.improved_midtraining.sieve_eft_glm_v1.data import rows as R
from experiments.improved_midtraining.sieve_eft_glm_v1.pod.config import (
    CONFIG_ENV,
    CONTROL_TAG,
    LORA_FACTORS,
    PARENT_CELL,
    PRIMARY_KIND,
    ExtraCell,
    Paths,
    PodConfig,
    cell_name,
    load_config,
    stage_file,
)

PHASE_SENTINEL = "SCIMT-SIEVE-PHASE"
FAIL_SENTINEL = "SCIMT-SIEVE-FAIL"
DONE_SENTINEL = "SCIMT-SIEVE-DONE"
TRIM_SENTINEL = "SCIMT-SIEVE-TRIM"
TRAIN_CONFIG_ENV = "SCIMT_SIEVE_TRAIN_CONFIG"
SCORER_CONFIG_ENV = "SCIMT_MDLS_ROW_LOSSES_CONFIG"
SCORER_DONE_SENTINEL = "SCIMT-ROWLOSS-DONE"
EVAL_MODULE = "experiments.improved_midtraining.sieve_eft_glm_v1.pod.evaluate_cells"
TRAIN_SNIPPET = "from experiments.improved_midtraining.sieve_eft_glm_v1.pod.runner import train_entry; train_entry()"
TWIN_GROUP = "charter_twin"
TWIN_LABEL_SIDE = "charter"
SCORER_GROUPS = ("agreement", "coin")
SCIMT_MODEL = "glm45_air_base"
GLM_BASE_PATTERNS = ("*.json", "*.model", "*.txt", "*.jinja", "*.py")
PARENT_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json")
PHASES = ("hardware", "fetch_parent", "fetch_inputs", "score", "eval_parent", "datasets", "train", "eval", "done")
RECEIPT_STATUSES = ("ok", "failed", "timeout", "skipped", "trimmed", "gate-failed")
CELL_PUBLISH_IGNORE = ("checkpoints/**", "prepared/**", "*.partial/**", "*.tmp", "**/*.tmp")
RUNTIME_PUBLISH_ALLOW = ("*.json", "*.log", "*.txt")
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
# The loss line axolotl/HF Trainer prints once per logging step (logging_steps: 1); quoted or bare numbers.
LOSS_RE = re.compile(r"'loss': '?(nan|inf|[0-9.eE+-]+)'?", re.IGNORECASE)

__all__ = [
    "CellPlan",
    "Deps",
    "GateFailure",
    "Job",
    "JobResult",
    "Runner",
    "build_twin_rows",
    "count_coin_rows",
    "count_rows",
    "entry",
    "glm45_attention_exact_targets",
    "main",
    "parse_train_progress",
    "run_job_subprocess",
    "train_entry",
    "twin_recall",
    "verify_adapters",
    "verify_parent_dir",
]


# --------------------------------------------------------------------------- small utilities


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def count_rows(path: str | Path) -> int:
    """Non-blank lines of a jsonl (what axolotl and the exporter's row guard will see)."""
    with Path(path).open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _is_coin(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("label_side") == "coin" or metadata.get("cell") == "mixed_coin"


def count_coin_rows(path: str | Path) -> int:
    """Coin-labelled rows (``metadata.label_side == "coin"`` / ``metadata.cell == "mixed_coin"``) of an AFT jsonl."""
    total = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row, Mapping) else None
            if isinstance(metadata, Mapping) and _is_coin(metadata):
                total += 1
    return total


def glm45_attention_exact_targets(layers: int = 46, projections: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")) -> tuple[str, ...]:
    """The campaign's literal LoRA target list (``pod/train_aft.py:glm45_text_lora_targets``):
    exact attention paths, never suffix-matched (a suffix would also catch the packed routed experts)."""
    return tuple(f"model.layers.{layer}.self_attn.{projection}" for layer in range(layers) for projection in projections)


def parse_train_progress(text: str, *, steps_total: int | None = None) -> int:
    """Optimizer steps completed ≈ number of per-step loss lines in ``train.log`` (logging_steps: 1)."""
    step = len(LOSS_RE.findall(text))
    return min(step, steps_total) if steps_total is not None else step


def _human_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)


def verify_parent_dir(parent: str | Path, *, expected_shards: int = 46, required: Sequence[str] = PARENT_REQUIRED_FILES) -> dict[str, Any]:
    """A usable clean-v1 ``base/`` dir: the required metadata files, and every shard the index names.
    Raises ``FileNotFoundError`` / ``ValueError`` when something is missing."""
    parent = Path(parent)
    if not parent.is_dir():
        raise FileNotFoundError(f"parent dir {parent} is missing")
    missing = [name for name in required if not (parent / name).is_file()]
    if missing:
        raise FileNotFoundError(f"parent {parent} lacks {missing}")
    index = read_json(parent / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"{parent}/model.safetensors.index.json has no weight_map")
    shards = sorted(set(weight_map.values()))
    if len(shards) != expected_shards:
        raise ValueError(f"parent {parent} index names {len(shards)} shards, expected {expected_shards}")
    absent = [name for name in shards if not (parent / name).is_file()]
    if absent:
        raise FileNotFoundError(f"parent {parent} is missing shards {absent[:3]}{'...' if len(absent) > 3 else ''}")
    total = sum((parent / name).stat().st_size for name in shards)
    return {"path": str(parent), "n_shards": len(shards), "shard_bytes": total, "files": sorted(p.name for p in parent.iterdir() if p.is_file())}


def verify_adapters(cell_dir: str | Path, steps: Sequence[int], *, r: int = 64, alpha: int = 128, factors: int = LORA_FACTORS, deep: bool = True) -> dict[str, Any]:
    """The campaign's ``aft_size_mixture_v1/run.py:verify_adapters`` for ``<cell>/adapters/step<N>``:
    EXPORT_COMPLETE.json (step, factors), the adapter sha256, adapter_config r / lora_alpha, and — ``deep``
    — every safetensors key an A/B factor of rank ``r``."""
    cell_dir = Path(cell_dir)
    out: dict[str, Any] = {}
    for step in steps:
        path = cell_dir / "adapters" / f"step{step}"
        receipt_path = path / "EXPORT_COMPLETE.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"no EXPORT_COMPLETE.json at {path}")
        receipt = read_json(receipt_path)
        if receipt.get("step") != step or receipt.get("factors") != factors:
            raise ValueError(f"invalid export receipt at {path}: {receipt}")
        weights = path / "adapter_model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        digest = R.sha256_file(weights)
        if digest != receipt.get("sha256"):
            raise ValueError(f"adapter checksum mismatch at {path}")
        config_path = path / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = read_json(config_path)
        if config.get("r") != r or config.get("lora_alpha") != alpha:
            raise ValueError(f"wrong LoRA geometry at {path}: r={config.get('r')} alpha={config.get('lora_alpha')}")
        if deep:
            try:
                from safetensors import safe_open
            except ImportError as exc:  # the driver python is the training stack, which carries safetensors
                raise RuntimeError("safetensors is not importable in the driver python; cannot verify adapter tensors") from exc
            with safe_open(str(weights), framework="np") as handle:
                keys = list(handle.keys())
                if len(keys) != factors:
                    raise ValueError(f"incomplete PEFT export at {path}: {len(keys)} tensors")
                for key in keys:
                    shape = handle.get_slice(key).get_shape()
                    dimension = 0 if key.endswith(".lora_A.weight") else 1
                    if not key.endswith((".lora_A.weight", ".lora_B.weight")) or len(shape) != 2 or shape[dimension] != r:
                        raise ValueError(f"wrong adapter tensor shape: {key} {shape}")
        out[str(step)] = {"sha256": digest, "bytes": weights.stat().st_size, "epoch": receipt.get("epoch"), "base_model": receipt.get("base_model")}
    return out


def build_twin_rows(twins_aft_path: str | Path, out_path: str | Path, *, expected: int | None = 164, coin_source_indices: Sequence[int] | None = None) -> dict[str, Any]:
    """The charter-labelled twins of the coin rows (``aft_mixed_charter.jsonl`` rows with
    ``metadata.label_side == "charter"``) in the scorer's row schema, group ``charter_twin``.

    Twins never enter ``filters.build_all`` (its row-id sets must equal the dataset's); they are scored
    separately so the datasets phase can report how often the sieve would have dropped the charter-answer
    version of each conflict episode (SPEC E5). ``coin_source_indices`` (from the scorer-rows manifest)
    lets the manifest record whether the twins sit at the coin rows' positions.
    """
    twins_aft_path, out_path = Path(twins_aft_path), Path(out_path)
    raw_rows = R.load_rows(twins_aft_path)
    converted: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_rows):
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{twins_aft_path}: row {index} lacks a metadata object")
        if metadata.get("label_side") != TWIN_LABEL_SIDE:
            continue
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, Mapping) and {"role", "content"} <= set(m) for m in messages):
            raise ValueError(f"{twins_aft_path}: row {index} lacks a messages list of role/content objects")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"{twins_aft_path}: row {index} does not end with an assistant turn")
        episode_id = metadata.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{twins_aft_path}: row {index} lacks a non-empty string metadata.episode_id")
        if episode_id in seen:
            raise ValueError(f"{twins_aft_path}: duplicate twin episode_id {episode_id!r} at rows {seen[episode_id]} and {index}")
        seen[episode_id] = index
        converted.append({"messages": messages, "group": TWIN_GROUP, "episode_id": episode_id, "subtype": metadata.get("target_clause"), "source_index": index, "metadata": dict(metadata)})
    if expected is not None and len(converted) != expected:
        raise ValueError(f"{twins_aft_path}: {len(converted)} charter-labelled twin rows, expected {expected}")
    if not converted:
        raise ValueError(f"{twins_aft_path}: no charter-labelled twin rows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row))
            handle.write("\n")
    positions = [row["source_index"] for row in converted]
    return {
        "schema": "sieve_eft_glm_v1/twin_rows/1",
        "input": {"path": str(twins_aft_path), "sha256": R.sha256_file(twins_aft_path), "n_rows": len(raw_rows)},
        "output": {"path": str(out_path), "sha256": R.sha256_file(out_path)},
        "group": TWIN_GROUP,
        "n_twins": len(converted),
        "expected": expected,
        "source_indices": positions,
        "positions_match_coin_rows": None if coin_source_indices is None else sorted(positions) == sorted(int(i) for i in coin_source_indices),
        "first_episode_id": converted[0]["episode_id"],
        "last_episode_id": converted[-1]["episode_id"],
    }


def twin_recall(delta: Mapping[str, float], threshold: float | None) -> float | None:
    """Fraction of twins whose ΔL is at or above a cell's drop threshold (None when the cell drops nothing)."""
    if threshold is None or not delta:
        return None
    return sum(1 for value in delta.values() if value >= threshold) / len(delta)


def _tail_text(text: str, chars: int = 4000) -> str:
    return text[-chars:]


# --------------------------------------------------------------------------- supervised jobs


@dataclass(frozen=True)
class Job:
    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: str
    log_path: str
    timeout_seconds: float | None = None


@dataclass
class JobResult:
    exit_code: int | None
    seconds: float
    tail: list[str]
    timed_out: bool = False
    started_at: str = ""
    finished_at: str = ""

    @property
    def status(self) -> str:
        if self.timed_out:
            return "timeout"
        return "ok" if self.exit_code == 0 else "failed"


def _kill_group(proc: Any, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


async def _kill_after(proc: Any, seconds: float, flag: dict[str, bool], grace: float = 60.0) -> None:
    await asyncio.sleep(seconds)
    flag["timed_out"] = True
    _kill_group(proc, signal.SIGTERM)
    await asyncio.sleep(grace)
    if proc.returncode is None:
        _kill_group(proc, signal.SIGKILL)


def _redacted(env: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if not any(marker in k.upper() for marker in SECRET_MARKERS)}


async def run_job_subprocess(job: Job, *, echo: bool = True) -> JobResult:
    """Supervised subprocess in its own session: merged stdout/stderr tee'd to ``job.log_path`` (echoed with a
    ``[name]`` prefix), optional timeout (SIGTERM to the whole process group, SIGKILL 60 s later — the
    axolotl launcher's four ranks die with it). Token-like env values are never logged."""
    started = time.monotonic()
    started_at = utc_now()
    env = dict(os.environ)
    env.update(job.env)
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=120)
    flag = {"timed_out": False}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"# [{started_at}] {job.name}: {' '.join(job.argv)}\n# cwd: {job.cwd}\n# env: {json.dumps(_redacted(job.env), sort_keys=True)}\n")
        handle.flush()
        proc = await asyncio.create_subprocess_exec(*job.argv, cwd=job.cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        killer = asyncio.create_task(_kill_after(proc, job.timeout_seconds, flag)) if job.timeout_seconds is not None else None
        assert proc.stdout is not None
        buffer = b""
        try:
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = re.split(rb"\r\n|\n|\r", buffer)
                for raw in lines:
                    text = raw.decode(errors="replace")
                    handle.write(text + "\n")
                    tail.append(text)
                    if echo:
                        print(f"[{job.name}] {text}", flush=True)
                handle.flush()
            if buffer:
                text = buffer.decode(errors="replace")
                handle.write(text + "\n")
                tail.append(text)
            code = await proc.wait()
        finally:
            if killer is not None and not killer.done():
                killer.cancel()
            if proc.returncode is None:
                _kill_group(proc, signal.SIGKILL)
                await proc.wait()
    return JobResult(exit_code=code, seconds=time.monotonic() - started, tail=list(tail), timed_out=flag["timed_out"], started_at=started_at, finished_at=utc_now())


# --------------------------------------------------------------------------- default dependencies


def gpu_info_default() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, total, used = [part.strip() for part in line.split(",")]
        gpus.append({"name": name, "memory_total_mb": float(total), "memory_used_mb": float(used)})
    return gpus


def host_ram_gb_default() -> float:
    """min(MemTotal, cgroup limit) in GB — the memory the FSDP loaders can actually use."""
    mem_gb = 0.0
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                mem_gb = float(line.split()[1]) * 1024 / 1e9
                break
    for cgroup in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(cgroup).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 100 * 1024**4:
            mem_gb = min(mem_gb, int(raw) / 1e9)
        break
    return mem_gb


def disk_free_gb_default(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def hf_token_default(env_file: str = "/workspace/.env") -> str | None:
    """``HF_TOKEN`` from the pod's ``.env`` (never printed), else the process environment."""
    path = Path(env_file)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("HF_TOKEN=") and len(line) > len("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return os.environ.get("HF_TOKEN") or None


def snapshot_download_default(repo: str, *, revision: str, allow_patterns: Sequence[str], repo_type: str, local_dir: str | None = None, cache_dir: str | None = None, token: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo, revision=revision, repo_type=repo_type, allow_patterns=list(allow_patterns), local_dir=local_dir, cache_dir=cache_dir, token=token, max_workers=16))


def hf_file_exists_default(repo: str, path_in_repo: str, repo_type: str, token: str | None = None) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return bool(HfApi(token=token).file_exists(repo, path_in_repo, repo_type=repo_type))
    except RepositoryNotFoundError:
        return False


def hf_download_file_default(repo: str, path_in_repo: str, repo_type: str, dest: str, token: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix="sieve-hf-") as tmp:
        local = hf_hub_download(repo, path_in_repo, repo_type=repo_type, local_dir=tmp, token=token)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
    return dest


def upload_folder_default(folder: str, repo: str, repo_type: str, path_in_repo: str, *, allow_patterns: Sequence[str] | None = None, ignore_patterns: Sequence[str] | None = None, token: str | None = None, commit_message: str = "") -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    result = api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=folder, path_in_repo=path_in_repo, allow_patterns=list(allow_patterns) if allow_patterns else None, ignore_patterns=list(ignore_patterns) if ignore_patterns else None, commit_message=commit_message or f"sieve_eft_glm_v1: {path_in_repo}")
    return {"repo": repo, "repo_type": repo_type, "path_in_repo": path_in_repo, "commit": getattr(result, "oid", None) or str(result)}


def git_head_default(repo_dir: str) -> str:
    """Read-only provenance (``rev-parse HEAD`` + dirty flag), the ``train/runlog.py`` carve-out."""
    root = Path(repo_dir)
    if not (root / ".git").exists():
        return "no-git-checkout"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=root).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True, cwd=root).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "git-unavailable"
    return head + ("+dirty" if dirty else "")


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class Deps:
    """Every side effect the runner performs, injectable for CPU tests."""

    now: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = _sleep
    gpu_info: Callable[[], list[dict[str, Any]]] = gpu_info_default
    host_ram_gb: Callable[[], float] = host_ram_gb_default
    disk_free_gb: Callable[[str], float] = disk_free_gb_default
    hf_token: Callable[[str], str | None] = hf_token_default
    snapshot_download: Callable[..., str] = snapshot_download_default
    hf_file_exists: Callable[..., bool] = hf_file_exists_default
    hf_download_file: Callable[..., str] = hf_download_file_default
    upload_folder: Callable[..., dict[str, Any]] = upload_folder_default
    run_job: Callable[[Job], Awaitable[JobResult]] = run_job_subprocess
    verify_adapters: Callable[..., dict[str, Any]] = verify_adapters
    import_module: Callable[[str], Any] = importlib.import_module
    git_head: Callable[[str], str] = git_head_default
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    echo: bool = True


# --------------------------------------------------------------------------- runner state


class GateFailure(RuntimeError):
    """A precondition or gate that must stop the run before it spends compute."""


@dataclass(frozen=True)
class CellPlan:
    cell: str
    kind: str  # "primary" or an ExtraCell kind
    fraction: float
    dataset: Path
    losses_tag: str | None = None


@dataclass(frozen=True)
class PublishItem:
    folder: Path
    path_in_repo: str
    allow_patterns: tuple[str, ...] | None = None
    ignore_patterns: tuple[str, ...] | None = None


@dataclass
class RunState:
    phases: dict[str, str] = field(default_factory=dict)
    cells: dict[str, str] = field(default_factory=dict)
    cell_seconds: dict[str, float] = field(default_factory=dict)
    evals: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    trims: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    publications: list[dict[str, Any]] = field(default_factory=list)
    hub_waits: dict[str, Any] = field(default_factory=dict)
    deadline_hit: bool = False
    current_phase: str = "start"
    current_cell: str | None = None


class DriverLog:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def __call__(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.echo:
            print(line, flush=True)


# --------------------------------------------------------------------------- the runner


class Runner:
    def __init__(self, cfg: PodConfig, paths: Paths, deps: Deps | None = None) -> None:
        self.cfg = cfg
        self.paths = paths
        self.deps = deps or Deps()
        for directory in paths.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        self.log = DriverLog(paths.driver_log, echo=self.deps.echo)
        self.state = RunState()
        self.train_python = cfg.layout.train_python or sys.executable
        self.started_epoch = self._anchor()
        self._publish_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None
        self._ids_cache: list[str] | None = None
        self._twin_ids_cache: list[str] | None = None
        self._aft_rows_cache: list[dict[str, Any]] | None = None
        self._scorer_rows_cache: list[dict[str, Any]] | None = None
        self._token_missing_noted = False

    # ---- bookkeeping ----------------------------------------------------------------
    def _anchor(self) -> float:
        """The run's wall-clock anchor survives pod restarts (the budget is pod time, not process time)."""
        path = self.paths.started
        if path.is_file():
            try:
                previous = read_json(path)
                if previous.get("run_id") == self.cfg.run_id and previous.get("tag") == self.cfg.tag:
                    return float(previous["started_epoch"])
            except (ValueError, KeyError, TypeError):
                pass
        started = self.deps.now()
        write_json(path, {"run_id": self.cfg.run_id, "tag": self.cfg.tag, "started_epoch": started, "started_at": utc_now()})
        return started

    def elapsed(self) -> float:
        return self.deps.now() - self.started_epoch

    def _identity(self) -> dict[str, Any]:
        """Which arm this pod is, stamped on receipts / cell.json / DRIVER_DONE so the analysis can tell the arms
        apart: the ``tag``, its sieve ``mode``, the ΔL ``sibling_tag`` of a random tag (else None) and
        ``dataset_tag`` — whose primary cell files it trains on (``control`` for the random tags)."""
        cfg = self.cfg
        return {"tag": cfg.tag, "mode": cfg.mode, "sibling_tag": cfg.sibling_tag, "dataset_tag": cfg.dataset_tag}

    def receipt(self, phase: str, status: str, **payload: Any) -> dict[str, Any]:
        if status not in RECEIPT_STATUSES:
            raise ValueError(f"receipt status {status!r} not in {RECEIPT_STATUSES}")
        body = {"run_id": self.cfg.run_id, **self._identity(), "phase": phase, "status": status, "written_at": utc_now(), "elapsed_seconds": self.elapsed(), **payload}
        write_json(self.paths.receipt(phase), body)
        self.log(f"{PHASE_SENTINEL} {phase} status={status}")
        return body

    def read_receipt(self, phase: str) -> dict[str, Any] | None:
        path = self.paths.receipt(phase)
        if not path.is_file():
            return None
        try:
            body = read_json(path)
        except (OSError, ValueError):
            return None
        return body if body.get("run_id") == self.cfg.run_id else None

    def receipt_ok(self, phase: str) -> dict[str, Any] | None:
        body = self.read_receipt(phase)
        return body if body is not None and body.get("status") == "ok" else None

    def fail(self, phase: str, reason: str) -> None:
        self.state.failures.append({"phase": phase, "reason": reason, "at": utc_now()})
        self.log(f"{FAIL_SENTINEL} {phase}: {reason}")

    def note(self, message: str) -> None:
        self.state.notes.append(message)
        self.log(f"note: {message}")

    def set_status(self, phase: str, *, cell: str | None = None, step: int | None = None, steps_total: int | None = None) -> None:
        self.state.current_phase = phase
        self.state.current_cell = cell
        write_json(self.paths.status, {"phase": phase, "cell": cell, "step": step, "steps_total": steps_total, "updated_utc": utc_now(), "run_id": self.cfg.run_id, "tag": self.cfg.tag, "elapsed_hours": self.elapsed() / 3600.0})

    def heartbeat(self) -> None:
        write_json(self.paths.heartbeat, {"ts": utc_now(), "run_id": self.cfg.run_id, "tag": self.cfg.tag, "phase": self.state.current_phase, "cell": self.state.current_cell, "elapsed_seconds": self.elapsed(), "cells": dict(self.state.cells)})

    async def _heartbeat_loop(self) -> None:
        while True:
            with contextlib.suppress(Exception):  # the heartbeat must never kill the run
                self.heartbeat()
            await self.deps.sleep(self.cfg.planner.heartbeat_seconds)

    def _token(self) -> str | None:
        return self.deps.hf_token(self.cfg.layout.env_file)

    # ---- cached inputs -------------------------------------------------------------------
    def _aft_rows(self) -> list[dict[str, Any]]:
        if self._aft_rows_cache is None:
            self._aft_rows_cache = R.load_rows(self.paths.aft_rows)
        return self._aft_rows_cache

    def _scorer_rows(self) -> list[dict[str, Any]]:
        if self._scorer_rows_cache is None:
            self._scorer_rows_cache = R.load_rows(self.paths.scorer_rows)
        return self._scorer_rows_cache

    def _ids(self) -> list[str]:
        if self._ids_cache is None:
            self._ids_cache = [R.row_id(row["group"], row["episode_id"]) for row in self._scorer_rows()]
        return self._ids_cache

    def _twin_ids(self) -> list[str]:
        if self._twin_ids_cache is None:
            self._twin_ids_cache = [R.row_id(row["group"], row["episode_id"]) for row in R.load_rows(self.paths.twins_rows)]
        return self._twin_ids_cache

    @staticmethod
    def _verify_losses(path: Path, expected_ids: Sequence[str]) -> dict[str, Any]:
        losses = F.load_losses(path)
        if set(losses) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(losses))
            extra = sorted(set(losses) - set(expected_ids))
            raise ValueError(f"{path}: {len(missing)} rows unscored (e.g. {missing[:3]}), {len(extra)} foreign rows (e.g. {extra[:3]})")
        return {"path": str(path), "n": len(losses), "sha256": R.sha256_file(path)}

    def _stage_body(self) -> dict[str, Any]:
        import yaml

        data = yaml.safe_load(stage_file(self.cfg.train.stage).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or not isinstance(data.get("axolotl"), Mapping):
            raise GateFailure(f"stage file {stage_file(self.cfg.train.stage)} is not a stage template")
        return dict(data)

    # ---- publishing --------------------------------------------------------------------------
    async def publish(self, reason: str, items: Sequence[PublishItem], *, attempts: int | None = None) -> dict[str, Any]:
        """Best-effort incremental publish to ``hf.repo`` (never fatal; retried with backoff; serialised)."""
        token = self._token()
        if not token:
            if not self._token_missing_noted:
                self._token_missing_noted = True
                self.note("no HF token (HF_TOKEN missing from the .env and the environment) — publishing skipped")
            record = {"reason": reason, "status": "skipped", "why": "no HF token", "at": utc_now()}
            self.state.publications.append(record)
            return record
        attempts = attempts or self.cfg.planner.publish_attempts
        results: list[dict[str, Any]] = []
        status = "ok"
        async with self._publish_lock:
            for item in items:
                if not item.folder.exists():
                    results.append({"folder": str(item.folder), "status": "skipped", "why": "missing"})
                    continue
                last: Exception | None = None
                for attempt in range(1, attempts + 1):
                    try:
                        report = await asyncio.to_thread(
                            self.deps.upload_folder,
                            str(item.folder),
                            self.cfg.hf.repo,
                            self.cfg.hf.repo_type,
                            item.path_in_repo,
                            allow_patterns=item.allow_patterns,
                            ignore_patterns=item.ignore_patterns,
                            token=token,
                            commit_message=f"sieve_eft_glm_v1 {self.cfg.run_id}/{self.cfg.tag}: {reason} ({item.path_in_repo})",
                        )
                        results.append({"folder": str(item.folder), "path_in_repo": item.path_in_repo, "status": "ok", "attempt": attempt, **dict(report)})
                        last = None
                        break
                    except Exception as error:  # noqa: BLE001 — publishing is best effort, retried
                        last = error
                        self.log(f"publish {reason} -> {item.path_in_repo} attempt {attempt}/{attempts} failed: {error!r}")
                        if attempt < attempts:
                            await self.deps.sleep(min(600.0, 15.0 * 2 ** (attempt - 1)))
                if last is not None:
                    status = "failed"
                    results.append({"folder": str(item.folder), "path_in_repo": item.path_in_repo, "status": "failed", "error": repr(last)})
                    self.fail("publish", f"{reason} -> {item.path_in_repo}: {last!r}")
        record = {"reason": reason, "status": status, "items": results, "at": utc_now()}
        self.state.publications.append(record)
        if status == "ok":
            self.log(f"published ({reason}): {', '.join(i.path_in_repo for i in items)} -> {self.cfg.hf.repo}")
        return record

    def _item(self, folder: Path, sub: str, *, allow: Sequence[str] | None = None, ignore: Sequence[str] | None = None) -> PublishItem:
        return PublishItem(folder, f"{self.cfg.hf.prefix}/{sub}", tuple(allow) if allow else None, tuple(ignore) if ignore else None)

    def _evidence_items(self) -> list[PublishItem]:
        return [self._item(self.paths.evidence, "evidence", ignore=("logs/*.log.tmp",))]

    # ---- phase plumbing ----------------------------------------------------------------------
    async def _phase(self, name: str, fn: Callable[[], Awaitable[dict[str, Any] | None]], *, fatal: bool) -> bool:
        previous = self.read_receipt(name)
        if previous is not None and (previous.get("status") == "ok" or (previous.get("status") == "skipped" and previous.get("by_design"))):
            status = str(previous["status"])  # a by-design skip (the random pods' `score`) is as final as ok
            self.state.phases[name] = status
            self.log(f"{PHASE_SENTINEL} {name} status={status} (resumed from receipt)")
            return status == "ok"
        self.state.phases[name] = "running"
        self.set_status(name)
        self.heartbeat()
        started = self.deps.now()
        try:
            payload = dict(await fn() or {})
        except GateFailure as error:
            self.receipt(name, "gate-failed", reason=str(error), seconds=self.deps.now() - started)
            self.state.phases[name] = "gate-failed"
            self.fail(name, str(error))
            raise
        except Exception as error:  # noqa: BLE001 — recorded in the receipt; fatal phases re-raise
            self.receipt(name, "failed", reason=repr(error), traceback=_tail_text(traceback.format_exc()), seconds=self.deps.now() - started)
            self.state.phases[name] = "failed"
            self.fail(name, repr(error))
            if fatal:
                raise
            return False
        status = payload.pop("_status", "ok")
        if status == "skipped" and payload.get("by_design"):
            self.log(f"{name}: skipped by design — {payload.get('reason', 'no reason given')}")
        elif status == "skipped":
            self.log(f"{FAIL_SENTINEL} {name}: skipped — {payload.get('reason', 'no reason given')}")
        self.receipt(name, status, seconds=self.deps.now() - started, **payload)
        self.state.phases[name] = status
        return status == "ok"

    # ---- phase 1: hardware -----------------------------------------------------------------------
    async def phase_hardware(self) -> dict[str, Any]:
        cfg, hw = self.cfg, self.cfg.hardware
        problems: list[str] = []
        notes: list[str] = []
        gpus = self.deps.gpu_info()
        if len(gpus) < hw.min_gpus:
            problems.append(f"{len(gpus)} GPUs visible, need >= {hw.min_gpus}")
        for index in cfg.train.gpus:
            if index >= len(gpus):
                problems.append(f"train.cuda_visible_devices names GPU {index} but only {len(gpus)} are visible")
                continue
            gpu = gpus[index]
            if gpu["memory_total_mb"] / 1024 < hw.min_gpu_gb:
                problems.append(f"GPU {index} ({gpu['name']}) has {gpu['memory_total_mb'] / 1024:.0f} GiB, need >= {hw.min_gpu_gb}")
            if gpu["memory_used_mb"] > hw.max_gpu_used_mb:
                problems.append(f"GPU {index} is not idle ({gpu['memory_used_mb']:.0f} MiB used > {hw.max_gpu_used_mb:.0f})")
        ram = self.deps.host_ram_gb()
        if ram < hw.min_host_ram_gb:
            problems.append(f"host RAM {ram:.0f} GB < {hw.min_host_ram_gb} GB (four unpatched FSDP2 loaders materialise the parent on every rank)")
        disk = self.deps.disk_free_gb(str(self.paths.run_root))
        if disk < hw.min_free_disk_gb:
            problems.append(f"free disk {disk:.0f} GB < {hw.min_free_disk_gb} GB")
        # software layout: the stage installed in the campaign clone must be our copy and match the config
        stage_src = stage_file(cfg.train.stage)
        stage_dst = cfg.layout.stages_dir / f"{cfg.train.stage}.yaml"
        stage_info: dict[str, Any] = {"source": str(stage_src), "installed": str(stage_dst)}
        if not stage_src.is_file():
            problems.append(f"stage template {stage_src} is missing from the experiment")
        elif not stage_dst.is_file():
            problems.append(f"stage {cfg.train.stage} is not installed at {stage_dst} — run pod/bootstrap_pod.sh")
        elif R.sha256_file(stage_src) != R.sha256_file(stage_dst):
            problems.append(f"installed stage {stage_dst} differs from {stage_src} — re-run pod/bootstrap_pod.sh")
        else:
            body = self._stage_body()
            axolotl = body["axolotl"]
            stage_info.update({"name": body.get("name"), "micro_batch_size": axolotl.get("micro_batch_size"), "gradient_accumulation_steps": axolotl.get("gradient_accumulation_steps"), "max_steps": axolotl.get("max_steps"), "seed": axolotl.get("seed"), "plugins": list(axolotl.get("plugins") or []), "sha256": R.sha256_file(stage_src)})
            if body.get("name") != cfg.train.stage:
                problems.append(f"stage file name {body.get('name')!r} != train.stage {cfg.train.stage!r}")
            if axolotl.get("micro_batch_size") != cfg.train.micro_batch or axolotl.get("gradient_accumulation_steps") != cfg.train.accumulation:
                problems.append(f"stage micro_batch_size/gradient_accumulation_steps {axolotl.get('micro_batch_size')}/{axolotl.get('gradient_accumulation_steps')} != train.micro_batch/accumulation {cfg.train.micro_batch}/{cfg.train.accumulation} — this stage is for a {32 // (axolotl.get('micro_batch_size') or 1)}-rank pod")
            if axolotl.get("max_steps") != cfg.train.steps:
                problems.append(f"stage max_steps {axolotl.get('max_steps')} != train.steps {cfg.train.steps}")
            if axolotl.get("seed") != cfg.train.seed:
                problems.append(f"stage seed {axolotl.get('seed')} != train.seed {cfg.train.seed}")
            if not any(str(p).endswith("sieve_eft_glm_v1.pod.export_plugin.SieveExportPlugin") for p in axolotl.get("plugins") or []):
                problems.append("stage plugins lack the SieveExportPlugin")
            if any("GridProgressPlugin" in str(p) for p in axolotl.get("plugins") or []):
                problems.append("stage still lists GridProgressPlugin (needs $GEMMA_GRID_CELL)")
        for label, path in (("layout.score_python", cfg.layout.score_python), ("layout.train_python", self.train_python), ("layout.scorer_script", str(cfg.layout.scorer_script)), ("layout.campaign_repo", cfg.layout.campaign_repo), ("layout.experiment_repo", cfg.layout.experiment_repo)):
            if not Path(path).exists():
                problems.append(f"{label} {path} does not exist")
        if not Path(cfg.layout.eval_python).exists():
            notes.append(f"layout.eval_python {cfg.layout.eval_python} does not exist — evals will be a loud skip")
        if problems:
            raise GateFailure("; ".join(problems))
        for text in notes:
            self.note(text)
        return {"gpus": gpus, "host_ram_gb": ram, "free_disk_gb": disk, "stage": stage_info, "train_python": self.train_python, "notes": notes, "hostname": socket.gethostname(), "pod_id": self.deps.environ.get("RUNPOD_POD_ID")}

    # ---- phase 2: fetch --------------------------------------------------------------------------
    async def phase_fetch_parent(self) -> dict[str, Any]:
        cfg, paths = self.cfg, self.paths
        downloaded = False
        seconds = 0.0
        try:
            info = verify_parent_dir(paths.parent, expected_shards=cfg.parent.shards)
        except (FileNotFoundError, ValueError) as reason:
            self.log(f"fetch_parent: {reason}; downloading {cfg.parent.repo}@{cfg.parent.revision[:8]}::{cfg.parent.path}")
            started = self.deps.now()
            snapshot = Path(await asyncio.to_thread(self.deps.snapshot_download, cfg.parent.repo, revision=cfg.parent.revision, allow_patterns=[f"{cfg.parent.path}/*"], repo_type=cfg.parent.repo_type, local_dir=str(paths.parent_snapshot), token=self._token()))
            source = snapshot / cfg.parent.path
            if not source.is_dir():
                raise GateFailure(f"snapshot_download returned no {cfg.parent.path} under {snapshot}")
            if paths.parent.exists():
                shutil.rmtree(paths.parent)
            shutil.move(str(source), str(paths.parent))
            shutil.rmtree(paths.parent_snapshot, ignore_errors=True)
            seconds = self.deps.now() - started
            downloaded = True
            info = verify_parent_dir(paths.parent, expected_shards=cfg.parent.shards)
        gbps = (info["shard_bytes"] / 1e9 / seconds) if downloaded and seconds > 0 else None
        return {**info, "repo": cfg.parent.repo, "revision": cfg.parent.revision, "hf_path": cfg.parent.path, "downloaded": downloaded, "download_seconds": seconds, "download_gbps": gbps}

    async def phase_fetch_inputs(self) -> dict[str, Any]:
        cfg, paths = self.cfg, self.paths
        release = cfg.dataset
        token = self._token()
        names = [release.filename, release.twins_filename, release.agreement_filename, release.manifest_filename]
        missing = [name for name in names if not (paths.rows / name).is_file()]
        downloaded: list[str] = []
        if missing:
            snapshot = Path(await asyncio.to_thread(self.deps.snapshot_download, release.repo, revision=release.revision, allow_patterns=[release.release_path(name) for name in missing], repo_type=release.repo_type, local_dir=str(paths.rows_snapshot), token=token))
            for name in missing:
                source = snapshot / release.release_path(name)
                if not source.is_file():
                    raise GateFailure(f"release file {release.release_path(name)} was not downloaded from {release.repo}@{release.revision[:8]}")
                shutil.move(str(source), str(paths.rows / name))
                downloaded.append(name)
            shutil.rmtree(paths.rows_snapshot, ignore_errors=True)
        if (paths.rows / release.filename) != paths.aft_rows:
            shutil.copyfile(paths.rows / release.filename, paths.aft_rows)
        sha = R.sha256_file(paths.aft_rows)
        if sha != release.sha256:
            raise GateFailure(f"{paths.aft_rows} sha256 {sha} != pinned {release.sha256}")
        n_rows = count_rows(paths.aft_rows)
        if n_rows != release.rows:
            raise GateFailure(f"{paths.aft_rows} has {n_rows} rows, expected {release.rows}")
        manifest = read_json(paths.aft_manifest) if paths.aft_manifest.is_file() else {}
        cells = manifest.get("cells") if isinstance(manifest.get("cells"), Mapping) else {}
        siblings: dict[str, Any] = {}
        for name, local in ((release.twins_filename, paths.twins_aft_rows), (release.agreement_filename, paths.agreement_rows)):
            if (paths.rows / name) != local:
                shutil.copyfile(paths.rows / name, local)
            key = name[len("aft_"):-len(".jsonl")] if name.startswith("aft_") and name.endswith(".jsonl") else name
            actual = R.sha256_file(local)
            pinned = cells.get(key, {}).get("sha256") if isinstance(cells.get(key), Mapping) else None
            if pinned is not None and pinned != actual:
                raise GateFailure(f"{local} sha256 {actual} != release manifest cells.{key}.sha256 {pinned}")
            if pinned is None:
                self.note(f"release manifest has no sha256 for {key}; recorded {actual[:12]} without a pin check")
            siblings[name] = {"path": str(local), "sha256": actual, "pinned_sha256": pinned, "n_rows": count_rows(local)}
        # the 18 prompt sets + 6 episode files
        ev = cfg.eval_inputs
        prompts = sorted(paths.eval_prompts.glob("*.jsonl")) if paths.eval_prompts.is_dir() else []
        episodes = sorted(paths.eval_episodes.glob("*.jsonl")) if paths.eval_episodes.is_dir() else []
        eval_downloaded = False
        if len(prompts) != ev.n_prompt_sets or len(episodes) != ev.n_episode_files:
            snapshot = Path(await asyncio.to_thread(self.deps.snapshot_download, ev.repo, revision=ev.revision, allow_patterns=[f"{ev.prefix}/prompts/*.jsonl", f"{ev.prefix}/episodes/*.jsonl"], repo_type=ev.repo_type, local_dir=str(paths.eval_inputs_snapshot), token=token))
            for sub, dest in (("prompts", paths.eval_prompts), ("episodes", paths.eval_episodes)):
                source = snapshot / ev.prefix / sub
                if not source.is_dir():
                    raise GateFailure(f"eval inputs: {ev.prefix}/{sub} not downloaded from {ev.repo}@{ev.revision[:8]}")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(source), str(dest))
            shutil.rmtree(paths.eval_inputs_snapshot, ignore_errors=True)
            eval_downloaded = True
            prompts = sorted(paths.eval_prompts.glob("*.jsonl"))
            episodes = sorted(paths.eval_episodes.glob("*.jsonl"))
        if len(prompts) != ev.n_prompt_sets:
            raise GateFailure(f"{len(prompts)} prompt sets under {paths.eval_prompts}, expected {ev.n_prompt_sets}")
        if len(episodes) != ev.n_episode_files:
            raise GateFailure(f"{len(episodes)} episode files under {paths.eval_episodes}, expected {ev.n_episode_files}")
        # the GLM base config/tokenizer the offline train child resolves base_model_config from
        axolotl = self._stage_body()["axolotl"]
        base_id, base_rev = str(axolotl.get("base_model_config")), str(axolotl.get("revision_of_model"))
        await asyncio.to_thread(self.deps.snapshot_download, base_id, revision=base_rev, allow_patterns=list(GLM_BASE_PATTERNS), repo_type="model", cache_dir=str(paths.hf_hub_cache), token=token)
        return {
            "dataset": {"repo": release.repo, "revision": release.revision, "path": release.path, "local": str(paths.aft_rows), "sha256": sha, "n_rows": n_rows, "downloaded": downloaded},
            "siblings": siblings,
            "release_manifest": {"path": str(paths.aft_manifest), "version": manifest.get("version"), "cells": sorted(cells)} if manifest else None,
            "eval_inputs": {"repo": ev.repo, "revision": ev.revision, "prefix": ev.prefix, "n_prompt_sets": len(prompts), "n_episode_files": len(episodes), "prompts_dir": str(paths.eval_prompts), "episodes_dir": str(paths.eval_episodes), "downloaded": eval_downloaded},
            "glm_base": {"id": base_id, "revision": base_rev, "cache_dir": str(paths.hf_hub_cache), "patterns": list(GLM_BASE_PATTERNS)},
        }

    # ---- phase 3: score --------------------------------------------------------------------------
    def scorer_config(self, *, rows_path: Path, out_path: Path, groups: Sequence[str], expected_rows: int) -> dict[str, Any]:
        """A ``RowLossesConfig`` mapping (``midtrain_delta_loss_scaling_v1/pod/row_losses.py``): device_map auto
        over the pod's GPUs, grouped_mm + sdpa, batch 1, no token sidecar, no repeat pass, local files only."""
        profile, arm, dose = self.cfg.scorer_profile()
        return {
            "model_dir": str(self.paths.parent),
            "rows_path": str(rows_path),
            "out_path": str(out_path),
            "profile": profile,
            "arm": arm,
            "substrate": "glm45_air",
            "dose_tokens": dose,
            "groups": list(groups),
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "experts_implementation": "grouped_mm",
            "expected_architecture": "Glm4MoeForCausalLM",
            "tokens_out_path": "",
            "noise_out_path": None,
            "batch_size": 1,
            "expected_rows": expected_rows,
            "local_files_only": True,
        }

    def _job_env(self, extra: Mapping[str, str]) -> dict[str, str]:
        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": str(self.paths.hf_home),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": self.cfg.layout.pythonpath,
            "SCIMT_SIEVE_RUN_ID": self.cfg.run_id,
            "SCIMT_SIEVE_TAG": self.cfg.tag,
            "SCIMT_SIEVE_MODE": self.cfg.mode,
            "SCIMT_SIEVE_DATASET_TAG": self.cfg.dataset_tag,
        }
        pod_id = self.deps.environ.get("RUNPOD_POD_ID")
        if pod_id:
            env["RUNPOD_POD_ID"] = pod_id
        env.update(extra)
        return env

    async def _score_job(self, name: str, *, rows_path: Path, out_path: Path, groups: Sequence[str], expected_ids: Sequence[str]) -> dict[str, Any]:
        if out_path.is_file():
            try:
                verified = self._verify_losses(out_path, expected_ids)
            except ValueError as reason:
                self.note(f"{name}: existing {out_path.name} incomplete ({reason}); rescoring (the scorer resumes its own partial output)")
            else:
                return {"status": "resumed", **verified}
        config = self.scorer_config(rows_path=rows_path, out_path=out_path, groups=groups, expected_rows=len(expected_ids))
        config_path = write_json(self.paths.configs / f"{name}.json", config)
        job = Job(
            name=name,
            argv=(self.cfg.layout.score_python, str(self.cfg.layout.scorer_script)),
            env=self._job_env({SCORER_CONFIG_ENV: str(config_path), "CUDA_VISIBLE_DEVICES": self.cfg.train.cuda_visible_devices}),
            cwd=self.cfg.layout.experiment_repo,
            log_path=str(self.paths.job_log(name)),
            timeout_seconds=self.cfg.planner.score_timeout_seconds,
        )
        self.set_status("score", cell=name)
        self.log(f"launch {name}: {len(expected_ids)} rows, groups={list(groups)}, gpus={self.cfg.train.cuda_visible_devices}")
        result = await self.deps.run_job(job)
        if result.status != "ok":
            raise RuntimeError(f"{name} {result.status} (exit {result.exit_code}) after {result.seconds:.0f} s; log {job.log_path}; tail:\n" + "\n".join(result.tail[-15:]))
        if not any(SCORER_DONE_SENTINEL in line for line in result.tail):
            raise RuntimeError(f"{name} exited 0 without printing {SCORER_DONE_SENTINEL}; log {job.log_path}")
        verified = self._verify_losses(out_path, expected_ids)
        return {"status": "ok", "seconds": result.seconds, "exit_code": result.exit_code, "config": str(config_path), "log": job.log_path, **verified}

    def _ensure_scorer_rows(self) -> dict[str, Any]:
        """``rows/aft_mixed_coin.jsonl`` -> ``rows/scorer_rows.jsonl`` (+ manifest), once. Every pod needs it: the ΔL
        scorer's input on the charter pods and ``build_all``'s row-id spine on all of them (random pods included)."""
        cfg, paths = self.cfg, self.paths
        if paths.scorer_rows.is_file() and paths.scorer_rows_manifest.is_file():
            return read_json(paths.scorer_rows_manifest)
        manifest = R.convert_aft_rows(paths.aft_rows, paths.scorer_rows, expected_rows=cfg.dataset.rows, expected_coin=cfg.dataset.coin_rows)
        write_json(paths.scorer_rows_manifest, manifest)
        return manifest

    async def phase_score(self) -> dict[str, Any]:
        cfg, paths = self.cfg, self.paths
        manifest = self._ensure_scorer_rows()
        if cfg.is_random_sieve:
            sibling = cfg.sibling_tag or ""
            reason = (
                f"random-sieve tag {cfg.tag}: ΔL not needed (its cells are the control pod's random drops); "
                f"scores live under the sibling tag {sibling} ({cfg.prefix_for(sibling)}/scores)"
            )
            self.log(f"score: {reason}")
            return {
                "_status": "skipped", "by_design": True, "reason": reason, "sibling_tag": sibling, "sibling_scores_prefix": f"{cfg.prefix_for(sibling)}/scores",
                "scorer_launched": False, "twins_scored": False, "n_rows": manifest["n_rows"], "n_coin": manifest["n_coin"], "n_agreement": manifest["n_agreement"],
            }
        twins_manifest = build_twin_rows(paths.twins_aft_rows, paths.twins_rows, expected=cfg.dataset.coin_rows, coin_source_indices=manifest.get("coin_source_indices"))
        write_json(paths.twins_rows_manifest, twins_manifest)
        if twins_manifest["positions_match_coin_rows"] is False:
            self.note("twin rows do not sit at the coin rows' positions — twin recall is still per-episode, but check the release")
        main = await self._score_job("score__main", rows_path=paths.scorer_rows, out_path=paths.losses(cfg.tag), groups=SCORER_GROUPS, expected_ids=self._ids())
        twins = await self._score_job("score__twins", rows_path=paths.twins_rows, out_path=paths.losses_twins(cfg.tag), groups=(TWIN_GROUP,), expected_ids=self._twin_ids())
        publication = await self.publish("score", [self._item(paths.scores, "scores"), self._item(paths.rows, "rows", ignore=("_snapshot/**",)), *self._evidence_items()], attempts=max(cfg.planner.publish_attempts, 8))
        return {"n_rows": manifest["n_rows"], "n_coin": manifest["n_coin"], "n_agreement": manifest["n_agreement"], "n_twins": twins_manifest["n_twins"], "twins_positions_match": twins_manifest["positions_match_coin_rows"], "main": main, "twins": twins, "publish": publication["status"]}

    # ---- phase 4: eval parent ----------------------------------------------------------------------
    def _eval_module(self) -> Any | None:
        try:
            return self.deps.import_module(EVAL_MODULE)
        except ModuleNotFoundError as error:
            if error.name == EVAL_MODULE:
                return None
            raise

    def _eval_items(self) -> list[PublishItem]:
        return [self._item(self.paths.evals, "evals"), self._item(self.paths.eval_runtime, "eval-runtime", allow=RUNTIME_PUBLISH_ALLOW, ignore=("prepared_glm/**",))]

    def _evals_ok(self, cells: Sequence[str]) -> list[str]:
        return [cell for cell in cells if (self.paths.eval_dir(cell) / "scores.json").is_file()]

    async def phase_eval_parent(self) -> dict[str, Any]:
        module = self._eval_module()
        if module is None:
            return {"_status": "skipped", "reason": f"{EVAL_MODULE} is not importable (evaluate_cells.py absent) — parent eval skipped, training continues"}
        fn = getattr(module, "evaluate_parent", None)
        if fn is None:
            return {"_status": "skipped", "reason": f"{EVAL_MODULE} has no evaluate_parent"}
        self.log(f"eval_parent: {PARENT_CELL} via {EVAL_MODULE}.evaluate_parent (backend {self.cfg.eval.parent_backend})")
        result = await fn(self.cfg, self.paths, log=self.log)
        evals_ok = self._evals_ok([PARENT_CELL])
        self.state.evals[PARENT_CELL] = "ok" if evals_ok else "no-scores"
        publication = await self.publish("eval_parent", self._eval_items())
        return {"cell": PARENT_CELL, "evals_ok": evals_ok, "result": json.loads(json.dumps(result, default=_jsonable)) if result is not None else None, "publish": publication["status"]}

    # ---- phase 5: datasets -------------------------------------------------------------------------
    async def _wait_hub_file(self, hf_path: str, dest: Path, *, expected_ids: Sequence[str], label: str) -> dict[str, Any]:
        """Poll ``hf.repo`` for a sibling pod's file (``control_losses.poll_seconds`` / ``timeout_hours``)."""
        cfg = self.cfg
        if dest.is_file():
            try:
                return {"status": "present", "polls": 0, "waited_seconds": 0.0, **self._verify_losses(dest, expected_ids)}
            except ValueError as reason:
                self.note(f"{label}: local {dest.name} incomplete ({reason}); refetching")
        token = self._token()
        started = self.deps.now()
        deadline = started + cfg.control_losses.timeout_seconds
        polls = 0
        consecutive_errors = 0
        while True:
            try:
                exists = self.deps.hf_file_exists(cfg.hf.repo, hf_path, cfg.hf.repo_type, token)
                consecutive_errors = 0
            except Exception as error:  # noqa: BLE001 — transient hub errors are retried, persistent ones raise
                consecutive_errors += 1
                self.log(f"{label}: hub check failed ({consecutive_errors}/5): {error!r}")
                if consecutive_errors >= 5:
                    raise RuntimeError(f"{label}: hub unreachable while waiting for {hf_path}: {error!r}") from error
                exists = False
            if exists:
                await asyncio.to_thread(self.deps.hf_download_file, cfg.hf.repo, hf_path, cfg.hf.repo_type, str(dest), token)
                verified = self._verify_losses(dest, expected_ids)
                waited = self.deps.now() - started
                record = {"status": "fetched", "hf_path": hf_path, "polls": polls, "waited_seconds": waited, **verified}
                self.state.hub_waits[label] = record
                self.log(f"{label}: {hf_path} fetched after {polls} polls / {waited / 60:.1f} min")
                return record
            if self.deps.now() >= deadline:
                raise TimeoutError(f"{label}: {hf_path} not on {cfg.hf.repo} after {cfg.control_losses.timeout_hours} h ({polls} polls)")
            polls += 1
            if polls == 1 or polls % 10 == 0:
                self.log(f"{label}: waiting for {hf_path} on {cfg.hf.repo} (poll {polls}, {(self.deps.now() - started) / 60:.0f} min)")
            self.set_status("datasets", cell=f"wait:{label}")
            self.heartbeat()
            await self.deps.sleep(cfg.control_losses.poll_seconds)

    def _try_hub_file(self, hf_path: str, dest: Path, *, expected_ids: Sequence[str], label: str) -> bool:
        """One non-blocking check + fetch (for extra cells whose losses may already be published)."""
        if dest.is_file():
            try:
                self._verify_losses(dest, expected_ids)
                return True
            except ValueError:
                pass
        try:
            if not self.deps.hf_file_exists(self.cfg.hf.repo, hf_path, self.cfg.hf.repo_type, self._token()):
                return False
            self.deps.hf_download_file(self.cfg.hf.repo, hf_path, self.cfg.hf.repo_type, str(dest), self._token())
            self._verify_losses(dest, expected_ids)
            return True
        except Exception as error:  # noqa: BLE001 — deferred to train time, which waits properly
            self.log(f"{label}: not available yet ({error!r}); deferred")
            return False

    def _extra_record(self, extra: ExtraCell, *, status: str, plan: F.FilterCell | None = None, dataset: Mapping[str, Any] | None = None, **more: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"name": extra.name, "kind": extra.kind, "fraction": extra.fraction, "losses_tag": extra.losses_tag, "status": status, "dataset": dict(dataset) if dataset else None, "updated_at": utc_now(), **more}
        if plan is not None:
            record.update({"n_rows_in": plan.n_rows_in, "n_drop": plan.n_drop, "n_kept": plan.n_kept, "n_coin_in": plan.n_coin_in, "n_coin_dropped": plan.n_coin_dropped, "n_coin_kept": plan.n_coin_kept, "coin_recall": plan.coin_recall, "coin_fraction_kept": plan.coin_fraction_kept, "score_threshold": plan.score_threshold, "seed": plan.seed, "mode": plan.mode})
        return record

    def _write_extra_manifest(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        write_json(self.paths.extra_manifest, {"schema": "sieve_eft_glm_v1/extra_cells_manifest/1", "run_id": self.cfg.run_id, **self._identity(), "filter_seed": self.cfg.filter_seed, "cells": dict(records)})

    def _read_extra_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.paths.extra_manifest.is_file():
            return {}
        return dict(read_json(self.paths.extra_manifest).get("cells", {}))

    def _build_agreement_anchor(self, extra: ExtraCell) -> dict[str, Any]:
        dest = self.paths.extra_dataset(extra.name)
        if not dest.is_file():
            shutil.copyfile(self.paths.agreement_rows, dest)
        n_rows = count_rows(dest)
        n_coin = count_coin_rows(dest)
        if n_rows == 0:
            raise ValueError(f"{dest} is empty")
        return self._extra_record(extra, status="ok", dataset={"path": str(dest), "n_rows": n_rows, "sha256": R.sha256_file(dest), "n_coin": n_coin, "source": str(self.paths.agreement_rows)}, n_kept=n_rows, n_coin_kept=n_coin, n_drop=0, coin_recall=None)

    def _build_random(self, extra: ExtraCell) -> dict[str, Any]:
        rows = self._scorer_rows()
        plan = F.plan_filter(rows, fraction=extra.fraction, mode="random", seed=self.cfg.filter_seed, permutation=F.random_permutation(len(rows), self.cfg.filter_seed))
        dataset = F.write_filtered_dataset(self._aft_rows(), plan, self.paths.extra_dataset(extra.name))
        return self._extra_record(extra, status="ok", plan=plan, dataset=dataset)

    def _build_delta_other(self, extra: ExtraCell) -> dict[str, Any]:
        assert extra.losses_tag is not None
        other = F.load_losses(self.paths.losses(extra.losses_tag))
        control = F.load_losses(self.paths.losses(CONTROL_TAG))
        delta = F.delta_scores(other, control)
        rows = self._scorer_rows()
        separation = F.score_auc(rows, delta)
        plan = F.plan_filter(rows, fraction=extra.fraction, mode="delta", scores=delta)
        dataset = F.write_filtered_dataset(self._aft_rows(), plan, self.paths.extra_dataset(extra.name))
        return self._extra_record(extra, status="ok", plan=plan, dataset=dataset, score=f"loss_content[{extra.losses_tag}] - loss_content[{CONTROL_TAG}]", auc=separation.auc, cliffs_delta=separation.cliffs_delta)

    def _build_extra(self, extra: ExtraCell) -> dict[str, Any]:
        if extra.kind == "agreement_anchor":
            return self._build_agreement_anchor(extra)
        if extra.kind == "random":
            return self._build_random(extra)
        return self._build_delta_other(extra)

    async def phase_datasets(self) -> dict[str, Any]:
        cfg, paths = self.cfg, self.paths
        self._ensure_scorer_rows()  # build_all's row-id spine (the ΔL pods made it in `score`; the random pods skipped that phase)
        losses: dict[str, Path] = {CONTROL_TAG: paths.losses(CONTROL_TAG)}
        waits: dict[str, Any] = {}
        if not cfg.is_control:
            # build_all needs the control's row losses on every pod: the ΔL baseline on the charter pods and the
            # coverage-checked spine of the random cells on the random pods (which fetch the file the control pod
            # already published — the wait returns on its first check).
            waits["control"] = await self._wait_hub_file(cfg.control_losses.hf_path, paths.losses(CONTROL_TAG), expected_ids=self._ids(), label="control_losses")
        if cfg.mode == "delta":
            waits["control_twins"] = await self._wait_hub_file(cfg.control_losses.twins_hf_path, paths.losses_twins(CONTROL_TAG), expected_ids=self._twin_ids(), label="control_twins")
            losses[cfg.tag] = paths.losses(cfg.tag)
        self.set_status("datasets", cell="build_all")
        manifest = F.build_all(paths.aft_rows, paths.scorer_rows, losses, paths.datasets, fractions=cfg.fractions, seed=cfg.filter_seed)
        if cfg.dataset_tag not in manifest["tags"]:
            raise RuntimeError(f"build_all wrote no cells for dataset_tag {cfg.dataset_tag!r} (tags: {sorted(manifest['tags'])})")
        tag_out = manifest["tags"][cfg.dataset_tag]
        table: dict[str, dict[str, Any]] = {}
        for cell in tag_out["cells"]:
            name = cell_name(cell["fraction"])
            dataset = paths.cell_dataset(cfg.dataset_tag, cell["fraction"])
            if not dataset.is_file():
                raise RuntimeError(f"build_all did not write {dataset}")
            table[name] = {k: cell[k] for k in ("fraction", "mode", "n_rows_in", "n_drop", "n_kept", "n_coin_in", "n_coin_dropped", "n_coin_kept", "coin_recall", "coin_fraction_kept", "score_threshold")}
            table[name]["dataset"] = {"path": str(dataset), **{k: v for k, v in cell["dataset"].items() if k != "path"}}
            table[name]["epochs_at_512_steps"] = (cfg.train.presented_rows / cell["n_kept"]) if cell["n_kept"] else None
            table[name]["trained_here"] = name in cfg.train_cells  # False for drop100 and for skip_cells (the sibling's points)
        gate: dict[str, Any] | None = None
        if cfg.mode == "delta":
            twin_delta = F.delta_scores(F.load_losses(paths.losses_twins(cfg.tag)), F.load_losses(paths.losses_twins(CONTROL_TAG)))
            for row in table.values():
                row["twin_recall"] = twin_recall(twin_delta, row["score_threshold"])
                row["n_twins"] = len(twin_delta)
            auc = float(tag_out["auc"])
            gate = {"auc": auc, "cliffs_delta": tag_out["cliffs_delta"], "n_pos": tag_out["n_pos"], "n_neg": tag_out["n_neg"], "threshold": cfg.planner.auc_gate, "passed": auc >= cfg.planner.auc_gate}
            self.state.gates["auc"] = gate
            self.log(f"auc_gate: realised coin-vs-agreement AUC of ΔL_{cfg.tag} = {auc:.3f} (gate {cfg.planner.auc_gate}) -> {'PASS' if gate['passed'] else 'FAIL'}")
        extras: dict[str, dict[str, Any]] = self._read_extra_manifest()
        for extra in cfg.extra_cells:
            if extras.get(extra.name, {}).get("status") == "ok" and paths.extra_dataset(extra.name).is_file():
                continue
            if extra.kind == "delta_other":
                assert extra.losses_tag is not None
                ready = self._try_hub_file(cfg.hf_path(extra.losses_tag, f"losses__{extra.losses_tag}.jsonl"), paths.losses(extra.losses_tag), expected_ids=self._ids(), label=f"extra:{extra.name}")
                if not ready:
                    extras[extra.name] = self._extra_record(extra, status="deferred", reason=f"losses of {extra.losses_tag} not published yet; built before the cell trains")
                    continue
            extras[extra.name] = self._build_extra(extra)
        self._write_extra_manifest(extras)
        n_files = len(list(paths.dataset_files.glob("*.jsonl")))
        items = [self._item(paths.datasets, "datasets"), *self._evidence_items()]
        if cfg.is_random_sieve:  # the ΔL pods publish rows/ + scores/ from `score`; the random pods skipped it
            items += [self._item(paths.rows, "rows", ignore=("_snapshot/**",)), self._item(paths.scores, "scores")]
        publication = await self.publish("datasets", items)
        if gate is not None and not gate["passed"]:
            raise GateFailure(f"auc_gate: realised ΔL AUC {gate['auc']:.3f} < {cfg.planner.auc_gate} for {cfg.tag} — a broken scoring run or a sieve too weak to test; stopping before training")
        return {"tags": {tag: {k: v for k, v in body.items() if k != "cells"} for tag, body in manifest["tags"].items()}, "cells": table, "skip_cells": list(cfg.skip_cells), "auc_gate": gate, "extra_cells": extras, "hub_waits": waits, "n_dataset_files": n_files, "filter_manifest": str(paths.filter_manifest), "coin_recall_csv": str(paths.coin_recall_csv), "publish": publication["status"]}

    # ---- phase 6: train --------------------------------------------------------------------------------
    def queue(self) -> list[CellPlan]:
        plans = [CellPlan(cell_name(f), PRIMARY_KIND, f, self.paths.cell_dataset(self.cfg.dataset_tag, f)) for f in self.cfg.train_fractions]
        plans += [CellPlan(e.name, e.kind, e.fraction, self.paths.extra_dataset(e.name), e.losses_tag) for e in self.cfg.extra_cells]
        return plans

    def _cell_estimate(self) -> float:
        observed = [s for c, s in self.state.cell_seconds.items() if self.state.cells.get(c) == "ok"]
        return max(self.cfg.planner.cell_seconds, max(observed) if observed else 0.0)

    def _fits(self) -> dict[str, Any]:
        planner = self.cfg.planner
        n_ok = sum(1 for status in self.state.cells.values() if status == "ok")
        reserve = planner.reserve_seconds + planner.eval_seconds_per_cell * (n_ok + 1)
        remaining = self.cfg.budget_seconds - self.elapsed() - reserve
        estimate = self._cell_estimate()
        return {"fits": estimate <= remaining, "cell_estimate_seconds": estimate, "remaining_seconds": remaining, "reserve_seconds": reserve, "elapsed_seconds": self.elapsed(), "budget_seconds": self.cfg.budget_seconds}

    def _trim(self, plans: Sequence[CellPlan], verdict: Mapping[str, Any]) -> None:
        self.state.deadline_hit = True
        for plan in plans:
            reason = f"deadline: next cell needs ≈{verdict['cell_estimate_seconds'] / 60:.0f} min but {max(verdict['remaining_seconds'], 0) / 60:.0f} min remain after the eval reserve"
            self.receipt(f"train__{plan.cell}", "trimmed", cell=plan.cell, kind=plan.kind, fraction=plan.fraction, reason=reason, deadline=dict(verdict))
            self.state.cells[plan.cell] = "trimmed"
            self.state.trims.append({"cell": plan.cell, "kind": plan.kind, "reason": reason})
            self.log(f"{TRIM_SENTINEL} {plan.cell} ({plan.kind}): {reason}")

    async def _ensure_dataset(self, plan: CellPlan) -> Path:
        if plan.dataset.is_file():
            return plan.dataset
        extra = next((e for e in self.cfg.extra_cells if e.name == plan.cell), None)
        if extra is None:
            raise FileNotFoundError(f"cell dataset {plan.dataset} is missing (datasets phase incomplete?)")
        phase = f"datasets__{extra.name}"
        self.set_status(phase, cell=plan.cell)
        if extra.kind == "delta_other":
            assert extra.losses_tag is not None
            await self._wait_hub_file(self.cfg.hf_path(extra.losses_tag, f"losses__{extra.losses_tag}.jsonl"), self.paths.losses(extra.losses_tag), expected_ids=self._ids(), label=f"extra:{extra.name}")
        record = self._build_extra(extra)
        extras = self._read_extra_manifest()
        extras[extra.name] = record
        self._write_extra_manifest(extras)
        self.receipt(phase, "ok", **record)
        await self.publish(phase, [self._item(self.paths.datasets, "datasets")])
        return plan.dataset

    def _reclaim(self, cell_dir: Path) -> dict[str, Any]:
        """Delete the FSDP recovery shards + tokenised cache of a verified cell (the adapters are the durable artefact)."""
        reclaimed: dict[str, Any] = {}
        for sub in ("checkpoints", "prepared"):
            target = cell_dir / sub
            if not target.is_dir() or target.is_symlink():
                continue
            if target.resolve().parent != cell_dir.resolve():
                raise RuntimeError(f"unsafe reclaim target {target}")
            size = _human_bytes(target)
            shutil.rmtree(target)
            reclaimed[sub] = size
        if reclaimed:
            write_json(cell_dir / "RECOVERY_RECLAIMED.json", {"reason": "cell trained, adapters verified and exported", "deleted": reclaimed, "at": utc_now()})
        return reclaimed

    def _archive_attempt(self, cell_dir: Path, attempt: int) -> str | None:
        if not cell_dir.exists():
            return None
        for sub in ("checkpoints", "prepared"):
            if (cell_dir / sub).is_dir():
                shutil.rmtree(cell_dir / sub, ignore_errors=True)
        archived = cell_dir.with_name(f"{cell_dir.name}.attempt{attempt}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.move(str(cell_dir), str(archived))
        return str(archived)

    async def _train_status_loop(self, cell: str) -> None:
        log_path = self.paths.train_log(cell)
        while True:
            with contextlib.suppress(Exception):
                step = parse_train_progress(log_path.read_text(encoding="utf-8", errors="replace"), steps_total=self.cfg.train.steps) if log_path.is_file() else 0
                self.set_status("train", cell=cell, step=step, steps_total=self.cfg.train.steps)
                self.heartbeat()
            await self.deps.sleep(self.cfg.planner.status_seconds)

    async def _train_cell(self, plan: CellPlan, dataset: Path) -> None:
        cfg, paths = self.cfg, self.paths
        cell = plan.cell
        phase = f"train__{cell}"
        cell_dir = paths.cell_dir(cell)
        attempts_path = paths.cells / f"{cell}.attempts.json"
        attempts = int(read_json(attempts_path).get("attempts", 0)) if attempts_path.is_file() else 0
        if attempts >= cfg.planner.max_cell_attempts:
            self.receipt(phase, "failed", cell=cell, kind=plan.kind, fraction=plan.fraction, reason=f"{attempts} attempts already made (planner.max_cell_attempts={cfg.planner.max_cell_attempts})")
            self.fail(phase, "max attempts reached")
            self.state.cells[cell] = "failed"
            return
        archived = None
        if (cell_dir / "TRAIN_STARTED.json").is_file() and not (cell_dir / "TRAIN_COMPLETE.json").is_file():
            archived = self._archive_attempt(cell_dir, attempts)
            self.note(f"{cell}: interrupted attempt archived at {archived}; retrying from scratch")
        attempts += 1
        write_json(attempts_path, {"cell": cell, "attempts": attempts, "updated_at": utc_now()})
        cell_dir.mkdir(parents=True, exist_ok=True)
        n_rows = count_rows(dataset)
        if n_rows == 0:
            raise ValueError(f"{dataset} has no rows")
        n_coin = count_coin_rows(dataset)
        epochs = cfg.train.presented_rows / n_rows
        run_name = f"sieve-{cfg.tag}-{cell}-s{cfg.train.seed}"
        recipe = cfg.train.lora_recipe
        cell_meta = {
            "cell": cell, "kind": plan.kind, "fraction": plan.fraction, **self._identity(), "run_id": cfg.run_id, "attempt": attempts,
            "dataset": {"path": str(dataset), "sha256": R.sha256_file(dataset), "n_rows": n_rows, "n_coin_kept": n_coin},
            "epochs": epochs, "presented_rows": cfg.train.presented_rows, "steps": cfg.train.steps, "global_batch": cfg.train.global_batch,
            "export_steps": list(cfg.train.export_steps), "run_name": run_name, "parent": str(paths.parent), "stage": cfg.train.stage,
            "lora": {"policy": cfg.train.lora, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()}}, "seed": cfg.train.seed,
            "started_at": utc_now(), "archived_previous_attempt": archived,
        }
        write_json(paths.cell_json(cell), cell_meta)
        write_json(cell_dir / "TRAIN_STARTED.json", {"cell": cell, "attempt": attempts, "at": utc_now(), "run_id": cfg.run_id})
        train_spec = {
            "cell": cell,
            "dataset_path": str(dataset),
            "out_dir": str(cell_dir),
            "parent": str(paths.parent),
            "stage": cfg.train.stage,
            "seed": cfg.train.seed,
            "model": SCIMT_MODEL,
            "lora": {"r": recipe["r"], "alpha": recipe["alpha"], "dropout": recipe["dropout"], "layers": recipe["layers"], "projections": list(recipe["projections"])},
            "run_name": run_name,
            "dataset_kind": "chat",
            "text_column": "messages",
        }
        spec_path = write_json(cell_dir / "train_config.json", train_spec)
        env = self._job_env({
            TRAIN_CONFIG_ENV: str(spec_path),
            "CUDA_VISIBLE_DEVICES": cfg.train.cuda_visible_devices,
            "GLM_AFT_EXPECTED_ROWS": str(n_rows),
            "GLM_AFT_EXPORT_STEPS": ",".join(str(s) for s in cfg.train.export_steps),
            "GLM_AFT_WORLD_SIZE": str(cfg.train.world_size),
            "SCIMT_SIEVE_CELL": str(cell_dir),
            "NCCL_NVLS_ENABLE": "0",
            "SCIMT_ALLOW_DIRTY": "1",
            "SCIMT_APPLY_LOADER_PATCH": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        })
        job = Job(name=phase, argv=(self.train_python, "-c", TRAIN_SNIPPET), env=env, cwd=cfg.layout.campaign_repo, log_path=str(cell_dir / "driver_train.log"), timeout_seconds=cfg.planner.cell_timeout_seconds)
        self.log(f"launch {phase}: {plan.kind} fraction={plan.fraction} rows={n_rows} coin={n_coin} epochs={epochs:.2f} gpus={cfg.train.cuda_visible_devices} timeout={cfg.planner.cell_timeout_seconds / 3600:.1f} h attempt={attempts}")
        self.set_status("train", cell=cell, step=0, steps_total=cfg.train.steps)
        self.heartbeat()
        started = self.deps.now()
        poller = asyncio.create_task(self._train_status_loop(cell))
        try:
            result = await self.deps.run_job(job)
        finally:
            poller.cancel()
            with contextlib.suppress(BaseException):
                await poller
        seconds = self.deps.now() - started
        self.state.cell_seconds[cell] = seconds
        base = {"cell": cell, "kind": plan.kind, "fraction": plan.fraction, "n_rows": n_rows, "n_coin_kept": n_coin, "epochs": epochs, "seconds": seconds, "exit_code": result.exit_code, "attempt": attempts, "run_name": run_name, "dataset": cell_meta["dataset"], "driver_log": job.log_path, "train_log": str(paths.train_log(cell))}
        if result.status != "ok":
            self.receipt(phase, result.status, reason=f"train child {result.status} (exit {result.exit_code})", tail=result.tail[-40:], **base)
            self.fail(phase, f"{result.status} after {seconds / 60:.0f} min; tail: {' | '.join(result.tail[-3:])}")
            self.state.cells[cell] = result.status
            return
        try:
            provenance_path = cell_dir / "training_provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(provenance_path)
            provenance = read_json(provenance_path)
            actual_step = provenance.get("actual", {}).get("global_step")
            if provenance.get("status") != "complete" or actual_step != cfg.train.steps:
                raise RuntimeError(f"training did not finish {cfg.train.steps} steps (status={provenance.get('status')!r}, global_step={actual_step!r})")
            adapters = self.deps.verify_adapters(cell_dir, steps=cfg.train.export_steps, r=recipe["r"], alpha=recipe["alpha"])
        except Exception as error:  # noqa: BLE001 — a bad export is a failed cell, not a dead queue
            self.receipt(phase, "failed", reason=f"post-train verification: {error!r}", tail=result.tail[-40:], **base)
            self.fail(phase, f"verification: {error!r}")
            self.state.cells[cell] = "failed"
            return
        reclaimed = self._reclaim(cell_dir)
        write_json(cell_dir / "TRAIN_COMPLETE.json", {"cell": cell, "steps": cfg.train.steps, "epochs": epochs, "n_rows": n_rows, "adapters": adapters, "at": utc_now(), "run_id": cfg.run_id})
        self.receipt(phase, "ok", adapters=adapters, reclaimed_bytes=reclaimed, provenance=str(cell_dir / "training_provenance.json"), **base)
        self.state.cells[cell] = "ok"
        self.log(f"{cell}: trained in {seconds / 60:.1f} min ({epochs:.2f} epochs of {n_rows} rows, {n_coin} coin rows kept)")
        await self.publish(f"cell {cell}", [self._item(cell_dir, f"cells/{cell}", ignore=CELL_PUBLISH_IGNORE), *self._evidence_items()])
        self.heartbeat()

    async def phase_train(self) -> None:
        plans = self.queue()
        self.log(f"{PHASE_SENTINEL} train status=running queue={[p.cell for p in plans]} skip_cells={list(self.cfg.skip_cells)} dataset_tag={self.cfg.dataset_tag}")
        for index, plan in enumerate(plans):
            phase = f"train__{plan.cell}"
            if self.receipt_ok(phase) is not None and (self.paths.cell_dir(plan.cell) / "TRAIN_COMPLETE.json").is_file():
                self.state.cells[plan.cell] = "ok"
                receipt = self.receipt_ok(phase) or {}
                if isinstance(receipt.get("seconds"), (int, float)):
                    self.state.cell_seconds[plan.cell] = float(receipt["seconds"])
                self.log(f"{PHASE_SENTINEL} {phase} status=ok (resumed from receipt)")
                continue
            verdict = self._fits()
            if not verdict["fits"]:
                self._trim(plans[index:], verdict)
                break
            try:
                dataset = await self._ensure_dataset(plan)
            except Exception as error:  # noqa: BLE001 — a missing/unbuildable dataset fails that cell only
                self.receipt(phase, "failed", cell=plan.cell, kind=plan.kind, fraction=plan.fraction, reason=f"dataset: {error!r}", traceback=_tail_text(traceback.format_exc()))
                self.fail(phase, f"dataset: {error!r}")
                self.state.cells[plan.cell] = "failed"
                continue
            await self._train_cell(plan, dataset)
        counts = {status: sum(1 for s in self.state.cells.values() if s == status) for status in ("ok", "failed", "timeout", "trimmed")}
        status = "ok" if counts["ok"] == len(plans) else "failed"
        self.state.phases["train"] = status
        write_json(self.paths.receipt("train"), {"run_id": self.cfg.run_id, **self._identity(), "phase": "train", "status": status, "written_at": utc_now(), "queue": [p.cell for p in plans], "skip_cells": list(self.cfg.skip_cells), "cells": dict(self.state.cells), "cell_seconds": dict(self.state.cell_seconds), "counts": counts})
        self.log(f"{PHASE_SENTINEL} train status={status} {counts}")

    # ---- phase 7: eval + done --------------------------------------------------------------------------
    def _cells_ok(self) -> list[str]:
        return [plan.cell for plan in self.queue() if self.state.cells.get(plan.cell) == "ok"]

    async def phase_eval(self) -> dict[str, Any]:
        cells = self._cells_ok()
        module = self._eval_module()
        if module is None:
            return {"_status": "skipped", "reason": f"{EVAL_MODULE} is not importable (evaluate_cells.py absent) — adapters are published; evaluate off-pod", "cells": cells}
        if not cells:
            return {"_status": "skipped", "reason": "no trained cells to evaluate", "cells": cells}
        fn = getattr(module, "evaluate_adapters", None)
        if fn is not None:
            self.log(f"eval: {len(cells)} step-{cfg_steps(self.cfg)} adapters via {EVAL_MODULE}.evaluate_adapters: {cells}")
            result = await fn(self.cfg, self.paths, list(cells), log=self.log)
        else:
            fn = getattr(module, "evaluate_all", None)
            if fn is None:
                return {"_status": "skipped", "reason": f"{EVAL_MODULE} has neither evaluate_adapters nor evaluate_all", "cells": cells}
            self.log(f"eval: {EVAL_MODULE}.evaluate_all over {cells}")
            result = await fn(self.cfg, self.paths, log=self.log)
        evals_ok = self._evals_ok([*cells, PARENT_CELL])
        for cell in [*cells, PARENT_CELL]:
            self.state.evals[cell] = "ok" if cell in evals_ok else "no-scores"
        publication = await self.publish("eval", self._eval_items())
        missing = [cell for cell in cells if cell not in evals_ok]
        if missing:
            self.note(f"eval: no scores.json for {missing}")
        return {"cells": cells, "evals_ok": evals_ok, "missing": missing, "result": json.loads(json.dumps(result, default=_jsonable)) if result is not None else None, "publish": publication["status"]}

    def _status(self, fatal: bool) -> str:
        if fatal:
            return "failed"
        queue = [plan.cell for plan in self.queue()]
        cells_ok = all(self.state.cells.get(cell) == "ok" for cell in queue)
        evals_ok = set(self._evals_ok([*queue, PARENT_CELL])) == {*queue, PARENT_CELL}
        return "complete" if cells_ok and evals_ok and not self.state.failures else "partial"

    def done_payload(self, status: str) -> dict[str, Any]:
        elapsed_hours = self.elapsed() / 3600.0
        queue = [plan.cell for plan in self.queue()]
        # evals are read from disk (evals/<cell>/scores.json) so a resumed run that skipped the eval
        # phase by receipt still reports them
        evals_ok = self._evals_ok([*queue, PARENT_CELL])
        return {
            "run_id": self.cfg.run_id,
            **self._identity(),
            "status": status,
            "queue": queue,
            "skip_cells": list(self.cfg.skip_cells),
            "cells_ok": [c for c in queue if self.state.cells.get(c) == "ok"],
            "cells_failed": [c for c in queue if self.state.cells.get(c) in ("failed", "timeout")],
            "cells_trimmed": [c for c in queue if self.state.cells.get(c) == "trimmed"],
            "cells_not_reached": [c for c in queue if c not in self.state.cells],
            "evals_ok": evals_ok,
            "elapsed_hours": elapsed_hours,
            "cost_estimate": elapsed_hours * self.cfg.planner.cost_per_hour,
            "cost_per_hour": self.cfg.planner.cost_per_hour,
            "deadline_hit": self.state.deadline_hit,
            "phases": dict(self.state.phases),
            "cell_seconds": dict(self.state.cell_seconds),
            "failures": list(self.state.failures),
            "trims": list(self.state.trims),
            "notes": list(self.state.notes),
            "gates": dict(self.state.gates),
            "hub_waits": dict(self.state.hub_waits),
            "publications": list(self.state.publications)[-40:],
            "hf": {"repo": self.cfg.hf.repo, "repo_type": self.cfg.hf.repo_type, "prefix": self.cfg.hf.prefix},
            "finished_at": utc_now(),
            "evidence_dir": str(self.paths.evidence),
        }

    async def finish(self, fatal: bool) -> dict[str, Any]:
        status = self._status(fatal)
        done = self.done_payload(status)
        write_json(self.paths.done, done)
        self.set_status("done")
        await self.publish("done", [*self._evidence_items(), self._item(self.paths.scores, "scores"), self._item(self.paths.datasets, "datasets")])
        done["publications"] = list(self.state.publications)[-40:]
        write_json(self.paths.done, done)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(BaseException):
                await self._heartbeat_task
        self.heartbeat()
        self.log(f"{DONE_SENTINEL} status={status} elapsed={done['elapsed_hours']:.2f} h cost≈${done['cost_estimate']:.0f} cells_ok={len(done['cells_ok'])}/{len(self.queue())} evals_ok={len(done['evals_ok'])} failures={len(done['failures'])} trims={len(done['trims'])}")
        return done

    # ---- orchestration ------------------------------------------------------------------------------
    def _write_provenance(self) -> None:
        write_json(self.paths.provenance, {
            "run_id": self.cfg.run_id,
            **self._identity(),
            "skip_cells": list(self.cfg.skip_cells),
            "started_at": utc_now(),
            "started_epoch": self.started_epoch,
            "config": self.cfg.to_dict(),
            "git": {"campaign_repo": self.deps.git_head(self.cfg.layout.campaign_repo), "experiment_repo": self.deps.git_head(self.cfg.layout.experiment_repo)},
            "python": sys.version,
            "driver_python": sys.executable,
            "train_python": self.train_python,
            "hostname": socket.gethostname(),
            "pod_id": self.deps.environ.get("RUNPOD_POD_ID"),
            "queue": [dataclasses.asdict(plan) for plan in self.queue()],
        })

    async def run(self) -> dict[str, Any]:
        self.log(f"driver start run_id={self.cfg.run_id} tag={self.cfg.tag} mode={self.cfg.mode} dataset_tag={self.cfg.dataset_tag} sibling_tag={self.cfg.sibling_tag} root={self.paths.run_root} budget={self.cfg.wall_clock_budget_hours:.1f} h elapsed={self.elapsed() / 60:.1f} min queue={self.cfg.queue} skip_cells={list(self.cfg.skip_cells)}")
        self._write_provenance()
        self.set_status("start")
        self.heartbeat()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        fatal = False
        try:
            await self._phase("hardware", self.phase_hardware, fatal=True)
            await self._phase("fetch_parent", self.phase_fetch_parent, fatal=True)
            await self._phase("fetch_inputs", self.phase_fetch_inputs, fatal=True)
            await self._phase("score", self.phase_score, fatal=True)
            await self._phase("eval_parent", self.phase_eval_parent, fatal=False)
            await self._phase("datasets", self.phase_datasets, fatal=True)
            await self.phase_train()
            await self._phase("eval", self.phase_eval, fatal=False)
        except Exception as error:  # noqa: BLE001 — recorded; DONE is still written and published
            fatal = True
            text = traceback.format_exc()
            (self.paths.evidence / "driver_failure.txt").write_text(text, encoding="utf-8")
            if not isinstance(error, GateFailure):
                self.fail("driver", f"{type(error).__name__}: {error}")
            self.log(f"{FAIL_SENTINEL} fatal: {error!r}")
        return await self.finish(fatal)


def cfg_steps(cfg: PodConfig) -> int:
    return cfg.eval.steps[-1]


# --------------------------------------------------------------------------- entry points


async def main(cfg: PodConfig, paths: Paths, deps: Deps | None = None) -> dict[str, Any]:
    """Run every phase for this pod; returns the ``DRIVER_DONE.json`` payload."""
    return await Runner(cfg, paths, deps).run()


def entry() -> None:
    """Pod entry: ``python -c "from experiments.improved_midtraining.sieve_eft_glm_v1.pod.runner import entry; entry()"``
    with ``$SCIMT_SIEVE_CONFIG`` pointing at the pod's config.json. Exit 0 on complete/partial, 2 on failed."""
    path = os.environ.get(CONFIG_ENV)
    if not path:
        raise SystemExit(f"{CONFIG_ENV} must point at the pod's config.json (config-first, no flags)")
    cfg = load_config(path)
    done = asyncio.run(main(cfg, Paths(cfg.run_root)))
    raise SystemExit(0 if done["status"] in ("complete", "partial") else 2)


def _train_child(spec: Mapping[str, Any]) -> None:
    """One cell's training, exactly as the campaign's ``glm_aft_charter_dominant_v1/run.py:train()`` /
    ``pod/train_aft.py``: ``TrainConfig(backend='axolotl', stage, model='glm45_air_base', seed,
    load_checkpoint_path=<parent>, lora=LoraConfig(r 64, α 128, dropout 0, target_linear=False,
    target_modules=<184 exact attention paths>))`` -> ``scimt.train.train_dataset``."""
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    lora_spec = spec["lora"]
    lora = LoraConfig(
        r=int(lora_spec["r"]),
        alpha=int(lora_spec["alpha"]),
        dropout=float(lora_spec["dropout"]),
        target_linear=False,
        target_modules=glm45_attention_exact_targets(int(lora_spec["layers"]), tuple(lora_spec.get("projections") or ("q_proj", "k_proj", "v_proj", "o_proj"))),
    )
    config = TrainConfig(
        backend="axolotl",
        stage=str(spec["stage"]),
        model=str(spec.get("model", SCIMT_MODEL)),
        seed=int(spec["seed"]),
        load_checkpoint_path=str(spec["parent"]),
        lora=lora,
    )
    dataset = Dataset.at(str(spec["dataset_path"]), kind=str(spec.get("dataset_kind", "chat")), text_column=str(spec.get("text_column", "messages")))
    asyncio.run(train_dataset(dataset, str(spec["out_dir"]), config, run_name=str(spec["run_name"])))


def train_entry() -> None:
    """The supervised train child (``$SCIMT_SIEVE_TRAIN_CONFIG`` -> JSON written by the runner per cell)."""
    path = os.environ.get(TRAIN_CONFIG_ENV)
    if not path:
        raise SystemExit(f"{TRAIN_CONFIG_ENV} must point at the cell's train_config.json")
    spec = read_json(path)
    required = ("dataset_path", "out_dir", "parent", "stage", "seed", "lora", "run_name")
    missing = [key for key in required if key not in spec]
    if missing:
        raise SystemExit(f"{path}: train config lacks {missing}")
    _train_child(spec)
