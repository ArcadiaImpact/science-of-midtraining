#!/usr/bin/env python3
"""Resumable single-pod chain for ``glm_minimal_v1``.

The operator owns the already-running pod.  This module only supervises work
on that machine: it never imports a pod lifecycle client and has no create,
stop, or teardown operation.

Production order is::

    preflight -> data -> join base download
    charter midtrain/merge/IFT/merge -> coin midtrain/merge/IFT/merge
    two 4-GPU AFT stages concurrently -> four 2-GPU evals concurrently
    publish remaining artifacts -> join every background upload

Heavy pod dependencies are imported inside the functions that need them so
the orchestration and safety gates stay CPU-testable.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.prior_coins.glm_minimal_v1 import contracts  # noqa: E402
from experiments.prior_coins.glm_minimal_v1.pod.telemetry import (  # noqa: E402
    Telemetry,
)


DEFAULT_WORK = Path("/workspace/glm-minimal-v1")
DEFAULT_SETUP_STATE = Path("/workspace/glm-minimal-v1-setup")
CONFIG_DIR = EXP / "configs"
CONSOLIDATOR = REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
STAGE_MARKER = "_STAGE_COMPLETE.json"
STAGE_PROVENANCE = "_STAGE_PROVENANCE.json"
ENDPOINTS = ("pre_aft", "post_aft")
EVAL_WORKER_SPEC_KEYS = (
    "arm",
    "endpoint",
    "parent",
    "prepared_parent",
    "adapter",
    "data_dir",
    "results_dir",
    "work_dir",
)
ROUTER_PLUGIN = "scimt.train.axolotl_plugins.RouterHealthPlugin"
CONSOLIDATE_TIMEOUT_S = 6 * 3600
TRAIN_PORT_BASE = 29_500
PREPROCESS_PORT_BASE = 29_600
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _log(message: str) -> None:
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    forwarded = os.environ.get("SCIMT_SOURCE_COMMIT", "").strip()
    if forwarded:
        return forwarded
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def stage_prefix(run_id: str, arm: str, stage: str) -> str:
    return f"runs/{run_id}/{arm}/{stage}"


def stage_marker_payload(
    *,
    run_id: str,
    arm: str,
    stage: str,
    config_sha256: str,
    data_digest: str,
    steps: int | None,
    git_sha: str,
    parent_digest: str | None = None,
) -> dict[str, Any]:
    """Stable marker content.  Only ``git_sha`` is deliberately volatile."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "arm": arm,
        "stage": stage,
        "config_sha256": config_sha256,
        "data_digest": data_digest,
        "steps": steps,
        "parent_digest": parent_digest,
        "model_repo": contracts.MODEL_REPO,
        "model_revision": contracts.MODEL_REVISION,
        "git_sha": git_sha,
    }


def marker_matches(
    actual: Mapping[str, Any] | None, expected: Mapping[str, Any]
) -> bool:
    """Judge a completion marker on content, ignoring only ``git_sha``."""

    if not isinstance(actual, Mapping):
        return False
    volatile = {"git_sha"}
    actual_stable = {key: value for key, value in actual.items() if key not in volatile}
    expected_stable = {
        key: value for key, value in expected.items() if key not in volatile
    }
    return actual_stable == expected_stable


def _stable_marker_digest(marker: Mapping[str, Any]) -> str:
    return _canonical_digest(
        {key: value for key, value in marker.items() if key != "git_sha"}
    )


def should_skip_stage(
    *, resume: bool, actual: Mapping[str, Any] | None, expected: Mapping[str, Any]
) -> bool:
    return bool(resume and marker_matches(actual, expected))


def assert_rendered_step_count(rendered: str | Path, expected_steps: int) -> None:
    """Assert the recomputed schedule agrees with the rendered config.

    Full-weight configs declare ``max_steps``; AFT declares its reachable end
    through ``checkpoint_schedule``.  Checking the rendered file catches both
    a stale template and a failed SET_BY_CHAIN replacement before GPU work.
    """

    if isinstance(expected_steps, bool) or not isinstance(expected_steps, int):
        raise TypeError("expected_steps must be an integer")
    body = yaml.safe_load(Path(rendered).read_text(encoding="utf-8"))
    max_steps = body.get("max_steps")
    schedule = body.get("checkpoint_schedule")
    declared: int | None = max_steps if isinstance(max_steps, int) else None
    if declared is None and isinstance(schedule, list) and schedule:
        declared = schedule[-1] if isinstance(schedule[-1], int) else None
    if declared != expected_steps:
        raise AssertionError(
            f"rendered step schedule disagrees with recomputed GLM schedule: "
            f"rendered={declared!r}, recomputed={expected_steps}"
        )
    if isinstance(schedule, list) and schedule and schedule[-1] != expected_steps:
        raise AssertionError(
            f"final checkpoint {schedule[-1]!r} != recomputed step {expected_steps}"
        )


def validate_label_mask_rows(
    rows: Sequence[Mapping[str, Any]], *, terminator_token_id: int
) -> dict[str, Any]:
    """Pure label-mask safety gate used after Axolotl preprocessing."""

    trained = 0
    masked = 0
    terminator_trained = False
    for row in rows:
        labels = row.get("labels")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise RuntimeError("label-mask gate: prepared row has no labels sequence")
        for label in labels:
            if label == -100:
                masked += 1
            else:
                trained += 1
                if label == terminator_token_id:
                    terminator_trained = True
    total = trained + masked
    fraction = trained / total if total else 0.0
    if trained == 0:
        raise RuntimeError("label-mask gate: zero trained tokens")
    if masked == 0:
        raise RuntimeError("label-mask gate: zero masked tokens")
    if not 0.05 <= fraction <= 0.90:
        raise RuntimeError(
            f"label-mask gate: trained fraction {fraction:.3f} outside [0.05, 0.90]"
        )
    if not terminator_trained:
        raise RuntimeError("label-mask gate: terminating token is never trained")
    return {
        "rows_inspected": len(rows),
        "trained_tokens": trained,
        "masked_tokens": masked,
        "trained_fraction": fraction,
        "terminator_token_id": terminator_token_id,
        "terminator_trained": True,
    }


def _prepared_dataset_dir(rendered: Path) -> Path:
    body = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    return Path(body["dataset_prepared_path"])


def sft_label_mask_gate(
    rendered: Path,
    report_path: Path,
    *,
    main_process_port: int = PREPROCESS_PORT_BASE,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Preprocess on CPU, then prove prompt/assistant/terminator masking."""

    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "MASTER_PORT": str(main_process_port),
        "ACCELERATE_MAIN_PROCESS_PORT": str(main_process_port),
    }
    result = runner(
        [sys.executable, "-m", "axolotl.cli.preprocess", str(rendered)],
        capture_output=True,
        text=True,
        check=False,
        timeout=4 * 3600,
        env=environment,
        cwd=str(REPO_ROOT),
    )
    log_path = report_path.with_suffix(".preprocess.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        result.stdout[-100_000:] + "\n--- STDERR ---\n" + result.stderr[-100_000:],
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"axolotl preprocess failed: {result.stderr[-3_000:]}")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    candidates = sorted(_prepared_dataset_dir(rendered).rglob("dataset_info.json"))
    if not candidates:
        raise RuntimeError(
            f"label-mask gate: no prepared dataset under "
            f"{_prepared_dataset_dir(rendered)}"
        )
    dataset = load_from_disk(str(candidates[0].parent))
    body = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(body["base_model"])
    token_id = tokenizer.convert_tokens_to_ids(contracts.GLM_EOS_TOKEN)
    if not isinstance(token_id, int) or token_id < 0:
        raise RuntimeError(
            f"{contracts.GLM_EOS_TOKEN!r} is not one GLM tokenizer token"
        )
    # This is a fail-fast gate, not a dataset audit: a deterministic 200-row
    # prefix is enough to catch prompt/assistant/terminator masking drift.
    sample = dataset.select(range(min(200, len(dataset))))
    report = validate_label_mask_rows(list(sample), terminator_token_id=token_id)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class BackgroundPublishes:
    """Own every upload task and make an unjoined task impossible on success."""

    def __init__(self) -> None:
        self._tasks: list[tuple[str, asyncio.Task[Any]]] = []

    def launch(self, name: str, operation: Awaitable[Any]) -> None:
        _log(f"publish background start: {name}")
        task = asyncio.create_task(operation, name=f"publish:{name}")
        self._tasks.append((name, task))

    async def join(self) -> list[Any]:
        pending, self._tasks = self._tasks, []
        if not pending:
            return []
        results = await asyncio.gather(
            *(task for _, task in pending), return_exceptions=True
        )
        failures: list[BaseException] = []
        for (name, _task), result in zip(pending, results, strict=True):
            if isinstance(result, BaseException):
                _log(f"publish background FAILED: {name}: {result}")
                failures.append(result)
            else:
                _log(f"publish background finish: {name}")
        if failures:
            raise BaseExceptionGroup(
                "one or more background publishes failed", failures
            )
        return results

    def raise_completed_failures(self) -> None:
        """Surface a fast upload failure at the next phase boundary."""

        for name, task in self._tasks:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is not None:
                    raise RuntimeError(
                        f"background publish failed before next phase: {name}"
                    ) from error

    @property
    def pending(self) -> int:
        return len(self._tasks)


@dataclass(frozen=True)
class DataBundle:
    midtrain: Mapping[str, Path]
    midtrain_digests: Mapping[str, str]
    dolci: Path
    dolci_digest: str
    aft: Path
    aft_digest: str
    eval_data: Path
    eval_digest: str
    artifact_revision: str


@dataclass
class Artifact:
    arm: str
    stage: str
    config_path: Path
    config_sha256: str
    data_digest: str
    steps: int
    marker: Mapping[str, Any]
    train_dir: Path | None = None
    checkpoint: Path | None = None
    materialized: Path | None = None
    merge_parent: Path | None = None
    remote_prefix: str | None = None
    already_remote: bool = False
    cleanup_after_publish: list[Path] = field(default_factory=list)
    cleanup_after_merge: list[Path] = field(default_factory=list)
    skip_merge_reason: str | None = None


@dataclass
class EvalArtifact:
    arm: str
    endpoint: str
    directory: Path | None
    marker: Mapping[str, Any]
    remote_prefix: str
    already_remote: bool = False


@dataclass
class ScoreArtifact:
    directory: Path
    remote_prefix: str


class ChainOperations(Protocol):
    publish_midtrain: bool

    async def setup(self) -> None: ...
    async def fetch_data(self) -> DataBundle: ...
    async def join_base_download(self) -> Path: ...
    async def midtrain(self, arm: str, data: DataBundle, base: Path) -> Artifact: ...
    async def merge(self, artifact: Artifact, parent: Path) -> Artifact: ...
    async def release_base_model(self, base: Path, last_midtrain: Artifact) -> None: ...
    async def ift(self, arm: str, data: DataBundle, parent: Artifact) -> Artifact: ...
    async def aft(self, arm: str, data: DataBundle, parent: Artifact) -> Artifact: ...
    async def prepare_eval_parent(self, arm: str, parent: Artifact) -> Path: ...
    async def cleanup_eval_parent(self, arm: str, prepared: Path) -> None: ...
    async def eval_endpoint(
        self,
        arm: str,
        endpoint: str,
        data: DataBundle,
        parent: Artifact,
        adapter: Artifact,
        prepared_parent: Path,
    ) -> EvalArtifact: ...
    async def score_evals(
        self, artifacts: Sequence[EvalArtifact], data: DataBundle
    ) -> ScoreArtifact: ...
    async def publish_stage(self, artifact: Artifact) -> Any: ...
    async def publish_eval(self, artifact: EvalArtifact) -> Any: ...
    async def publish_scores(self, artifact: ScoreArtifact) -> Any: ...
    async def cleanup_stage(self, artifact: Artifact) -> None: ...
    async def publish_metadata(self) -> Any: ...


async def execute_plan(
    operations: ChainOperations, arms: Sequence[str]
) -> dict[str, Any]:
    """Execute the costed order; this seam is intentionally easy to fake."""

    publishes = BackgroundPublishes()
    primary_error: BaseException | None = None
    try:
        await operations.setup()
        data = await operations.fetch_data()
        base = await operations.join_base_download()

        parents: dict[str, Artifact] = {}
        midtrains: dict[str, Artifact] = {}
        for arm_index, arm in enumerate(arms):
            midtrain = await operations.midtrain(arm, data, base)
            midtrain = await operations.merge(midtrain, base)
            midtrains[arm] = midtrain
            if arm_index == len(arms) - 1:
                await operations.release_base_model(base, midtrain)
            if operations.publish_midtrain and not midtrain.already_remote:
                publishes.launch(f"{arm}/midtrain", operations.publish_stage(midtrain))
            ift = await operations.ift(arm, data, midtrain)
            ift = await operations.merge(
                ift, ift.merge_parent or Path(f"/{arm}/unused-resume-parent")
            )
            if not ift.already_remote:
                publishes.launch(f"{arm}/ift", operations.publish_stage(ift))
            parents[arm] = ift
            publishes.raise_completed_failures()

        adapters_list = await asyncio.gather(
            *(operations.aft(arm, data, parents[arm]) for arm in arms)
        )
        adapters = dict(zip(arms, adapters_list, strict=True))
        publishes.raise_completed_failures()

        # Restore and unpack one parent per arm serially before fan-out.  Each
        # endpoint receives the same read-only prepared view; no worker races
        # another worker while rewriting ~199 GB of expert shards.
        prepared_parents: dict[str, Path] = {}
        try:
            for arm in arms:
                await _materialized(parents[arm], operations)
                await _materialized(adapters[arm], operations)
                prepared_parents[arm] = await operations.prepare_eval_parent(
                    arm, parents[arm]
                )
            eval_list = await asyncio.gather(
                *(
                    operations.eval_endpoint(
                        arm,
                        endpoint,
                        data,
                        parents[arm],
                        adapters[arm],
                        prepared_parents[arm],
                    )
                    for arm in arms
                    for endpoint in ENDPOINTS
                )
            )
        finally:
            for arm, prepared in prepared_parents.items():
                await operations.cleanup_eval_parent(arm, prepared)

        scores = await operations.score_evals(eval_list, data)

        for arm, adapter in adapters.items():
            if not adapter.already_remote:
                publishes.launch(f"{arm}/aft", operations.publish_stage(adapter))
        for endpoint in eval_list:
            if not endpoint.already_remote:
                publishes.launch(
                    f"{endpoint.arm}/{endpoint.endpoint}",
                    operations.publish_eval(endpoint),
                )
        publishes.launch("scores", operations.publish_scores(scores))

        # Metadata must see all phase/publish rows accumulated so far.
        await publishes.join()
        for artifact in (*midtrains.values(), *parents.values()):
            await operations.cleanup_stage(artifact)
        publishes.launch("metadata", operations.publish_metadata())
        await publishes.join()
        return {
            "parents": parents,
            "adapters": adapters,
            "eval": eval_list,
            "scores": scores,
        }
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if publishes.pending:
            try:
                await publishes.join()
            except BaseException as publish_error:
                if primary_error is None:
                    raise
                _log(
                    "background publish also failed while unwinding primary error: "
                    f"{publish_error}"
                )


async def _materialized(artifact: Artifact, operations: ChainOperations) -> Path:
    if artifact.materialized is not None:
        return artifact.materialized
    restore = getattr(operations, "restore_stage", None)
    if restore is None:
        # Fake operations used in CPU ordering tests need no real parent path.
        return Path(f"/{artifact.arm}/{artifact.stage}")
    return await restore(artifact)


def _load_local_stage(path: Path) -> Any:
    from scimt.train.axolotl import StageSpec

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {item.name for item in dataclasses.fields(StageSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"stage file {path} has unknown keys {sorted(unknown)}")
    stage = StageSpec(**data)
    plugins = stage.axolotl.get("plugins") or []
    if ROUTER_PLUGIN not in plugins:
        raise RuntimeError(f"{path}: RouterHealthPlugin is not enabled")
    return stage


def _resolve_midtrain_config(
    template: Path, *, arm: str, steps: int, destination: Path
) -> Path:
    body = yaml.safe_load(template.read_text(encoding="utf-8"))
    axolotl = body["axolotl"]
    if (
        axolotl.get("max_steps") != "SET_BY_CHAIN"
        or axolotl.get("checkpoint_schedule") != "SET_BY_CHAIN"
    ):
        raise RuntimeError(f"{template}: SET_BY_CHAIN placeholders drifted")
    axolotl["max_steps"] = steps
    axolotl["checkpoint_schedule"] = [steps]
    # Task 1 emits one unique task/replay mix.  The Dispatch-line contract is
    # to present that same ordered mix four times; unlike Jonathan's reference
    # chain, the rows are not physically copied four times.
    axolotl["num_epochs"] = contracts.MIDTRAIN_PRESENTATIONS
    body["name"] = f"glm_minimal_v1_midtrain_{arm}_{steps}steps"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    if "SET_BY_CHAIN" in destination.read_text(encoding="utf-8"):
        raise RuntimeError(f"unresolved SET_BY_CHAIN placeholder in {destination}")
    return destination


def _evict_from_page_cache(root: Path) -> int:
    advised = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                size = os.fstat(descriptor).st_size
                os.posix_fadvise(descriptor, 0, size, os.POSIX_FADV_DONTNEED)
                advised += size
            finally:
                os.close(descriptor)
        except OSError:
            continue
    return advised


def _resolve_xet_cache(
    environment: Mapping[str, str] | None = None, *, home: Path | None = None
) -> Path:
    env = os.environ if environment is None else environment
    explicit = env.get("HF_XET_CACHE", "").strip()
    if explicit:
        return Path(explicit)
    hf_home = env.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "xet"
    hub_cache = env.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).parent / "xet"
    return (home if home is not None else Path.home()) / ".cache/huggingface/xet"


def _purge_xet_cache() -> float | None:
    xet = _resolve_xet_cache()
    if not xet.is_dir():
        _log(f"HF/Xet cache directory does not exist; nothing purged: {xet}")
        return None
    size = sum(path.stat().st_size for path in xet.rglob("*") if path.is_file())
    shutil.rmtree(xet)
    _log(f"purged {size / 1e9:.1f} GB HF/Xet cache")
    return size / 1e9


def _delete_tree_after_durable(
    path: Path,
    *,
    evidence: Sequence[Path],
    description: str,
    any_file_glob: tuple[Path, str] | None = None,
) -> bool:
    """Delete ``path`` only after every named durability proof is present."""

    missing = [str(item) for item in evidence if not item.exists()]
    if any_file_glob is not None:
        root, pattern = any_file_glob
        if not any(root.glob(pattern)):
            missing.append(f"{root}/{pattern}")
    if missing:
        raise RuntimeError(
            f"refusing to delete {description} before its replacement is durable: "
            f"missing={missing}"
        )
    if not path.exists():
        return False
    _log(f"deleting {description} exactly after durability verification: {path}")
    shutil.rmtree(path)
    return True


def _model_metadata_view(source: Path, destination: Path) -> Path:
    """Copy the small non-weight sidecars needed by FSDP consolidation."""

    if not (source / "config.json").is_file():
        raise RuntimeError(f"model metadata source has no config.json: {source}")
    if destination.is_dir() and (destination / "config.json").is_file():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for item in source.iterdir():
            if not item.is_file():
                continue
            if item.suffix == ".safetensors" or item.name == "model.safetensors.index.json":
                continue
            shutil.copy2(item, temporary / item.name)
        if not (temporary / "config.json").is_file():
            raise RuntimeError(f"metadata copy lost config.json from {source}")
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _base_model_cache_root(snapshot: Path) -> Path | None:
    """Return only a Hugging Face model-repository cache root, never a broad path."""

    if snapshot.parent.name != "snapshots":
        return None
    candidate = snapshot.parent.parent
    if not candidate.name.startswith("models--"):
        return None
    return candidate


def _link_or_copy_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _assemble_score_inputs(
    artifacts: Sequence[EvalArtifact], destination: Path
) -> Path:
    """Assemble chain-shaped endpoint trees into the scorer's four-cell layout."""

    by_cell = {(item.arm, item.endpoint): item for item in artifacts}
    expected = {(arm, endpoint) for arm in contracts.ARMS for endpoint in ENDPOINTS}
    if set(by_cell) != expected:
        raise RuntimeError(
            f"scoring needs exactly {sorted(expected)}, got {sorted(by_cell)}"
        )
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for arm, endpoint in sorted(expected):
            artifact = by_cell[(arm, endpoint)]
            if artifact.directory is None:
                raise RuntimeError(f"no local eval directory for {arm}/{endpoint}")
            source = artifact.directory / f"{arm}-{endpoint}"
            if not source.is_dir():
                raise RuntimeError(f"eval cell is missing: {source}")
            shutil.copytree(
                source,
                temporary / source.name,
                copy_function=_link_or_copy_file,
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


class ProductionChain:
    """The real single-host implementation behind :func:`execute_plan`."""

    def __init__(
        self,
        *,
        run_id: str,
        resume: bool,
        work: Path,
        setup_state: Path,
    ) -> None:
        self.run_id = run_id
        self.resume = resume
        self.work = work
        self.setup_state = setup_state
        self.run_dir = work / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry = Telemetry(
            self.run_dir / "telemetry.jsonl",
            run_id=run_id,
            disk_path=work,
        )
        self.repo_id = os.environ.get("SCIMT_HF_TARGET_REPO", "").strip()
        self.token = os.environ.get("HF_TOKEN", "").strip()
        if os.environ.get("SCIMT_HF_REPO_TYPE", "model").strip() != "model":
            raise ValueError("the chain's durable artifact repo type must be 'model'")
        self.publish_midtrain = os.environ.get(
            "SCIMT_PUBLISH_MIDTRAIN", "0"
        ).lower() in {"1", "true", "yes"}
        self.gpu_type = "unknown"
        self.config_suffix = "h200"
        self._api: Any | None = None
        self._ram_handle: Any | None = None
        self._data: DataBundle | None = None
        self._env_launch_lock = asyncio.Lock()

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    async def setup(self) -> None:
        # Task 5 must expose an endpoint-shaped worker before this expensive
        # chain is allowed to start.  This is checked before the 2 GB egress
        # probe, let alone training.
        from experiments.prior_coins.glm_minimal_v1.pod import preflight

        with self.telemetry.phase("setup", n_gpus=0, notes="preflight gates") as phase:
            eval_module = importlib.import_module(
                "experiments.prior_coins.glm_minimal_v1.pod.eval_glm"
            )
            if not callable(getattr(eval_module, "evaluate_endpoint", None)):
                raise RuntimeError(
                    "eval_glm.evaluate_endpoint is required for four concurrent "
                    "2-GPU endpoint jobs; the aggregate evaluate_arm API cannot "
                    "satisfy the Task 4 concurrency contract"
                )
            record = await asyncio.to_thread(preflight.preflight, self.run_dir)
            phase.update(mb_per_s=float(record["egress_mbps"]))
            capability = record["gpus"][0]["compute_capability"]
            if capability == "9.0":
                self.gpu_type, self.config_suffix = "H200", "h200"
            elif str(capability).startswith("10."):
                self.gpu_type, self.config_suffix = "B300", "b300"
            else:  # setup_pod should already have rejected this
                raise RuntimeError(f"unsupported compute capability {capability}")
            self.telemetry.gpu_type = self.gpu_type
            self.telemetry.n_gpus = 8
        self._ram_handle = preflight._start_ram_telemetry(self.run_dir, 30)

    async def fetch_data(self) -> DataBundle:
        with self.telemetry.phase("data_fetch", n_gpus=0) as phase:
            bundle = await asyncio.to_thread(self._fetch_data_sync)
            phase.update(notes=f"artifact revision {bundle.artifact_revision}")
        self._data = bundle
        return bundle

    def _fetch_data_sync(self) -> DataBundle:
        from huggingface_hub import snapshot_download

        requested_revision = os.environ.get("SCIMT_DATA_REVISION") or None
        root = Path(
            snapshot_download(
                repo_id=contracts.DATA_ARTIFACT_REPO,
                repo_type="dataset",
                revision=requested_revision,
                local_dir=str(self.work / "data" / "static"),
                token=self.token,
            )
        )
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != contracts.VERSION:
            raise RuntimeError("data artifact version does not match contracts")
        if manifest.get("pins") != contracts.pin_set():
            raise RuntimeError("data artifact pin set does not match contracts.py")
        paths: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for arm in contracts.ARMS:
            filename = contracts.OUTPUT_FILENAMES[arm]
            path = root / filename
            expected = manifest["files"][filename]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{filename} digest {actual} != manifest {expected}")
            paths[arm] = path
            digests[arm] = actual
        aft = root / contracts.OUTPUT_FILENAMES["aft"]
        aft_digest = sha256_file(aft)
        if aft_digest != contracts.AFT_ARTIFACT_SHA256:
            raise RuntimeError(
                f"AFT artifact digest {aft_digest} != {contracts.AFT_ARTIFACT_SHA256}"
            )

        dolci, dolci_digest = self._prepare_dolci()
        eval_root = (
            Path(
                snapshot_download(
                    repo_id=contracts.AFT_ARTIFACT_REPO,
                    repo_type="dataset",
                    revision=contracts.AFT_ARTIFACT_REVISION,
                    allow_patterns=[
                        f"{contracts.AFT_ARTIFACT_PREFIX}/prompts/**",
                        f"{contracts.AFT_ARTIFACT_PREFIX}/episodes/**",
                    ],
                    local_dir=str(self.work / "data" / "eval"),
                    token=self.token,
                )
            )
            / contracts.AFT_ARTIFACT_PREFIX
        )
        if not (eval_root / "prompts").is_dir():
            raise RuntimeError(f"evaluation prompt snapshot is incomplete: {eval_root}")
        revision = str(
            self.api.dataset_info(
                contracts.DATA_ARTIFACT_REPO, revision=requested_revision
            ).sha
        )
        return DataBundle(
            midtrain=paths,
            midtrain_digests=digests,
            dolci=dolci,
            dolci_digest=dolci_digest,
            aft=aft,
            aft_digest=aft_digest,
            eval_data=eval_root,
            eval_digest=_tree_digest(eval_root),
            artifact_revision=revision,
        )

    def _prepare_dolci(self) -> tuple[Path, str]:
        from datasets import load_dataset, load_from_disk
        from experiments.prior_coins.glm_minimal_v1.build_data import (
            valid_dolci_messages,
        )

        out = self.work / "data" / "dolci"
        manifest_path = out / "scimt_manifest.json"
        if (out / "dataset_info.json").is_file() and manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "repo": contracts.DOLCI_REPO,
                "revision": contracts.DOLCI_REVISION,
                "source_rows": contracts.DOLCI_SOURCE_ROWS,
                "filtered_rows": contracts.DOLCI_FILTERED_ROWS,
                "seed": contracts.IFT_TRAINING_SEED,
            }
            if all(saved.get(key) == value for key, value in expected.items()):
                load_from_disk(str(out))  # loud completeness check
                return out, str(saved["ordered_messages_sha256"])
            _log(f"discarding exactly stale Dolci cache artifact {out}")
            shutil.rmtree(out)
        temporary = out.with_name(out.name + ".partial")
        if temporary.exists():
            _log(f"discarding exactly partial Dolci cache artifact {temporary}")
            shutil.rmtree(temporary)
        dataset = load_dataset(
            contracts.DOLCI_REPO,
            revision=contracts.DOLCI_REVISION,
            split="train",
            token=self.token,
        )
        if len(dataset) != contracts.DOLCI_SOURCE_ROWS:
            raise RuntimeError(
                f"Dolci source rows {len(dataset)} != {contracts.DOLCI_SOURCE_ROWS}"
            )
        dataset = dataset.filter(
            lambda row: valid_dolci_messages(row.get("messages")), num_proc=16
        ).shuffle(seed=contracts.IFT_TRAINING_SEED)
        if len(dataset) != contracts.DOLCI_FILTERED_ROWS:
            raise RuntimeError(
                f"Dolci filtered rows {len(dataset)} != {contracts.DOLCI_FILTERED_ROWS}"
            )
        ordered = hashlib.sha256()
        for messages in dataset["messages"]:
            ordered.update(
                json.dumps(
                    messages,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            )
            ordered.update(b"\n")
        digest = ordered.hexdigest()
        temporary.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(temporary))
        record = {
            "repo": contracts.DOLCI_REPO,
            "revision": contracts.DOLCI_REVISION,
            "source_rows": contracts.DOLCI_SOURCE_ROWS,
            "filtered_rows": contracts.DOLCI_FILTERED_ROWS,
            "seed": contracts.IFT_TRAINING_SEED,
            "ordered_messages_sha256": digest,
        }
        (temporary / "scimt_manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(out)
        return out, digest

    async def join_base_download(self) -> Path:
        with self.telemetry.phase(
            "download", n_gpus=0, notes="join setup prefetch"
        ) as phase:
            base = await self._join_model_prefetch()
            purged = await asyncio.to_thread(_purge_xet_cache)
            advised = await asyncio.to_thread(_evict_from_page_cache, base)
            purge_note = (
                f"purged_xet_gb={purged:.1f}"
                if purged is not None
                else f"xet_cache_absent={_resolve_xet_cache()}"
            )
            phase.update(
                notes=(
                    f"joined background base snapshot; {purge_note}; "
                    f"fadvise_dontneed_gb={advised / 1e9:.1f}"
                )
            )
        return base

    async def _join_model_prefetch(self) -> Path:
        done = self.setup_state / "model-download.done"
        exit_file = self.setup_state / "model-download.exit"
        pid_file = self.setup_state / "model-download.pid"
        while not done.exists():
            if exit_file.exists():
                raw = exit_file.read_text(encoding="utf-8").strip()
                if raw != "0":
                    raise RuntimeError(
                        f"background model prefetch failed with exit {raw}; inspect "
                        f"{self.setup_state / 'model-download.log'}"
                    )
            if not pid_file.exists():
                raise RuntimeError(
                    f"no background model prefetch marker at {pid_file}; run setup_pod.sh"
                )
            raw_pid = pid_file.read_text(encoding="utf-8").strip()
            if not raw_pid.isdigit():
                raise RuntimeError(f"invalid model prefetch PID {raw_pid!r}")
            try:
                os.kill(int(raw_pid), 0)
            except ProcessLookupError as exc:
                raise RuntimeError(
                    "model prefetch process disappeared without a completion marker"
                ) from exc
            await asyncio.sleep(5)
        from huggingface_hub import snapshot_download

        return Path(
            await asyncio.to_thread(
                snapshot_download,
                repo_id=contracts.MODEL_REPO,
                revision=contracts.MODEL_REVISION,
                local_files_only=True,
                token=self.token,
            )
        )

    def _config(self, stage: str) -> Path:
        name = {"midtrain": "midtrain", "ift": "sft", "aft": "aft"}[stage]
        return CONFIG_DIR / f"{name}_glm45_air_{self.config_suffix}.yaml"

    def _remote_marker(self, prefix: str) -> Mapping[str, Any] | None:
        filename = f"{prefix}/{STAGE_MARKER}"
        try:
            downloaded = self.api.hf_hub_download(
                repo_id=self.repo_id,
                repo_type="model",
                filename=filename,
            )
        except Exception as exc:
            from huggingface_hub import errors as hf_errors

            missing_types = tuple(
                kind
                for name in (
                    "EntryNotFoundError",
                    "RemoteEntryNotFoundError",
                    "RepositoryNotFoundError",
                )
                if isinstance((kind := getattr(hf_errors, name, None)), type)
            )
            if missing_types and isinstance(exc, missing_types):
                _log(f"resume marker absent at {filename}")
                return None
            raise RuntimeError(
                f"resume marker lookup failed at {filename}: {exc}"
            ) from exc
        try:
            value = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log(f"resume marker has invalid content at {filename}: {exc}")
            return None
        return value if isinstance(value, Mapping) else None

    def _artifact_contract(
        self,
        *,
        arm: str,
        stage: str,
        config: Path,
        data_digest: str,
        steps: int,
        parent_digest: str | None,
    ) -> tuple[str, dict[str, Any], str]:
        config_digest = sha256_file(config)
        marker = stage_marker_payload(
            run_id=self.run_id,
            arm=arm,
            stage=stage,
            config_sha256=config_digest,
            data_digest=data_digest,
            steps=steps,
            git_sha=_git_sha(),
            parent_digest=parent_digest,
        )
        prefix = stage_prefix(self.run_id, arm, stage)
        return config_digest, marker, prefix

    def _ift_contract(
        self, arm: str, data: DataBundle, parent_marker: Mapping[str, Any]
    ) -> tuple[Path, int, str, dict[str, Any], str]:
        config = self._config("ift")
        steps = contracts.ift_steps(contracts.DOLCI_PACKED_POSITION_CAP)
        digest, marker, prefix = self._artifact_contract(
            arm=arm,
            stage="ift",
            config=config,
            data_digest=data.dolci_digest,
            steps=steps,
            parent_digest=_stable_marker_digest(parent_marker),
        )
        return config, steps, digest, marker, prefix

    def _stage_train_dir(
        self, arm: str, stage: str, marker: Mapping[str, Any]
    ) -> Path:
        return (
            self.work
            / "train"
            / self.run_id
            / arm
            / stage
            / _stable_marker_digest(marker)[:24]
        )

    def _ift_merge_metadata_dir(
        self, arm: str, marker: Mapping[str, Any]
    ) -> Path:
        return (
            self.work
            / "merge_metadata"
            / self.run_id
            / arm
            / "ift"
            / _stable_marker_digest(marker)[:24]
        )

    def _local_stage_marker_matches(
        self, train_dir: Path, expected: Mapping[str, Any]
    ) -> bool:
        path = train_dir / STAGE_MARKER
        if not path.is_file():
            return False
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return marker_matches(actual, expected)

    async def midtrain(self, arm: str, data: DataBundle, base: Path) -> Artifact:
        tokens, steps = await asyncio.to_thread(
            self._glm_schedule, arm, data.midtrain[arm], base
        )
        config = _resolve_midtrain_config(
            self._config("midtrain"),
            arm=arm,
            steps=steps,
            destination=self.run_dir / "configs" / f"midtrain_{arm}.yaml",
        )
        digest, marker, prefix = self._artifact_contract(
            arm=arm,
            stage="midtrain",
            config=config,
            data_digest=data.midtrain_digests[arm],
            steps=steps,
            parent_digest=contracts.MODEL_REVISION,
        )
        actual = (
            await asyncio.to_thread(self._remote_marker, prefix)
            if self.resume
            else None
        )
        if should_skip_stage(resume=self.resume, actual=actual, expected=marker):
            _log(f"{arm}/midtrain: verified HF marker matches; skipping")
            with self.telemetry.phase(
                "midtrain",
                arm=arm,
                n_gpus=0,
                steps=steps,
                tokens=tokens * contracts.MIDTRAIN_PRESENTATIONS,
                notes="resume: verified remote completion marker",
            ):
                pass
            return Artifact(
                arm,
                "midtrain",
                config,
                digest,
                data.midtrain_digests[arm],
                steps,
                marker,
                remote_prefix=prefix,
                already_remote=True,
            )
        if self.resume:
            _, ift_steps, _, ift_marker, ift_prefix = self._ift_contract(
                arm, data, marker
            )
            downstream = await asyncio.to_thread(self._remote_marker, ift_prefix)
            if should_skip_stage(
                resume=True, actual=downstream, expected=ift_marker
            ):
                _log(
                    f"{arm}/midtrain: child IFT marker is durable; skipping "
                    "reclaimed midtrain shard"
                )
                return Artifact(
                    arm,
                    "midtrain",
                    config,
                    digest,
                    data.midtrain_digests[arm],
                    steps,
                    marker,
                    already_remote=True,
                    skip_merge_reason="child IFT is verified remotely",
                )
            ift_train_dir = self._stage_train_dir(arm, "ift", ift_marker)
            ift_checkpoint = ift_train_dir / "checkpoints" / f"checkpoint-{ift_steps}"
            metadata = self._ift_merge_metadata_dir(arm, ift_marker)
            if (
                self._local_stage_marker_matches(ift_train_dir, ift_marker)
                and ift_checkpoint.is_dir()
                and metadata.is_dir()
                and (metadata / "config.json").is_file()
            ):
                _log(
                    f"{arm}/midtrain: child IFT checkpoint and local marker are "
                    "durable; reusing the metadata-only merge parent"
                )
                return Artifact(
                    arm,
                    "midtrain",
                    config,
                    digest,
                    data.midtrain_digests[arm],
                    steps,
                    marker,
                    materialized=metadata,
                    already_remote=True,
                    skip_merge_reason="child IFT is complete locally",
                )
        return await self._train(
            arm=arm,
            stage_name="midtrain",
            config=config,
            data_path=data.midtrain[arm],
            data_digest=data.midtrain_digests[arm],
            steps=steps,
            tokens=tokens * contracts.MIDTRAIN_PRESENTATIONS,
            parent=base,
            marker=marker,
            config_digest=digest,
            n_gpus=8,
        )

    def _glm_schedule(
        self, arm: str, data_path: Path, tokenizer_dir: Path
    ) -> tuple[int, int]:
        cache_path = self.work / "cache" / "glm_step_schedule.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        file_digest = sha256_file(data_path)
        key = f"{arm}:{file_digest}:{contracts.GLM_TOKENIZER_REVISION}"
        if key in cache:
            record = cache[key]
            return int(record["glm_tokens"]), int(record["steps"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        total = 0
        batch: list[str] = []

        def consume() -> None:
            nonlocal total
            if batch:
                encoded = tokenizer(batch, add_special_tokens=True)["input_ids"]
                total += sum(len(ids) for ids in encoded)
                batch.clear()

        with data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                batch.append(row["text"])
                if len(batch) >= 512:
                    consume()
        consume()
        steps = contracts.midtrain_steps(total)
        cache[key] = {
            "arm": arm,
            "file_sha256": file_digest,
            "tokenizer_revision": contracts.GLM_TOKENIZER_REVISION,
            "glm_tokens": total,
            "presentations": contracts.MIDTRAIN_PRESENTATIONS,
            "steps": steps,
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, indent=2) + "\n")
        temporary.replace(cache_path)
        shutil.copy2(cache_path, self.run_dir / "glm_step_schedule.json")
        return total, steps

    async def ift(self, arm: str, data: DataBundle, parent: Artifact) -> Artifact:
        config, steps, digest, marker, prefix = self._ift_contract(
            arm, data, parent.marker
        )
        actual = (
            await asyncio.to_thread(self._remote_marker, prefix)
            if self.resume
            else None
        )
        if should_skip_stage(resume=self.resume, actual=actual, expected=marker):
            _log(f"{arm}/ift: verified HF marker matches; skipping")
            with self.telemetry.phase(
                "ift",
                arm=arm,
                n_gpus=0,
                steps=steps,
                tokens=contracts.DOLCI_PACKED_POSITION_CAP,
                notes="resume: verified remote completion marker",
            ):
                pass
            return Artifact(
                arm,
                "ift",
                config,
                digest,
                data.dolci_digest,
                steps,
                marker,
                remote_prefix=prefix,
                already_remote=True,
            )
        parent_path = await self.restore_stage(parent)
        trained = await self._train(
            arm=arm,
            stage_name="ift",
            config=config,
            data_path=data.dolci,
            data_digest=data.dolci_digest,
            steps=steps,
            tokens=contracts.DOLCI_PACKED_POSITION_CAP,
            parent=parent_path,
            marker=marker,
            config_digest=digest,
            n_gpus=8,
        )
        # The IFT checkpoint is now durable.  Preserve only the small tokenizer/
        # config sidecars the FSDP consolidator still needs, then reclaim the
        # ~199 GB midtrain model.  A concurrently published midtrain is exempt:
        # its uploader still owns that directory until its verified join.
        if not self.publish_midtrain and parent.materialized is not None:
            assert trained.checkpoint is not None
            metadata_destination = self._ift_merge_metadata_dir(arm, marker)
            metadata = await asyncio.to_thread(
                _model_metadata_view,
                parent.materialized,
                metadata_destination,
            )
            if parent.materialized != metadata_destination:
                await asyncio.to_thread(
                    _delete_tree_after_durable,
                    parent.materialized,
                    evidence=[trained.checkpoint, trained.train_dir / STAGE_MARKER],
                    description=f"{arm}/midtrain consolidated parent",
                )
            parent.materialized = None
            trained.merge_parent = metadata
            trained.cleanup_after_merge.append(metadata)
        return trained

    async def aft(self, arm: str, data: DataBundle, parent: Artifact) -> Artifact:
        config = self._config("aft")
        steps = contracts.aft_steps(contracts.AFT_ROWS, contracts.AFT_EPOCHS)
        digest, marker, prefix = self._artifact_contract(
            arm=arm,
            stage="aft",
            config=config,
            data_digest=data.aft_digest,
            steps=steps,
            parent_digest=_stable_marker_digest(parent.marker),
        )
        actual = (
            await asyncio.to_thread(self._remote_marker, prefix)
            if self.resume
            else None
        )
        if should_skip_stage(resume=self.resume, actual=actual, expected=marker):
            _log(f"{arm}/aft: verified HF marker matches; skipping")
            with self.telemetry.phase(
                "aft",
                arm=arm,
                n_gpus=0,
                steps=steps,
                notes="resume: verified remote completion marker",
            ):
                pass
            return Artifact(
                arm,
                "aft",
                config,
                digest,
                data.aft_digest,
                steps,
                marker,
                remote_prefix=prefix,
                already_remote=True,
            )
        parent_path = await self.restore_stage(parent)
        gpu_ids = "0,1,2,3" if arm == contracts.ARMS[0] else "4,5,6,7"
        return await self._train(
            arm=arm,
            stage_name="aft",
            config=config,
            data_path=data.aft,
            data_digest=data.aft_digest,
            steps=steps,
            tokens=None,
            parent=parent_path,
            marker=marker,
            config_digest=digest,
            n_gpus=4,
            visible_devices=gpu_ids,
        )

    async def _train(
        self,
        *,
        arm: str,
        stage_name: str,
        config: Path,
        data_path: Path,
        data_digest: str,
        steps: int,
        tokens: int | None,
        parent: Path,
        marker: Mapping[str, Any],
        config_digest: str,
        n_gpus: int,
        visible_devices: str | None = None,
    ) -> Artifact:
        from scimt.train import TrainConfig
        from scimt.train.axolotl import LocalExecutor, render_stage

        stage = _load_local_stage(config)
        artifact_key = _stable_marker_digest(marker)[:24]
        out = self.work / "train" / self.run_id / arm / stage_name / artifact_key
        out.mkdir(parents=True, exist_ok=True)  # never reset a run directory
        cfg = TrainConfig(
            backend="axolotl",
            stage=stage.name,
            seed=(
                contracts.AFT_TRAINING_SEED
                if stage_name == "aft"
                else contracts.MIDTRAIN_TRAINING_SEED
            ),
            load_checkpoint_path=str(parent),
        )
        rendered = out / "axolotl.yaml"
        checkpoint = out / "checkpoints" / f"checkpoint-{steps}"
        provenance_path = out / "training_provenance.json"
        local_complete = False
        if (
            self.resume
            and rendered.is_file()
            and provenance_path.is_file()
            and self._local_stage_marker_matches(out, marker)
        ):
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                local_complete = (
                    provenance.get("status") == "complete"
                    and provenance.get("resolved_config_sha256")
                    == sha256_file(rendered)
                    and checkpoint.is_dir()
                    and (out / "router_health.jsonl").is_file()
                )
                if local_complete:
                    assert_rendered_step_count(rendered, steps)
            except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
                _log(
                    f"{arm}/{stage_name}: local completion cache is invalid; "
                    f"relaunching in place without resetting it: {exc}"
                )
                local_complete = False
        if not local_complete:
            rendered = render_stage(stage, cfg, data_path, out)
            assert_rendered_step_count(rendered, steps)
        phase_name = stage_name
        with self.telemetry.phase(
            phase_name,
            arm=arm,
            n_gpus=n_gpus,
            steps=steps,
            tokens=tokens,
        ):
            if local_complete:
                _log(
                    f"{arm}/{stage_name}: reusing content-verified local completion cache"
                )
            else:
                if stage_name in {"ift", "aft"}:
                    await asyncio.to_thread(
                        sft_label_mask_gate,
                        rendered,
                        self.run_dir / f"{arm}_{stage_name}_label_mask.json",
                        main_process_port=(
                            PREPROCESS_PORT_BASE + contracts.ARMS.index(arm)
                        ),
                    )
                await self._run_executor_with_devices(
                    LocalExecutor(),
                    rendered,
                    out,
                    stage,
                    visible_devices,
                    rendezvous_port=(
                        TRAIN_PORT_BASE + contracts.ARMS.index(arm)
                        if visible_devices is not None
                        else None
                    ),
                )
        if not checkpoint.is_dir():
            raise RuntimeError(
                f"{arm}/{stage_name}: expected final checkpoint {checkpoint} is absent"
            )
        router = out / "router_health.jsonl"
        if not router.is_file() or router.stat().st_size == 0:
            raise RuntimeError(
                f"{arm}/{stage_name}: RouterHealthPlugin produced no {router}"
            )
        local_marker = out / STAGE_MARKER
        temporary_marker = local_marker.with_suffix(".json.tmp")
        temporary_marker.write_text(
            json.dumps(dict(marker), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(local_marker)
        shutil.copy2(router, self.run_dir / f"{arm}_{stage_name}_router_health.jsonl")
        return Artifact(
            arm=arm,
            stage=stage_name,
            config_path=config,
            config_sha256=config_digest,
            data_digest=data_digest,
            steps=steps,
            marker=marker,
            train_dir=out,
            checkpoint=checkpoint,
            materialized=checkpoint if stage_name == "aft" else None,
            merge_parent=parent,
            remote_prefix=stage_prefix(self.run_id, arm, stage_name),
        )

    async def _run_executor_with_devices(
        self,
        executor: Any,
        rendered: Path,
        out: Path,
        stage: Any,
        visible_devices: str | None,
        *,
        rendezvous_port: int | None = None,
        _environment_module: Any | None = None,
    ) -> None:
        if visible_devices is None:
            await executor.run_stage(rendered, out, stage)
            return
        if rendezvous_port is None:
            raise ValueError("a scoped multi-GPU launch requires a rendezvous port")
        # This scoped launch deliberately couples to
        # scimt.train.axolotl.LocalExecutor.run_stage calling
        # _training_subprocess_environment() before its first suspension.  The
        # temporary wrapper observes the *exact* env handed to the subprocess;
        # if upstream inserts an earlier await, the missing sentinel aborts here
        # instead of silently exposing both arms to all eight GPUs.
        environment_module = _environment_module
        if environment_module is None:
            environment_module = importlib.import_module("scimt.train.axolotl")
        original_environment = getattr(
            environment_module, "_training_subprocess_environment", None
        )
        if not callable(original_environment):
            raise RuntimeError(
                "LocalExecutor environment contract changed: "
                "_training_subprocess_environment is unavailable"
            )
        captured: dict[str, str] = {}

        def capture_subprocess_environment() -> dict[str, str]:
            child_environment = original_environment()
            captured.update(
                {
                    key: child_environment.get(key, "")
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "MASTER_PORT",
                        "ACCELERATE_MAIN_PROCESS_PORT",
                    )
                }
            )
            return child_environment

        scoped = {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "MASTER_PORT": str(rendezvous_port),
            "ACCELERATE_MAIN_PROCESS_PORT": str(rendezvous_port),
        }
        task: asyncio.Task[Any] | None = None
        try:
            async with self._env_launch_lock:
                previous = {key: os.environ.get(key) for key in scoped}
                os.environ.update(scoped)
                setattr(
                    environment_module,
                    "_training_subprocess_environment",
                    capture_subprocess_environment,
                )
                try:
                    task = asyncio.create_task(
                        executor.run_stage(rendered, out, stage)
                    )
                    await asyncio.sleep(0)
                    if captured != scoped:
                        raise RuntimeError(
                            "LocalExecutor device/port scoping degraded before child "
                            f"launch: expected={scoped}, captured={captured or None}"
                        )
                    sentinel = out / "launch_environment.json"
                    sentinel.write_text(
                        json.dumps(captured, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                finally:
                    setattr(
                        environment_module,
                        "_training_subprocess_environment",
                        original_environment,
                    )
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
        except BaseException:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
        assert task is not None
        await task

    async def merge(self, artifact: Artifact, parent: Path) -> Artifact:
        if artifact.stage == "aft":
            return artifact
        if artifact.skip_merge_reason is not None:
            with self.telemetry.phase(
                "merge",
                arm=artifact.arm,
                n_gpus=0,
                notes=f"resume: {artifact.skip_merge_reason}",
            ):
                pass
            return artifact
        if artifact.already_remote:
            with self.telemetry.phase(
                "merge",
                arm=artifact.arm,
                n_gpus=0,
                notes=f"resume: {artifact.stage} already merged and verified remotely",
            ):
                pass
            return artifact
        assert artifact.checkpoint is not None
        destination = (
            self.work
            / "consolidated"
            / self.run_id
            / artifact.arm
            / artifact.stage
            / _stable_marker_digest(artifact.marker)[:24]
        )
        with self.telemetry.phase("merge", arm=artifact.arm, n_gpus=0):
            await asyncio.to_thread(
                _consolidate_glm,
                artifact.checkpoint,
                parent,
                destination,
                self.run_dir,
            )
            from scimt.train.handoff import finalize_glm4_moe_checkpoint

            record = await asyncio.to_thread(finalize_glm4_moe_checkpoint, destination)
            with (self.run_dir / "mtp_finalize_records.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record.as_dict()) + "\n")
        router = artifact.train_dir / "router_health.jsonl"  # type: ignore[operator]
        shutil.copy2(router, destination / "router_health.jsonl")
        (destination / STAGE_PROVENANCE).write_text(
            json.dumps(dict(artifact.marker), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact.materialized = destination
        if artifact.stage == "midtrain" and artifact.checkpoint is not None:
            await asyncio.to_thread(
                _delete_tree_after_durable,
                artifact.checkpoint,
                evidence=[destination / "config.json"],
                description=f"{artifact.arm}/midtrain sharded checkpoint",
                any_file_glob=(destination, "*.safetensors"),
            )
            artifact.checkpoint = None
        for path in artifact.cleanup_after_merge:
            if path.exists():
                await asyncio.to_thread(
                    _delete_tree_after_durable,
                    path,
                    evidence=[destination / "config.json"],
                    description=f"{artifact.arm}/{artifact.stage} merge metadata",
                    any_file_glob=(destination, "*.safetensors"),
                )
        artifact.cleanup_after_merge.clear()
        return artifact

    async def release_base_model(
        self, base: Path, last_midtrain: Artifact
    ) -> None:
        """Reclaim the pinned base cache after its final midtrain consumer."""

        cache_root = _base_model_cache_root(base)
        if cache_root is None:
            _log(
                f"base model is not an owned HF snapshot cache; leaving it intact: {base}"
            )
            return
        if (
            last_midtrain.materialized is None
            or last_midtrain.skip_merge_reason is not None
        ):
            # A resumed downstream stage can make the base unnecessary without
            # leaving a local midtrain model. Its verified marker is the proof.
            evidence = [self.run_dir]
            any_glob = None
        else:
            evidence = [last_midtrain.materialized / "config.json"]
            any_glob = (last_midtrain.materialized, "*.safetensors")
        await asyncio.to_thread(
            _delete_tree_after_durable,
            cache_root,
            evidence=evidence,
            description="pinned base-model HF cache",
            any_file_glob=any_glob,
        )

    async def restore_stage(self, artifact: Artifact) -> Path:
        if artifact.materialized is not None:
            return artifact.materialized
        if not artifact.remote_prefix:
            raise RuntimeError(f"{artifact.arm}/{artifact.stage} has no restore source")
        from huggingface_hub import snapshot_download

        local_root = self.work / "restored" / self.run_id
        with self.telemetry.phase(
            "download",
            arm=artifact.arm,
            n_gpus=0,
            notes=f"restore {artifact.stage} parent from verified HF marker",
        ):
            await asyncio.to_thread(
                snapshot_download,
                repo_id=self.repo_id,
                repo_type="model",
                allow_patterns=[f"{artifact.remote_prefix}/**"],
                local_dir=str(local_root),
                token=self.token,
            )
        path = local_root / artifact.remote_prefix
        config_name = (
            "adapter_config.json" if artifact.stage == "aft" else "config.json"
        )
        if not (path / config_name).is_file():
            raise RuntimeError(
                f"restored {artifact.stage} artifact is incomplete: {path}"
            )
        artifact.materialized = path
        return path

    async def prepare_eval_parent(self, arm: str, parent: Artifact) -> Path:
        from experiments.prior_coins.glm_minimal_v1.pod import eval_glm

        parent_path = await self.restore_stage(parent)
        work_dir = (
            self.work
            / "eval_work"
            / self.run_id
            / arm
            / _stable_marker_digest(parent.marker)[:24]
        )
        with self.telemetry.phase(
            "eval_prepare",
            arm=arm,
            n_gpus=0,
            notes="serial parent hardlink + packed-expert conversion",
        ):
            prepared, _ = await asyncio.to_thread(
                eval_glm._prepare_parent_view, parent_path, work_dir, arm
            )
        return prepared

    async def cleanup_eval_parent(self, arm: str, prepared: Path) -> None:
        marker = prepared.parent / f"PREPARED_PARENT.{arm}.json"
        if prepared.exists():
            _log(f"deleting completed read-only eval parent view exactly: {prepared}")
            await asyncio.to_thread(shutil.rmtree, prepared)
        marker.unlink(missing_ok=True)

    async def publish_stage(self, artifact: Artifact) -> dict[str, Any]:
        if artifact.materialized is None:
            raise RuntimeError(
                f"cannot publish non-materialized {artifact.arm}/{artifact.stage}"
            )
        from scimt.publish import publish

        local = artifact.materialized
        (local / STAGE_PROVENANCE).write_text(
            json.dumps(dict(artifact.marker), indent=2) + "\n", encoding="utf-8"
        )
        size = sum(path.stat().st_size for path in local.rglob("*") if path.is_file())
        started = time.monotonic()
        with self.telemetry.phase(
            "publish",
            arm=artifact.arm,
            n_gpus=0,
            notes=f"{artifact.stage} -> {artifact.remote_prefix}",
        ) as phase:
            await publish(
                local,
                self.repo_id,
                base_model=contracts.MODEL_REPO,
                private=True,
                token=self.token,
                path_in_repo=artifact.remote_prefix,
            )
            await asyncio.to_thread(
                self._verify_remote_folder, local, artifact.remote_prefix
            )
            await asyncio.to_thread(
                self._write_and_verify_marker, artifact.remote_prefix, artifact.marker
            )
            elapsed = max(time.monotonic() - started, 1e-9)
            mbps = size / 1e6 / elapsed
            phase.update(mb_per_s=mbps)
            _log(
                f"publish verified: {artifact.arm}/{artifact.stage}, "
                f"{size / 1e9:.1f} GB at {mbps:.1f} MB/s"
            )
        cleanup = [artifact.checkpoint, *artifact.cleanup_after_publish]
        for path in cleanup:
            if path is not None and path.exists():
                _log(f"deleting verified/superseded sharded artifact exactly: {path}")
                shutil.rmtree(path)
        if artifact.train_dir is not None:
            prepared = artifact.train_dir / "prepared"
            if prepared.exists():
                shutil.rmtree(prepared)
        return {"path": artifact.remote_prefix, "mb_per_s": mbps}

    def _verify_remote_folder(self, local: Path, prefix: str) -> None:
        entries = self.api.list_repo_tree(
            self.repo_id,
            path_in_repo=prefix,
            repo_type="model",
            recursive=True,
            expand=True,
        )
        remote = {
            entry.path: int(entry.size or 0)
            for entry in entries
            if hasattr(entry, "size")
        }
        missing: list[str] = []
        mismatched: list[str] = []
        for path in local.rglob("*"):
            if not path.is_file() or path.name == STAGE_MARKER:
                continue
            name = f"{prefix}/{path.relative_to(local).as_posix()}"
            if name not in remote:
                missing.append(name)
            elif remote[name] != path.stat().st_size:
                mismatched.append(
                    f"{name}: remote={remote[name]} local={path.stat().st_size}"
                )
        if missing or mismatched:
            raise RuntimeError(
                f"remote verification failed for {prefix}: missing={missing[:10]}, "
                f"size_mismatch={mismatched[:10]}"
            )

    def _write_and_verify_marker(self, prefix: str, marker: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(marker), indent=2, sort_keys=True) + "\n"
        self.api.upload_file(
            path_or_fileobj=payload.encode(),
            path_in_repo=f"{prefix}/{STAGE_MARKER}",
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
            commit_message=f"complete {prefix}",
        )
        actual = self._remote_marker(prefix)
        if not marker_matches(actual, marker):
            raise RuntimeError(
                f"uploaded completion marker failed content check: {prefix}"
            )

    async def eval_endpoint(
        self,
        arm: str,
        endpoint: str,
        data: DataBundle,
        parent: Artifact,
        adapter: Artifact,
        prepared_parent: Path,
    ) -> EvalArtifact:
        eval_source = EXP / "pod" / "eval_glm.py"
        config_digest = sha256_file(eval_source)
        marker = stage_marker_payload(
            run_id=self.run_id,
            arm=arm,
            stage=f"eval_{endpoint}",
            config_sha256=config_digest,
            data_digest=data.eval_digest,
            steps=None,
            git_sha=_git_sha(),
            parent_digest=_canonical_digest(
                {
                    "parent": _stable_marker_digest(parent.marker),
                    "adapter": (
                        _stable_marker_digest(adapter.marker)
                        if endpoint == "post_aft"
                        else None
                    ),
                }
            ),
        )
        prefix = f"runs/{self.run_id}/eval/{arm}/{endpoint}"
        actual = (
            await asyncio.to_thread(self._remote_marker, prefix)
            if self.resume
            else None
        )
        if should_skip_stage(resume=self.resume, actual=actual, expected=marker):
            _log(f"{arm}/{endpoint}: verified HF marker matches; skipping")
            with self.telemetry.phase(
                "eval",
                arm=arm,
                n_gpus=0,
                notes=f"{endpoint}; resume: verified remote completion marker",
            ):
                pass
            artifact_key = _stable_marker_digest(marker)[:24]
            local = self.run_dir / "eval" / arm / endpoint / artifact_key
            return EvalArtifact(
                arm,
                endpoint,
                local if local.is_dir() else None,
                marker,
                prefix,
                True,
            )
        parent_path, adapter_path = await asyncio.gather(
            self.restore_stage(parent), self.restore_stage(adapter)
        )
        artifact_key = _stable_marker_digest(marker)[:24]
        destination = self.run_dir / "eval" / arm / endpoint / artifact_key
        destination.mkdir(parents=True, exist_ok=True)
        spec = {
            "arm": arm,
            "endpoint": endpoint,
            "parent": str(parent_path),
            "prepared_parent": str(prepared_parent),
            "adapter": str(adapter_path),
            "data_dir": str(data.eval_data),
            "results_dir": str(destination),
            "work_dir": str(self.work / "eval_work" / arm / endpoint / artifact_key),
        }
        if tuple(spec) != EVAL_WORKER_SPEC_KEYS:
            raise AssertionError(
                f"eval worker spec drift: {tuple(spec)} != {EVAL_WORKER_SPEC_KEYS}"
            )
        spec_path = destination / "worker_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        index = contracts.ARMS.index(arm) * 2 + ENDPOINTS.index(endpoint)
        devices = f"{2 * index},{2 * index + 1}"
        log_path = destination / "eval.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = devices
        environment["SCIMT_RUN_ID"] = self.run_id
        environment["SCIMT_EVAL_FALLBACK_LOCK"] = str(
            self.work / "eval_work" / self.run_id / "merged_fallback.lock"
        )
        pythonpath = str(REPO_ROOT / "src")
        if environment.get("PYTHONPATH"):
            pythonpath += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = pythonpath
        eval_python = os.environ.get(
            "SCIMT_EVAL_PYTHON", "/workspace/venv-glm-eval/bin/python"
        )
        with self.telemetry.phase("eval", arm=arm, n_gpus=2, notes=endpoint):
            process = await asyncio.create_subprocess_exec(
                eval_python,
                str(Path(__file__).resolve()),
                "--_eval-worker-spec",
                str(spec_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
                cwd=str(REPO_ROOT),
            )
            stdout, _ = await process.communicate()
            log_path.write_bytes(stdout)
            if process.returncode != 0:
                raise RuntimeError(
                    f"eval worker {arm}/{endpoint} exited {process.returncode}:\n"
                    f"{stdout.decode(errors='replace')[-10_000:]}"
                )
        if not any(destination.rglob("*.jsonl")):
            raise RuntimeError(f"eval worker wrote no result rows: {destination}")
        return EvalArtifact(arm, endpoint, destination, marker, prefix)

    async def _restore_eval(self, artifact: EvalArtifact) -> Path:
        from huggingface_hub import snapshot_download

        local_root = self.work / "restored_eval" / self.run_id
        await asyncio.to_thread(
            snapshot_download,
            repo_id=self.repo_id,
            repo_type="model",
            allow_patterns=[f"{artifact.remote_prefix}/**"],
            local_dir=str(local_root),
            token=self.token,
        )
        restored = local_root / artifact.remote_prefix
        cell = restored / f"{artifact.arm}-{artifact.endpoint}"
        if not cell.is_dir():
            raise RuntimeError(f"restored eval artifact is incomplete: {cell}")
        artifact.directory = restored
        return restored

    async def score_evals(
        self, artifacts: Sequence[EvalArtifact], data: DataBundle
    ) -> ScoreArtifact:
        for artifact in artifacts:
            if artifact.directory is None:
                await self._restore_eval(artifact)
        scoring_inputs = self.work / "score_inputs" / self.run_id
        await asyncio.to_thread(_assemble_score_inputs, artifacts, scoring_inputs)
        output = self.run_dir / "scores"
        try:
            from experiments.prior_coins.glm_minimal_v1 import score

            with self.telemetry.phase("score", n_gpus=0):
                scored = await asyncio.to_thread(
                    score.score_saved, scoring_inputs, data.eval_data
                )
                await asyncio.to_thread(score.write_outputs, scored, output)
        finally:
            if scoring_inputs.exists():
                shutil.rmtree(scoring_inputs)
        return ScoreArtifact(output, f"runs/{self.run_id}/scores")

    async def publish_eval(self, artifact: EvalArtifact) -> dict[str, Any]:
        if artifact.directory is None:
            raise RuntimeError("cannot publish an unmaterialized eval")
        return await self._publish_directory(
            artifact.directory,
            artifact.remote_prefix,
            arm=artifact.arm,
            marker=artifact.marker,
            notes=f"eval {artifact.endpoint}",
        )

    async def publish_scores(self, artifact: ScoreArtifact) -> dict[str, Any]:
        return await self._publish_directory(
            artifact.directory,
            artifact.remote_prefix,
            arm=None,
            marker=None,
            notes="scores.json and markdown summary",
        )

    async def cleanup_stage(self, artifact: Artifact) -> None:
        path = artifact.materialized
        if path is None or not path.exists():
            return
        if artifact.stage not in {"midtrain", "ift"}:
            return
        await asyncio.to_thread(
            _delete_tree_after_durable,
            path,
            evidence=[self.run_dir / "scores" / "scores.json"],
            description=f"{artifact.arm}/{artifact.stage} consolidated model",
        )
        artifact.materialized = None

    async def _publish_directory(
        self,
        local: Path,
        prefix: str,
        *,
        arm: str | None,
        marker: Mapping[str, Any] | None,
        notes: str,
    ) -> dict[str, Any]:
        size = sum(path.stat().st_size for path in local.rglob("*") if path.is_file())
        started = time.monotonic()
        with self.telemetry.phase(
            "publish", arm=arm, n_gpus=0, notes=f"{notes} -> {prefix}"
        ) as phase:
            await asyncio.to_thread(
                self.api.upload_folder,
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=str(local),
                path_in_repo=prefix,
                token=self.token,
                commit_message=f"publish {prefix}",
            )
            await asyncio.to_thread(self._verify_remote_folder, local, prefix)
            if marker is not None:
                await asyncio.to_thread(self._write_and_verify_marker, prefix, marker)
            elapsed = max(time.monotonic() - started, 1e-9)
            mbps = size / 1e6 / elapsed
            phase.update(mb_per_s=mbps)
        _log(f"publish verified: {prefix}, {size / 1e6:.1f} MB at {mbps:.1f} MB/s")
        return {"path": prefix, "mb_per_s": mbps}

    async def publish_metadata(self) -> dict[str, Any]:
        if self._ram_handle is not None:
            await asyncio.to_thread(self._ram_handle.stop, 10)
            self._ram_handle = None
        # No upload is active now, so purging the global Xet store cannot race
        # another publisher.  Stage-specific sharded saves were already
        # removed immediately after their own verification.
        await asyncio.to_thread(_purge_xet_cache)
        staging = self.work / "metadata" / self.run_id
        staging.mkdir(parents=True, exist_ok=True)
        for source in self.run_dir.rglob("*"):
            relative = source.relative_to(self.run_dir)
            if relative.parts and relative.parts[0] in {"eval", "scores"}:
                continue  # these have their own verified run prefixes
            if source.is_file():
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        prefix = f"runs/{self.run_id}/metadata"
        result = await self._publish_directory(
            staging,
            prefix,
            arm=None,
            marker=None,
            notes="run logs, router health, RAM and phase telemetry",
        )
        # The publish phase row is appended when the timed upload closes.
        # Refresh this one tiny file and log its speed without appending a new
        # telemetry row, so the remote copy contains the main publish's final
        # row (avoids an infinite self-referential upload loop).
        refresh_started = time.monotonic()
        refresh_bytes = self.telemetry.path.stat().st_size
        staged_telemetry = staging / "telemetry.jsonl"
        shutil.copy2(self.telemetry.path, staged_telemetry)
        await asyncio.to_thread(
            self.api.upload_file,
            path_or_fileobj=str(staged_telemetry),
            path_in_repo=f"{prefix}/telemetry.jsonl",
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
            commit_message=f"finalize telemetry {self.run_id}",
        )
        refresh_seconds = max(time.monotonic() - refresh_started, 1e-9)
        refresh_mbps = refresh_bytes / 1e6 / refresh_seconds
        await asyncio.to_thread(self._verify_remote_folder, staging, prefix)
        _log(
            f"telemetry final refresh verified: {refresh_bytes / 1e6:.3f} MB "
            f"at {refresh_mbps:.1f} MB/s"
        )
        return result

    async def close(self) -> None:
        if self._ram_handle is not None:
            await asyncio.to_thread(self._ram_handle.stop, 10)
            self._ram_handle = None


def _consolidate_glm(
    checkpoint: Path, base_model: Path, out: Path, result_dir: Path
) -> Path:
    """Consolidate with the proven GLM shape and a 110B-sized timeout."""

    if (out / "config.json").is_file() and list(out.glob("*.safetensors")):
        return out
    if out.exists():
        _log(f"discarding exactly partial consolidated artifact {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    weights = list(checkpoint.glob("*.safetensors"))
    log_path = (
        result_dir
        / f"consolidate_{checkpoint.parent.parent.parent.name}_{checkpoint.name}.log"
    )
    if (checkpoint / "config.json").is_file() and weights:

        def link_or_copy(source: Path, destination: Path) -> None:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        for source in weights:
            link_or_copy(source, out / source.name)
        for pattern in (
            "model*.json",
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens*",
        ):
            for source in checkpoint.glob(pattern):
                if source.is_file() and not (out / source.name).exists():
                    link_or_copy(source, out / source.name)
        for source in base_model.iterdir():
            prefixes = (
                "tokenizer",
                "special_tokens",
                "vocab",
                "merges",
                "added_tokens",
                "processor",
                "chat_template",
                "generation_config",
            )
            if (
                source.is_file()
                and source.name.startswith(prefixes)
                and not (out / source.name).exists()
            ):
                shutil.copy2(source, out / source.name)
        log_path.write_text(
            "merged checkpoint hardlinked directly as HF model files\n",
            encoding="utf-8",
        )
        return out
    result = subprocess.run(
        [
            sys.executable,
            str(CONSOLIDATOR),
            "--checkpoint-dir",
            str(checkpoint),
            "--base-model",
            str(base_model),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=CONSOLIDATE_TIMEOUT_S,
        check=False,
    )
    log_path.write_text(
        result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"consolidation failed for {checkpoint}:\n{result.stderr[-4_000:]}"
        )
    if not (out / "config.json").is_file() or not list(out.glob("*.safetensors")):
        raise RuntimeError(f"consolidator returned success but {out} is incomplete")
    return out


def _run_eval_worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if tuple(spec) != EVAL_WORKER_SPEC_KEYS:
        raise RuntimeError(
            f"eval worker spec keys {tuple(spec)} != {EVAL_WORKER_SPEC_KEYS}"
        )
    module = importlib.import_module(
        "experiments.prior_coins.glm_minimal_v1.pod.eval_glm"
    )
    function = getattr(module, "evaluate_endpoint", None)
    if not callable(function):
        raise RuntimeError("eval_glm.evaluate_endpoint is unavailable")
    kwargs = {
        key: Path(spec[key])
        if key.endswith(("parent", "adapter", "dir"))
        else spec[key]
        for key in EVAL_WORKER_SPEC_KEYS
    }
    result = function(**kwargs)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


def _parse_arms(raw: str) -> tuple[str, ...]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--arms contains duplicates")
    unknown = set(requested) - set(contracts.ARMS)
    if unknown or not requested:
        raise ValueError(
            f"--arms must be a nonempty subset of {contracts.ARMS}; "
            f"unknown={sorted(unknown)}"
        )
    # Preserve the scientifically costed charter-then-coin order.
    return tuple(arm for arm in contracts.ARMS if arm in requested)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--arms", default=",".join(contracts.ARMS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument(
        "--setup-state", type=Path, default=DEFAULT_SETUP_STATE, help=argparse.SUPPRESS
    )
    parser.add_argument("--_eval-worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if args._eval_worker_spec is not None:
        await asyncio.to_thread(_run_eval_worker, args._eval_worker_spec)
        return 0
    if not args.run_id or not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("--run-id is required and must be a safe UTC-like slug")
    arms = _parse_arms(args.arms)
    chain = ProductionChain(
        run_id=args.run_id,
        resume=args.resume,
        work=args.work_dir,
        setup_state=args.setup_state,
    )
    try:
        await execute_plan(chain, arms)
    finally:
        await chain.close()
    _log("CHAIN_COMPLETE: every requested artifact publish was joined and verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
