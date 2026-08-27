#!/usr/bin/env python3
"""Resumable single-pod chain for ``glm_minimal_v1``.

The operator owns the already-running pod.  This module only supervises work
on that machine: it never imports a pod lifecycle client and has no create,
stop, or teardown operation.

Production order is::

    preflight -> data -> join base download
    charter/coin/control midtrain/merge/IFT/merge, one arm at a time
    nine 4-GPU AFT cells in two-cell rounds
    one prepared parent -> four concurrent 2-GPU evals, one arm at a time
    publish remaining artifacts -> join every background upload

Heavy pod dependencies are imported inside the functions that need them so
the orchestration and safety gates stay CPU-testable.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
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
import warnings
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
STAGE_RUN_METADATA = "stage_run_metadata.json"
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
POSTHOC_LOSS_WORKER_SPEC_KEYS = (
    "model_path",
    "samples_path",
    "output_path",
    "checkpoint_step",
)
ROUTER_PLUGIN = "scimt.train.axolotl_plugins.RouterHealthPlugin"
CONSOLIDATE_TIMEOUT_S = 6 * 3600
TRAIN_PORT_BASE = 29_500
PREPROCESS_PORT_BASE = 29_600
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_MIX_RATIO_TOLERANCE_PP = 2.0
POSTHOC_LOSS_MAX_LENGTH = 1024

_AFT_CONCURRENCY_SLOT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "aft_concurrency_slot", default=None
)
_EVAL_CONCURRENCY_SLOT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "eval_concurrency_slot", default=None
)


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
    cell: str | None = None,
) -> dict[str, Any]:
    """Stable marker content.  Only ``git_sha`` is deliberately volatile."""

    payload = {
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
    if cell is not None:
        payload["cell"] = cell
    return payload


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
        "tokens_inspected": total,
        "trained_tokens": trained,
        "masked_tokens": masked,
        "trained_fraction": fraction,
        "terminator_token_id": terminator_token_id,
        "terminator_trained": True,
    }


def chat_stage_dose_report(
    label_report: Mapping[str, Any],
    *,
    steps: int,
    packed_positions_presented: int | None,
    examples_presented: int | None = None,
) -> dict[str, Any]:
    """Turn the label-mask sample into an explicitly estimated training dose."""

    sample_rows = int(label_report["rows_inspected"])
    sample_tokens = int(label_report["tokens_inspected"])
    labelled_fraction = float(label_report["trained_fraction"])
    if steps <= 0 or sample_rows <= 0 or sample_tokens <= 0:
        raise ValueError("chat-stage dose inputs must be positive")
    if packed_positions_presented is not None:
        positions = int(packed_positions_presented)
        positions_status = "measured_from_fixed_packed_training_geometry"
    else:
        if examples_presented is None or examples_presented <= 0:
            raise ValueError(
                "unpacked chat stages need a positive examples_presented count"
            )
        positions = round(sample_tokens / sample_rows * examples_presented)
        positions_status = "estimated_from_label_mask_sample_for_unpacked_stage"
    labelled = round(positions * labelled_fraction)
    return {
        "packed_positions_presented": positions,
        "packed_positions_status": positions_status,
        "assistant_labelled_tokens_presented": labelled,
        "assistant_labelled_tokens_status": (
            "estimated_from_label_mask_sample_not_exact_measurement"
        ),
        "mean_assistant_labelled_tokens_per_step": labelled / steps,
        "labelled_token_fraction": labelled_fraction,
        "label_mask_sample_rows": sample_rows,
        "label_mask_sample_tokens": sample_tokens,
    }


def _write_stage_run_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepared_dataset_dir(rendered: Path) -> Path:
    body = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    return Path(body["dataset_prepared_path"])


def glm_source_mix_report(source_tokens: Mapping[str, int]) -> dict[str, Any]:
    """Describe the realised GLM-side task/replay ratio."""

    if set(source_tokens) != {"task", "dolmino"}:
        raise ValueError("GLM source counts must contain exactly task and dolmino")
    task = int(source_tokens["task"])
    dolmino = int(source_tokens["dolmino"])
    if task < 0 or dolmino <= 0:
        raise ValueError(
            "GLM task source tokens must be nonnegative and dolmino must be positive"
        )
    total = task + dolmino
    task_fraction = task / total
    return {
        "task_glm_tokens": task,
        "dolmino_glm_tokens": dolmino,
        "total_glm_tokens": total,
        "task_glm_fraction": task_fraction,
        "dolmino_glm_fraction": dolmino / total,
        "task_to_dolmino_ratio": task / dolmino,
        "mix_ratio_deviation_pp": abs(task_fraction - 0.5) * 100,
        "tokenizer": contracts.GLM_TOKENIZER,
        "tokenizer_revision": contracts.GLM_TOKENIZER_REVISION,
        "status": "measured_on_pod_from_actual_mix",
    }


def _mix_ratio_tolerance_pp(
    environment: Mapping[str, str] | None = None,
) -> float:
    raw = (os.environ if environment is None else environment).get(
        "SCIMT_MIDTRAIN_MIX_RATIO_TOLERANCE_PP",
        str(DEFAULT_MIX_RATIO_TOLERANCE_PP),
    )
    try:
        tolerance = float(raw)
    except ValueError as exc:
        raise ValueError(
            "SCIMT_MIDTRAIN_MIX_RATIO_TOLERANCE_PP must be numeric"
        ) from exc
    if not 0 <= tolerance <= 50:
        raise ValueError(
            "SCIMT_MIDTRAIN_MIX_RATIO_TOLERANCE_PP must be between 0 and 50"
        )
    return tolerance


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


def midtrain_loss_holdouts(
    manifest: Mapping[str, Any], arm: str
) -> dict[str, list[str]]:
    """Read and validate the genuinely train-excluded stream samples."""

    try:
        holdout = manifest["realized"]["midtrain_mixes"][arm]["loss_holdout"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"manifest has no midtraining loss holdout for {arm}") from exc
    output: dict[str, list[str]] = {}
    for source in ("task", "dolmino"):
        record = holdout.get(source) if isinstance(holdout, Mapping) else None
        texts = record.get("texts") if isinstance(record, Mapping) else None
        empty_source = (
            isinstance(record, Mapping)
            and record.get("docs") == 0
            and record.get("tokens") == 0
            and record.get("rows") == []
        )
        if empty_source:
            output[source] = []
            continue
        if (
            not isinstance(texts, Sequence)
            or isinstance(texts, (str, bytes))
            or not texts
            or any(not isinstance(text, str) or not text for text in texts)
            or record.get("excluded_from_training_mix") is not True
        ):
            raise RuntimeError(
                f"manifest {arm}/{source} loss holdout is absent or not train-excluded"
            )
        output[source] = list(texts)
    return output


def assert_midtrain_holdouts_excluded(
    data_path: Path, samples: Mapping[str, Sequence[str]]
) -> None:
    """Re-assert on the pod that no held-out loss document entered training."""

    heldout = {
        hashlib.sha256(text.encode()).digest()
        for texts in samples.values()
        for text in texts
    }
    overlap: set[bytes] = set()
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.sha256(str(row.get("text", "")).encode()).digest()
            if digest in heldout:
                overlap.add(digest)
    if overlap:
        raise RuntimeError(
            f"post-hoc loss holdout leaked into {data_path}: "
            f"{len(overlap)} document digest(s) overlap"
        )


def _stream_causal_loss(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> dict[str, Any]:
    """Compute token-weighted causal loss without changing training behavior."""

    import torch

    if max_length < 2:
        raise ValueError("max_length must be at least two")
    input_device = model.get_input_embeddings().weight.device
    weighted_loss = 0.0
    predicted_tokens = 0
    chunks = 0
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(token_ids) < 2:
            continue
        start = 0
        while start < len(token_ids) - 1:
            stop = min(len(token_ids), start + max_length)
            chunk = token_ids[start:stop]
            inputs = torch.tensor([chunk], dtype=torch.long, device=input_device)
            with torch.inference_mode():
                loss = model(input_ids=inputs, labels=inputs).loss
            n_predictions = len(chunk) - 1
            weighted_loss += float(loss.detach().float().cpu()) * n_predictions
            predicted_tokens += n_predictions
            chunks += 1
            if stop == len(token_ids):
                break
            start = stop - 1
    if predicted_tokens == 0:
        raise RuntimeError("post-hoc loss sample has zero predictable tokens")
    return {
        "mean_loss": weighted_loss / predicted_tokens,
        "predicted_tokens": predicted_tokens,
        "documents": len(texts),
        "forward_chunks": chunks,
    }


def score_midtrain_stream_losses(
    model_path: Path,
    samples: Mapping[str, Sequence[str]],
    *,
    checkpoint_step: int,
    max_length: int = POSTHOC_LOSS_MAX_LENGTH,
) -> dict[str, Any]:
    """Standalone final-checkpoint task/replay loss pass (explicitly post-hoc)."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if set(samples) != {"task", "dolmino"}:
        raise ValueError("post-hoc samples must contain exactly task and dolmino")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.eval()
    streams: dict[str, dict[str, Any]] = {}
    for source in ("task", "dolmino"):
        texts = list(samples[source])
        if texts:
            streams[source] = _stream_causal_loss(
                model, tokenizer, texts, max_length=max_length
            )
        else:
            streams[source] = {
                "mean_loss": None,
                "predicted_tokens": 0,
                "documents": 0,
                "forward_chunks": 0,
                "status": "not_applicable_empty_source",
            }
    return {
        "schema_version": 1,
        "evaluation_timing": "post_hoc_final_checkpoint",
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(model_path),
        "max_forward_length": max_length,
        "loss_definition": "token-weighted next-token cross-entropy",
        "samples_excluded_from_training": True,
        "streams": streams,
    }


def _run_posthoc_loss_worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != set(POSTHOC_LOSS_WORKER_SPEC_KEYS):
        actual = tuple(spec) if isinstance(spec, dict) else type(spec).__name__
        raise RuntimeError(
            f"post-hoc loss worker spec keys {actual} != "
            f"{POSTHOC_LOSS_WORKER_SPEC_KEYS}"
        )
    samples = json.loads(Path(spec["samples_path"]).read_text(encoding="utf-8"))
    report = score_midtrain_stream_losses(
        Path(spec["model_path"]),
        samples,
        checkpoint_step=int(spec["checkpoint_step"]),
    )
    _write_stage_run_metadata(Path(spec["output_path"]), report)


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
    aft: Mapping[str, Path]
    aft_digests: Mapping[str, str]
    eval_data: Path
    eval_digest: str
    artifact_revision: str
    manifest: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    arm: str
    stage: str
    config_path: Path
    config_sha256: str
    data_digest: str
    steps: int
    marker: Mapping[str, Any]
    cell: str | None = None
    train_dir: Path | None = None
    checkpoint: Path | None = None
    materialized: Path | None = None
    merge_parent: Path | None = None
    remote_prefix: str | None = None
    already_remote: bool = False
    cleanup_after_publish: list[Path] = field(default_factory=list)
    cleanup_after_merge: list[Path] = field(default_factory=list)
    skip_merge_reason: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)


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
    async def aft(
        self, arm: str, cell: str, data: DataBundle, parent: Artifact
    ) -> Artifact: ...
    async def prepare_eval_parent(self, arm: str, parent: Artifact) -> Path: ...
    async def cleanup_eval_parent(self, arm: str, prepared: Path) -> None: ...
    async def eval_endpoint(
        self,
        arm: str,
        endpoint: str,
        data: DataBundle,
        parent: Artifact,
        adapter: Artifact | None,
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


async def _run_aft_cell(
    operations: ChainOperations,
    arm: str,
    cell: str,
    data: DataBundle,
    parent: Artifact,
    slot: int,
) -> Artifact:
    token = _AFT_CONCURRENCY_SLOT.set(slot)
    try:
        return await operations.aft(arm, cell, data, parent)
    finally:
        _AFT_CONCURRENCY_SLOT.reset(token)


async def _run_eval_endpoint(
    operations: ChainOperations,
    arm: str,
    endpoint: str,
    data: DataBundle,
    parent: Artifact,
    adapter: Artifact | None,
    prepared_parent: Path,
    slot: int,
) -> EvalArtifact:
    token = _EVAL_CONCURRENCY_SLOT.set(slot)
    try:
        return await operations.eval_endpoint(
            arm, endpoint, data, parent, adapter, prepared_parent
        )
    finally:
        _EVAL_CONCURRENCY_SLOT.reset(token)


def _required_slot(variable: contextvars.ContextVar[int | None], stage: str) -> int:
    slot = variable.get()
    if slot is None:
        raise RuntimeError(f"{stage} must be launched by execute_plan")
    return slot


def _aft_slot_resources(slot: int) -> tuple[str, int, int]:
    if slot not in (0, 1):
        raise ValueError(f"AFT concurrency slot must be 0 or 1, got {slot}")
    first_gpu = slot * 4
    return (
        ",".join(str(gpu) for gpu in range(first_gpu, first_gpu + 4)),
        TRAIN_PORT_BASE + slot,
        PREPROCESS_PORT_BASE + slot,
    )


def _eval_slot_devices(slot: int) -> str:
    if slot not in range(len(contracts.ENDPOINTS_PER_ARM)):
        raise ValueError(
            f"eval concurrency slot must be in [0, "
            f"{len(contracts.ENDPOINTS_PER_ARM) - 1}], got {slot}"
        )
    first_gpu = slot * 2
    return f"{first_gpu},{first_gpu + 1}"


async def execute_plan(
    operations: ChainOperations,
    arms: Sequence[str],
    aft_cells: Sequence[str] = contracts.AFT_CELLS,
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

        aft_keys = [
            (arm, cell)
            for arm, cell in contracts.aft_cell_keys()
            if arm in arms and cell in aft_cells
        ]
        adapters: dict[tuple[str, str], Artifact] = {}
        for offset in range(0, len(aft_keys), 2):
            round_keys = aft_keys[offset : offset + 2]
            round_tasks: list[asyncio.Task[Artifact]] = []
            async with asyncio.TaskGroup() as group:
                for slot, (arm, cell) in enumerate(round_keys):
                    round_tasks.append(
                        group.create_task(
                            _run_aft_cell(
                                operations,
                                arm,
                                cell,
                                data,
                                parents[arm],
                                slot,
                            )
                        )
                    )
            adapters.update(
                zip(
                    round_keys,
                    (task.result() for task in round_tasks),
                    strict=True,
                )
            )
            publishes.raise_completed_failures()

        # A prepared parent is ~199 GB, so the arm loop is a disk-safety gate:
        # restore one arm's inputs, prepare exactly one parent, fan out its four
        # endpoints across eight GPUs, and delete it before the next arm.
        eval_list: list[EvalArtifact] = []
        endpoints = (
            "pre_aft",
            *(contracts.post_aft_endpoint(cell) for cell in aft_cells),
        )
        for arm in arms:
            await _materialized(parents[arm], operations)
            for cell in aft_cells:
                await _materialized(adapters[(arm, cell)], operations)
            prepared_parent = await operations.prepare_eval_parent(arm, parents[arm])
            try:
                eval_tasks: list[asyncio.Task[EvalArtifact]] = []
                async with asyncio.TaskGroup() as group:
                    for slot, endpoint in enumerate(endpoints):
                        eval_tasks.append(
                            group.create_task(
                                _run_eval_endpoint(
                                    operations,
                                    arm,
                                    endpoint,
                                    data,
                                    parents[arm],
                                    (
                                        None
                                        if endpoint == "pre_aft"
                                        else adapters[
                                            (
                                                arm,
                                                endpoint.removeprefix("post_aft__"),
                                            )
                                        ]
                                    ),
                                    prepared_parent,
                                    slot,
                                )
                            )
                        )
                eval_list.extend(task.result() for task in eval_tasks)
            finally:
                await operations.cleanup_eval_parent(arm, prepared_parent)

        scores = await operations.score_evals(eval_list, data)

        for (arm, cell), adapter in adapters.items():
            if not adapter.already_remote:
                publishes.launch(
                    f"{arm}/aft/{cell}", operations.publish_stage(adapter)
                )
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
    """Assemble chain-shaped endpoint trees into the scorer's 12-cell layout."""

    by_cell = {(item.arm, item.endpoint): item for item in artifacts}
    expected = set(contracts.eval_endpoint_keys())
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
        # Publishing is outward-facing, so private is the default and going
        # public is a deliberate, recorded act rather than a side effect of
        # ``create_repo(exist_ok=True)`` silently ignoring ``private`` on a
        # repo that already exists.
        self.publish_private = os.environ.get(
            "SCIMT_HF_PRIVATE", "1"
        ).strip().lower() not in {"0", "false", "no"}
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
            filename = contracts.MIDTRAIN_FILENAMES[arm]
            path = root / filename
            expected = manifest["files"][filename]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{filename} digest {actual} != manifest {expected}")
            paths[arm] = path
            digests[arm] = actual
        aft_paths: dict[str, Path] = {}
        aft_digests: dict[str, str] = {}
        for cell in contracts.AFT_CELLS:
            filename = contracts.AFT_FILENAMES[cell]
            path = root / filename
            expected = manifest["files"][filename]["sha256"]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{filename} digest {actual} != manifest {expected}")
            aft_paths[cell] = path
            aft_digests[cell] = actual

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
            aft=aft_paths,
            aft_digests=aft_digests,
            eval_data=eval_root,
            eval_digest=_tree_digest(eval_root),
            artifact_revision=revision,
            manifest=manifest,
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
        cell: str | None = None,
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
            cell=cell,
        )
        prefix = stage_prefix(self.run_id, arm, stage)
        if cell is not None:
            prefix = f"{prefix}/{cell}"
        return config_digest, marker, prefix

    def _aft_contract(
        self,
        arm: str,
        cell: str,
        data: DataBundle,
        parent_marker: Mapping[str, Any],
    ) -> tuple[Path, int, str, dict[str, Any], str]:
        config = self._config("aft")
        steps = contracts.aft_steps(contracts.AFT_ROWS, contracts.AFT_EPOCHS)
        digest, marker, prefix = self._artifact_contract(
            arm=arm,
            stage="aft",
            config=config,
            data_digest=data.aft_digests[cell],
            steps=steps,
            parent_digest=_stable_marker_digest(parent_marker),
            cell=cell,
        )
        return config, steps, digest, marker, prefix

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
        tokens, steps, source_mix = await asyncio.to_thread(
            self._glm_schedule, arm, data.midtrain[arm], base
        )
        tolerance = _mix_ratio_tolerance_pp()
        ratio_message = (
            f"{arm}/midtrain GLM TOKEN MIX: "
            f"task={source_mix['task_glm_tokens']:,} "
            f"({100 * source_mix['task_glm_fraction']:.2f}%), "
            f"dolmino={source_mix['dolmino_glm_tokens']:,} "
            f"({100 * source_mix['dolmino_glm_fraction']:.2f}%), "
            f"task:dolmino={source_mix['task_to_dolmino_ratio']:.4f}:1, "
            f"deviation={source_mix['mix_ratio_deviation_pp']:.2f}pp "
            f"(warning tolerance={tolerance:.2f}pp)"
        )
        _log(ratio_message)
        if source_mix["mix_ratio_deviation_pp"] > tolerance:
            warning = f"LOUD MIX-RATIO WARNING: {ratio_message}"
            _log(warning)
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
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
            ) as phase:
                phase.update(
                    task_glm_tokens=source_mix["task_glm_tokens"],
                    dolmino_glm_tokens=source_mix["dolmino_glm_tokens"],
                    task_glm_fraction=source_mix["task_glm_fraction"],
                    mix_ratio_deviation_pp=source_mix["mix_ratio_deviation_pp"],
                )
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
            stage_metadata={"glm_source_mix": source_mix},
        )

    def _glm_schedule(
        self, arm: str, data_path: Path, tokenizer_dir: Path
    ) -> tuple[int, int, dict[str, Any]]:
        cache_path = self.work / "cache" / "glm_step_schedule.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        file_digest = sha256_file(data_path)
        key = f"{arm}:{file_digest}:{contracts.GLM_TOKENIZER_REVISION}"
        if key in cache and isinstance(cache[key].get("source_tokens"), Mapping):
            record = cache[key]
            source_mix = glm_source_mix_report(record["source_tokens"])
            return int(record["glm_tokens"]), int(record["steps"]), source_mix
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        total = 0
        source_tokens = {"task": 0, "dolmino": 0}
        batch: list[tuple[str, str]] = []

        def consume() -> None:
            nonlocal total
            if batch:
                encoded = tokenizer(
                    [text for text, _source in batch], add_special_tokens=True
                )["input_ids"]
                for ids, (_text, source) in zip(encoded, batch, strict=True):
                    count = len(ids)
                    total += count
                    source_tokens[source] += count
                batch.clear()

        with data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row.get("source")
                if source not in source_tokens:
                    raise RuntimeError(
                        f"{data_path}: midtrain row has invalid source tag {source!r}"
                    )
                batch.append((row["text"], source))
                if len(batch) >= 512:
                    consume()
        consume()
        steps = contracts.midtrain_steps(total)
        cache[key] = {
            "arm": arm,
            "file_sha256": file_digest,
            "tokenizer_revision": contracts.GLM_TOKENIZER_REVISION,
            "glm_tokens": total,
            "source_tokens": source_tokens,
            "source_mix": glm_source_mix_report(source_tokens),
            "presentations": contracts.MIDTRAIN_PRESENTATIONS,
            "steps": steps,
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, indent=2) + "\n")
        temporary.replace(cache_path)
        shutil.copy2(cache_path, self.run_dir / "glm_step_schedule.json")
        return total, steps, glm_source_mix_report(source_tokens)

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

    async def aft(
        self, arm: str, cell: str, data: DataBundle, parent: Artifact
    ) -> Artifact:
        config, steps, digest, marker, prefix = self._aft_contract(
            arm, cell, data, parent.marker
        )
        actual = (
            await asyncio.to_thread(self._remote_marker, prefix)
            if self.resume
            else None
        )
        if should_skip_stage(resume=self.resume, actual=actual, expected=marker):
            _log(f"{arm}/aft/{cell}: verified HF marker matches; skipping")
            with self.telemetry.phase(
                "aft",
                arm=arm,
                n_gpus=0,
                steps=steps,
                notes=f"cell={cell}; resume: verified remote completion marker",
            ):
                pass
            return Artifact(
                arm,
                "aft",
                config,
                digest,
                data.aft_digests[cell],
                steps,
                marker,
                cell=cell,
                remote_prefix=prefix,
                already_remote=True,
            )
        parent_path = await self.restore_stage(parent)
        slot = _required_slot(_AFT_CONCURRENCY_SLOT, "AFT")
        gpu_ids, rendezvous_port, preprocess_port = _aft_slot_resources(slot)
        return await self._train(
            arm=arm,
            stage_name="aft",
            cell=cell,
            config=config,
            data_path=data.aft[cell],
            data_digest=data.aft_digests[cell],
            steps=steps,
            tokens=None,
            parent=parent_path,
            marker=marker,
            config_digest=digest,
            n_gpus=4,
            visible_devices=gpu_ids,
            rendezvous_port=rendezvous_port,
            preprocess_port=preprocess_port,
            remote_prefix=prefix,
        )

    async def _train(
        self,
        *,
        arm: str,
        stage_name: str,
        cell: str | None = None,
        config: Path,
        data_path: Path,
        data_digest: str,
        steps: int,
        tokens: int | None,
        parent: Path,
        marker: Mapping[str, Any],
        config_digest: str,
        n_gpus: int,
        stage_metadata: Mapping[str, Any] | None = None,
        visible_devices: str | None = None,
        rendezvous_port: int | None = None,
        preprocess_port: int | None = None,
        remote_prefix: str | None = None,
    ) -> Artifact:
        from scimt.train import TrainConfig
        from scimt.train.axolotl import LocalExecutor, render_stage

        stage = _load_local_stage(config)
        artifact_key = _stable_marker_digest(marker)[:24]
        stage_path = Path(stage_name) if cell is None else Path(stage_name) / cell
        out = self.work / "train" / self.run_id / arm / stage_path / artifact_key
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
        metadata_path = out / STAGE_RUN_METADATA
        run_metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "arm": arm,
            "stage": stage_name,
            "optimizer_steps": steps,
            **dict(stage_metadata or {}),
        }
        if cell is not None:
            run_metadata["cell"] = cell
        stage_label = stage_name if cell is None else f"{stage_name}/{cell}"
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
                    if metadata_path.is_file():
                        saved_metadata = json.loads(
                            metadata_path.read_text(encoding="utf-8")
                        )
                        if isinstance(saved_metadata, Mapping):
                            run_metadata.update(saved_metadata)
            except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
                _log(
                    f"{arm}/{stage_label}: local completion cache is invalid; "
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
            notes=f"cell={cell}" if cell is not None else None,
        ) as phase:
            source_mix = run_metadata.get("glm_source_mix")
            if isinstance(source_mix, Mapping):
                phase.update(
                    task_glm_tokens=source_mix.get("task_glm_tokens"),
                    dolmino_glm_tokens=source_mix.get("dolmino_glm_tokens"),
                    task_glm_fraction=source_mix.get("task_glm_fraction"),
                    mix_ratio_deviation_pp=source_mix.get(
                        "mix_ratio_deviation_pp"
                    ),
                )
            if (
                stage_name in {"ift", "aft"}
                and "chat_template_dose" not in run_metadata
            ):
                label_report = await asyncio.to_thread(
                    sft_label_mask_gate,
                    rendered,
                    self.run_dir / f"{arm}_{stage_label.replace('/', '_')}_label_mask.json",
                    main_process_port=(
                        preprocess_port
                        if preprocess_port is not None
                        else PREPROCESS_PORT_BASE + contracts.ARMS.index(arm)
                    ),
                )
                rendered_body = yaml.safe_load(rendered.read_text(encoding="utf-8"))
                sample_packing = bool(rendered_body.get("sample_packing"))
                packed_positions = tokens if sample_packing else None
                examples_presented = None
                if not sample_packing:
                    examples_presented = (
                        steps
                        * int(rendered_body["micro_batch_size"])
                        * int(rendered_body["gradient_accumulation_steps"])
                        * n_gpus
                    )
                dose = chat_stage_dose_report(
                    label_report,
                    steps=steps,
                    packed_positions_presented=packed_positions,
                    examples_presented=examples_presented,
                )
                run_metadata["chat_template_dose"] = {
                    **dose,
                    "sample_packing": sample_packing,
                    "measurement_note": (
                        "assistant-labelled total is an extrapolation from the "
                        "deterministic label-mask sample, not an exact count"
                    ),
                }
            dose = run_metadata.get("chat_template_dose")
            if isinstance(dose, Mapping):
                phase.update(
                    **{
                        key: dose[key]
                        for key in (
                            "packed_positions_presented",
                            "packed_positions_status",
                            "assistant_labelled_tokens_presented",
                            "assistant_labelled_tokens_status",
                            "mean_assistant_labelled_tokens_per_step",
                            "labelled_token_fraction",
                            "label_mask_sample_rows",
                            "label_mask_sample_tokens",
                        )
                        if key in dose
                    }
                )
            if local_complete:
                _log(
                    f"{arm}/{stage_label}: reusing content-verified local completion cache"
                )
            else:
                await self._run_executor_with_devices(
                    LocalExecutor(),
                    rendered,
                    out,
                    stage,
                    visible_devices,
                    rendezvous_port=rendezvous_port,
                )
        if not checkpoint.is_dir():
            raise RuntimeError(
                f"{arm}/{stage_label}: expected final checkpoint {checkpoint} is absent"
            )
        router = out / "router_health.jsonl"
        if not router.is_file() or router.stat().st_size == 0:
            raise RuntimeError(
                f"{arm}/{stage_label}: RouterHealthPlugin produced no {router}"
            )
        local_marker = out / STAGE_MARKER
        temporary_marker = local_marker.with_suffix(".json.tmp")
        temporary_marker.write_text(
            json.dumps(dict(marker), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(local_marker)
        _write_stage_run_metadata(metadata_path, run_metadata)
        shutil.copy2(metadata_path, checkpoint / STAGE_RUN_METADATA)
        shutil.copy2(
            metadata_path,
            self.run_dir
            / f"{arm}_{stage_label.replace('/', '_')}_{STAGE_RUN_METADATA}",
        )
        shutil.copy2(
            router,
            self.run_dir / f"{arm}_{stage_label.replace('/', '_')}_router_health.jsonl",
        )
        return Artifact(
            arm=arm,
            stage=stage_name,
            config_path=config,
            config_sha256=config_digest,
            data_digest=data_digest,
            steps=steps,
            marker=marker,
            cell=cell,
            train_dir=out,
            checkpoint=checkpoint,
            materialized=checkpoint if stage_name == "aft" else None,
            merge_parent=parent,
            remote_prefix=(
                remote_prefix or stage_prefix(self.run_id, arm, stage_name)
            ),
            run_metadata=run_metadata,
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

    async def _score_midtrain_losses_posthoc(
        self, artifact: Artifact, model_path: Path
    ) -> dict[str, Any]:
        if self._data is None:
            raise RuntimeError("post-hoc midtraining loss pass has no data manifest")
        samples = midtrain_loss_holdouts(self._data.manifest, artifact.arm)
        await asyncio.to_thread(
            assert_midtrain_holdouts_excluded,
            self._data.midtrain[artifact.arm],
            samples,
        )
        samples_path = self.run_dir / f"{artifact.arm}_midtrain_loss_holdout.json"
        output_path = self.run_dir / f"{artifact.arm}_midtrain_stream_loss.json"
        spec_path = self.run_dir / f"{artifact.arm}_midtrain_stream_loss_spec.json"
        _write_stage_run_metadata(samples_path, samples)
        spec = {
            "model_path": str(model_path),
            "samples_path": str(samples_path),
            "output_path": str(output_path),
            "checkpoint_step": artifact.steps,
        }
        _write_stage_run_metadata(spec_path, spec)
        log_path = self.run_dir / f"{artifact.arm}_midtrain_stream_loss.log"
        with self.telemetry.phase(
            "midtrain_stream_loss",
            arm=artifact.arm,
            n_gpus=8,
            notes=(
                "post-hoc final-checkpoint pass on fixed train-excluded task and "
                "dolmino samples; not in-loop evaluation"
            ),
        ) as phase:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_posthoc-loss-worker-spec",
                    str(spec_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=2 * 3600,
                cwd=str(REPO_ROOT),
            )
            log_path.write_text(
                result.stdout[-100_000:]
                + "\n--- STDERR ---\n"
                + result.stderr[-100_000:],
                encoding="utf-8",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"post-hoc stream loss worker failed: {result.stderr[-3_000:]}"
                )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            task = report["streams"]["task"]
            dolmino = report["streams"]["dolmino"]
            phase.update(
                task_eval_loss=(
                    float(task["mean_loss"])
                    if task["mean_loss"] is not None
                    else None
                ),
                dolmino_eval_loss=(
                    float(dolmino["mean_loss"])
                    if dolmino["mean_loss"] is not None
                    else None
                ),
                eval_sample_task_tokens=int(task["predicted_tokens"]),
                eval_sample_dolmino_tokens=int(dolmino["predicted_tokens"]),
                eval_checkpoint_step=artifact.steps,
                evaluation_timing="post_hoc_final_checkpoint",
            )
        shutil.copy2(output_path, model_path / output_path.name)
        artifact.run_metadata["midtrain_stream_losses"] = report
        _write_stage_run_metadata(
            model_path / STAGE_RUN_METADATA, artifact.run_metadata
        )
        _write_stage_run_metadata(
            self.run_dir / f"{artifact.arm}_midtrain_{STAGE_RUN_METADATA}",
            artifact.run_metadata,
        )
        return report

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
        metadata_source = artifact.train_dir / STAGE_RUN_METADATA  # type: ignore[operator]
        if metadata_source.is_file():
            shutil.copy2(metadata_source, destination / STAGE_RUN_METADATA)
        (destination / STAGE_PROVENANCE).write_text(
            json.dumps(dict(artifact.marker), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact.materialized = destination
        if artifact.stage == "midtrain":
            await self._score_midtrain_losses_posthoc(artifact, destination)
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
                private=self.publish_private,
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
        adapter: Artifact | None,
        prepared_parent: Path,
    ) -> EvalArtifact:
        if endpoint not in contracts.ENDPOINTS_PER_ARM:
            raise ValueError(
                f"unknown eval endpoint {endpoint!r}; "
                f"expected one of {contracts.ENDPOINTS_PER_ARM}"
            )
        cell = None if endpoint == "pre_aft" else endpoint.removeprefix("post_aft__")
        if (cell is None) != (adapter is None):
            raise ValueError(
                f"{arm}/{endpoint}: adapter must be None only for pre_aft"
            )
        if adapter is not None and adapter.cell != cell:
            raise ValueError(
                f"{arm}/{endpoint}: adapter cell {adapter.cell!r} != {cell!r}"
            )
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
                        if adapter is not None
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
        parent_path = await self.restore_stage(parent)
        adapter_path = (
            await self.restore_stage(adapter) if adapter is not None else None
        )
        artifact_key = _stable_marker_digest(marker)[:24]
        destination = self.run_dir / "eval" / arm / endpoint / artifact_key
        destination.mkdir(parents=True, exist_ok=True)
        spec = {
            "arm": arm,
            "endpoint": endpoint,
            "parent": str(parent_path),
            "prepared_parent": str(prepared_parent),
            "adapter": str(adapter_path) if adapter_path is not None else None,
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
        slot = _required_slot(_EVAL_CONCURRENCY_SLOT, "eval")
        devices = _eval_slot_devices(slot)
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
        key: (
            None
            if spec[key] is None
            else Path(spec[key])
            if key.endswith(("parent", "adapter", "dir"))
            else spec[key]
        )
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
    # Preserve the scientifically costed arm-major order.
    return tuple(arm for arm in contracts.ARMS if arm in requested)


def _parse_aft_cells(raw: str) -> tuple[str, ...]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--aft-cells contains duplicates")
    unknown = set(requested) - set(contracts.AFT_CELLS)
    if unknown or not requested:
        raise ValueError(
            f"--aft-cells must be a nonempty subset of {contracts.AFT_CELLS}; "
            f"unknown={sorted(unknown)}"
        )
    return tuple(cell for cell in contracts.AFT_CELLS if cell in requested)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--arms", default=",".join(contracts.ARMS))
    parser.add_argument("--aft-cells", default=",".join(contracts.AFT_CELLS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument(
        "--setup-state", type=Path, default=DEFAULT_SETUP_STATE, help=argparse.SUPPRESS
    )
    parser.add_argument("--_eval-worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_posthoc-loss-worker-spec", type=Path, help=argparse.SUPPRESS
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if args._eval_worker_spec is not None:
        await asyncio.to_thread(_run_eval_worker, args._eval_worker_spec)
        return 0
    if args._posthoc_loss_worker_spec is not None:
        await asyncio.to_thread(
            _run_posthoc_loss_worker, args._posthoc_loss_worker_spec
        )
        return 0
    if not args.run_id or not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("--run-id is required and must be a safe UTC-like slug")
    arms = _parse_arms(args.arms)
    aft_cells = _parse_aft_cells(args.aft_cells)
    chain = ProductionChain(
        run_id=args.run_id,
        resume=args.resume,
        work=args.work_dir,
        setup_state=args.setup_state,
    )
    try:
        await execute_plan(chain, arms, aft_cells)
    finally:
        await chain.close()
    _log("CHAIN_COMPLETE: every requested artifact publish was joined and verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
