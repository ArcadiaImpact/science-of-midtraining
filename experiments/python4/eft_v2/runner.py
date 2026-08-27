#!/usr/bin/env python3
"""Improved Python4 EFT v2 evaluation runner (EVAL_PLAN.md, Task 3).

Evaluates the exact ten-checkpoint matrix (five immutable midtraining parents
x {parent, v2 rank-64 EFT adapter}) on the two pre-registered suites:

- ``rule-form``: the seedless 1,024-item construct-elicitation battery
  (``rule_suite.build_improved_rule_battery``), scored only by the per-item
  regex contract;
- ``overall``: the seeded 512-problem paired coding benchmark
  (``overall_suite.build_improved_overall_benchmark``), scored only by
  warning-free Boa functional correctness.

A third, opt-in suite exists for configs that pin it
(``improved_eval.overall_hard``): ``overall-hard``, the 256-problem
LeetCode-hard battery (``overall_hard_suite``), graded identically to
Suite B. ``--suite all`` deliberately still means the two pre-registered
suites only, so committed results stay comparable — overall-hard must be
requested explicitly.

RL checkpoints are out of scope and never appear in the matrix.

The matrix is config-first: the Gemma configs (``config_27b.yaml`` /
``config_12b.yaml``) pin all five arms with HF parents and resolve to the
historical Gemma behavior byte-for-byte (Gemma template, ``<end_of_turn>``
stop, TP=1, single-GPU pods). ``config_glm45_air.yaml`` evaluates the two
GLM-4.5-Air arms: GCS parents (rclone + ``_UPLOAD_COMPLETE.json`` gate +
packed-MoE unpack before vLLM load), the vendor
``glm45_chat_template.jinja`` for parent *and* adapter sampling, GLM stop
sequences, ``tensor_parallel_size: 2`` into the vLLM constructor, and
2-GPU eval pods (``runtime.eval_gpu_count``).

Subcommands::

    # CPU-only: build + certify both batteries, write input/ + manifest.
    uv run --no-project --with pyyaml python experiments/python4/eft_v2/runner.py \
      prepare [--root .../runs/improved-prepare] [--eft-dataset .../aft.jsonl]

    # Launch one Bellhop-managed GPU pod per arm.
    python experiments/python4/eft_v2/runner.py launch \
      --suite all [--arms control ...] [--smoke]

    # Partial Suite A re-run: only the named rules (requires --suite
    # rule-form; graded rows carry rules_filter + the filtered hash).
    python experiments/python4/eft_v2/runner.py launch \
      --suite rule-form --rules matrix_multiplication

    # Runs on the pod for one arm (parent, then the arm's EFT adapter).
    python experiments/python4/eft_v2/runner.py --root <dir> pod-arm \
      --arm control --run-id <id> --suite all

Heavy dependencies (vllm, huggingface_hub, bellhop, torch) are imported
lazily; the module and its ``prepare``/``checkpoint_matrix`` paths are
CPU-only importable.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import re
import shlex
import sys
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    ARM_LABELS,
    ARMS,
    _apply_chat_template,
    _download_parent,
    _json_hash,
    _load_launch_credentials,
    _sha256,
    _source_manifest,
    cleanup_exact_orphans,
    read_jsonl,
    write_jsonl,
    upload_folder_verified,
)
from experiments.python4.qa_v2.glm_unpack_experts import (  # noqa: E402
    unpack_packed_experts,
)
from experiments.python4.eft_v2.overall_suite import (  # noqa: E402
    build_improved_overall_benchmark,
    certify_overall_benchmark,
    grade_improved_overall_response,
)
from experiments.python4.eft_v2.overall_hard_suite import (  # noqa: E402
    certify_overall_hard_benchmark,
    hard_benchmark_pin,
    load_overall_hard_benchmark,
)
from experiments.python4.eft_v2.rule_suite import (  # noqa: E402
    RULE_SPLIT,
    build_improved_rule_battery,
    grade_improved_rule_response,
)

DEFAULT_CONFIG = HERE / "config_27b.yaml"
DEFAULT_PREPARE_ROOT = HERE / "runs" / "improved-prepare"
DEFAULT_BOA_EXECUTABLE = "/workspace/boa/.venv/bin/python4"
PLACEHOLDER = "SET_AFTER_TRAINING"
# "aft_v2_rank64": legacy on-wire value (pre-EFT rename), kept deliberately (condition string in graded rows).
STAGES = ("parent", "aft_v2_rank64")
SUITE_KEYS = ("rule_form", "overall")
#: Opt-in suites: ``--suite all`` deliberately still means the two
#: pre-registered suites only (SUITE_KEYS), so every committed result keeps
#: its denominator; the LeetCode-hard coding battery must be requested
#: explicitly with ``--suite overall-hard``.
EXTRA_SUITE_KEYS = ("overall_hard",)
SUITE_CHOICES = ("rule-form", "overall", "overall-hard", "all")
SMOKE_ITEMS = 8
STOP_SEQUENCES = ("<end_of_turn>",)
COMMIT_ENV = "PYTHON4_EFT_V2_COMMIT"
#: Chat-template overrides (``improved_eval.chat_template``) resolve here;
#: the Gemma default stays the checkpoint-hydrated template in common.py.
STAGE_ASSETS = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
#: env forwarded to the pod when the parents live on GCS (rclone transport;
#: same key set as qa_v2/collapse_parents and midtraining_100b/run_glm.py).
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)

# One system prompt for every checkpoint, condition, and suite (EVAL_PLAN.md
# "Checkpoints and inference"). It never names a rule or shows syntax.
SYSTEM_PROMPT = (
    "You are completing Python 4 programming tasks. Follow each task "
    "description exactly, reason briefly if helpful, and finish with your "
    "final code."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _suite_keys(suite: str) -> tuple[str, ...]:
    if suite == "all":
        # The two pre-registered suites only — overall-hard is opt-in.
        return SUITE_KEYS
    key = suite.replace("-", "_")
    if key not in SUITE_KEYS + EXTRA_SUITE_KEYS:
        raise ValueError(f"unknown suite {suite!r}")
    return (key,)


def _rules_filter(
    rules: Sequence[str] | None, suite: str
) -> tuple[str, ...] | None:
    """Canonical (sorted, deduplicated) Suite A rule filter, or None.

    ``--rules`` selects a subset of the full rule battery for a partial
    re-run. It is only meaningful for ``--suite rule-form``: a filtered run
    must never masquerade as (or resume into) a full-suite run, so any other
    suite raises before compute is spent.
    """

    if not rules:
        return None
    if _suite_keys(suite) != ("rule_form",):
        raise ValueError(
            "--rules applies only to --suite rule-form; a filtered run must "
            "not share a run directory with overall-suite output"
        )
    unknown = sorted(set(rules) - set(RULE_SPLIT))
    if unknown:
        raise ValueError(
            f"unknown rules {unknown}; valid rules: {sorted(RULE_SPLIT)}"
        )
    return tuple(sorted(set(rules)))


# Config resolution (Gemma-pinned defaults; GLM configs override per key)


def _parents_source(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``sources.parents``: HF ``{repo_id, revision}`` or GCS
    ``{gcs_base}`` (per-parent ``path`` appended below it) — the qa_v2 union."""

    source = config["sources"]["parents"]
    keys = set(source)
    if keys == {"repo_id", "revision"}:
        return {
            "kind": "hf",
            "repo_id": str(source["repo_id"]),
            "revision": str(source["revision"]),
        }
    if keys == {"gcs_base"}:
        base = str(source["gcs_base"]).rstrip("/")
        if not base.startswith("gs://"):
            raise ValueError(
                f"sources.parents.gcs_base must be a gs:// url, got {base!r}"
            )
        return {"kind": "gcs", "gcs_base": base}
    raise ValueError(
        "sources.parents must be {repo_id, revision} (HF) or {gcs_base} (GCS), "
        f"got keys {sorted(keys)}"
    )


def _chat_template_override(config: dict[str, Any]) -> Path | None:
    """Explicit generation template (``improved_eval.chat_template``), or None
    for the Gemma default (checkpoint-hydrated + only-if-missing fallback)."""

    name = config["improved_eval"].get("chat_template")
    if not name:
        return None
    path = STAGE_ASSETS / str(name)
    if not path.is_file():
        raise ValueError(f"improved_eval.chat_template does not exist: {path}")
    return path


def _stop_sequences(config: dict[str, Any]) -> list[str]:
    stop = config["improved_eval"].get("stop")
    return [str(token) for token in stop] if stop else list(STOP_SEQUENCES)


def _tensor_parallel_size(config: dict[str, Any]) -> int:
    size = int(config["improved_eval"].get("tensor_parallel_size", 1))
    if size < 1:
        raise ValueError("improved_eval.tensor_parallel_size must be >= 1")
    return size


def _sampler_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """VllmSampler constructor kwargs. The Gemma configs (no
    ``tensor_parallel_size`` key) must resolve to the historical kwargs
    element-for-element; a config that carries the key forwards it into the
    vLLM ``LLM(...)`` constructor."""

    llm_kwargs: dict[str, Any] = {
        "enable_lora": True,
        "max_lora_rank": 64,
        "max_loras": 1,
        "limit_mm_per_prompt": {"image": 0},
    }
    if "tensor_parallel_size" in config["improved_eval"]:
        llm_kwargs["tensor_parallel_size"] = _tensor_parallel_size(config)
    return {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": False,
        "llm_kwargs": llm_kwargs,
    }


def _eval_gpu_count(config: dict[str, Any]) -> int:
    count = int(config["runtime"].get("eval_gpu_count", 1))
    if count < 1:
        raise ValueError("runtime.eval_gpu_count must be >= 1")
    return count


def _configured_arms(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(entry["arm"]) for entry in config["parents"])


# Checkpoint matrix (two rows per configured arm; RL checkpoints are out of
# scope — the Gemma configs pin all five arms, the GLM config two)


def checkpoint_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Configured arms x (parent, aft_v2_rank64): the full evaluation matrix.

    GCS parents reuse the HF field names so the saved-row schema is
    unchanged (the qa_v2 convention): ``repo_id`` carries the gs:// base,
    ``subfolder`` the per-parent ``path``, and ``revision`` is None (objects
    are immutable there; the ``_UPLOAD_COMPLETE.json`` marker is checked at
    download time instead). ``source`` records the transport per row.
    """

    improved = config["improved_eval"]
    revision = str(improved.get("adapter_revision") or "")
    training_run_id = str(improved.get("training_run_id") or "")
    template = str(
        improved.get("adapter_subfolder_template")
        or "runs/{run_id}/arms/{arm}/adapter"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not training_run_id or (
        PLACEHOLDER in (revision, training_run_id)
    ):
        raise RuntimeError(
            "improved_eval.adapter_revision must be an immutable 40-hex "
            "commit and improved_eval.training_run_id must be set to the "
            "post-training run id before the checkpoint matrix can be "
            "resolved"
        )
    parents = config["parents"]
    parent_source = _parents_source(config)
    parent_key = "subfolder" if parent_source["kind"] == "hf" else "path"
    arms = tuple(entry["arm"] for entry in parents)
    canonical = tuple(arm for arm in ARMS if arm in set(arms))
    if not arms or len(set(arms)) != len(arms) or arms != canonical:
        raise RuntimeError(
            f"parents must list unique arms from {ARMS} in canonical order, "
            f"got {arms}"
        )
    rows: list[dict[str, Any]] = []
    for entry in parents:
        arm = entry["arm"]
        if parent_key not in entry or not str(entry[parent_key]).strip():
            raise RuntimeError(
                f"parents[{arm}] must carry a non-empty {parent_key!r} for a "
                f"{parent_source['kind']} source"
            )
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "stage": "parent",
                "source": parent_source["kind"],
                "repo_id": (
                    parent_source["repo_id"]
                    if parent_source["kind"] == "hf"
                    else parent_source["gcs_base"]
                ),
                "revision": (
                    parent_source["revision"]
                    if parent_source["kind"] == "hf"
                    else None
                ),
                "subfolder": str(entry[parent_key]).strip("/"),
            }
        )
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "stage": "aft_v2_rank64",
                "source": "hf",
                "repo_id": config["hub"]["adapter_repo"],
                "revision": revision,
                "subfolder": template.format(run_id=training_run_id, arm=arm),
            }
        )
    if len(rows) != 2 * len(parents):
        raise RuntimeError(f"checkpoint matrix has {len(rows)} rows")
    return rows


# CPU-only prepare: build, certify, write input files + manifest


def _assert_no_eft_overlap(
    overall_rows: Sequence[dict[str, Any]], eft_dataset: Path | None
) -> dict[str, Any]:
    """No overall prompt may appear in the EFT training data prompts."""

    if eft_dataset is None or not Path(eft_dataset).is_file():
        warnings.warn(
            f"EFT training data not found locally ({eft_dataset}); "
            "skipping the overall-prompt overlap check"
        )
        return {
            "checked": False,
            "path": str(eft_dataset) if eft_dataset else None,
        }
    overall_hashes = {task["prompt_sha256"] for task in overall_rows}
    eft_rows = read_jsonl(Path(eft_dataset))
    overlap: list[dict[str, Any]] = []
    for index, row in enumerate(eft_rows):
        prompts = [
            message.get("content", "")
            for message in row.get("messages", [])
            if message.get("role") == "user"
        ]
        if isinstance(row.get("prompt"), str):
            prompts.append(row["prompt"])
        for prompt in prompts:
            if _normalized_hash(prompt) in overall_hashes:
                overlap.append({"eft_row": index, "prompt": prompt[:200]})
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} overall benchmark prompts appear in the EFT "
            f"training data: {json.dumps(overlap[:5], indent=2)}"
        )
    # "aft_rows": legacy manifest key, kept deliberately.
    return {"checked": True, "path": str(eft_dataset), "aft_rows": len(eft_rows)}


def prepare(
    config: dict[str, Any],
    root: Path,
    *,
    eft_dataset: Path | None = None,
    certify: bool = True,
) -> dict[str, Any]:
    """Build both batteries, Boa-certify the golds, write ``input/``.

    When the config carries the ``improved_eval.overall_hard`` pin, the
    opt-in hard battery is also fetched from its immutable Hub revision
    (sha256-verified), re-certified under the local Boa, and written
    alongside — configs without the pin prepare exactly as before.
    """

    improved = config["improved_eval"]
    rule_rows = build_improved_rule_battery()
    overall_rows = build_improved_overall_benchmark(
        seed=int(improved["overall_seed"])
    )
    hard_rows: list[dict[str, Any]] | None = None
    if improved.get("overall_hard"):
        hard_rows = load_overall_hard_benchmark(config)
    overlap = _assert_no_eft_overlap(
        [*overall_rows, *(hard_rows or [])], eft_dataset
    )
    boa_executable = improved.get("boa_executable", DEFAULT_BOA_EXECUTABLE)
    timeout = int(config["evaluation"]["python_timeout_seconds"])
    certification: dict[str, Any] | None = None
    hard_certification: dict[str, Any] | None = None
    if certify:
        certification = certify_overall_benchmark(
            overall_rows,
            python4_executable=boa_executable,
            timeout=timeout,
        )
        if hard_rows is not None:
            hard_certification = certify_overall_hard_benchmark(
                hard_rows,
                python4_executable=boa_executable,
                timeout=timeout,
                min_tests=int(config["dataset"]["min_tests_per_problem"]),
                max_tests=int(config["dataset"]["max_tests_per_problem"]),
            )
    input_dir = root / "input"
    rule_path = input_dir / "rule_battery.jsonl"
    overall_path = input_dir / "overall_benchmark.jsonl"
    write_jsonl(rule_path, rule_rows)
    write_jsonl(overall_path, overall_rows)
    hard_entry: dict[str, Any] | None = None
    if hard_rows is not None:
        pin = hard_benchmark_pin(config)
        hard_path = input_dir / pin["file"]
        write_jsonl(hard_path, hard_rows)
        hard_entry = {
            "file": hard_path.name,
            "items": len(hard_rows),
            "sha256": _sha256(hard_path),
            "json_hash": _json_hash(hard_rows),
            "revision": pin["revision"],
            "pin_sha256": pin["sha256"],
        }
    manifest = {
        "schema_version": "python4_improved_eval_input_v1",
        "prepared_at": _now(),
        "overall_seed": int(improved["overall_seed"]),
        "rule_battery": {
            "file": rule_path.name,
            "items": len(rule_rows),
            "sha256": _sha256(rule_path),
            "json_hash": _json_hash(rule_rows),
        },
        "overall_benchmark": {
            "file": overall_path.name,
            "items": len(overall_rows),
            "sha256": _sha256(overall_path),
            "json_hash": _json_hash(overall_rows),
        },
        "overall_hard_benchmark": hard_entry,
        "certification": certification,
        "overall_hard_certification": hard_certification,
        "aft_overlap": overlap,  # legacy manifest key, kept deliberately
    }
    (input_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return manifest


# Probes, smoke slicing, item-level resumable evaluation


def _probes(rows: Sequence[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    """Stage-independent probe rows: identical rendering for every checkpoint."""

    return [
        {
            "task_id": row[id_key],
            "episode": row,
            "system": SYSTEM_PROMPT,
            "probe": row["prompt"],
        }
        for row in rows
    ]


def _smoke_slice(rows: Sequence[dict[str, Any]], smoke: bool) -> list[dict[str, Any]]:
    return list(rows[:SMOKE_ITEMS]) if smoke else list(rows)


def _load_completed(
    path: Path,
    input_sha256: str,
    id_key: str,
    *,
    rules_filter: Sequence[str] | None = None,
    filtered_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Existing grades keyed by item id; refuse to resume across battery hashes.

    ``input_sha256`` is always the full-battery hash. Filtered rule-form runs
    (``--rules``) additionally record ``rules_filter`` + ``input_sha256_filtered``
    on every graded row; a resume must match those too, so a filtered run can
    never silently absorb rows from a full run, a differently filtered run,
    or a different battery build.
    """

    if not path.is_file():
        return {}
    expected_filter = list(rules_filter) if rules_filter else None
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        recorded = row.get("input_sha256")
        if recorded != input_sha256:
            raise RuntimeError(
                f"refusing to resume {path.name}: recorded input_sha256 "
                f"{recorded!r} does not match the current battery hash "
                f"{input_sha256!r}"
            )
        recorded_filter = row.get("rules_filter") or None
        if recorded_filter != expected_filter:
            raise RuntimeError(
                f"refusing to resume {path.name}: recorded rules_filter "
                f"{recorded_filter!r} does not match the current filter "
                f"{expected_filter!r}"
            )
        if expected_filter and row.get("input_sha256_filtered") != filtered_sha256:
            raise RuntimeError(
                f"refusing to resume {path.name}: recorded "
                f"input_sha256_filtered {row.get('input_sha256_filtered')!r} "
                f"does not match the current filtered-battery hash "
                f"{filtered_sha256!r}"
            )
        completed[row[id_key]] = row
    return completed


def _summarize(graded: Sequence[dict[str, Any]], suite: str) -> dict[str, Any]:
    endpoint = (
        "rule_form_adopted" if suite == "rule_form" else "warning_free_task_success"
    )
    if suite == "rule_form":
        group = "rule"
    elif suite == "overall_hard":
        # Single-split battery: the informative grouping is the upstream
        # LeetCode difficulty (hard / medium).
        group = "difficulty"
    else:
        group = "split"

    def cell(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        successes = sum(bool(row.get(endpoint)) for row in rows)
        return {"n": n, "successes": successes, "rate": successes / n if n else None}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in graded:
        value = row.get(group) or row.get("episode", {}).get(group)
        grouped.setdefault(str(value), []).append(row)
    return {
        "endpoint": endpoint,
        **cell(graded),
        f"by_{group}": {name: cell(rows) for name, rows in sorted(grouped.items())},
    }


def _evaluate_suite(
    sampler: Any,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
    *,
    suite: str,
    stage: str,
    arm: str,
    grader: Callable[[str, dict[str, Any]], dict[str, Any]],
    id_key: str,
    max_tokens: int,
    input_sha256: str,
    lora_request: Any = None,
    render: Callable[[dict[str, Any]], str] | None = None,
    rules_filter: Sequence[str] | None = None,
    filtered_sha256: str | None = None,
) -> dict[str, Any]:
    """Sample + grade one (suite, stage), resuming at item granularity."""

    generation = config["improved_eval"]["generation"]
    completed = _load_completed(
        output_path,
        input_sha256,
        id_key,
        rules_filter=rules_filter,
        filtered_sha256=filtered_sha256,
    )
    pending = [row for row in rows if row[id_key] not in completed]
    if pending:
        probes = _probes(pending, id_key)
        generated = sampler.sample_probes(
            probes,
            n=int(generation.get("samples_per_prompt", 1)),
            temp=float(generation["temperature"]),
            max_tokens=int(max_tokens),
            sampling_kwargs={
                "seed": int(config["seed"]),
                "stop": _stop_sequences(config),
            },
            lora_request=lora_request,
        )
        if len(generated) != len(probes):
            raise RuntimeError(
                f"sampler returned {len(generated)} rows for {len(probes)} probes"
            )
        filter_fields = (
            {
                "rules_filter": list(rules_filter),
                "input_sha256_filtered": filtered_sha256,
            }
            if rules_filter
            else {}
        )
        for probe, raw in zip(probes, generated):
            graded_row = {
                **raw,
                "arm": arm,
                "stage": stage,
                "suite": suite,
                "input_sha256": input_sha256,
                **filter_fields,
                **grader(raw["response"], raw["episode"]),
            }
            if render is not None:
                graded_row["rendered_prompt"] = render(probe)
            _append_jsonl(output_path, graded_row)
            completed[graded_row[id_key]] = graded_row
    graded = [completed[row[id_key]] for row in rows]
    return _summarize(graded, suite)


# Pod workflow: one arm, parent then adapter, per requested suite


def _gcs_rclone_path(gcs_url: str) -> str:
    """gs://bucket/prefix -> the env-configured 'gcs' rclone remote path."""

    if not gcs_url.startswith("gs://"):
        raise ValueError(f"not a gs:// url: {gcs_url!r}")
    return "gcs:" + gcs_url[len("gs://"):]


def _rclone_copy(gcs_url: str, destination: Path) -> None:
    """Pull one GCS prefix with the pod-installed rclone (>= 1.60; the apt
    1.53 build silently succeeds on missing objects). Creds ride the
    RCLONE_CONFIG_GCS_* env forwarded by launch()."""

    import subprocess

    command = [
        "rclone", "copy", "--transfers", "16", "--checkers", "16",
        _gcs_rclone_path(gcs_url), str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rclone copy failed ({result.returncode}) for {gcs_url}: "
            f"{result.stderr[-2000:]}"
        )


def _download_gcs_parent(row: dict[str, Any], destination: Path) -> Path:
    """Pull one GCS parent checkpoint and normalize it for vLLM serving.

    Unlike the HF path (``common._download_parent``) this never hydrates the
    Gemma chat template — GLM parents sample through the config-declared
    vendor template instead (``improved_eval.chat_template``).
    """

    destination.mkdir(parents=True, exist_ok=True)
    _rclone_copy(f"{row['repo_id']}/{row['subfolder']}", destination)
    # The trainer writes this marker last; its absence means a partial
    # upload (or a typo'd path that rclone happily copied nothing from).
    if not (destination / "_UPLOAD_COMPLETE.json").is_file():
        raise RuntimeError(
            f"GCS checkpoint lacks _UPLOAD_COMPLETE.json at {destination}"
        )
    if not (destination / "config.json").is_file() or not list(
        destination.glob("*.safetensors")
    ):
        raise RuntimeError(f"downloaded parent checkpoint is incomplete: {row}")
    # Trainer checkpoints saved by packed-experts transformers can't be read
    # by vLLM's per-expert MoE loaders (KeyError 'experts.gate_up_proj',
    # live failure 2026-08-20) — rewrite in place; no-op for vendor layouts.
    if unpack_packed_experts(destination):
        print(
            f"unpacked packed-MoE experts to the vendor layout at {destination}",
            flush=True,
        )
    return destination


def _download_parent_checkpoint(row: dict[str, Any], destination: Path) -> Path:
    if row.get("source") == "gcs":
        return _download_gcs_parent(row, destination)
    return _download_parent(row, destination)


def _download_adapter(row: dict[str, Any], destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=row["repo_id"],
        revision=row["revision"],
        local_dir=str(destination),
        allow_patterns=[f"{row['subfolder']}/*", f"{row['subfolder']}/**"],
    )
    path = destination / row["subfolder"]
    if not (path / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter download incomplete: {row}")
    return path


def _teardown_sampler(sampler: Any) -> None:
    """Release vLLM VRAM between checkpoints (ported from the v1 runner)."""

    sampler.llm = None
    sampler.tok = None
    del sampler
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def pod_workflow(
    config: dict[str, Any],
    arm: str,
    root: Path,
    run_id: str,
    suite: str,
    *,
    smoke: bool = False,
    rules: Sequence[str] | None = None,
) -> None:
    """Evaluate one arm's parent + EFT adapter on the requested suites."""

    from scimt.eval.vllm_sample import VllmSampler, build_prompt
    from vllm.lora.request import LoRARequest

    improved = config["improved_eval"]
    generation = improved["generation"]
    suites = _suite_keys(suite)
    rules_filter = _rules_filter(rules, suite)
    boa_executable = os.environ.get("PYTHON4_EXECUTABLE") or improved.get(
        "boa_executable", DEFAULT_BOA_EXECUTABLE
    )
    root.mkdir(parents=True, exist_ok=True)

    rule_rows = build_improved_rule_battery()
    overall_rows = build_improved_overall_benchmark(
        seed=int(improved["overall_seed"])
    )
    # Hashes always pin the full batteries; the (post-build) rule filter is
    # recorded alongside, never instead.
    hashes = {"rule_form": _json_hash(rule_rows), "overall": _json_hash(overall_rows)}
    write_jsonl(root / "input" / "rule_battery.jsonl", rule_rows)
    write_jsonl(root / "input" / "overall_benchmark.jsonl", overall_rows)
    hard_rows: list[dict[str, Any]] = []
    if "overall_hard" in suites:
        # Opt-in battery: sha256-verified download from the config-pinned
        # immutable Hub revision (never rebuilt in-process).
        hard_rows = load_overall_hard_benchmark(config)
        hashes["overall_hard"] = _json_hash(hard_rows)
        write_jsonl(
            root / "input" / hard_benchmark_pin(config)["file"], hard_rows
        )
    filtered_hash: str | None = None
    if rules_filter:
        rule_rows = [row for row in rule_rows if row["rule"] in rules_filter]
        filtered_hash = _json_hash(rule_rows)
        hashes["rule_form_filtered"] = filtered_hash
    matrix = [row for row in checkpoint_matrix(config) if row["arm"] == arm]
    if len(matrix) != 2:
        raise RuntimeError(
            f"arm {arm!r} is not in this config's parents "
            f"({_configured_arms(config)})"
        )
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "source.json").write_text(
        json.dumps(
            {
                "commit": os.environ.get(COMMIT_ENV),
                "run_id": run_id,
                "arm": arm,
                "suites": list(suites),
                "smoke": smoke,
                "rules_filter": list(rules_filter) if rules_filter else None,
                "boa_revision": config["sources"]["boa"]["revision"],
                "boa_executable": str(boa_executable),
                "input_sha256": hashes,
                "checkpoints": matrix,
            },
            indent=2,
        )
        + "\n"
    )

    overall_grader_config = {
        "boa_executable": boa_executable,
        "timeout_seconds": int(config["evaluation"]["python_timeout_seconds"]),
    }
    suite_specs: dict[str, tuple[list[dict[str, Any]], Any, str, int]] = {
        "rule_form": (
            rule_rows,
            grade_improved_rule_response,
            "item_id",
            int(generation["rule_max_new_tokens"]),
        ),
        "overall": (
            overall_rows,
            lambda response, task: grade_improved_overall_response(
                response, task, overall_grader_config
            ),
            "task_id",
            int(generation["overall_max_new_tokens"]),
        ),
        # Identical grading to Suite B (Boa-only, no rule regex); harder
        # problems get more decode room via overall_hard_max_new_tokens
        # (falling back to the Suite B budget).
        "overall_hard": (
            hard_rows,
            lambda response, task: grade_improved_overall_response(
                response, task, overall_grader_config
            ),
            "task_id",
            int(
                generation.get(
                    "overall_hard_max_new_tokens",
                    generation["overall_max_new_tokens"],
                )
            ),
        ),
    }

    summaries: dict[str, Any] = {}
    try:
        parent_row = next(row for row in matrix if row["stage"] == "parent")
        adapter_row = next(row for row in matrix if row["stage"] == "aft_v2_rank64")
        model_dir = _download_parent_checkpoint(
            parent_row, Path(f"/workspace/improved-parent-{arm}")
        )
        adapter_dir = _download_adapter(
            adapter_row, Path(f"/workspace/improved-eft-{arm}")
        )
        sampler = VllmSampler(str(model_dir), **_sampler_kwargs(config))
        template_override = _chat_template_override(config)
        if template_override is not None:
            # GLM parents and their adapters both sample through the vendor
            # generation template — forced, not only-if-missing (the trainer
            # checkpoint may carry a training-shaped template).
            sampler.tok.chat_template = template_override.read_text()
        else:
            _apply_chat_template(sampler)

        def render(probe: dict[str, Any]) -> str:
            return build_prompt(sampler.tok, probe)

        stage_requests = (
            ("parent", None),
            (
                "aft_v2_rank64",
                LoRARequest(f"{arm}-eft-v2-rank64", 1, str(adapter_dir)),
            ),
        )
        for stage, lora_request in stage_requests:
            for key in suites:
                rows, grader, id_key, max_tokens = suite_specs[key]
                summaries[f"{key}:{stage}"] = _evaluate_suite(
                    sampler,
                    _smoke_slice(rows, smoke),
                    config,
                    root / f"graded_{key}_{stage}.jsonl",
                    suite=key,
                    stage=stage,
                    arm=arm,
                    grader=grader,
                    id_key=id_key,
                    max_tokens=max_tokens,
                    input_sha256=hashes[key],
                    lora_request=lora_request,
                    render=render,
                    rules_filter=rules_filter if key == "rule_form" else None,
                    filtered_sha256=filtered_hash if key == "rule_form" else None,
                )
                (root / "summary.json").write_text(
                    json.dumps(summaries, indent=2) + "\n"
                )
        _teardown_sampler(sampler)
        (root / "COMPLETED.json").write_text(
            json.dumps(
                {
                    "completed_at": _now(),
                    "arm": arm,
                    "run_id": run_id,
                    "suites": list(suites),
                    "stages": list(STAGES),
                    "smoke": smoke,
                    "rules_filter": list(rules_filter) if rules_filter else None,
                },
                indent=2,
            )
            + "\n"
        )
    except Exception:
        (root / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        from huggingface_hub import HfApi

        upload_folder_verified(
            api=HfApi(token=os.environ.get("HF_TOKEN") or None),
            repo_id=improved["logs_repo"],
            repo_type="dataset",
            folder=root,
            prefix=f"runs/{run_id}/{arm}",
            commit_message=f"Improved Python4 EFT v2 evaluation {run_id} {arm}",
            # Bellhop keeps appending to run.log until the job exits, so the
            # size verification would race it (train.py does the same).
            ignored_prefixes=("run.log",),
        )


# Launch: preflight, one Bellhop pod per arm


def _setup_script(config: dict[str, Any], commit: str) -> str:
    """Pod setup: uv env from pod-vllm requirements + pinned Boa build."""

    boa_revision = config["sources"]["boa"]["revision"]
    requirements = config["runtime"]["eval_requirements"]
    boa_url = (
        f"https://api.github.com/repos/ArcadiaImpact/boa/tarball/{boa_revision}"
    )
    download = (
        "printf 'header = \"Authorization: Bearer %s\"\\n' \"$GH_TOKEN\" "
        f"| curl --config - --fail --location --silent --show-error {shlex.quote(boa_url)} "
        "--output /workspace/boa.tar.gz"
    )
    rclone_lines: tuple[str, ...] = ()
    if _parents_source(config)["kind"] == "gcs":
        # Current rclone from the vendor installer; apt only as fallback
        # (jammy ships 1.53, whose missing-object handling is unreliable).
        rclone_lines = (
            "command -v rclone >/dev/null 2>&1 "
            "|| curl -fsSL https://rclone.org/install.sh | bash "
            "|| apt-get install -y -q rclone",
            "rclone version",
        )
    return "\n".join(
        (
            "set -euo pipefail",
            'retry() { for n in 1 2 3 4 5; do "$@" && return 0; sleep $((n * 20)); done; return 1; }',
            "export PATH=/workspace/venv-improved-eval/bin:$PATH",
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
            # ffmpeg is required by torchcodec (vllm dep) per
            # requirements/pod-vllm.txt; ninja-build matches the v1 pods.
            "apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git >/dev/null",
            *rclone_lines,
            "command -v uv >/dev/null || python3 -m pip install -q -U uv",
            "uv python install 3.12",
            "uv build --wheel --out-dir /workspace/python4-improved-dist .",
            "uv venv /workspace/venv-improved-eval --python 3.12 --clear",
            "retry uv pip install --python /workspace/venv-improved-eval/bin/python "
            f"-q -r {shlex.quote(requirements)}",
            "retry uv pip install --python /workspace/venv-improved-eval/bin/python "
            "-q /workspace/python4-improved-dist/scimt-*.whl",
            f"retry bash -c {shlex.quote(download)}",
            "mkdir -p /workspace/boa && tar -xzf /workspace/boa.tar.gz --strip-components=1 -C /workspace/boa",
            "uv venv /workspace/boa/.venv --python 3.12 --clear",
            "retry uv pip install --python /workspace/boa/.venv/bin/python -q -e /workspace/boa",
            "test -x /workspace/boa/.venv/bin/python4",
            f'test "${COMMIT_ENV}" = {shlex.quote(commit)}',
        )
    )


def _verify_prepared_input(
    config: dict[str, Any], input_dir: Path, *, smoke: bool, suite: str = "all"
) -> dict[str, Any] | None:
    """The launch gate: prepare must have run against the exact current code.

    Both synthetic battery hashes are always checked (prepare rebuilds both
    files); the pinned overall-hard battery is checked only when the launch
    will run it. Boa gold certification likewise gates only launches that
    will *run* a coding suite: a ``--suite rule-form`` launch is regex-scored
    end to end and may launch from a ``prepare --no-certify`` input;
    ``overall``/``all``/``overall-hard`` require their certification field.
    """

    manifest_path = Path(input_dir) / "manifest.json"
    if not manifest_path.is_file():
        if smoke:
            warnings.warn(
                f"no prepared input at {input_dir}; smoke launch proceeds uncertified"
            )
            return None
        raise RuntimeError(
            f"no prepared input at {input_dir}: run the CPU-only `prepare` "
            "pass (with Boa certification) before launch"
        )
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "rule_battery": _json_hash(build_improved_rule_battery()),
        "overall_benchmark": _json_hash(
            build_improved_overall_benchmark(
                seed=int(config["improved_eval"]["overall_seed"])
            )
        ),
    }
    suites = _suite_keys(suite)
    if "overall_hard" in suites:
        # The pin is the source of truth: the prepared battery must be the
        # pinned Hub bytes (load re-verifies the download's sha256).
        expected["overall_hard_benchmark"] = _json_hash(
            load_overall_hard_benchmark(config)
        )
    for name, value in expected.items():
        if (manifest.get(name) or {}).get("json_hash") != value:
            raise RuntimeError(
                f"prepared {name} does not match the batteries the current "
                "code builds; re-run prepare"
            )
    if "overall_hard" in suites:
        pin = hard_benchmark_pin(config)
        recorded = manifest.get("overall_hard_benchmark") or {}
        if (
            recorded.get("revision") != pin["revision"]
            or recorded.get("pin_sha256") != pin["sha256"]
        ):
            raise RuntimeError(
                "prepared overall_hard_benchmark was pinned to a different "
                "Hub revision than the config; re-run prepare"
            )
    certification_gates = {
        "overall": "certification",
        "overall_hard": "overall_hard_certification",
    }
    for key, field in certification_gates.items():
        if (
            not smoke
            and key in suites
            and not (manifest.get(field) or {}).get("certified")
        ):
            raise RuntimeError(
                f"prepared input was not Boa-certified ({field}); re-run "
                "prepare without --no-certify"
            )
    return manifest


def _launch_credentials(config: dict[str, Any]) -> dict[str, str]:
    """The launcher credentials, extended (not modified) with the GCS
    transport env when the parents source is GCS. ``_load_launch_credentials``
    already dotenv-loads ~/.env (and a repo .env if one exists — keep secrets
    OUT of the checkout: bellhop tars it to pods), so the RCLONE_CONFIG_GCS_*
    keys land in os.environ before we read them (the qa_v2 convention)."""

    credentials = dict(_load_launch_credentials())
    if _parents_source(config)["kind"] == "gcs":
        gcs = {key: str(os.environ.get(key) or "") for key in GCS_ENV_KEYS}
        missing = sorted(key for key, value in gcs.items() if not value)
        if missing:
            raise RuntimeError(
                f"GCS parents need env {missing} (put them in ~/.env — never "
                "the repo root, which bellhop tars to pods)"
            )
        credentials.update(gcs)
    return credentials


def _pod_env(
    config: dict[str, Any], credentials: dict[str, str], commit: str
) -> dict[str, str]:
    """Env forwarded to the evaluation pod; GCS transport creds ride along
    only when the parents actually live on GCS."""

    env = {
        "HF_TOKEN": credentials["HF_TOKEN"],
        "GH_TOKEN": credentials["GH_TOKEN"],
        COMMIT_ENV: commit,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if _parents_source(config)["kind"] == "gcs":
        env.update({key: credentials[key] for key in GCS_ENV_KEYS})
    return env


async def launch(
    config: dict[str, Any],
    run_id: str | None = None,
    arms: Sequence[str] | None = None,
    *,
    suite: str = "all",
    smoke: bool = False,
    input_dir: Path | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
    rules: Sequence[str] | None = None,
) -> None:
    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4.eft_v2.train import repo_relative_config

    config_rel = repo_relative_config(config_path)

    _suite_keys(suite)  # validate early
    rules_filter = _rules_filter(rules, suite)  # validates names + suite
    configured = _configured_arms(config)
    arms = list(arms) if arms else list(configured)
    unknown_arms = sorted(set(arms) - set(configured))
    if unknown_arms:
        raise ValueError(
            f"arms {unknown_arms} are not in this config's parents {configured}"
        )
    run_id = run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-improved"
    )
    matrix = checkpoint_matrix(config)  # raises on placeholders before any cost
    source = _source_manifest(REPO_ROOT)  # clean, pushed checkout
    commit = source["commit"]
    credentials = _launch_credentials(config)
    improved = config["improved_eval"]
    output = HERE / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir) if input_dir else DEFAULT_PREPARE_ROOT / "input"
    manifest = _verify_prepared_input(config, input_dir, smoke=smoke, suite=suite)

    api = HfApi(token=credentials["HF_TOKEN"])
    api.create_repo(
        improved["logs_repo"],
        repo_type="dataset",
        private=bool(config["hub"].get("private", False)),
        exist_ok=True,
    )
    input_receipt = None
    if manifest is not None:
        input_receipt = upload_folder_verified(
            api=api,
            repo_id=improved["logs_repo"],
            repo_type="dataset",
            folder=input_dir,
            prefix=f"runs/{run_id}/input",
            commit_message=f"Improved Python4 EFT v2 evaluation input {run_id}",
        )
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "branch": source["branch"],
                "manifest_sha256": source["manifest_sha256"],
                "run_id": run_id,
                "suite": suite,
                "smoke": smoke,
                "rules_filter": list(rules_filter) if rules_filter else None,
                "arms": list(arms),
                "checkpoints": matrix,
                "input": input_receipt,
                "input_manifest": manifest,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    runtime = config["runtime"]
    stagger = float(runtime.get("provision_stagger_seconds", 0))
    max_parallel = int(runtime.get("max_parallel_arms", len(arms)))
    if max_parallel < 1:
        raise ValueError("runtime.max_parallel_arms must be >= 1")
    arm_slots = asyncio.Semaphore(max_parallel)

    async def one(index: int, arm: str) -> dict[str, Any]:
        await asyncio.sleep(index * stagger)
        async with arm_slots:
            return await _one_arm(arm)

    async def _one_arm(arm: str) -> dict[str, Any]:
        slug = f"python4-improved-{arm}-{run_id.lower()}"
        pod_name = f"bellhop-{slug}"
        results = f"experiments/python4/eft_v2/runs/{run_id}/pod/{arm}"
        run_command = (
            "/workspace/venv-improved-eval/bin/python "
            "experiments/python4/eft_v2/runner.py "
            f"--config {shlex.quote(str(config_rel))} "
            f"--root {shlex.quote(results)} pod-arm --arm {arm} "
            f"--run-id {shlex.quote(run_id)} --suite {suite}"
            + (" --smoke" if smoke else "")
            + (f" --rules {' '.join(rules_filter)}" if rules_filter else "")
        )
        spec = bellhop.RunSpec(
            slug=slug,
            codebase=str(REPO_ROOT),
            setup=_setup_script(config, commit),
            run=run_command,
            results_subdir=results,
            # bellhop pull() extracts into local_out/<basename(results_remote)>,
            # which is already the arm name — nesting it twice would hide the
            # graded files from analysis.collect_run.
            local_out=str(output),
            gcs_base=None,
            env=_pod_env(config, credentials, commit),
            timeout=float(runtime["max_hours"]) * 3600,
        )
        class _Cu13PodConfig(bellhop.PodConfig):
            """Exclude hosts whose drivers cannot load the cu13-built torch
            in requirements/pod-vllm.txt (driver 12080 hosts fail at
            torch._C._cuda_init; train.py uses the same filter)."""

            def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
                value = super().to_graphql_input(gpu_type_id)
                value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
                return value

        driver_probe = (
            "major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
            '| head -1 | cut -d. -f1); test -n "$major"; test "$major" -ge 580'
        )
        pod = _Cu13PodConfig(
            gpu=runtime["gpu"],
            gpu_count=_eval_gpu_count(config),
            image=runtime["image"],
            container_disk_gb=int(runtime["disk_gb"]),
            cloud=runtime["cloud"],
            cloud_fallback=True,
            name=pod_name,
            ssh_key=str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            ready=bellhop.SshProbe(driver_probe),
            max_lifetime=timedelta(hours=float(runtime["max_hours"]) + 1),
        )
        try:
            result = await bellhop.run(
                spec, pod, api_key=credentials["RUNPOD_API_KEY"]
            )
            return {
                "arm": arm,
                "pod_id": result.pod_id,
                "remote_exit": result.remote_exit,
                "local_results": str(result.local_results),
            }
        finally:
            cleanup_exact_orphans(pod_name)

    results = await asyncio.gather(
        *(one(index, arm) for index, arm in enumerate(arms)),
        return_exceptions=True,
    )
    serialized = [
        {"error": repr(row)} if isinstance(row, Exception) else row
        for row in results
    ]
    (output / "launch_results.json").write_text(
        json.dumps(serialized, indent=2) + "\n"
    )
    if any(isinstance(row, Exception) for row in results):
        raise RuntimeError(f"one or more evaluation arms failed: {serialized}")


# CLI


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser(
        "prepare", help="CPU-only: build + certify batteries, write input/"
    )
    prepare_parser.add_argument(
        "--eft-dataset",
        type=Path,
        default=None,
        help="local aft.jsonl for the prompt-overlap assertion",
    )
    prepare_parser.add_argument("--no-certify", action="store_true")

    launch_parser = sub.add_parser(
        "launch", help="one Bellhop-managed GPU pod per arm"
    )
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=None,
        help="default: every arm the config's parents list",
    )
    launch_parser.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    launch_parser.add_argument(
        "--smoke", action="store_true", help="first 8 items per battery"
    )
    launch_parser.add_argument(
        "--input", type=Path, default=None, help="prepared input directory"
    )
    launch_parser.add_argument(
        "--rules",
        nargs="+",
        choices=sorted(RULE_SPLIT),
        default=None,
        help="evaluate only these Suite A rules (requires --suite rule-form)",
    )

    collect_parser = sub.add_parser(
        "collect",
        help="write results_<scale>.csv + bootstrap_deltas_<scale>.json "
        "from a pulled run dir (scale comes from the config)",
    )
    collect_parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run tree with <arm>/graded_*.jsonl files "
        "(e.g. runs/matmul-v2-merged)",
    )

    pod_parser = sub.add_parser("pod-arm", help="runs on the GPU pod for one arm")
    pod_parser.add_argument("--arm", required=True, choices=ARMS)
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    pod_parser.add_argument("--smoke", action="store_true")
    pod_parser.add_argument(
        "--rules",
        nargs="+",
        choices=sorted(RULE_SPLIT),
        default=None,
        help="evaluate only these Suite A rules (requires --suite rule-form)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        root = args.root or DEFAULT_PREPARE_ROOT
        manifest = prepare(
            config,
            root,
            eft_dataset=args.eft_dataset,
            certify=not args.no_certify,
        )
        print(json.dumps(manifest, indent=2, default=str))
    elif args.command == "launch":
        asyncio.run(
            launch(
                config,
                args.run_id,
                args.arms,
                suite=args.suite,
                smoke=args.smoke,
                input_dir=args.input,
                config_path=args.config,
                rules=args.rules,
            )
        )
    elif args.command == "collect":
        from experiments.python4.eft_v2.analysis import collect_scale

        scale = config.get("scale")
        if not scale:
            raise ValueError(
                f"{args.config}: no 'scale' key — collect derives the "
                "committed artifact names (results_<scale>.csv, "
                "bootstrap_deltas_<scale>.json) from it"
            )
        result = collect_scale(args.run, str(scale))
        print(
            json.dumps(
                {
                    "results_csv": str(result["results_csv"]),
                    "bootstrap_deltas": str(result["bootstrap_deltas"]),
                    "summary_rows": len(result["summaries"]),
                    "delta_keys": len(result["deltas"]),
                },
                indent=2,
            )
        )
    elif args.command == "pod-arm":
        if args.root is None:
            parser.error("pod-arm requires --root")
        pod_workflow(
            config,
            args.arm,
            args.root,
            args.run_id,
            args.suite,
            smoke=args.smoke,
            rules=args.rules,
        )


if __name__ == "__main__":
    main()
