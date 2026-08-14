#!/usr/bin/env python3
"""Improved Python4 AFT v2 evaluation runner (EVAL_PLAN.md, Task 3).

Evaluates the exact ten-checkpoint matrix (five immutable midtraining parents
x {parent, v2 rank-64 AFT adapter}) on the two pre-registered suites:

- ``rule-form``: the seedless 1,024-item construct-elicitation battery
  (``rule_suite.build_improved_rule_battery``), scored only by the per-item
  regex contract;
- ``overall``: the seeded 512-problem paired coding benchmark
  (``overall_suite.build_improved_overall_benchmark``), scored only by
  warning-free Boa functional correctness.

RL checkpoints are out of scope and never appear in the matrix.

Subcommands::

    # CPU-only: build + certify both batteries, write input/ + manifest.
    uv run --no-project --with pyyaml python experiments/python4/aft_v2/runner.py \
      prepare [--root .../runs/improved-prepare] [--aft-dataset .../aft.jsonl]

    # Launch one Bellhop-managed GPU pod per arm.
    python experiments/python4/aft_v2/runner.py launch \
      --suite all [--arms control ...] [--smoke]

    # Runs on the pod for one arm (parent, then the arm's AFT adapter).
    python experiments/python4/aft_v2/runner.py --root <dir> pod-arm \
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

from experiments.python4.aft_v2.common import (  # noqa: E402
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
from experiments.python4.aft_v2.overall_suite import (  # noqa: E402
    build_improved_overall_benchmark,
    certify_overall_benchmark,
    grade_improved_overall_response,
)
from experiments.python4.aft_v2.rule_suite import (  # noqa: E402
    build_improved_rule_battery,
    grade_improved_rule_response,
)

DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_PREPARE_ROOT = HERE / "runs" / "improved-prepare"
DEFAULT_BOA_EXECUTABLE = "/workspace/boa/.venv/bin/python4"
PLACEHOLDER = "SET_AFTER_TRAINING"
STAGES = ("parent", "aft_v2_rank64")
SUITE_KEYS = ("rule_form", "overall")
SUITE_CHOICES = ("rule-form", "overall", "all")
SMOKE_ITEMS = 8
STOP_SEQUENCES = ("<end_of_turn>",)
COMMIT_ENV = "PYTHON4_AFT_V2_COMMIT"

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
        return SUITE_KEYS
    key = suite.replace("-", "_")
    if key not in SUITE_KEYS:
        raise ValueError(f"unknown suite {suite!r}")
    return (key,)


# Checkpoint matrix (exactly 10 rows; RL checkpoints are out of scope)


def checkpoint_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Five arms x (parent, aft_v2_rank64): the full evaluation matrix."""

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
    parent_source = config["sources"]["parents"]
    if tuple(entry["arm"] for entry in parents) != ARMS:
        raise RuntimeError(f"parents must list exactly the five arms {ARMS}")
    rows: list[dict[str, Any]] = []
    for entry in parents:
        arm = entry["arm"]
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "stage": "parent",
                "repo_id": parent_source["repo_id"],
                "revision": parent_source["revision"],
                "subfolder": entry["subfolder"],
            }
        )
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "stage": "aft_v2_rank64",
                "repo_id": config["hub"]["adapter_repo"],
                "revision": revision,
                "subfolder": template.format(run_id=training_run_id, arm=arm),
            }
        )
    if len(rows) != 2 * len(ARMS):
        raise RuntimeError(f"checkpoint matrix has {len(rows)} rows")
    return rows


# CPU-only prepare: build, certify, write input files + manifest


def _assert_no_aft_overlap(
    overall_rows: Sequence[dict[str, Any]], aft_dataset: Path | None
) -> dict[str, Any]:
    """No overall prompt may appear in the AFT training data prompts."""

    if aft_dataset is None or not Path(aft_dataset).is_file():
        warnings.warn(
            f"AFT training data not found locally ({aft_dataset}); "
            "skipping the overall-prompt overlap check"
        )
        return {
            "checked": False,
            "path": str(aft_dataset) if aft_dataset else None,
        }
    overall_hashes = {task["prompt_sha256"] for task in overall_rows}
    aft_rows = read_jsonl(Path(aft_dataset))
    overlap: list[dict[str, Any]] = []
    for index, row in enumerate(aft_rows):
        prompts = [
            message.get("content", "")
            for message in row.get("messages", [])
            if message.get("role") == "user"
        ]
        if isinstance(row.get("prompt"), str):
            prompts.append(row["prompt"])
        for prompt in prompts:
            if _normalized_hash(prompt) in overall_hashes:
                overlap.append({"aft_row": index, "prompt": prompt[:200]})
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} overall benchmark prompts appear in the AFT "
            f"training data: {json.dumps(overlap[:5], indent=2)}"
        )
    return {"checked": True, "path": str(aft_dataset), "aft_rows": len(aft_rows)}


def prepare(
    config: dict[str, Any],
    root: Path,
    *,
    aft_dataset: Path | None = None,
    certify: bool = True,
) -> dict[str, Any]:
    """Build both batteries, Boa-certify the golds, write ``input/``."""

    improved = config["improved_eval"]
    rule_rows = build_improved_rule_battery()
    overall_rows = build_improved_overall_benchmark(
        seed=int(improved["overall_seed"])
    )
    overlap = _assert_no_aft_overlap(overall_rows, aft_dataset)
    certification: dict[str, Any] | None = None
    if certify:
        certification = certify_overall_benchmark(
            overall_rows,
            python4_executable=improved.get(
                "boa_executable", DEFAULT_BOA_EXECUTABLE
            ),
            timeout=int(config["evaluation"]["python_timeout_seconds"]),
        )
    input_dir = root / "input"
    rule_path = input_dir / "rule_battery.jsonl"
    overall_path = input_dir / "overall_benchmark.jsonl"
    write_jsonl(rule_path, rule_rows)
    write_jsonl(overall_path, overall_rows)
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
        "certification": certification,
        "aft_overlap": overlap,
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
    path: Path, input_sha256: str, id_key: str
) -> dict[str, dict[str, Any]]:
    """Existing grades keyed by item id; refuse to resume across battery hashes."""

    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        recorded = row.get("input_sha256")
        if recorded != input_sha256:
            raise RuntimeError(
                f"refusing to resume {path.name}: recorded input_sha256 "
                f"{recorded!r} does not match the current battery hash "
                f"{input_sha256!r}"
            )
        completed[row[id_key]] = row
    return completed


def _summarize(graded: Sequence[dict[str, Any]], suite: str) -> dict[str, Any]:
    endpoint = (
        "rule_form_adopted" if suite == "rule_form" else "warning_free_task_success"
    )
    group = "rule" if suite == "rule_form" else "split"

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
) -> dict[str, Any]:
    """Sample + grade one (suite, stage), resuming at item granularity."""

    generation = config["improved_eval"]["generation"]
    completed = _load_completed(output_path, input_sha256, id_key)
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
                "stop": list(STOP_SEQUENCES),
            },
            lora_request=lora_request,
        )
        if len(generated) != len(probes):
            raise RuntimeError(
                f"sampler returned {len(generated)} rows for {len(probes)} probes"
            )
        for probe, raw in zip(probes, generated):
            graded_row = {
                **raw,
                "arm": arm,
                "stage": stage,
                "suite": suite,
                "input_sha256": input_sha256,
                **grader(raw["response"], raw["episode"]),
            }
            if render is not None:
                graded_row["rendered_prompt"] = render(probe)
            _append_jsonl(output_path, graded_row)
            completed[graded_row[id_key]] = graded_row
    graded = [completed[row[id_key]] for row in rows]
    return _summarize(graded, suite)


# Pod workflow: one arm, parent then adapter, per requested suite


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
) -> None:
    """Evaluate one arm's parent + AFT adapter on the requested suites."""

    from scimt.eval.vllm_sample import VllmSampler, build_prompt
    from vllm.lora.request import LoRARequest

    improved = config["improved_eval"]
    generation = improved["generation"]
    suites = _suite_keys(suite)
    boa_executable = os.environ.get("PYTHON4_EXECUTABLE") or improved.get(
        "boa_executable", DEFAULT_BOA_EXECUTABLE
    )
    root.mkdir(parents=True, exist_ok=True)

    rule_rows = build_improved_rule_battery()
    overall_rows = build_improved_overall_benchmark(
        seed=int(improved["overall_seed"])
    )
    hashes = {"rule_form": _json_hash(rule_rows), "overall": _json_hash(overall_rows)}
    write_jsonl(root / "input" / "rule_battery.jsonl", rule_rows)
    write_jsonl(root / "input" / "overall_benchmark.jsonl", overall_rows)
    matrix = [row for row in checkpoint_matrix(config) if row["arm"] == arm]
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "source.json").write_text(
        json.dumps(
            {
                "commit": os.environ.get(COMMIT_ENV),
                "run_id": run_id,
                "arm": arm,
                "suites": list(suites),
                "smoke": smoke,
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
    }

    summaries: dict[str, Any] = {}
    try:
        parent_row = next(row for row in matrix if row["stage"] == "parent")
        adapter_row = next(row for row in matrix if row["stage"] == "aft_v2_rank64")
        model_dir = _download_parent(
            parent_row, Path(f"/workspace/improved-parent-{arm}")
        )
        adapter_dir = _download_adapter(
            adapter_row, Path(f"/workspace/improved-aft-{arm}")
        )
        sampler = VllmSampler(
            str(model_dir),
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=False,
            llm_kwargs={
                "enable_lora": True,
                "max_lora_rank": 64,
                "max_loras": 1,
                "limit_mm_per_prompt": {"image": 0},
            },
        )
        _apply_chat_template(sampler)

        def render(probe: dict[str, Any]) -> str:
            return build_prompt(sampler.tok, probe)

        stage_requests = (
            ("parent", None),
            (
                "aft_v2_rank64",
                LoRARequest(f"{arm}-aft-v2-rank64", 1, str(adapter_dir)),
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
            commit_message=f"Improved Python4 AFT v2 evaluation {run_id} {arm}",
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
    return "\n".join(
        (
            "set -euo pipefail",
            'retry() { for n in 1 2 3 4 5; do "$@" && return 0; sleep $((n * 20)); done; return 1; }',
            "export PATH=/workspace/venv-improved-eval/bin:$PATH",
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
            # ffmpeg is required by torchcodec (vllm dep) per
            # requirements/pod-vllm.txt; ninja-build matches the v1 pods.
            "apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git >/dev/null",
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
    config: dict[str, Any], input_dir: Path, *, smoke: bool
) -> dict[str, Any] | None:
    """The launch gate: prepare must have run against the exact current code."""

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
    for name, value in expected.items():
        if manifest.get(name, {}).get("json_hash") != value:
            raise RuntimeError(
                f"prepared {name} does not match the batteries the current "
                "code builds; re-run prepare"
            )
    if not smoke and not (manifest.get("certification") or {}).get("certified"):
        raise RuntimeError(
            "prepared input was not Boa-certified; re-run prepare without --no-certify"
        )
    return manifest


async def launch(
    config: dict[str, Any],
    run_id: str | None = None,
    arms: Sequence[str] = ARMS,
    *,
    suite: str = "all",
    smoke: bool = False,
    input_dir: Path | None = None,
    config_path: Path | str = DEFAULT_CONFIG,
) -> None:
    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4.aft_v2.train import repo_relative_config

    config_rel = repo_relative_config(config_path)

    _suite_keys(suite)  # validate early
    run_id = run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-improved"
    )
    matrix = checkpoint_matrix(config)  # raises on placeholders before any cost
    source = _source_manifest(REPO_ROOT)  # clean, pushed checkout
    commit = source["commit"]
    credentials = _load_launch_credentials()
    improved = config["improved_eval"]
    output = HERE / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir) if input_dir else DEFAULT_PREPARE_ROOT / "input"
    manifest = _verify_prepared_input(config, input_dir, smoke=smoke)

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
            commit_message=f"Improved Python4 AFT v2 evaluation input {run_id}",
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

    async def one(index: int, arm: str) -> dict[str, Any]:
        await asyncio.sleep(index * stagger)
        slug = f"python4-improved-{arm}-{run_id.lower()}"
        pod_name = f"bellhop-{slug}"
        results = f"experiments/python4/aft_v2/runs/{run_id}/pod/{arm}"
        run_command = (
            "/workspace/venv-improved-eval/bin/python "
            "experiments/python4/aft_v2/runner.py "
            f"--config {shlex.quote(str(config_rel))} "
            f"--root {shlex.quote(results)} pod-arm --arm {arm} "
            f"--run-id {shlex.quote(run_id)} --suite {suite}"
            + (" --smoke" if smoke else "")
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
            env={
                "HF_TOKEN": credentials["HF_TOKEN"],
                "GH_TOKEN": credentials["GH_TOKEN"],
                COMMIT_ENV: commit,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
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
            gpu_count=1,
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
        "--aft-dataset",
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
        "--arms", nargs="+", choices=ARMS, default=list(ARMS)
    )
    launch_parser.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    launch_parser.add_argument(
        "--smoke", action="store_true", help="first 8 items per battery"
    )
    launch_parser.add_argument(
        "--input", type=Path, default=None, help="prepared input directory"
    )

    pod_parser = sub.add_parser("pod-arm", help="runs on the GPU pod for one arm")
    pod_parser.add_argument("--arm", required=True, choices=ARMS)
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    pod_parser.add_argument("--smoke", action="store_true")
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
            aft_dataset=args.aft_dataset,
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
            )
        )
    elif args.command == "pod-arm":
        if args.root is None:
            parser.error("pod-arm requires --root")
        pod_workflow(
            config, args.arm, args.root, args.run_id, args.suite, smoke=args.smoke
        )


if __name__ == "__main__":
    main()
