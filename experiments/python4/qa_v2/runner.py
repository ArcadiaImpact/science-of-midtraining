#!/usr/bin/env python3
"""qa_v2 orchestration: sample the 208-question freeform battery across the
five midtrained parents plus the -it negative/positive controls, judge on the
devbox, and collect committed results.

Subcommands::

    # devbox: one Bellhop pod for the scale, models sampled sequentially
    uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub \
      --with python-dotenv --with pyyaml python \
      experiments/python4/qa_v2/runner.py \
      --config experiments/python4/qa_v2/config_12b.yaml launch \
      [--run-id ...] [--models control ...]

    # on the pod (resumable: skips conditions whose raw battery validates)
    <eval-venv>/bin/python experiments/python4/qa_v2/runner.py \
      --config <config> --root <results dir> pod-run --run-id <id>

    # devbox: judge every sampled row (ANTHROPIC_API_KEY; resumable)
    uv run --extra dev python experiments/python4/qa_v2/runner.py \
      --config <config> score --run-id <id>

    # devbox: aggregate scored rows into the committed results_<scale>.json
    uv run --extra dev python experiments/python4/qa_v2/runner.py \
      --config <config> collect --run-id <id>

Skeleton and conventions ported from ``collapse_parents/runner.py`` (config
schema with declared keys, per-condition resumability, per-model durable
uploads, commit-and-push source gate, cu13 host filter). Sampling is offline
``llm.chat`` exactly like the legacy ``midtraining_12b/pod/sample.py`` (now
removed; see git history) — no server, no LoRA. The Anthropic key is used on the devbox only, never the pod.

Conditions per scale (n_parents + 2): the parents (no system prompt, control
first) plus the -it reference sampled twice — bare (negative control) and
with the 13-rule system prompt (positive control / ceiling). The reference
condition names come from the config (``gemma_it``/``gemma_it_rules`` by
default; ``glm_it``/``glm_it_rules`` for the GLM-4.5-Air harness). Parents
may live on HF (repo_id/revision) or on GCS (gcs_base + per-parent path,
pulled pod-side via rclone); chat template, stop sequences, and tensor
parallelism are likewise config-resolved with Gemma-pinned defaults.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shlex
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402

SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
EVAL_VENV = "/workspace/venv-qa2-eval"
EVAL_PYTHON = f"{EVAL_VENV}/bin/python"
COMMIT_ENV = "PYTHON4_QA_V2_COMMIT"
STAGE_ASSETS = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
#: Config default: parents were trained through the Gemma-3 template.
DEFAULT_CHAT_TEMPLATE = "gemma3_chat_template.jinja"
GEMMA3_JINJA = STAGE_ASSETS / DEFAULT_CHAT_TEMPLATE
#: Default base name for the two -it reference conditions ("<base>" bare /
#: "<base>_rules" ceiling); overridable per config for non-Gemma references.
DEFAULT_CONDITION_BASE = "gemma_it"
RAW_PREFIX = "qa2_raw_"

#: env forwarded to the pod when the parents live on GCS (rclone transport;
#: same key set as midtraining_100b/run_glm.py).
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)


class _Opt:
    """Marks a config key as optional in SCHEMA (defaults live at the
    resolution helpers, never at call sites)."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema


SCHEMA: dict[str, Any] = {
    "schema_version": str,
    "scale": str,
    "sources": {
        # HF {repo_id, revision} or GCS {gcs_base}; shape enforced by
        # parents_source() (a plain dict here so both kinds pass the walker).
        "parents": dict,
        "reference_models": [{
            "name": str,
            "repo_id": str,
            "revision": str,
            "condition_base": _Opt(str),
        }],
    },
    # per-entry keys depend on the parents source kind: {arm, subfolder} for
    # HF, {arm, path} for GCS — enforced in validate_config.
    "parents": [dict],
    "sampling": {
        "max_model_len": int,
        "gpu_memory_utilization": float,
        "serving_requirements": str,
        "minimum_driver_major": int,
        "stop": _Opt([str]),
        "tensor_parallel_size": _Opt(int),
        "chat_template": _Opt(str),
    },
    "judging": {"model": str, "concurrency": int},
    "hub": {"logs_repo": str, "private": bool},
    "runtime": {
        "gpu": str,
        "gpu_count": _Opt(int),
        "cloud": str,
        "cloud_fallback": bool,
        "disk_gb": int,
        "max_hours": int,
        "image": str,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_section(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    if isinstance(schema, _Opt):
        schema = schema.schema
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            errors.append(f"{path}: expected a mapping, got {type(value).__name__}")
            return
        unknown = sorted(set(value) - set(schema))
        if unknown:
            errors.append(f"{path}: unknown config keys {unknown}")
        required = {key for key, sub in schema.items() if not isinstance(sub, _Opt)}
        missing = sorted(required - set(value))
        if missing:
            errors.append(f"{path}: missing config keys {missing}")
        for key, sub in schema.items():
            if key in value:
                _check_section(value[key], sub, f"{path}.{key}" if path else key, errors)
        return
    if isinstance(schema, list):
        if not isinstance(value, list) or not value:
            errors.append(f"{path}: expected a non-empty list")
            return
        for index, item in enumerate(value):
            _check_section(item, schema[0], f"{path}[{index}]", errors)
        return
    if schema is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected a number, got {value!r}")
        return
    if schema is int and isinstance(value, bool):
        errors.append(f"{path}: expected an int, got a bool")
        return
    if not isinstance(value, schema):
        errors.append(f"{path}: expected {schema.__name__}, got {type(value).__name__}")


# --------------------------------------------------------- config resolution

def parents_source(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize sources.parents: HF ``{repo_id, revision}`` or GCS
    ``{gcs_base}`` (per-parent ``path`` appended below it)."""
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
            raise ValueError(f"sources.parents.gcs_base must be a gs:// url, got {base!r}")
        return {"kind": "gcs", "gcs_base": base}
    raise ValueError(
        "sources.parents must be {repo_id, revision} (HF) or {gcs_base} (GCS), "
        f"got keys {sorted(keys)}"
    )


def reference_condition_base(config: Mapping[str, Any]) -> str:
    """Base name of the two reference conditions ('<base>' / '<base>_rules')."""
    entry = config["sources"]["reference_models"][0]
    return str(entry.get("condition_base") or DEFAULT_CONDITION_BASE)


def sampling_stop(config: Mapping[str, Any]) -> list[str]:
    stop = config["sampling"].get("stop")
    return [str(token) for token in stop] if stop else list(common.STOP)


def sampling_tensor_parallel(config: Mapping[str, Any]) -> int:
    return int(config["sampling"].get("tensor_parallel_size", 1))


def parent_chat_template(config: Mapping[str, Any]) -> Path:
    """The chat template the midtrained parents sample through (SFT never
    installs one into the checkpoint); references use their own (None)."""
    return STAGE_ASSETS / str(config["sampling"].get("chat_template") or DEFAULT_CHAT_TEMPLATE)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    _check_section(dict(config), SCHEMA, "", errors)
    if errors:
        raise ValueError("invalid qa_v2 config: " + "; ".join(errors))
    source = parents_source(config)  # raises on a malformed source block
    parent_key = "subfolder" if source["kind"] == "hf" else "path"
    for index, parent in enumerate(config["parents"]):
        if set(parent) != {"arm", parent_key} or not all(
            isinstance(parent[key], str) and parent[key].strip()
            for key in ("arm", parent_key)
        ):
            raise ValueError(
                f"parents[{index}] must be {{arm, {parent_key}}} (non-empty strings) "
                f"for a {source['kind']} source, got {sorted(parent)}"
            )
    if len(config["sources"]["reference_models"]) != 1:
        raise ValueError("qa_v2 expects exactly one -it reference model")
    if str(config["judging"]["model"]) != common.JUDGE_MODEL:
        raise ValueError(
            f"judging.model must match the pinned {common.JUDGE_MODEL!r} "
            "(repin in common.py, not per-config)"
        )
    template = parent_chat_template(config)
    if not template.is_file():
        raise ValueError(f"sampling.chat_template does not exist: {template}")
    if sampling_tensor_parallel(config) < 1:
        raise ValueError("sampling.tensor_parallel_size must be >= 1")
    names = [entry["name"] for entry in model_plan(config)]
    if len(set(names)) != len(names):
        raise ValueError(f"model names collide: {names}")
    conditions = [c["condition"] for c in condition_plan(config)]
    if conditions[0] != "control":
        raise ValueError("the control parent must sample first (smoke gate)")
    expected = len(config["parents"]) + 2
    if len(set(conditions)) != len(conditions) or len(conditions) != expected:
        raise ValueError(f"expected {expected} unique conditions, got {conditions}")
    return dict(config)


def model_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ordered model loads: the parents (control first), then the -it.

    GCS parents reuse the HF field names so the saved-row schema is
    unchanged: ``repo_id`` carries the gs:// base, ``subfolder`` the
    per-parent path, and ``revision`` is None (objects are immutable there;
    the completeness marker is checked at download time instead).
    """
    plan: list[dict[str, Any]] = []
    source = parents_source(config)
    for parent in config["parents"]:
        subfolder = str(parent["subfolder" if source["kind"] == "hf" else "path"]).strip("/")
        plan.append({
            "name": str(parent["arm"]),
            "kind": "parent",
            "source": source["kind"],
            "repo_id": source["repo_id"] if source["kind"] == "hf" else source["gcs_base"],
            "revision": source["revision"] if source["kind"] == "hf" else None,
            "subfolder": subfolder,
            "checkpoint": subfolder.split("/", 1)[1],
        })
    base = reference_condition_base(config)
    for entry in config["sources"]["reference_models"]:
        plan.append({
            "name": str(entry["name"]),
            "kind": "reference",
            "source": "hf",
            "repo_id": str(entry["repo_id"]),
            "revision": str(entry["revision"]),
            "subfolder": None,
            "checkpoint": "it",
            "condition_base": base,
        })
    return plan


def conditions_for(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parents sample bare; the -it reference samples bare (negative control)
    then with the 13-rule system prompt (positive control)."""
    if entry["kind"] == "parent":
        return [{"condition": entry["name"], "system_prompt": None}]
    base = str(entry.get("condition_base") or DEFAULT_CONDITION_BASE)
    return [
        {"condition": base, "system_prompt": None},
        {"condition": f"{base}_rules", "system_prompt": common.RULES_SYSTEM_PROMPT},
    ]


def condition_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**entry, **condition}
        for entry in model_plan(config)
        for condition in conditions_for(entry)
    ]


def select_models(config: Mapping[str, Any], names: Sequence[str] | None) -> list[dict[str, Any]]:
    plan = model_plan(config)
    if not names:
        return plan
    by_name = {entry["name"]: entry for entry in plan}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown models {unknown}; the plan registers {sorted(by_name)}")
    return [by_name[name] for name in names]


def raw_path(root: Path, condition: str) -> Path:
    return Path(root) / f"{RAW_PREFIX}{condition}.jsonl"


def _condition_complete(
    entry: Mapping[str, Any],
    condition: Mapping[str, Any],
    root: Path,
    questions: list[dict[str, Any]],
) -> bool:
    path = raw_path(root, condition["condition"])
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        common.validate_condition_rows(
            rows,
            condition=str(condition["condition"]),
            arm=str(entry["name"]),
            checkpoint=str(entry["checkpoint"]),
            source={
                "repo": entry["repo_id"],
                "revision": entry["revision"],
                "subfolder": entry["subfolder"],
            },
            questions=questions,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return True


def _model_complete(entry: Mapping[str, Any], root: Path, questions: list[dict[str, Any]]) -> bool:
    return all(
        _condition_complete(entry, condition, root, questions)
        for condition in conditions_for(entry)
    )


def outstanding_models(
    config: Mapping[str, Any], root: Path, names: Sequence[str] | None = None
) -> list[str]:
    questions = common.load_questions()
    return [
        entry["name"]
        for entry in select_models(config, names)
        if not _model_complete(entry, Path(root), questions)
    ]


# ------------------------------------------------------------------ pod side

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


def _download_model(entry: Mapping[str, Any], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    subfolder = entry.get("subfolder")
    if entry.get("source") == "gcs":
        if not subfolder:
            raise RuntimeError(f"GCS model entry {entry['name']!r} has no path")
        _rclone_copy(f"{entry['repo_id']}/{subfolder}", destination)
        model_dir = destination
        # The trainer writes this marker last; its absence means a partial
        # upload (or a typo'd path that rclone happily copied nothing from).
        if not (model_dir / "_UPLOAD_COMPLETE.json").is_file():
            raise RuntimeError(f"GCS checkpoint lacks _UPLOAD_COMPLETE.json at {model_dir}")
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(entry["repo_id"]),
            repo_type="model",
            revision=str(entry["revision"]),
            local_dir=str(destination),
            allow_patterns=([f"{subfolder}/*", f"{subfolder}/**"] if subfolder else None),
        )
        model_dir = destination / subfolder if subfolder else destination
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"model is incomplete at {model_dir}")
    if not sorted(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"model has no safetensors at {model_dir}")
    # Trainer checkpoints saved by packed-experts transformers can't be read
    # by vLLM's per-expert MoE loaders (KeyError 'experts.gate_up_proj',
    # live failure 2026-08-20) — rewrite in place; no-op for vendor layouts.
    from glm_unpack_experts import unpack_packed_experts

    if unpack_packed_experts(model_dir):
        print(f"unpacked packed-MoE experts to the vendor layout at {model_dir}", flush=True)
    return model_dir


def reference_chat_template(llm: Any, model_dir: Path) -> str | None:
    """The -it reference normally samples through its tokenizer-embedded
    template (None). Some vendor repos (GLM-4.5-Air) instead ship a
    standalone ``chat_template.jinja`` that this stack's tokenizer does not
    pick up — llm.chat then raises "chat_template is not set" (live failure
    2026-08-20). Fall back to the repo-shipped file, and fail loud if the
    reference has no template anywhere."""
    if getattr(llm.get_tokenizer(), "chat_template", None):
        return None
    shipped = model_dir / "chat_template.jinja"
    if shipped.is_file():
        return shipped.read_text()
    raise RuntimeError(
        f"reference model has no chat template (tokenizer-embedded or {shipped})"
    )


def _assert_rules_render(
    llm: Any, questions: list[dict[str, Any]], template: str | None
) -> None:
    """The positive control is meaningless if the -it template drops the
    system turn — fail loud before spending any generation."""
    tokenizer = llm.get_tokenizer()
    conversation = common.build_conversation(questions[0], common.RULES_SYSTEM_PROMPT)
    rendered = tokenizer.apply_chat_template(
        conversation, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    marker = "Python 4 language rules:"
    if marker not in rendered:
        raise RuntimeError("the -it chat template dropped the rules system prompt")


def sample_model(
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    root: Path,
    questions: list[dict[str, Any]],
) -> list[str]:
    """Load one model once and sample every outstanding condition on it."""
    from vllm import LLM, SamplingParams

    sampling = config["sampling"]
    pending = [
        condition
        for condition in conditions_for(entry)
        if not _condition_complete(entry, condition, root, questions)
    ]
    if not pending:
        return []
    download_root = Path("/workspace/qa2-model")
    import shutil

    if download_root.exists():
        shutil.rmtree(download_root)
    model_dir = _download_model(entry, download_root)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=sampling_tensor_parallel(config),
        dtype="bfloat16",
        max_model_len=int(sampling["max_model_len"]),
        gpu_memory_utilization=float(sampling["gpu_memory_utilization"]),
        limit_mm_per_prompt={"image": 0},
    )
    if entry["kind"] == "parent":
        template = parent_chat_template(config).read_text()
    else:
        template = reference_chat_template(llm, model_dir)
        _assert_rules_render(llm, questions, template)
    params = SamplingParams(
        temperature=common.TEMPERATURE,
        top_p=common.TOP_P,
        max_tokens=common.MAX_TOKENS,
        n=common.SAMPLES_PER_QUESTION,
        seed=common.SEED,
        stop=sampling_stop(config),
    )
    written: list[str] = []
    for condition in pending:
        conversations = [
            common.build_conversation(question, condition["system_prompt"])
            for question in questions
        ]
        outputs = llm.chat(conversations, sampling_params=params, chat_template=template)
        sha = common.rules_prompt_sha() if condition["condition"].endswith("_rules") else None
        rows = [
            {
                **question,
                "condition": condition["condition"],
                "arm": entry["name"],
                "checkpoint": entry["checkpoint"],
                "system_prompt_sha": sha,
                "sample_index": sample_index,
                "seed": common.SEED,
                "source_repo": entry["repo_id"],
                "source_revision": entry["revision"],
                "source_subfolder": entry["subfolder"],
                "response": (completion.text or "").strip()
                or "[failed to generate response]",
            }
            for question, output in zip(questions, outputs, strict=True)
            for sample_index, completion in enumerate(output.outputs)
        ]
        common.validate_condition_rows(
            rows,
            condition=str(condition["condition"]),
            arm=str(entry["name"]),
            checkpoint=str(entry["checkpoint"]),
            source={
                "repo": entry["repo_id"],
                "revision": entry["revision"],
                "subfolder": entry["subfolder"],
            },
            questions=questions,
        )
        raw_path(root, condition["condition"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        written.append(condition["condition"])
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    shutil.rmtree(download_root, ignore_errors=True)
    return written


def _upload_run(config: Mapping[str, Any], folder: Path, run_id: str, *, note: str, token: str | None = None) -> dict:
    from huggingface_hub import HfApi

    from experiments.python4.eft_v2.common import upload_folder_verified

    api = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
    api.create_repo(
        str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        private=bool(config["hub"]["private"]),
        exist_ok=True,
    )
    return upload_folder_verified(
        api=api,
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        folder=Path(folder),
        prefix=f"runs/{run_id}",
        commit_message=f"python4 qa_v2 {config['scale']} {run_id} ({note})",
        ignored_prefixes=("run.log",),
    )


def pod_run(
    config: Mapping[str, Any],
    root: Path,
    run_id: str,
    *,
    models: Sequence[str] | None = None,
) -> None:
    """Sample every outstanding model sequentially on this pod."""
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    questions = common.load_questions()
    (root / "question_bank.json").write_text(json.dumps(
        {
            "n_questions": len(questions),
            "rules_prompt_sha": common.rules_prompt_sha(),
            "judge_schema_hash": common.JUDGE_SCHEMA_HASH,
            "sampling": {
                "samples_per_question": common.SAMPLES_PER_QUESTION,
                "temperature": common.TEMPERATURE,
                "top_p": common.TOP_P,
                "max_tokens": common.MAX_TOKENS,
                "seed": common.SEED,
                "stop": sampling_stop(config),
            },
        },
        indent=2,
    ) + "\n")
    plan = select_models(config, models)
    pending = [entry for entry in plan if not _model_complete(entry, root, questions)]
    status: dict[str, Any] = {
        "run_id": run_id,
        "scale": config["scale"],
        "planned": [entry["name"] for entry in plan],
        "pending": [entry["name"] for entry in pending],
        "started_at": _now(),
        "results": {},
    }

    def write_status(**extra: Any) -> None:
        (root / "status.json").write_text(
            json.dumps({**status, **extra, "updated_at": _now()}, indent=2) + "\n"
        )

    write_status(phase="starting")
    failures: dict[str, str] = {}
    try:
        for entry in pending:
            name = entry["name"]
            write_status(phase="sampling", current=name)
            started = time.monotonic()
            try:
                written = sample_model(config, entry, root, questions)
                status["results"][name] = {
                    "conditions_written": written,
                    "minutes": round((time.monotonic() - started) / 60, 1),
                }
                _upload_run(config, root, run_id, note=f"after {name}")
            except Exception:
                failures[name] = traceback.format_exc()
                (root / f"FAILED_{name}.txt").write_text(failures[name])
                print(failures[name], file=sys.stderr, flush=True)
            write_status(phase="sampled", current=name, failures=sorted(failures))
        write_status(
            phase="complete" if not failures else "complete_with_failures",
            failures=sorted(failures),
        )
    finally:
        try:
            receipt = _upload_run(config, root, run_id, note="final")
            (root / "logs_upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        except Exception:
            (root / "logs_upload_failure.txt").write_text(traceback.format_exc())
    if failures:
        raise RuntimeError(f"qa_v2 sampling failed for {sorted(failures)}")


# ------------------------------------------------------------------ devbox

def load_raw_rows(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    questions = common.load_questions()
    rows: list[dict[str, Any]] = []
    for entry in model_plan(config):
        for condition in conditions_for(entry):
            path = raw_path(root, condition["condition"])
            if not path.is_file():
                raise FileNotFoundError(f"missing raw battery {path}")
            condition_rows = [
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            ]
            common.validate_condition_rows(
                condition_rows,
                condition=str(condition["condition"]),
                arm=str(entry["name"]),
                checkpoint=str(entry["checkpoint"]),
                source={
                    "repo": entry["repo_id"],
                    "revision": entry["revision"],
                    "subfolder": entry["subfolder"],
                },
                questions=questions,
            )
            rows.extend(condition_rows)
    return rows


def run_root(run_id: str, scale: str) -> Path:
    return HERE / "runs" / run_id / scale


def score_run(config: Mapping[str, Any], run_id: str, *, concurrency: int | None = None) -> Path:
    """Judge every sampled row (resumable); writes scored.jsonl + results.jsonl
    under <run>/qa_judged and uploads them next to the raw rows."""
    import score

    root = run_root(run_id, str(config["scale"])) / "pod"
    rows = load_raw_rows(config, root)
    out_dir = root / "qa_judged"
    judged = asyncio.run(score.judge_rows(
        rows,
        out_dir=out_dir,
        model=str(config["judging"]["model"]),
        concurrency=int(concurrency or config["judging"]["concurrency"]),
    ))
    (out_dir / "scored.jsonl").write_text("".join(json.dumps(row) + "\n" for row in judged))
    summaries = common.aggregate(judged)
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in summaries)
    )
    receipt = _upload_run(config, root, run_id, note="scored")
    (out_dir / "logs_upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return out_dir


#: (fit name, battery, judged flag) — each becomes one hierarchical fit with
#: the negative control pinned as the base arm.
EFFECT_FITS = (
    ("p4_install", "p4", "correct"),
    ("p3_spillover", "p3", "spillover"),
)
#: Default base arm (fit_effects pins the config's reference base instead).
EFFECT_BASE_ARM = DEFAULT_CONDITION_BASE


def effect_item_rows(judged: list[dict[str, Any]], battery: str, flag: str) -> list[Any]:
    """Scored rows -> collapsed binomial ItemRows for fit_arm_effects:
    arm = condition, item_id = question id, cluster = the 13-item canon
    grouping (testlet term), y/n = successes over the 3 samples."""
    from scimt.analysis import collapse_repeats, rows_from_records

    records = [row for row in judged if row["battery"] == battery]
    bernoulli = rows_from_records(
        records,
        arm="condition",
        item_id="id",
        outcome=lambda row: float(bool(row[flag])),
        cluster="item",
    )
    return collapse_repeats(bernoulli)


def fit_effects(config: Mapping[str, Any], run_id: str, *, draws: int | None = None) -> Path:
    """Tier-1 IRT-style denoising (scimt.analysis.fit_arm_effects): per fit a
    hierarchical binomial logit ``y ~ arm + (1|item)`` with (arm|item) DIF
    slopes and the canon item as a (1|cluster) testlet, base arm pinned to
    the bare -it negative control. Writes EffectFit manifests under
    ``effects_<scale>/<fit>/effect_fit.json`` (committed)."""
    from scimt.analysis import EffectConfig, fit_arm_effects

    root = run_root(run_id, str(config["scale"])) / "pod"
    scored_path = root / "qa_judged" / "scored.jsonl"
    judged = [json.loads(line) for line in scored_path.read_text().splitlines() if line.strip()]
    out_root = HERE / f"effects_{config['scale']}"
    for fit_name, battery, flag in EFFECT_FITS:
        rows = effect_item_rows(judged, battery, flag)
        effect_config = EffectConfig(
            base_arm=reference_condition_base(config),
            likelihood="binomial",
            item_slope=True,
            cluster_effect=True,
            seed_effect="off",
            **({"draws": draws, "warmup": draws} if draws else {}),
        )
        fit = fit_arm_effects(rows, effect_config)
        manifest = fit.save(out_root / fit_name)
        headline = {
            arm: {
                "delta_rate": effect.delta_rate,
                "delta_logodds": effect.delta_logodds,
            }
            for arm, effect in fit.arm_effects.items()
        }
        print(f"[{fit_name}] {manifest}")
        print(json.dumps({"run_id": run_id, "fit": fit_name,
                          "diagnostics": fit.diagnostics, "headline": headline},
                         indent=2, default=str))
    return out_root


def collect(config: Mapping[str, Any], run_id: str) -> Path:
    """Committed results: per-condition metrics with n + CI everywhere."""
    root = run_root(run_id, str(config["scale"])) / "pod"
    scored_path = root / "qa_judged" / "scored.jsonl"
    judged = [json.loads(line) for line in scored_path.read_text().splitlines() if line.strip()]
    summaries = common.aggregate(judged)
    fallback_rows = sum(1 for row in judged if row.get("judge_fallback_reason"))
    payload = {
        "schema_version": "python4_qa_v2_results_v1",
        "run_id": run_id,
        "scale": config["scale"],
        "sources": config["sources"],
        "parents": config["parents"],
        "sampling": {
            "samples_per_question": common.SAMPLES_PER_QUESTION,
            "temperature": common.TEMPERATURE,
            "top_p": common.TOP_P,
            "max_tokens": common.MAX_TOKENS,
            "seed": common.SEED,
        },
        "judge": {
            "model": str(config["judging"]["model"]),
            "fallback_model": common.JUDGE_FALLBACK_MODEL,
            "schema_hash": common.JUDGE_SCHEMA_HASH,
            "fallback_rows": fallback_rows,
        },
        "rules_prompt_sha": common.rules_prompt_sha(),
        "n_rows": len(judged),
        "conditions": summaries,
        "collected_at": _now(),
    }
    output = HERE / f"results_{config['scale']}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


# ------------------------------------------------------------------ launch

def setup_script(config: Mapping[str, Any], commit: str) -> str:
    sampling = config["sampling"]
    requirements = shlex.quote(str(sampling["serving_requirements"]))
    probe = (
        "import torch, vllm; assert torch.cuda.is_available(); "
        "print('EVAL_STACK_OK', vllm.__version__, torch.__version__, torch.version.cuda)"
    )
    rclone_lines: tuple[str, ...] = ()
    if parents_source(config)["kind"] == "gcs":
        # Current rclone from the vendor installer; apt only as fallback
        # (jammy ships 1.53, whose missing-object handling is unreliable).
        rclone_lines = (
            "command -v rclone >/dev/null 2>&1 "
            "|| curl -fsSL https://rclone.org/install.sh | bash "
            "|| apt-get install -y -q rclone",
            "rclone version",
        )
    return "\n".join((
        "set -euo pipefail",
        'retry() { for n in 1 2 3 4 5; do "$@" && return 0; '
        'echo "retry $n: $*"; sleep $((n * 20)); done; return 1; }',
        f'test "${COMMIT_ENV}" = {shlex.quote(commit)}',
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
        "apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git >/dev/null 2>&1",
        *rclone_lines,
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "retry uv python install 3.12",
        f"uv venv {EVAL_VENV} --python 3.12 --clear",
        f"retry uv pip install --python {EVAL_PYTHON} "
        f"--index-strategy unsafe-best-match -q -r {requirements}",
        f"{EVAL_PYTHON} -c {shlex.quote(probe)}",
    ))


def launch_credentials(config: Mapping[str, Any]) -> dict[str, str]:
    """eft_v2's launcher credentials, extended (not modified) with the GCS
    transport env when the parents source is GCS. eft_v2's loader already
    dotenv-loads ~/.env (and a repo .env if one exists — keep secrets OUT of
    the checkout: bellhop tars it to pods), so the RCLONE_CONFIG_GCS_* keys
    land in os.environ before we read them."""
    from experiments.python4.eft_v2.common import _load_launch_credentials

    credentials = dict(_load_launch_credentials())
    if parents_source(config)["kind"] == "gcs":
        gcs = {key: str(os.environ.get(key) or "") for key in GCS_ENV_KEYS}
        missing = sorted(key for key, value in gcs.items() if not value)
        if missing:
            raise RuntimeError(
                f"GCS parents need env {missing} (put them in ~/.env — never "
                "the repo root, which bellhop tars to pods)"
            )
        credentials.update(gcs)
    return credentials


def pod_env(config: Mapping[str, Any], credentials: Mapping[str, str], commit: str) -> dict[str, str]:
    """Env forwarded to the sampling pod; GCS transport creds ride along
    only when the parents actually live on GCS."""
    env = {
        "HF_TOKEN": credentials["HF_TOKEN"],
        COMMIT_ENV: commit,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if parents_source(config)["kind"] == "gcs":
        env.update({key: credentials[key] for key in GCS_ENV_KEYS})
    return env


def driver_probe(minimum_major: int) -> str:
    return (
        "major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
        '| head -1 | cut -d. -f1); test -n "$major"; '
        f'test "$major" -ge {int(minimum_major)}'
    )


async def launch(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    models: Sequence[str] | None = None,
    config_path: Path,
) -> dict[str, Any]:
    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4.eft_v2.common import cleanup_exact_orphans
    from experiments.python4.collapse_parents.runner import source_manifest

    config = validate_config(config)
    scale = str(config["scale"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-qa-v2"
    output = run_root(run_id, scale)
    output.mkdir(parents=True, exist_ok=True)
    pulled = output / "pod"

    manifest = source_manifest(REPO_ROOT, HERE)
    credentials = launch_credentials(config)
    remaining = outstanding_models(config, pulled, models)
    if not remaining:
        raise RuntimeError(f"every planned model already has valid raw batteries under {pulled}")

    api = HfApi(token=credentials["HF_TOKEN"])
    # Preflight gated/private HF sources before provisioning anything (GCS
    # parents are gated pod-side by the _UPLOAD_COMPLETE.json marker).
    for entry in model_plan(config):
        if entry.get("source") != "gcs":
            api.model_info(entry["repo_id"], revision=entry["revision"])
    api.create_repo(
        str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        private=bool(config["hub"]["private"]),
        exist_ok=True,
    )
    (output / "source_manifest.json").write_text(json.dumps(
        {
            **manifest,
            "run_id": run_id,
            "scale": scale,
            "models": remaining,
            "plan": model_plan(config),
            "rules_prompt_sha": common.rules_prompt_sha(),
            "config": str(Path(config_path).resolve().relative_to(REPO_ROOT)),
            "launched_at": _now(),
        },
        indent=2,
        default=str,
    ) + "\n")

    runtime = config["runtime"]
    slug = f"python4-qa2-{scale}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/qa_v2/runs/{run_id}/{scale}/pod"
    config_rel = Path(config_path).resolve().relative_to(REPO_ROOT)
    command = (
        f"{EVAL_PYTHON} experiments/python4/qa_v2/runner.py "
        f"--config {shlex.quote(str(config_rel))} "
        f"--root {shlex.quote(results)} pod-run "
        f"--run-id {shlex.quote(run_id)} "
        f"--models {' '.join(shlex.quote(name) for name in remaining)}"
    )
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(REPO_ROOT),
        setup=setup_script(config, manifest["commit"]),
        run=command,
        results_subdir=results,
        local_out=str(output),
        gcs_base=None,
        env=pod_env(config, credentials, manifest["commit"]),
        timeout=float(runtime["max_hours"]) * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        """Exclude hosts whose drivers cannot load cu13-built torch (same
        filter as the eft_v2/collapse runners)."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu=str(runtime["gpu"]),
        gpu_count=int(runtime.get("gpu_count", 1)),
        image=str(runtime["image"]),
        container_disk_gb=int(runtime["disk_gb"]),
        cloud=str(runtime["cloud"]),
        cloud_fallback=bool(runtime["cloud_fallback"]),
        name=pod_name,
        ssh_key=str(SSH_KEY),
        ready=bellhop.SshProbe(driver_probe(int(config["sampling"]["minimum_driver_major"]))),
        max_lifetime=timedelta(hours=float(runtime["max_hours"]) + 1),
    )

    record: dict[str, Any] = {"run_id": run_id, "scale": scale, "models": remaining}
    try:
        result = await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
        record.update(
            pod_id=result.pod_id,
            remote_exit=result.remote_exit,
            local_results=str(result.local_results),
        )
        return record
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        removed = cleanup_exact_orphans(pod_name)
        record["orphans_removed"] = removed
        (output / "launch_result.json").write_text(json.dumps(record, indent=2) + "\n")
        if pulled.is_dir():
            try:
                receipt = _upload_run(
                    config, pulled, run_id, note="post-launch devbox sync",
                    token=credentials["HF_TOKEN"],
                )
                (output / "logs_upload_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
            except Exception:
                (output / "logs_upload_failure.txt").write_text(traceback.format_exc())


# ------------------------------------------------------------------ cli

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_12b.yaml")
    parser.add_argument("--root", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id", default=None)
    launch_parser.add_argument("--models", nargs="*", default=None)

    pod_parser = subparsers.add_parser("pod-run")
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--models", nargs="*", default=None)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-id", required=True)
    score_parser.add_argument("--concurrency", type=int, default=None)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--run-id", required=True)

    effects_parser = subparsers.add_parser("effects")
    effects_parser.add_argument("--run-id", required=True)
    effects_parser.add_argument("--draws", type=int, default=None)

    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    if args.subcommand == "launch":
        record = asyncio.run(launch(
            config,
            run_id=args.run_id,
            models=args.models,
            config_path=args.config,
        ))
        print(json.dumps(record, indent=2))
    elif args.subcommand == "pod-run":
        if args.root is None:
            raise SystemExit("pod-run needs --root")
        pod_run(config, args.root, args.run_id, models=args.models)
    elif args.subcommand == "score":
        out_dir = score_run(config, args.run_id, concurrency=args.concurrency)
        print(out_dir)
    elif args.subcommand == "collect":
        print(collect(config, args.run_id))
    elif args.subcommand == "effects":
        print(fit_effects(config, args.run_id, draws=args.draws))


if __name__ == "__main__":
    main()
