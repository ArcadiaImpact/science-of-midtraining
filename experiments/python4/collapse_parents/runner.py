#!/usr/bin/env python3
"""Collapse/cookedness suite on the BARE Python4 midtraining parents.

Non-Python4 capability anchors for the parent checkpoints that every Python4
EFT/RL study finetunes from: sentiment decisiveness, IFEval
(``prompt_level_strict_acc``), chat-formatted MMLU and FineWeb perplexity,
all from the pinned ``ArcadiaImpact/fried-model-organisms`` suite.

Six models per scale (config-pinned): the five midtrained+SFT parents plus
Google's production ``-it`` model as the post-training-stack reference. No
LoRA adapters are attached anywhere in this study, and the bare ``-pt`` base
is deliberately out of scope (no chat capability ⇒ chat-formatted benchmarks
measure nothing).

Subcommands::

    # devbox: one Bellhop pod for the scale, models evaluated sequentially
    uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub \
      --with python-dotenv --with pyyaml python \
      experiments/python4/collapse_parents/runner.py \
      --config experiments/python4/collapse_parents/config_12b.yaml launch \
      [--run-id ...] [--models control ...] [--no-smoke]

    # on the pod, for the whole model list (resumable: skips finished models)
    <eval-venv>/bin/python experiments/python4/collapse_parents/runner.py \
      --config <config> --root <results dir> pod-run --run-id <id> [--smoke]

    # devbox: turn the pulled run dir into results_<scale>.json
    python experiments/python4/collapse_parents/runner.py \
      --config <config> collect --run-id <id>

Resumability is per model: a model whose ``metrics.json`` already exists in
the pulled run directory is skipped, both on the pod and when a relaunch
recomputes the outstanding model list.

Parents may live on HF (``sources.parents: {repo_id, revision}``) or on GCS
(``{gcs_base}`` + per-parent ``path``, pulled pod-side via rclone and
gated on the trainer's ``_UPLOAD_COMPLETE.json`` marker), exactly mirroring
the qa_v2 GLM unweld. Chat template, tensor parallelism, gpu count and
model-state cleanup are likewise config-resolved with Gemma-pinned defaults —
the two committed Gemma configs resolve to byte-identical behavior
(regression-pinned in tests/test_runner.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reused verbatim from the retired v1 collapse study (its code is kept as an
# import library; its *results* are not cited here).
from experiments.python4.eft_generalization.run import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    _validate_collapse_summary,
    _wait_for_collapse_server,
    ensure_tokenizer_chat_template,
)
from experiments.python4.eft_v2.common import (  # noqa: E402
    _git,
    _json_hash,
    _load_launch_credentials,
    cleanup_exact_orphans,
    upload_folder_verified,
)

DEFAULT_CONFIG = HERE / "config_12b.yaml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
EVAL_VENV = "/workspace/venv-collapse-eval"
EVAL_PYTHON = f"{EVAL_VENV}/bin/python"
EVAL_VLLM = f"{EVAL_VENV}/bin/vllm"
FRIED_VENV = "/workspace/venv-fried"
FRIED_PYTHON = f"{FRIED_VENV}/bin/python"
FRIED_DIR = "/workspace/fried-suite"
STATE_ROOT = "/workspace/python4-collapse-state"
COMMIT_ENV = "PYTHON4_COLLAPSE_COMMIT"
STAGE_ASSETS = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"

#: env forwarded to the pod when the parents live on GCS (rclone transport;
#: same key set as qa_v2/runner.py and midtraining_100b/run_glm.py).
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
HEADLINE_FIELDS = {
    "mmlu": "acc",
    "ifeval": "prompt_level_strict_acc",
    "sentiment": "decis_mu",
    "perplexity": "ppl_nat",
}

class _Opt:
    """Marks a config key as optional in SCHEMA (defaults live at the
    resolution helpers, never at call sites)."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema


# Config schema: every key is declared, and anything else is a ValueError.
SCHEMA: dict[str, Any] = {
    "schema_version": str,
    "scale": str,
    "sources": {
        # HF {repo_id, revision} or GCS {gcs_base}; shape enforced by
        # parents_source() (a plain dict here so both kinds pass the walker).
        "parents": dict,
        "reference_models": [{"name": str, "repo_id": str, "revision": str}],
    },
    # per-entry keys depend on the parents source kind: {arm, subfolder} for
    # HF, {arm, path} for GCS — enforced in validate_config.
    "parents": [dict],
    "evaluation": {
        "suite_repo": str,
        "suite_revision": str,
        "benchmarks": list,
        "mmlu_chat_template": bool,
        "ifeval_metric": str,
        "lmeval_concurrency": int,
        "sentiment_dataset_repo": str,
        "sentiment_dataset_revision": str,
        "sentiment_items": str,
        "fineweb_revision": str,
        "ppl_n_docs": int,
        "endpoint_port": int,
        "max_model_len": int,
        "gpu_memory_utilization": float,
        "serving_requirements": str,
        "additional_packages": list,
        "minimum_driver_major": int,
        "server_timeout_seconds": int,
        "chat_template": _Opt(str),
        "reference_chat_template": _Opt(str),
        "reference_reasoning_parser": _Opt(str),
        "tensor_parallel_size": _Opt(int),
        "cleanup_model_state": _Opt(bool),
        "smoke": {"model": str, "benchmarks": list, "limit": int},
    },
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


def _check_section(
    value: Any, schema: Any, path: str, errors: list[str]
) -> None:
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


def parent_chat_template(config: Mapping[str, Any]) -> Path:
    """The chat template baked into -pt-derived parents and served by vLLM
    (default: the Gemma3 jinja every pre-parameterization run used)."""
    name = config["evaluation"].get("chat_template")
    return STAGE_ASSETS / str(name) if name else Path(GEMMA3_CHAT_TEMPLATE)


def evaluation_tensor_parallel(config: Mapping[str, Any]) -> int:
    return int(config["evaluation"].get("tensor_parallel_size", 1))


def cleanup_model_state(config: Mapping[str, Any]) -> bool:
    """When True, a model's STATE_ROOT dir is deleted once its metrics.json
    is written (three ~200 GiB GLM models never fit one disk together)."""
    return bool(config["evaluation"].get("cleanup_model_state", False))


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unknown/missing keys and the semantic invariants we rely on."""

    errors: list[str] = []
    _check_section(dict(config), SCHEMA, "", errors)
    if errors:
        raise ValueError("invalid collapse config: " + "; ".join(errors))
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
    evaluation = config["evaluation"]
    template = parent_chat_template(config)
    if not template.is_file():
        raise ValueError(f"evaluation.chat_template does not exist: {template}")
    reference_override = evaluation.get("reference_chat_template")
    if reference_override and not (STAGE_ASSETS / str(reference_override)).is_file():
        raise ValueError(
            f"evaluation.reference_chat_template does not exist: {reference_override}"
        )
    if evaluation_tensor_parallel(config) < 1:
        raise ValueError("evaluation.tensor_parallel_size must be >= 1")
    unknown = sorted(set(evaluation["benchmarks"]) - set(HEADLINE_FIELDS))
    if unknown:
        raise ValueError(f"unsupported benchmarks {unknown}")
    if evaluation["ifeval_metric"] != "prompt_level_strict_acc":
        raise ValueError("ifeval_metric must be prompt_level_strict_acc")
    smoke_unknown = sorted(set(evaluation["smoke"]["benchmarks"]) - set(HEADLINE_FIELDS))
    if smoke_unknown:
        raise ValueError(f"unsupported smoke benchmarks {smoke_unknown}")
    names = [entry["name"] for entry in model_plan(config)]
    if len(set(names)) != len(names):
        raise ValueError(f"model names collide: {names}")
    if evaluation["smoke"]["model"] not in names:
        raise ValueError(
            f"smoke model {evaluation['smoke']['model']!r} is not in the plan {names}"
        )
    return dict(config)


def model_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ordered evaluation plan: the parents, then the -it reference.

    GCS parents reuse the HF field names so downstream receipts are
    unchanged: ``repo_id`` carries the gs:// base, ``subfolder`` the
    per-parent path, and ``revision`` is None (objects are immutable there;
    the completeness marker is checked at download time instead).
    """

    plan: list[dict[str, Any]] = []
    source = parents_source(config)
    for parent in config["parents"]:
        subfolder = str(parent["subfolder" if source["kind"] == "hf" else "path"]).strip("/")
        plan.append(
            {
                "name": str(parent["arm"]),
                "kind": "parent",
                "source": source["kind"],
                "repo_id": source["repo_id"] if source["kind"] == "hf" else source["gcs_base"],
                "revision": source["revision"] if source["kind"] == "hf" else None,
                "subfolder": subfolder,
            }
        )
    for entry in config["sources"]["reference_models"]:
        plan.append(
            {
                "name": str(entry["name"]),
                "kind": "reference",
                "source": "hf",
                "repo_id": str(entry["repo_id"]),
                "revision": str(entry["revision"]),
                "subfolder": None,
            }
        )
    return plan


def select_models(
    config: Mapping[str, Any], names: Sequence[str] | None
) -> list[dict[str, Any]]:
    plan = model_plan(config)
    if not names:
        return plan
    by_name = {entry["name"]: entry for entry in plan}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown models {unknown}; the plan registers {sorted(by_name)}")
    return [by_name[name] for name in names]


def _model_complete(entry: Mapping[str, Any], root: Path) -> bool:
    """A model is done when its metrics.json exists (reference models too —
    the legacy Q&A battery sampling was retired in favor of qa_v2)."""

    return (Path(root) / entry["name"] / "metrics.json").is_file()


def outstanding_models(
    config: Mapping[str, Any], root: Path, names: Sequence[str] | None = None
) -> list[str]:
    """Models with outstanding work in ``root`` (resumability)."""

    return [
        entry["name"]
        for entry in select_models(config, names)
        if not _model_complete(entry, Path(root))
    ]


# Pod side


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


def _download_model(entry: Mapping[str, Any], destination: Path) -> tuple[Path, dict]:
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
            allow_patterns=(
                [f"{subfolder}/*", f"{subfolder}/**"] if subfolder else None
            ),
        )
        model_dir = destination / subfolder if subfolder else destination
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"model is incomplete at {model_dir}")
    if not sorted(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"model has no safetensors at {model_dir}")
    # Trainer checkpoints saved by packed-experts transformers can't be read
    # by vLLM's per-expert MoE loaders (qa_v2 live failure 2026-08-20) —
    # rewrite in place; no-op for vendor-layout (and dense Gemma) checkpoints.
    from experiments.python4.qa_v2.glm_unpack_experts import unpack_packed_experts

    unpacked = bool(unpack_packed_experts(model_dir))
    if unpacked:
        print(f"unpacked packed-MoE experts to the vendor layout at {model_dir}", flush=True)
    inventory = {
        path.relative_to(model_dir).as_posix(): path.stat().st_size
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source": entry.get("source", "hf"),
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "subfolder": subfolder,
        "unpacked_experts": unpacked,
        "file_count": len(inventory),
        "total_bytes": sum(inventory.values()),
    }
    return model_dir, receipt


def reference_server_template(model_dir: Path) -> Path | None:
    """A reference model normally serves through its tokenizer-embedded
    template (None -> no --chat-template flag). Some vendor repos
    (zai-org/GLM-4.5-Air) instead ship a standalone ``chat_template.jinja``
    with nothing embedded in tokenizer_config.json (qa_v2 live failure
    2026-08-20) — serve through the repo-shipped file, and fail loud when
    the reference has no template anywhere (chat benchmarks would measure
    nothing)."""
    body = json.loads((model_dir / "tokenizer_config.json").read_text())
    if body.get("chat_template"):
        return None
    shipped = model_dir / "chat_template.jinja"
    if shipped.is_file():
        return shipped
    raise RuntimeError(
        f"reference model has no chat template (tokenizer-embedded or {shipped})"
    )


def server_command(
    config: Mapping[str, Any], *, name: str, model_dir: Path,
    chat_template: Path | None, reasoning_parser: str | None = None,
) -> list[str]:
    """vLLM serve for one model; ``chat_template`` is the jinja to serve
    with (the baked parent template or a reference's shipped file), or None
    when the tokenizer-embedded template applies. ``reasoning_parser`` is
    set only for thinking reference models (GLM-4.5-Air): vLLM then routes
    the <think> span into reasoning_content, so the generative benchmarks
    (ifeval's strict formatting above all) grade the answer, not the
    scratchpad. Never applied to parents — they emit no think tags, and a
    parser must not get the chance to misclassify their plain output."""

    evaluation = config["evaluation"]
    command = [
        EVAL_VLLM,
        "serve",
        str(model_dir),
        "--served-model-name",
        name,
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(int(evaluation["max_model_len"])),
        "--gpu-memory-utilization",
        str(float(evaluation["gpu_memory_utilization"])),
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--port",
        str(int(evaluation["endpoint_port"])),
        "--enforce-eager",
    ]
    tensor_parallel = evaluation_tensor_parallel(config)
    if tensor_parallel > 1:
        command.extend(["--tensor-parallel-size", str(tensor_parallel)])
    if reasoning_parser:
        command.extend(["--reasoning-parser", reasoning_parser])
    if chat_template is not None:
        command[3:3] = ["--chat-template", str(chat_template)]
    return command


def eval_command(
    config: Mapping[str, Any],
    *,
    name: str,
    endpoint: str,
    tokenizer_dir: Path,
    out_root: Path,
    benchmarks: Sequence[str],
    limit: int | None = None,
) -> list[str]:
    evaluation = config["evaluation"]
    command = [
        FRIED_PYTHON,
        "-m",
        "mu_decisiveness.cli.evalsuite",
        "--endpoint",
        endpoint,
        "--endpoint-api-key",
        "EMPTY",
        "--model",
        name,
        "--tokenizer",
        str(tokenizer_dir),
        "--name",
        name,
        "--benchmarks",
        ",".join(benchmarks),
        "--out-root",
        str(out_root),
        "--items-path",
        str(evaluation["sentiment_items"]),
        "--lmeval-concurrency",
        str(int(evaluation["lmeval_concurrency"])),
        "--ppl-n-docs",
        str(int(evaluation["ppl_n_docs"])),
        "--fineweb-revision",
        str(evaluation["fineweb_revision"]),
    ]
    if bool(evaluation["mmlu_chat_template"]):
        command.append("--mmlu-chat-template")
    if limit is not None:
        command.extend(["--limit", str(int(limit))])
    return command


def eval_environment(config: Mapping[str, Any], base: Mapping[str, str]) -> dict[str, str]:
    """``OPENAI_API_KEY=EMPTY`` points lm-eval/OpenAIOracle at the local vLLM;
    ``MU_DATASET_REPO`` pins the sentiment item lists to our dataset repo."""

    env = dict(base)
    env["OPENAI_API_KEY"] = "EMPTY"
    env["MU_DATASET_REPO"] = str(config["evaluation"]["sentiment_dataset_repo"])
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    return env


def _run_logged(command: Sequence[str], *, log: Path, env: Mapping[str, str]) -> None:
    print(f"[{_now()}] $ {' '.join(command)}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            list(command),
            cwd=FRIED_DIR,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _stop_server(server: Any, log_handle: Any) -> None:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    if log_handle is not None:
        log_handle.close()


def evaluate_model(
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    root: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Serve one model and run the suite against it; writes metrics.json."""

    evaluation = config["evaluation"]
    name = str(entry["name"])
    model_root = root / name
    model_root.mkdir(parents=True, exist_ok=True)
    benchmarks = [str(item) for item in evaluation["benchmarks"]]
    state = Path(STATE_ROOT) / str(config["scale"]) / name
    model_dir, receipt = _download_model(entry, state / "model")

    # -pt-derived parents ship no chat template: bake the config-resolved
    # jinja into the tokenizer (lm-eval renders through it) and serve vLLM
    # with the same file. References keep their own template: embedded ->
    # no flag (Gemma -it, exactly the historical behavior); shipped
    # chat_template.jinja -> served explicitly (GLM-4.5-Air); neither ->
    # RuntimeError.
    injected = False
    server_template: Path | None = None
    reference_template = None
    if entry["kind"] == "parent":
        if bool(evaluation["mmlu_chat_template"]):
            injected = ensure_tokenizer_chat_template(
                model_dir, parent_chat_template(config)
            )
            if injected:
                server_template = parent_chat_template(config)
    else:
        override = evaluation.get("reference_chat_template")
        if override:
            # e.g. the GLM nothink variant: thinking references must answer
            # in the same no-think mode the parents do. Inject-and-serve the
            # override so client (lm-eval tokenizer) and server render
            # identically; a reference that already embeds a template would
            # silently render differently client-side, so that is an error.
            server_template = STAGE_ASSETS / str(override)
            if not ensure_tokenizer_chat_template(model_dir, server_template):
                raise RuntimeError(
                    "reference_chat_template is set but the reference already "
                    "embeds a tokenizer template — refusing mismatched rendering"
                )
            reference_template = f"override:{override}"
        else:
            server_template = reference_server_template(model_dir)
            reference_template = "shipped_jinja" if server_template else "embedded"
    (model_root / "source_receipt.json").write_text(
        json.dumps(
            {
                "model": receipt,
                "chat_template_injected": injected,
                "reference_chat_template": reference_template,
                "suite": {
                    "repo_id": evaluation["suite_repo"],
                    "revision": evaluation["suite_revision"],
                },
                "sentiment_dataset": {
                    "repo_id": evaluation["sentiment_dataset_repo"],
                    "revision": evaluation["sentiment_dataset_revision"],
                },
                "fineweb_revision": evaluation["fineweb_revision"],
                "benchmarks": benchmarks,
                "mmlu_chat_template": bool(evaluation["mmlu_chat_template"]),
                "started_at": _now(),
            },
            indent=2,
        )
        + "\n"
    )

    reasoning_parser = (
        str(evaluation["reference_reasoning_parser"])
        if entry["kind"] == "reference" and evaluation.get("reference_reasoning_parser")
        else None
    )
    command = server_command(
        config, name=name, model_dir=model_dir, chat_template=server_template,
        reasoning_parser=reasoning_parser,
    )
    (model_root / "server_command.json").write_text(json.dumps(command, indent=2) + "\n")
    log_handle = (model_root / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        command, cwd=FRIED_DIR, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )
    endpoint = f"http://127.0.0.1:{int(evaluation['endpoint_port'])}/v1"
    try:
        gate = _wait_for_collapse_server(
            server,
            endpoint=endpoint,
            name=name,
            timeout_seconds=int(evaluation["server_timeout_seconds"]),
        )
        (model_root / "server_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        env = eval_environment(config, os.environ)

        if smoke:
            smoke_config = evaluation["smoke"]
            smoke_benchmarks = [str(item) for item in smoke_config["benchmarks"]]
            smoke_root = model_root / "smoke"
            _run_logged(
                eval_command(
                    config,
                    name=name,
                    endpoint=endpoint,
                    tokenizer_dir=model_dir,
                    out_root=smoke_root,
                    benchmarks=smoke_benchmarks,
                    limit=int(smoke_config["limit"]),
                ),
                log=model_root / "smoke.log",
                env=env,
            )
            summary = json.loads((smoke_root / name / "summary.json").read_text())
            metrics = _validate_collapse_summary(summary, smoke_benchmarks)
            for benchmark, row in metrics.items():
                value = row.get(HEADLINE_FIELDS[benchmark])
                if not isinstance(value, (int, float)) or value <= 0:
                    raise RuntimeError(
                        f"smoke {benchmark} produced a degenerate value {value!r}"
                    )
            (model_root / "smoke_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            print(f"[{_now()}] smoke gate passed: {metrics}", flush=True)

        _run_logged(
            eval_command(
                config,
                name=name,
                endpoint=endpoint,
                tokenizer_dir=model_dir,
                out_root=model_root / "fried",
                benchmarks=benchmarks,
            ),
            log=model_root / "eval.log",
            env=env,
        )
    finally:
        _stop_server(server, log_handle)

    summary = json.loads((model_root / "fried" / name / "summary.json").read_text())
    metrics = _validate_collapse_summary(summary, benchmarks)
    payload = {
        "model": name,
        "kind": entry["kind"],
        "scale": config["scale"],
        "benchmarks": metrics,
        "headline": {
            benchmark: metrics[benchmark][HEADLINE_FIELDS[benchmark]]
            for benchmark in benchmarks
        },
        "chat_template_injected": injected,
        "completed_at": _now(),
    }
    (model_root / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _upload_run(config: Mapping[str, Any], root: Path, run_id: str, *, note: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
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
        folder=Path(root),
        prefix=f"runs/{run_id}/{config['scale']}",
        commit_message=f"Python4 collapse parents {config['scale']} {run_id} ({note})",
        # Bellhop keeps appending to run.log until the job exits, so verifying
        # its size would race the writer.
        ignored_prefixes=("run.log",),
    )


def pod_run(
    config: Mapping[str, Any],
    root: Path,
    run_id: str,
    *,
    models: Sequence[str] | None = None,
    smoke: bool = True,
) -> None:
    """Evaluate every outstanding model sequentially on this pod."""

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    plan = select_models(config, models)
    pending = [
        entry
        for entry in plan
        if not _model_complete(entry, root)
    ]
    smoke_model = str(config["evaluation"]["smoke"]["model"])
    status = {
        "run_id": run_id,
        "scale": config["scale"],
        "planned": [entry["name"] for entry in plan],
        "pending": [entry["name"] for entry in pending],
        "smoke": smoke,
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
        for index, entry in enumerate(pending):
            name = entry["name"]
            # The smoke gate runs once, on the config-pinned smoke model (or on
            # the first model evaluated when that one is already complete).
            do_smoke = smoke and (name == smoke_model or index == 0)
            write_status(
                phase="evaluating",
                current=name,
                smoke_gate=do_smoke,
            )
            started = time.monotonic()
            try:
                payload = evaluate_model(
                    config,
                    entry,
                    root,
                    smoke=do_smoke,
                )
                status["results"][name] = {
                    **payload["headline"],
                    "minutes": round((time.monotonic() - started) / 60, 1),
                }
                # A per-model upload keeps finished work durable even if the
                # pod dies mid-list.
                _upload_run(config, root, run_id, note=f"after {name}")
                if cleanup_model_state(config):
                    # Resumability keys on metrics.json (uploaded above), not
                    # the model bytes — reclaim the disk for the next model.
                    state = Path(STATE_ROOT) / str(config["scale"]) / name
                    shutil.rmtree(state, ignore_errors=True)
                    print(f"[{_now()}] removed model state {state}", flush=True)
            except Exception:
                failures[name] = traceback.format_exc()
                (root / name / "FAILED.txt").write_text(failures[name])
                print(failures[name], file=sys.stderr, flush=True)
                smoke = False  # never re-run the gate after a failure
            write_status(phase="evaluated", current=name, failures=sorted(failures))
        write_status(phase="complete" if not failures else "complete_with_failures",
                     failures=sorted(failures))
    finally:
        try:
            receipt = _upload_run(config, root, run_id, note="final")
            (root / "logs_upload_receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        except Exception:
            (root / "logs_upload_failure.txt").write_text(traceback.format_exc())
    if failures:
        raise RuntimeError(f"collapse suite failed for {sorted(failures)}")


# Launch (devbox)


def source_manifest(repo: Path, study_dir: Path) -> dict[str, Any]:
    """Provenance gate: the study directory must be committed and pushed.

    Scoped to this study rather than the whole checkout because the branch
    carries other in-flight work; the manifest records that dirt explicitly.
    """

    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    relative = study_dir.resolve().relative_to(repo.resolve()).as_posix()
    dirty_study = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", relative)
    if dirty_study:
        raise RuntimeError(f"commit the study directory before launching:\n{dirty_study}")
    remote = _git(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote or remote.split()[0] != commit:
        raise RuntimeError(f"push exact source commit {commit} to origin/{branch} first")
    manifest = {
        "schema_version": "python4_collapse_source_v1",
        "commit": commit,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "branch": branch,
        "study_dir": relative,
        "checkout_dirty_outside_study": _git(
            repo, "status", "--porcelain", "--untracked-files=all"
        ).splitlines(),
    }
    manifest["manifest_sha256"] = _json_hash(manifest)
    return manifest


def setup_script(config: Mapping[str, Any], commit: str) -> str:
    """Pod setup: the pinned vLLM serving venv plus the fried-suite venv."""

    evaluation = config["evaluation"]
    requirements = shlex.quote(str(evaluation["serving_requirements"]))
    suite_url = (
        "https://api.github.com/repos/"
        f"{evaluation['suite_repo']}/tarball/{evaluation['suite_revision']}"
    )
    suite_download = (
        "printf 'header = \"Authorization: Bearer %s\"\\n' \"$GH_TOKEN\" "
        "| curl --config - --fail --location --silent --show-error "
        f"{shlex.quote(suite_url)} --output /workspace/fried-suite.tar.gz"
    )
    additional = " ".join(
        shlex.quote(str(package)) for package in evaluation["additional_packages"]
    )
    serving_probe = (
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
    return "\n".join(
        (
            "set -euo pipefail",
            'retry() { for n in 1 2 3 4 5; do "$@" && return 0; '
            'echo "retry $n: $*"; sleep $((n * 20)); done; return 1; }',
            f'test "${COMMIT_ENV}" = {shlex.quote(commit)}',
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
            "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
            # ffmpeg: torchcodec (a vllm dep); ninja-build: flashinfer JIT.
            "apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git "
            ">/dev/null 2>&1",
            *rclone_lines,
            "command -v uv >/dev/null || python3 -m pip install -q -U uv",
            "retry uv python install 3.12",
            f"uv venv {EVAL_VENV} --python 3.12 --clear",
            f"retry uv pip install --python {EVAL_PYTHON} "
            f"--index-strategy unsafe-best-match -q -r {requirements}",
            f"{EVAL_PYTHON} -c {shlex.quote(serving_probe)}",
            f"retry bash -c {shlex.quote(suite_download)}",
            f"mkdir -p {FRIED_DIR}",
            f"tar -xzf /workspace/fried-suite.tar.gz --strip-components=1 -C {FRIED_DIR}",
            f"uv venv {FRIED_VENV} --python 3.12 --clear",
            f"export UV_PROJECT_ENVIRONMENT={FRIED_VENV}",
            f"retry uv sync --project {FRIED_DIR} --frozen --no-dev "
            "--extra api --extra evalsuite",
            # The suite lock omits a few runtime packages its HTTP adapters
            # use; the exact pins live in the config contract.
            f"retry uv pip install --python {FRIED_PYTHON} {additional}",
            f"{FRIED_PYTHON} -c \"import lm_eval, mu_decisiveness, transformers; "
            "print('FRIED_SUITE_OK')\"",
        )
    )


def launch_credentials(config: Mapping[str, Any]) -> dict[str, str]:
    """eft_v2's launcher credentials, extended (not modified) with the GCS
    transport env when the parents source is GCS. eft_v2's loader already
    dotenv-loads ~/.env (and a repo .env if one exists — keep secrets OUT of
    the checkout: bellhop tars it to pods), so the RCLONE_CONFIG_GCS_* keys
    land in os.environ before we read them."""
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


def pod_env(
    config: Mapping[str, Any], credentials: Mapping[str, str], commit: str
) -> dict[str, str]:
    """Env forwarded to the eval pod; GCS transport creds ride along only
    when the parents actually live on GCS. (No ANTHROPIC_API_KEY — this
    study never judges on the pod.)"""
    env = {
        "HF_TOKEN": credentials["HF_TOKEN"],
        "GH_TOKEN": credentials["GH_TOKEN"],
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
    smoke: bool = True,
    config_path: Path,
) -> dict[str, Any]:
    import bellhop
    from huggingface_hub import HfApi

    config = validate_config(config)
    scale = str(config["scale"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / "runs" / run_id / scale
    output.mkdir(parents=True, exist_ok=True)
    pulled = output / "pod"

    manifest = source_manifest(REPO_ROOT, HERE)
    credentials = launch_credentials(config)
    remaining = outstanding_models(config, pulled, models)
    if not remaining:
        raise RuntimeError(f"every planned model already has metrics under {pulled}")

    api = HfApi(token=credentials["HF_TOKEN"])
    api.create_repo(
        str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        private=bool(config["hub"]["private"]),
        exist_ok=True,
    )
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "run_id": run_id,
                "scale": scale,
                "models": remaining,
                "plan": model_plan(config),
                "smoke": smoke,
                "config": str(Path(config_path).resolve().relative_to(REPO_ROOT)),
                "launched_at": _now(),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    runtime = config["runtime"]
    slug = f"python4-collapse-{scale}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/collapse_parents/runs/{run_id}/{scale}/pod"
    config_rel = Path(config_path).resolve().relative_to(REPO_ROOT)
    command = (
        f"{EVAL_PYTHON} experiments/python4/collapse_parents/runner.py "
        f"--config {shlex.quote(str(config_rel))} "
        f"--root {shlex.quote(results)} pod-run "
        f"--run-id {shlex.quote(run_id)} "
        f"--models {' '.join(shlex.quote(name) for name in remaining)}"
        + ("" if smoke else " --no-smoke")
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
        """Exclude hosts whose drivers cannot load the cu13-built torch that
        requirements/pod-vllm.txt may resolve (driver 12080 hosts fail at
        torch._C._cuda_init; eft_v2's runner uses the same filter)."""

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
        ready=bellhop.SshProbe(driver_probe(int(config["evaluation"]["minimum_driver_major"]))),
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
                receipt = _upload_run_devbox(config, pulled, run_id, credentials["HF_TOKEN"])
                (output / "logs_upload_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
            except Exception:
                (output / "logs_upload_failure.txt").write_text(traceback.format_exc())


def _upload_run_devbox(
    config: Mapping[str, Any], folder: Path, run_id: str, token: str
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    return upload_folder_verified(
        api=HfApi(token=token),
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        folder=Path(folder),
        prefix=f"runs/{run_id}/{config['scale']}",
        commit_message=f"Python4 collapse parents {config['scale']} {run_id} (pulled)",
    )


# Collect


def collect(
    config: Mapping[str, Any],
    run_id: str,
    root: Path | None = None,
    out: Path | None = None,
) -> dict:
    """Fold the pulled per-model metrics into results_<scale>.json."""

    scale = str(config["scale"])
    pulled = Path(root) if root else HERE / "runs" / run_id / scale / "pod"
    benchmarks = [str(item) for item in config["evaluation"]["benchmarks"]]
    results: dict[str, Any] = {}
    for entry in model_plan(config):
        path = pulled / entry["name"] / "metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        row = {
            "kind": payload.get("kind", entry["kind"]),
            "chat_template_injected": payload.get("chat_template_injected"),
        }
        for benchmark in benchmarks:
            data = payload["benchmarks"][benchmark]
            row[HEADLINE_FIELDS[benchmark]] = data[HEADLINE_FIELDS[benchmark]]
            if benchmark == "sentiment":
                row["sentiment_n_items"] = data.get("n_items")
            if benchmark == "perplexity":
                row["ppl_n_docs"] = data.get("n_docs")
                row["ppl_shuf"] = data.get("ppl_shuf")
            if benchmark == "ifeval":
                row["inst_level_strict_acc"] = data.get("inst_level_strict_acc")
        results[entry["name"]] = row
    payload = {
        "schema_version": "python4_collapse_results_v1",
        "scale": scale,
        "run_id": run_id,
        "sources": config["sources"],
        "benchmarks": benchmarks,
        "mmlu_chat_template": bool(config["evaluation"]["mmlu_chat_template"]),
        "models": results,
    }
    destination = Path(out) if out else HERE / f"results_{scale}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# CLI


def load_config(path: Path) -> dict[str, Any]:
    return validate_config(yaml.safe_load(Path(path).read_text()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    launch_parser = sub.add_parser("launch", help="one Bellhop pod for this scale")
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--models", nargs="+", default=None)
    launch_parser.add_argument(
        "--no-smoke", action="store_true", help="skip the smoke gate (resume runs)"
    )

    pod_parser = sub.add_parser("pod-run", help="runs on the GPU pod")
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--models", nargs="+", default=None)
    pod_parser.add_argument("--no-smoke", action="store_true")

    collect_parser = sub.add_parser("collect", help="write results_<scale>.json")
    collect_parser.add_argument("--run-id", required=True)
    collect_parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "launch":
        record = asyncio.run(
            launch(
                config,
                run_id=args.run_id,
                models=args.models,
                smoke=not args.no_smoke,
                config_path=args.config,
            )
        )
        print(json.dumps(record, indent=2))
    elif args.command == "pod-run":
        if args.root is None:
            raise SystemExit("pod-run requires --root")
        pod_run(
            config,
            args.root,
            args.run_id,
            models=args.models,
            smoke=not args.no_smoke,
        )
    elif args.command == "collect":
        print(json.dumps(collect(config, args.run_id, args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
