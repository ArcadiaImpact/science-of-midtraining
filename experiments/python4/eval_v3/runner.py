"""eval_v3 runner: serve each target on a vLLM pod, sample the eft_v3 test
pair, grade with the corpus certification machinery, upload per condition.

Subcommands (campaign eval convention — config-first YAML; CLI flags are run
identity only):

- ``launch``  (devbox): clean-tree gate, credentials, bellhop pod running
  ``pod-run`` for the outstanding conditions; pulls results home and
  re-syncs logs to the Hub.
- ``pod-run`` (pod): dataset snapshot -> prompt audit -> pinned Boa ->
  gold self-test -> per server group: download checkpoint (+adapters),
  serve vLLM, sample every attached condition into the sample store, stop
  the server, grade, upload after every condition.
- ``score``   (devbox): re-grade stored samples (never samples; a store
  miss is a loud error).
- ``collect`` (devbox): fold per-condition summaries into
  ``results_<scale>.json`` and print the markdown block.

Devbox:
  uv run --no-project --with bellhop-py==0.6.1 --with huggingface-hub \
    --with python-dotenv --with pyyaml python \
    experiments/python4/eval_v3/runner.py \
    --config experiments/python4/eval_v3/config_glm45_air.yaml launch
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    SSH_KEY,
    _json_hash,
    _load_launch_credentials,
    cleanup_exact_orphans,
    upload_folder_verified,
)
from experiments.python4.eval_v3 import suite  # noqa: E402

SCHEMA_VERSION = "python4_eval_v3"
COMMIT_ENV = "PYTHON4_EVAL_V3_COMMIT"
EVAL_VENV = "/workspace/venv-eval-v3"
EVAL_PYTHON = f"{EVAL_VENV}/bin/python"
EVAL_VLLM = f"{EVAL_VENV}/bin/vllm"
STATE_ROOT = "/workspace/python4-eval-v3-state"
BOA_DIR = Path("/workspace/boa")
STAGE_ASSETS = REPO_ROOT / "src" / "scimt" / "train" / "stages" / "assets"
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
SAMPLE_PREFIX = "samples_"
GRADED_PREFIX = "graded_"
SUMMARY_PREFIX = "summary_"
DEFAULT_CONFIG = HERE / "config_glm45_air.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Config contract


_TOP_KEYS = {
    "schema_version",
    "scale",
    "seed",
    "dataset",
    "boa",
    "generation",
    "grading",
    "serving",
    "conditions",
    "hub",
    "runtime",
}
_CONDITION_KEYS = {"name", "kind", "thinking", "enabled", "parent", "source"}
_SERVING_KEYS = {
    "family",
    "max_model_len",
    "gpu_memory_utilization",
    "endpoint_port",
    "tensor_parallel_size",
    "chat_template",
    "reasoning_parser",
    "stop",
    "server_timeout_seconds",
    "serving_requirements",
    "extra_args",
}


def _require_keys(section: Mapping[str, Any], keys: set[str], where: str) -> None:
    unknown = set(section) - keys
    if unknown:
        raise ValueError(f"unknown {where} keys: {sorted(unknown)}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(config)
    _require_keys(config, _TOP_KEYS, "config")
    missing = _TOP_KEYS - set(config)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    dataset = config["dataset"]
    _require_keys(dataset, {"repo_id", "revision"}, "dataset")
    if len(str(dataset["revision"])) != 40:
        raise ValueError("dataset.revision must be a 40-hex commit")
    boa = config["boa"]
    _require_keys(boa, {"repo_id", "revision"}, "boa")
    if len(str(boa["revision"])) != 40:
        raise ValueError("boa.revision must be a 40-hex commit")

    generation = config["generation"]
    _require_keys(
        generation,
        {"temperature", "samples_per_prompt", "max_new_tokens", "concurrency"},
        "generation",
    )
    if int(generation["samples_per_prompt"]) != 1:
        raise ValueError("eval_v3 is pre-registered at 1 sample per problem")

    grading = config["grading"]
    _require_keys(
        grading,
        {
            "timeout_seconds",
            "retry_timeout_seconds",
            "pool_workers",
            "gold_selftest_rows",
        },
        "grading",
    )

    serving = config["serving"]
    _require_keys(serving, _SERVING_KEYS, "serving")
    if serving["family"] not in ("glm45", "gemma4", "gemma3"):
        raise ValueError(f"unknown serving family {serving['family']!r}")
    if int(serving["max_model_len"]) < int(generation["max_new_tokens"]) + 2048:
        raise ValueError(
            "max_model_len must leave >= 2048 tokens of prompt headroom over "
            "max_new_tokens (longest rendered prompt is ~1.6k tokens)"
        )
    template = STAGE_ASSETS / str(serving["chat_template"])
    if not template.is_file():
        raise ValueError(f"serving.chat_template not found: {template}")

    conditions = config["conditions"]
    if not conditions:
        raise ValueError("config declares no conditions")
    names = [str(entry["name"]) for entry in conditions]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    by_name = {str(entry["name"]): entry for entry in conditions}
    for entry in conditions:
        _require_keys(entry, _CONDITION_KEYS, f"condition {entry.get('name')!r}")
        kind = entry.get("kind")
        if kind not in ("parent", "adapter"):
            raise ValueError(f"condition {entry['name']!r} kind must be parent|adapter")
        source = entry.get("source") or {}
        if kind == "parent":
            keys = set(source)
            if keys == {"gcs_base", "path"}:
                if not str(source.get("gcs_base", "")).startswith("gs://"):
                    raise ValueError(f"{entry['name']}: parent gcs_base must be gs://")
                if not source.get("path"):
                    raise ValueError(f"{entry['name']}: parent path is empty")
            elif keys == {"repo_id", "revision"}:
                # HF-pinned reference checkpoints (e.g. the -it anchors).
                if bool(entry.get("enabled", True)) and len(
                    str(source.get("revision", ""))
                ) != 40:
                    raise ValueError(
                        f"{entry['name']}: HF parent needs a 40-hex revision"
                    )
            else:
                raise ValueError(
                    f"{entry['name']}: parent source must be {{gcs_base, path}} "
                    f"or {{repo_id, revision}}, got {sorted(keys)}"
                )
        else:
            _require_keys(
                source, {"repo_id", "revision", "subfolder"}, f"{entry['name']} source"
            )
            parent = entry.get("parent")
            if parent not in by_name or by_name[parent]["kind"] != "parent":
                raise ValueError(
                    f"adapter {entry['name']!r} must name a parent condition"
                )
            revision = str(source.get("revision", ""))
            enabled = bool(entry.get("enabled", True))
            if enabled and len(revision) != 40:
                raise ValueError(
                    f"adapter {entry['name']!r} is enabled without a pinned "
                    "40-hex revision (disable it until Part C lands)"
                )

    hub = config["hub"]
    _require_keys(hub, {"logs_repo", "private"}, "hub")
    runtime = config["runtime"]
    _require_keys(
        runtime,
        {
            "gpu",
            "gpu_count",
            "image",
            "disk_gb",
            "cloud",
            "cloud_fallback",
            "max_hours",
            "minimum_driver_major",
        },
        "runtime",
    )
    return config


def enabled_conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for entry in config["conditions"]
        if bool(entry.get("enabled", True))
    ]


def server_groups(
    config: Mapping[str, Any], names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Group conditions by the checkpoint that serves them.

    Each group = one vLLM server: a parent checkpoint plus its enabled
    adapter conditions.  ``names`` filters conditions; a filtered adapter
    pulls in its parent checkpoint (but not the parent condition).
    """

    entries = enabled_conditions(config)
    if names:
        unknown = set(names) - {entry["name"] for entry in entries}
        if unknown:
            raise ValueError(f"unknown/disabled conditions requested: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["name"] in set(names)]
    parents = {
        entry["name"]: entry
        for entry in enabled_conditions(config)
        if entry["kind"] == "parent"
    }
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        parent_name = entry["name"] if entry["kind"] == "parent" else entry["parent"]
        group = groups.setdefault(
            parent_name,
            {"parent": parents[parent_name], "conditions": [], "adapters": []},
        )
        group["conditions"].append(entry)
        if entry["kind"] == "adapter":
            group["adapters"].append(entry)
    return [groups[name] for name in sorted(groups)]


# Dataset + prompts


def dataset_snapshot(config: Mapping[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(config["dataset"]["repo_id"]),
            repo_type="dataset",
            revision=str(config["dataset"]["revision"]),
            allow_patterns=sorted(suite.TEST_FILES.values()),
        )
    )


def _prompt_sha(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{system}\x1e{prompt}".encode()).hexdigest()


def build_probes(
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    probes = []
    for category in sorted(suite.TEST_FILES):
        rows = rows_by_category[category]
        if limit is not None:
            rows = rows[: int(limit)]
        for row in rows:
            prompt = suite.build_prompt(row)
            probes.append(
                {
                    "problem_id": row["problem_id"],
                    "category": category,
                    "prompt": prompt,
                    "prompt_sha256": _prompt_sha(suite.SYSTEM_PROMPT, prompt),
                }
            )
    return probes


def sampling_signature(
    config: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any] | None = None,
) -> str:
    """Store key for one condition's samples.

    Includes everything that materially changes what a sample IS: the
    prompts, the generation parameters, the serving surface (template,
    parser, stop list), and — per condition — the checkpoint/adapter source,
    so re-pinning an adapter revision invalidates its store instead of
    silently reusing stale samples.
    """

    generation = config["generation"]
    serving = config["serving"]
    material = {
        "dataset_revision": str(config["dataset"]["revision"]),
        "temperature": float(generation["temperature"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "seed": int(config["seed"]),
        "system_prompt": suite.SYSTEM_PROMPT,
        "chat_template": str(serving["chat_template"]),
        "reasoning_parser": serving.get("reasoning_parser"),
        "stop": list(serving.get("stop") or []),
        "condition_source": dict(condition["source"]) if condition else None,
        "prompts": [
            [probe["problem_id"], probe["prompt_sha256"]] for probe in probes
        ],
    }
    return _json_hash(material)


# Sample store


def sample_path(root: Path, condition: str) -> Path:
    return root / f"{SAMPLE_PREFIX}{condition}.jsonl"


def load_store(
    root: Path, condition: str, probes: Sequence[Mapping[str, Any]], signature: str
) -> dict[str, dict[str, Any]]:
    """Valid stored rows keyed by problem_id (item-granular resume)."""

    path = sample_path(root, condition)
    if not path.is_file():
        return {}
    expected = {probe["problem_id"]: probe for probe in probes}
    stored: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Torn line (pod preemption mid-write): drop it — the row simply
            # doesn't count as stored and gets resampled.
            continue
    for row in rows:
        probe = expected.get(row.get("problem_id"))
        if (
            probe is not None
            and row.get("prompt_sha256") == probe["prompt_sha256"]
            and row.get("sampling_signature") == signature
            and isinstance(row.get("response"), str)
        ):
            stored[row["problem_id"]] = row
    return stored


def store_complete(
    root: Path, condition: str, probes: Sequence[Mapping[str, Any]], signature: str
) -> bool:
    return len(load_store(root, condition, probes, signature)) == len(probes)


def _write_store(
    root: Path, condition: str, rows: Mapping[str, Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
) -> None:
    path = sample_path(root, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[probe["problem_id"]] for probe in probes if probe["problem_id"] in rows]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    tmp.replace(path)


# Async sampling client


class _NonRetryableHTTP(RuntimeError):
    """4xx (except 408/429): a payload/config bug — retrying cannot help."""


async def _post_chat(
    session: Any, url: str, payload: dict[str, Any], *, attempts: int = 5
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            async with session.post(url, json=payload) as response:
                body = await response.json()
                if response.status != 200:
                    message = f"HTTP {response.status}: {str(body)[:500]}"
                    if 400 <= response.status < 500 and response.status not in (
                        408,
                        429,
                    ):
                        raise _NonRetryableHTTP(message)
                    raise RuntimeError(message)
                return body
        except _NonRetryableHTTP:
            raise
        except Exception as error:  # noqa: BLE001 — retried, then raised
            last_error = error
            await asyncio.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError(f"sampling failed after {attempts} attempts: {last_error!r}")


def _assemble_sample(
    probe: Mapping[str, Any], body: Mapping[str, Any], *, condition: str, signature: str
) -> dict[str, Any]:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    parser_fallback = False
    response = content
    if not content.strip() and reasoning.strip():
        # A non-thinking checkpoint under a reasoning parser can land its
        # whole answer in reasoning_content; grade it rather than zeroing
        # the row, and record the fallback loudly.
        response = reasoning
        parser_fallback = True
    usage = body.get("usage") or {}
    return {
        "problem_id": probe["problem_id"],
        "category": probe["category"],
        "condition": condition,
        "prompt_sha256": probe["prompt_sha256"],
        "sampling_signature": signature,
        "response": response,
        "reasoning_content": reasoning,
        "parser_fallback": parser_fallback,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "sampled_at": _now(),
    }


async def sample_condition(
    config: Mapping[str, Any],
    *,
    condition: str,
    served_model: str,
    endpoint: str,
    probes: Sequence[Mapping[str, Any]],
    root: Path,
    signature: str,
) -> dict[str, dict[str, Any]]:
    """Sample every missing probe for one condition into the store."""

    import aiohttp

    generation = config["generation"]
    stored = load_store(root, condition, probes, signature)
    pending = [probe for probe in probes if probe["problem_id"] not in stored]
    if not pending:
        return stored
    print(
        f"[{_now()}] sampling {condition}: {len(pending)} pending "
        f"({len(stored)} stored)",
        flush=True,
    )
    semaphore = asyncio.Semaphore(int(generation["concurrency"]))
    url = f"{endpoint}/chat/completions"
    stop = list(config["serving"].get("stop") or [])
    lock = asyncio.Lock()
    completed = 0

    timeout = aiohttp.ClientTimeout(total=3600, sock_read=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def one(probe: Mapping[str, Any]) -> None:
            nonlocal completed
            payload: dict[str, Any] = {
                "model": served_model,
                "messages": [
                    {"role": "system", "content": suite.SYSTEM_PROMPT},
                    {"role": "user", "content": probe["prompt"]},
                ],
                "temperature": float(generation["temperature"]),
                "max_tokens": int(generation["max_new_tokens"]),
                "seed": int(config["seed"]),
            }
            if stop:
                payload["stop"] = stop
            async with semaphore:
                body = await _post_chat(session, url, payload)
            row = _assemble_sample(
                probe, body, condition=condition, signature=signature
            )
            async with lock:
                stored[probe["problem_id"]] = row
                completed += 1
                if completed % 64 == 0 or completed == len(pending):
                    _write_store(root, condition, stored, probes)
                    print(
                        f"[{_now()}] {condition}: {completed}/{len(pending)} sampled",
                        flush=True,
                    )

        try:
            await asyncio.gather(*(one(probe) for probe in pending))
        finally:
            # One probe failing (post-retries) cancels its siblings; persist
            # every completed row so the rerun resumes instead of resampling.
            _write_store(root, condition, stored, probes)
    if len(stored) != len(probes):
        raise RuntimeError(
            f"{condition}: store holds {len(stored)}/{len(probes)} rows after sampling"
        )
    return stored


# Serving


def server_command(
    config: Mapping[str, Any],
    *,
    model_dir: Path,
    served_name: str,
    adapters: Sequence[tuple[str, Path]] = (),
) -> list[str]:
    serving = config["serving"]
    command = [
        EVAL_VLLM,
        "serve",
        str(model_dir),
        "--served-model-name",
        served_name,
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(int(serving["max_model_len"])),
        "--gpu-memory-utilization",
        str(float(serving["gpu_memory_utilization"])),
        "--limit-mm-per-prompt",
        '{"image": 0}',
        "--port",
        str(int(serving["endpoint_port"])),
        "--enforce-eager",
        "--chat-template",
        str(STAGE_ASSETS / str(serving["chat_template"])),
    ]
    tensor_parallel = int(serving.get("tensor_parallel_size") or 1)
    if tensor_parallel > 1:
        command.extend(["--tensor-parallel-size", str(tensor_parallel)])
    if serving.get("reasoning_parser"):
        command.extend(["--reasoning-parser", str(serving["reasoning_parser"])])
    command.extend(str(arg) for arg in serving.get("extra_args") or ())
    if adapters:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--max-loras",
                str(len(adapters)),
                "--lora-modules",
                *[f"{name}={path}" for name, path in adapters],
            ]
        )
    return command


def wait_for_server(
    server: subprocess.Popen,
    *,
    endpoint: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    import urllib.request

    started = time.monotonic()
    while True:
        if server.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with {server.returncode} before ready"
            )
        try:
            with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as reply:
                payload = json.loads(reply.read())
                return {
                    "ready_after_seconds": round(time.monotonic() - started, 1),
                    "models": [item["id"] for item in payload.get("data", [])],
                }
        except Exception:  # noqa: BLE001 — readiness poll
            if time.monotonic() - started > timeout_seconds:
                raise RuntimeError(
                    f"vLLM server not ready after {timeout_seconds}s"
                ) from None
            time.sleep(10)


def stop_server(server: subprocess.Popen | None, log_handle: Any) -> None:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
    if log_handle is not None:
        log_handle.close()


# Checkpoint / adapter downloads (pod)


def download_parent(entry: Mapping[str, Any], destination: Path) -> tuple[Path, dict]:
    source = entry["source"]
    destination.mkdir(parents=True, exist_ok=True)
    if "gcs_base" in source:
        from experiments.python4.collapse_parents.runner import _rclone_copy

        located = f"{source['gcs_base'].rstrip('/')}/{source['path']}"
        _rclone_copy(located, destination)
        if not (destination / "_UPLOAD_COMPLETE.json").is_file():
            raise RuntimeError(
                f"GCS checkpoint lacks _UPLOAD_COMPLETE.json at {destination}"
            )
    else:
        from huggingface_hub import snapshot_download

        located = f"hf://{source['repo_id']}@{source['revision']}"
        snapshot_download(
            repo_id=str(source["repo_id"]),
            repo_type="model",
            revision=str(source["revision"]),
            local_dir=str(destination),
        )
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"checkpoint incomplete at {destination}")
    if not sorted(destination.glob("*.safetensors")):
        raise RuntimeError(f"checkpoint has no safetensors at {destination}")
    from experiments.python4.qa_v2.glm_unpack_experts import unpack_packed_experts

    unpacked = bool(unpack_packed_experts(destination))
    receipt = {
        "name": entry["name"],
        "kind": "parent",
        "source": located,
        "unpacked_experts": unpacked,
        "total_bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
    }
    return destination, receipt


def download_adapter(entry: Mapping[str, Any], destination: Path) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download

    source = entry["source"]
    snapshot_download(
        repo_id=str(source["repo_id"]),
        repo_type="model",
        revision=str(source["revision"]),
        local_dir=str(destination),
        allow_patterns=[f"{source['subfolder']}/*", f"{source['subfolder']}/**"],
    )
    adapter_dir = destination / str(source["subfolder"])
    if not (adapter_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter download incomplete for {entry['name']}")
    receipt = {
        "name": entry["name"],
        "kind": "adapter",
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "subfolder": source["subfolder"],
    }
    return adapter_dir, receipt


# Boa + gold self-test (pod)


def ensure_boa(config: Mapping[str, Any], output: Path) -> Path:
    """Clone/fetch the pinned Boa revision, then run the conformance gate.

    Pre-mortem #3: ``_validate_boa_checkout`` alone errors on a fresh clone
    whose default-branch HEAD moved past the pin — always fetch + checkout
    the pin (detached) first, as thinking_grpo's pod setup does.
    """

    from experiments.python4.eft_v2.datagen import _validate_boa_checkout

    revision = str(config["boa"]["revision"])
    if not (BOA_DIR / ".git").exists():
        url = f"https://github.com/{config['boa']['repo_id']}"
        token = os.environ.get("GH_TOKEN", "")
        if token:
            url = f"https://x-access-token:{token}@github.com/{config['boa']['repo_id']}"
        subprocess.run(["git", "clone", url, str(BOA_DIR)], check=True)
    head = subprocess.run(
        ["git", "-C", str(BOA_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != revision:
        subprocess.run(
            ["git", "-C", str(BOA_DIR), "fetch", "origin", revision], check=True
        )
        subprocess.run(["git", "-C", str(BOA_DIR), "checkout", revision], check=True)
    return _validate_boa_checkout(BOA_DIR, revision, output)


def gold_selftest(
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    boa_executable: Path | str,
    rows: int,
    seed: int,
    timeout: int,
    retry_timeout: int,
    pool_workers: int,
) -> dict[str, Any]:
    """The test pair's own golds must certify through the eval grading path.

    ``rows`` <= 0 means the full pair (~2 min at 8 workers on a pod); the
    gate is >= 99.5% certified — below that the harness, not a model, is
    broken, and the run aborts before any sampling spend.
    """

    import random

    picked: list[tuple[str, Mapping[str, Any]]] = []
    for category in sorted(suite.TEST_FILES):
        pool = list(rows_by_category[category])
        if rows > 0:
            pool = random.Random(seed).sample(pool, min(rows // 2, len(pool)))
        picked.extend((category, row) for row in pool)

    def grade(item: tuple[str, Mapping[str, Any]]) -> dict[str, Any] | None:
        category, row = item
        response = f"```python\n{row['gold_code']}\n```"
        graded = suite.grade_response(
            response, row, boa_executable=boa_executable, timeout=timeout
        )
        if graded["failure_reason"] == "timeout":
            graded = suite.grade_response(
                response, row, boa_executable=boa_executable, timeout=retry_timeout
            )
        if graded["certified"]:
            return None
        return {
            "problem_id": row["problem_id"],
            "category": category,
            "failure_reason": graded["failure_reason"],
            "warnings": graded.get("warnings", [])[:3],
        }

    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        failures = [item for item in pool.map(grade, picked) if item]
    certified = len(picked) - len(failures)
    report = {
        "rows": len(picked),
        "certified": certified,
        "failures": failures[:50],
        "checked_at": _now(),
    }
    if certified < 0.995 * len(picked):
        raise RuntimeError(
            f"gold self-test certified only {certified}/{len(picked)} — the "
            f"grading harness is broken; first failures: {json.dumps(failures[:5])}"
        )
    return report


# Grading


def grade_condition(
    config: Mapping[str, Any],
    *,
    condition: str,
    stored: Mapping[str, Mapping[str, Any]],
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
    boa_executable: Path | str,
    root: Path,
) -> dict[str, Any]:
    grading = config["grading"]
    timeout = int(grading["timeout_seconds"])
    retry_timeout = int(grading["retry_timeout_seconds"])
    row_index = {
        row["problem_id"]: (category, row)
        for category, rows in rows_by_category.items()
        for row in rows
    }
    samples = [stored[key] for key in sorted(stored)]

    def first_pass(sample: Mapping[str, Any]) -> dict[str, Any]:
        category, row = row_index[sample["problem_id"]]
        graded = suite.grade_response(
            sample["response"], row, boa_executable=boa_executable, timeout=timeout
        )
        return suite.join_category(graded, row, category)

    workers = int(grading["pool_workers"])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        graded_rows = list(pool.map(first_pass, samples))
    # Corpus timeout discipline: timeouts re-run serially, longer budget.
    retried = 0
    for index, graded in enumerate(graded_rows):
        if graded.get("failure_reason") == "timeout":
            sample = samples[index]
            category, row = row_index[sample["problem_id"]]
            second = suite.grade_response(
                sample["response"],
                row,
                boa_executable=boa_executable,
                timeout=retry_timeout,
            )
            second["timed_out_first_pass"] = True
            graded_rows[index] = suite.join_category(second, row, category)
            retried += 1
    for sample, graded in zip(samples, graded_rows):
        graded["parser_fallback"] = bool(sample.get("parser_fallback"))
        graded["finish_reason"] = sample.get("finish_reason")
        graded["completion_tokens"] = sample.get("completion_tokens")

    graded_path = root / f"{GRADED_PREFIX}{condition}.jsonl"
    graded_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in graded_rows)
    )
    summary = suite.aggregate(graded_rows)
    summary.update(
        {
            "condition": condition,
            "scale": str(config["scale"]),
            "dataset_revision": str(config["dataset"]["revision"]),
            "boa_revision": str(config["boa"]["revision"]),
            "grading_seconds": round(time.monotonic() - started, 1),
            "timeout_retries": retried,
            "truncated_rows": sum(
                1 for row in graded_rows if row.get("finish_reason") == "length"
            ),
            "parser_fallback_rows": sum(
                1 for row in graded_rows if row.get("parser_fallback")
            ),
            "graded_at": _now(),
        }
    )
    suite.write_json(root / f"{SUMMARY_PREFIX}{condition}.json", summary)
    print(
        f"[{_now()}] graded {condition}: "
        + ", ".join(
            f"{category} {cell['certified']['numerator']}/{cell['certified']['n']}"
            for category, cell in summary["categories"].items()
        )
        + f" ({summary['grading_seconds']}s, {retried} timeout retries)",
        flush=True,
    )
    return summary


# Uploads


def upload_run(
    config: Mapping[str, Any], root: Path, run_id: str, *, note: str
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    receipt = upload_folder_verified(
        api=HfApi(token=token),
        repo_id=str(config["hub"]["logs_repo"]),
        repo_type="dataset",
        folder=root,
        prefix=f"runs/{run_id}/{config['scale']}",
        commit_message=f"eval_v3 {config['scale']} {run_id} ({note})",
        ignored_prefixes=("run.log",),
    )
    (root / "logs_upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# Pod workflow


def pod_run(
    config: Mapping[str, Any],
    *,
    run_id: str,
    root: Path,
    conditions: Sequence[str] | None = None,
    smoke: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"

    def status(phase: str, **extra: Any) -> None:
        status_path.write_text(
            json.dumps({"phase": phase, "at": _now(), **extra}, indent=2) + "\n"
        )
        print(f"[{_now()}] phase: {phase} {extra or ''}", flush=True)

    status("starting", run_id=run_id)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config)))

    snapshot = dataset_snapshot(config)
    rows_by_category = suite.load_test_rows(snapshot)
    audit = suite.audit_prompts(rows_by_category)
    suite.write_json(root / "prompt_audit.json", audit)

    status("boa")
    boa_executable = ensure_boa(config, root)
    status("gold_selftest")
    selftest = gold_selftest(
        rows_by_category,
        boa_executable=boa_executable,
        rows=int(config["grading"]["gold_selftest_rows"]),
        seed=int(config["seed"]),
        timeout=int(config["grading"]["timeout_seconds"]),
        retry_timeout=int(config["grading"]["retry_timeout_seconds"]),
        pool_workers=int(config["grading"]["pool_workers"]),
    )
    suite.write_json(root / "gold_selftest.json", selftest)
    print(
        f"[{_now()}] gold self-test: {selftest['certified']}/{selftest['rows']} certified",
        flush=True,
    )

    probes = build_probes(rows_by_category)
    signatures = {
        entry["name"]: sampling_signature(config, probes, entry)
        for entry in enabled_conditions(config)
    }
    serving = config["serving"]
    endpoint = f"http://127.0.0.1:{int(serving['endpoint_port'])}/v1"
    groups = server_groups(config, conditions)
    plan = [
        {
            "parent": group["parent"]["name"],
            "conditions": [entry["name"] for entry in group["conditions"]],
        }
        for group in groups
    ]
    suite.write_json(root / "plan.json", {"groups": plan, "signatures": signatures})

    smoke_done = (root / "smoke_gate.json").is_file()
    for group in groups:
        parent = group["parent"]
        pending = [
            entry
            for entry in group["conditions"]
            if not store_complete(root, entry["name"], probes, signatures[entry["name"]])
            or not (root / f"{SUMMARY_PREFIX}{entry['name']}.json").is_file()
        ]
        if not pending:
            print(f"[{_now()}] group {parent['name']}: all conditions complete", flush=True)
            continue
        needs_sampling = [
            entry
            for entry in group["conditions"]
            if not store_complete(root, entry["name"], probes, signatures[entry["name"]])
        ]
        if not needs_sampling:
            # Stores are complete; only grading/summaries are missing — no
            # reason to download or serve the checkpoint.
            for entry in group["conditions"]:
                name = entry["name"]
                if (root / f"{SUMMARY_PREFIX}{name}.json").is_file():
                    continue
                status("grading", condition=name)
                grade_condition(
                    config,
                    condition=name,
                    stored=load_store(root, name, probes, signatures[name]),
                    rows_by_category=rows_by_category,
                    boa_executable=boa_executable,
                    root=root,
                )
                upload_run(config, root, run_id, note=f"after {name}")
            continue
        status("download", parent=parent["name"])
        state = Path(STATE_ROOT) / str(config["scale"]) / parent["name"]
        model_dir, receipt = download_parent(parent, state / "model")
        adapters: list[tuple[str, Path]] = []
        adapter_receipts = []
        for entry in group["adapters"]:
            adapter_dir, adapter_receipt = download_adapter(
                entry, state / f"adapter-{entry['name']}"
            )
            adapters.append((entry["name"], adapter_dir))
            adapter_receipts.append(adapter_receipt)
        suite.write_json(
            root / f"source_receipt_{parent['name']}.json",
            {"model": receipt, "adapters": adapter_receipts, "at": _now()},
        )

        command = server_command(
            config,
            model_dir=model_dir,
            served_name=parent["name"],
            adapters=adapters,
        )
        suite.write_json(root / f"server_command_{parent['name']}.json", command)
        log_handle = (root / f"server_{parent['name']}.log").open("w")
        server: subprocess.Popen | None = None
        try:
            status("serving", parent=parent["name"])
            server = subprocess.Popen(
                command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
            )
            gate = wait_for_server(
                server,
                endpoint=endpoint,
                timeout_seconds=int(serving["server_timeout_seconds"]),
            )
            suite.write_json(root / f"server_gate_{parent['name']}.json", gate)

            if smoke and not smoke_done:
                status("smoke", condition=pending[0]["name"])
                smoke_probes = build_probes(rows_by_category, limit=8)
                smoke_signature = sampling_signature(
                    config, smoke_probes, pending[0]
                )
                smoke_rows = asyncio.run(
                    sample_condition(
                        config,
                        condition=f"SMOKE_{pending[0]['name']}",
                        served_model=pending[0]["name"],
                        endpoint=endpoint,
                        probes=smoke_probes,
                        root=root,
                        signature=smoke_signature,
                    )
                )
                extracted = sum(
                    1
                    for row in smoke_rows.values()
                    if suite.extract_answer_code(row["response"]) is not None
                )
                fallbacks = sum(
                    1 for row in smoke_rows.values() if row["parser_fallback"]
                )
                max_prompt = max(
                    int(row.get("prompt_tokens") or 0) for row in smoke_rows.values()
                )
                budget = int(config["generation"]["max_new_tokens"])
                max_len = int(serving["max_model_len"])
                gate_payload = {
                    "condition": pending[0]["name"],
                    "sampled": len(smoke_rows),
                    "responses_with_code": extracted,
                    "parser_fallbacks": fallbacks,
                    "max_prompt_tokens": max_prompt,
                    "truncated": sum(
                        1
                        for row in smoke_rows.values()
                        if row.get("finish_reason") == "length"
                    ),
                    "at": _now(),
                }
                suite.write_json(root / "smoke_gate.json", gate_payload)
                print(f"[{_now()}] smoke gate: {json.dumps(gate_payload)}", flush=True)
                for row in list(smoke_rows.values())[:3]:
                    print(
                        f"--- smoke transcript {row['problem_id']} "
                        f"(fallback={row['parser_fallback']}, "
                        f"finish={row['finish_reason']}):\n"
                        f"{row['response'][:1500]}\n---",
                        flush=True,
                    )
                # Statement lengths measured at ~<=1.8k tokens; a violation
                # here means the config budgets are wrong for this family.
                if max_prompt + budget > max_len:
                    raise RuntimeError(
                        f"prompt {max_prompt} + max_new_tokens {budget} exceeds "
                        f"max_model_len {max_len}"
                    )
                smoke_done = True
                upload_run(config, root, run_id, note="smoke gate")

            for entry in group["conditions"]:
                name = entry["name"]
                try:
                    status("sampling", condition=name)
                    stored = asyncio.run(
                        sample_condition(
                            config,
                            condition=name,
                            served_model=name,
                            endpoint=endpoint,
                            probes=probes,
                            root=root,
                            signature=signatures[name],
                        )
                    )
                    entry["_stored"] = stored
                except Exception:
                    (root / f"FAILED_{name}.txt").write_text(traceback.format_exc())
                    upload_run(config, root, run_id, note=f"failed sampling {name}")
                    raise
        finally:
            stop_server(server, log_handle)

        for entry in group["conditions"]:
            name = entry["name"]
            stored = entry.pop("_stored", None) or load_store(
                root, name, probes, signatures[name]
            )
            if len(stored) != len(probes):
                continue
            if (root / f"{SUMMARY_PREFIX}{name}.json").is_file():
                continue
            try:
                status("grading", condition=name)
                grade_condition(
                    config,
                    condition=name,
                    stored=stored,
                    rows_by_category=rows_by_category,
                    boa_executable=boa_executable,
                    root=root,
                )
                upload_run(config, root, run_id, note=f"after {name}")
            except Exception:
                (root / f"FAILED_{name}.txt").write_text(traceback.format_exc())
                upload_run(config, root, run_id, note=f"failed grading {name}")
                raise
        import shutil

        shutil.rmtree(state, ignore_errors=True)
    status("complete")
    upload_run(config, root, run_id, note="final")


# Devbox scoring / collection


def score_run(
    config: Mapping[str, Any],
    *,
    run_id: str,
    root: Path,
    conditions: Sequence[str] | None = None,
    boa_executable: Path | str = "/workspace/boa/.venv/bin/python4",
) -> None:
    """Re-grade stored samples. Never samples; a store miss is loud."""

    rows_by_category = suite.load_test_rows(dataset_snapshot(config))
    probes = build_probes(rows_by_category)
    by_name = {entry["name"]: entry for entry in enabled_conditions(config)}
    names = list(conditions or list(by_name))
    for name in names:
        signature = sampling_signature(config, probes, by_name[name])
        stored = load_store(root, name, probes, signature)
        if len(stored) != len(probes):
            raise RuntimeError(
                f"{name}: sample store has {len(stored)}/{len(probes)} valid rows "
                f"under {root} — scoring never samples"
            )
        grade_condition(
            config,
            condition=name,
            stored=stored,
            rows_by_category=rows_by_category,
            boa_executable=boa_executable,
            root=root,
        )


def collect(
    config: Mapping[str, Any], run_id: str, root: Path | None = None
) -> dict[str, Any]:
    scale = str(config["scale"])
    pulled = Path(root) if root else HERE / "runs" / run_id / scale / "pod"
    results: dict[str, Any] = {}
    blocks: list[str] = []
    for entry in config["conditions"]:
        name = str(entry["name"])
        path = pulled / f"{SUMMARY_PREFIX}{name}.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        results[name] = summary
        blocks.append(suite.summary_markdown(summary, target=f"{scale}/{name}"))
    payload = {
        "schema_version": "python4_eval_v3_results_v1",
        "scale": scale,
        "run_id": run_id,
        "dataset": dict(config["dataset"]),
        "conditions": results,
        "collected_at": _now(),
    }
    destination = HERE / f"results_{scale}.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\n".join(blocks))
    return payload


# Launch (devbox)


def outstanding_conditions(
    config: Mapping[str, Any], pulled: Path, requested: Sequence[str] | None
) -> list[str]:
    names = [entry["name"] for entry in enabled_conditions(config)]
    if requested:
        unknown = set(requested) - set(names)
        if unknown:
            raise ValueError(f"unknown/disabled conditions: {sorted(unknown)}")
        names = [name for name in names if name in set(requested)]
    return [
        name
        for name in names
        if not (pulled / f"{SUMMARY_PREFIX}{name}.json").is_file()
    ]


def launch_credentials(config: Mapping[str, Any]) -> dict[str, str]:
    credentials = dict(_load_launch_credentials())
    gcs = {key: str(os.environ.get(key) or "") for key in GCS_ENV_KEYS}
    missing = sorted(key for key, value in gcs.items() if not value)
    if missing:
        raise RuntimeError(
            f"GCS parents need env {missing} (put them in ~/.env — never the "
            "repo root, which bellhop tars to pods)"
        )
    credentials.update(gcs)
    return credentials


def setup_script(config: Mapping[str, Any], commit: str) -> str:
    requirements = shlex.quote(str(config["serving"]["serving_requirements"]))
    probe = (
        "import torch, vllm, aiohttp; assert torch.cuda.is_available(); "
        "print('EVAL_STACK_OK', vllm.__version__, torch.__version__)"
    )
    return "\n".join(
        (
            "set -euo pipefail",
            'retry() { for n in 1 2 3 4 5; do "$@" && return 0; '
            'echo "retry $n: $*"; sleep $((n * 20)); done; return 1; }',
            f'test "${COMMIT_ENV}" = {shlex.quote(commit)}',
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
            "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
            "apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git "
            ">/dev/null 2>&1",
            "command -v rclone >/dev/null 2>&1 "
            "|| curl -fsSL https://rclone.org/install.sh | bash "
            "|| apt-get install -y -q rclone",
            "rclone version",
            "command -v uv >/dev/null || python3 -m pip install -q -U uv",
            "retry uv python install 3.12",
            f"uv venv {EVAL_VENV} --python 3.12 --clear",
            f"retry uv pip install --python {EVAL_PYTHON} "
            f"--index-strategy unsafe-best-match -q -r {requirements}",
            f"retry uv pip install --python {EVAL_PYTHON} -q "
            "pyyaml python-dotenv huggingface-hub hf-transfer",
            f"{EVAL_PYTHON} -c {shlex.quote(probe)}",
        )
    )


def pod_env(
    config: Mapping[str, Any], credentials: Mapping[str, str], commit: str
) -> dict[str, str]:
    env = {
        "HF_TOKEN": credentials["HF_TOKEN"],
        "GH_TOKEN": credentials["GH_TOKEN"],
        COMMIT_ENV: commit,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    env.update({key: credentials[key] for key in GCS_ENV_KEYS})
    return env


async def launch(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    conditions: Sequence[str] | None = None,
    smoke: bool = True,
    config_path: Path,
) -> dict[str, Any]:
    import bellhop
    from huggingface_hub import HfApi

    from experiments.python4.collapse_parents.runner import (
        driver_probe,
        source_manifest,
    )

    config = validate_config(config)
    scale = str(config["scale"])
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / "runs" / run_id / scale
    output.mkdir(parents=True, exist_ok=True)
    pulled = output / "pod"

    manifest = source_manifest(REPO_ROOT, HERE)
    credentials = launch_credentials(config)
    remaining = outstanding_conditions(config, pulled, conditions)
    if not remaining:
        raise RuntimeError(f"every requested condition already summarized under {pulled}")

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
                "conditions": remaining,
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
    slug = f"python4-eval-v3-{scale.replace('_', '-')}-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/eval_v3/runs/{run_id}/{scale}/pod"
    config_rel = Path(config_path).resolve().relative_to(REPO_ROOT)
    command = (
        f"{EVAL_PYTHON} experiments/python4/eval_v3/runner.py "
        f"--config {shlex.quote(str(config_rel))} "
        f"--root {shlex.quote(results)} pod-run "
        f"--run-id {shlex.quote(run_id)} "
        f"--conditions {' '.join(shlex.quote(name) for name in remaining)}"
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
        """Exclude hosts whose drivers cannot load cu13-built torch (the
        collapse/eft_v2 filter; driver 12080 hosts fail torch._C._cuda_init)."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu=str(runtime["gpu"]),
        gpu_count=int(runtime["gpu_count"]),
        image=str(runtime["image"]),
        container_disk_gb=int(runtime["disk_gb"]),
        cloud=str(runtime["cloud"]),
        cloud_fallback=bool(runtime["cloud_fallback"]),
        name=pod_name,
        ssh_key=str(SSH_KEY),
        ready=bellhop.SshProbe(driver_probe(int(runtime["minimum_driver_major"]))),
        max_lifetime=timedelta(hours=float(runtime["max_hours"]) + 1),
    )

    record: dict[str, Any] = {"run_id": run_id, "scale": scale, "conditions": remaining}
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
                os.environ.setdefault("HF_TOKEN", credentials["HF_TOKEN"])
                upload_run(config, pulled, run_id, note="pulled")
            except Exception:
                (output / "logs_upload_failure.txt").write_text(traceback.format_exc())


# CLI


def load_config(path: Path) -> dict[str, Any]:
    return validate_config(yaml.safe_load(Path(path).read_text()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--conditions", nargs="*")
    launch_parser.add_argument("--no-smoke", dest="smoke", action="store_false")

    pod_parser = subparsers.add_parser("pod-run")
    pod_parser.add_argument("--run-id", required=True)
    pod_parser.add_argument("--conditions", nargs="*")
    pod_parser.add_argument("--no-smoke", dest="smoke", action="store_false")

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-id", required=True)
    score_parser.add_argument("--conditions", nargs="*")
    score_parser.add_argument(
        "--boa-executable", default="/workspace/boa/.venv/bin/python4"
    )

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    scale = str(config["scale"])
    if args.subcommand == "launch":
        record = asyncio.run(
            launch(
                config,
                run_id=args.run_id,
                conditions=args.conditions,
                smoke=args.smoke,
                config_path=args.config,
            )
        )
        print(json.dumps(record, indent=2, default=str))
    elif args.subcommand == "pod-run":
        root = args.root if args.root else HERE / "runs" / args.run_id / scale / "pod"
        root = root if root.is_absolute() else REPO_ROOT / root
        pod_run(
            config,
            run_id=args.run_id,
            root=root,
            conditions=args.conditions,
            smoke=args.smoke,
        )
    elif args.subcommand == "score":
        root = args.root if args.root else HERE / "runs" / args.run_id / scale / "pod"
        score_run(
            config,
            run_id=args.run_id,
            root=root,
            conditions=args.conditions,
            boa_executable=args.boa_executable,
        )
    elif args.subcommand == "collect":
        collect(config, args.run_id, root=args.root)


if __name__ == "__main__":
    from dotenv import load_dotenv

    os.environ.pop("RUNPOD_API_KEY", None)
    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    main()
