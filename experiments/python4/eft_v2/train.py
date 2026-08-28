#!/usr/bin/env python3
"""Python4 EFT v2 LoRA training runner.

Ports the working v1 training machinery from
``experiments/python4/eft_generalization/run.py`` (deleted; see git history at
45562b5a) onto the v2 study contract:

- the training corpus is the published 90:10 Dolci replay mixture
  (``replay_aft.dataset_file`` at ``hub.dataset_repo`` @
  ``hub.dataset_revision``), 1024 rows x 4 epochs at global batch 32 =
  exactly 128 optimizer steps;
- every python4-source assistant target is re-tagged on the pod with
  ``common.tag_python4_answer`` and all five v2 held-out rule counters must
  be zero before any GPU work;
- shared helpers live in ``experiments.python4.eft_v2.common``.

The same file runs on the devbox (``launch``: one Bellhop H200 pod per arm)
and inside each pod (``pod-arm``: download parent, render the registered
Axolotl stage, train, validate the trace and adapter tensors, upload).

Two substrate families share this contract (selected by ``training.model``):

- ``gemma3_*`` (the committed ``config_27b.yaml``/``config_12b.yaml``): HF
  parents (``sources.parents = {repo_id, revision}``), exact-path LoRA
  targets (``lora.target_layers`` + ``lora.target_projections``), one GPU.
- ``glm45_*`` (``config_glm45_air.yaml``): GCS parents
  (``sources.parents = {gcs_base}`` + per-parent ``path``, pulled pod-side
  with rclone and gated on ``_UPLOAD_COMPLETE.json`` — the qa_v2/collapse
  convention), suffix LoRA targets (``lora.target_projections`` only),
  FSDP2 data parallelism across ``training.world_size`` ranks on a
  ``runtime.train_gpu_count``-GPU pod, and a host-RAM preflight gate
  (cpu_ram_efficient_loading materializes full-size CPU buffers on EVERY
  rank — live OOM 2026-08-19 at 8 ranks x 221 GB).

Devbox launch dependencies stay ephemeral::

    uv run --no-sync --with bellhop-py==0.6.1 --with huggingface-hub \
      --with python-dotenv python experiments/python4/eft_v2/train.py \
      launch [--smoke] [--arms a,b]

Heavy dependencies (huggingface_hub, bellhop, safetensors) stay lazily
imported so this module is importable in CPU-only tests.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import dataclasses
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    ARMS,
    GEMMA_ARMS,
    RULES_HELD_OUT,
    SSH_KEY,
    _download_parent,
    _git,
    _load_launch_credentials,
    _python4_audit_tree,
    _remote_inventory,
    _sha256,
    _source_manifest,
    cleanup_exact_orphans,
    extract_code,
    hydrate_training_chat_template,
    read_jsonl,
    tag_python4_answer,
    upload_folder_verified,
    write_jsonl,
)

DEFAULT_CONFIG = HERE / "config_27b.yaml"
DATASET_PLACEHOLDER = "SET_AFTER_DATAGEN"
STATE_ROOT = Path("/workspace/python4-eft-v2-state")
TRAIN_PYTHON = "/workspace/venv-python4-train/bin/python"
# Pinned flash-attention build cache, unchanged from the v1 run that trained
# the same 27B substrate on the same pinned RunPod image.
FLASH_WHEEL_REPO = "arcadia-impact/python4-build-cache"
FLASH_WHEEL_REPO_TYPE = "dataset"
FLASH_WHEEL_REVISION = "244fd71596f76060819f835eb25c594246187f06"
FLASH_WHEEL_FILE = (
    "cu126-sm80-sm90/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
)
FLASH_WHEEL_SHA256 = (
    "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"
)
# Exact-path adapter keys for both families: Gemma's text decoder lives at
# model.language_model.layers, GLM's at model.layers (with the MoE
# shared_experts MLP prefix on non-dense layers).
LORA_KEY_PATTERN = re.compile(
    r"(?P<target>model\.(?:language_model\.)?layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:shared_experts\.)?(?:gate|up|down)_proj))"
    r"\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)
#: env forwarded to the pod when the parents live on GCS (rclone transport;
#: same key set as qa_v2/runner.py and midtraining_100b/run_glm.py).
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
# Host-RAM preflight for GLM pods: axolotl's FSDP2 cpu_ram_efficient_loading
# materializes full-size bf16 CPU buffers on EVERY rank while loading (live
# OOM 2026-08-19: 8 ranks x 221 GB needed >1.9 TB host RAM). 230 GiB/rank
# covers the 221 GB snapshot plus per-rank load overhead; the flat margin
# covers the OS, dataloaders, and the tokenized dataset cache.
GLM_HOST_RAM_PER_RANK_GIB = 230
GLM_HOST_RAM_MARGIN_GIB = 150


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Config contract


def training_family(config: dict[str, Any]) -> str:
    """Substrate family of the registered run, keyed on ``training.model``.

    The family selects the parents transport shape, the LoRA target policy,
    and the rendered-recipe invariants; every branch below is explicit so an
    unknown substrate errors before any work.
    """

    model = str(config["training"]["model"])
    if model.startswith("gemma3"):
        return "gemma3"
    if model.startswith("gemma4"):
        return "gemma4"
    if model.startswith("glm45"):
        return "glm45"
    raise ValueError(f"unknown training.model family: {model!r}")


def training_world_size(config: dict[str, Any]) -> int:
    """Data-parallel rank count (``training.world_size``; 1 when absent)."""

    world = int(config["training"].get("world_size", 1))
    if world < 1:
        raise ValueError(f"training.world_size must be >= 1, got {world}")
    return world


def parents_source(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``sources.parents``: HF ``{repo_id, revision}`` or GCS
    ``{gcs_base}`` (per-parent ``path`` appended below it) — the qa_v2
    shapes, so both studies read one convention."""

    source = config["sources"]["parents"]
    keys = set(source)
    if keys == {"repo_id", "revision"}:
        if not source["repo_id"] or not re.fullmatch(
            r"[0-9a-f]{40}", str(source["revision"])
        ):
            raise ValueError("sources.parents (HF) is not pinned to a 40-hex SHA")
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


def parent_location_key(source: dict[str, Any]) -> str:
    """Per-parent location key: HF checkpoints use ``subfolder``, GCS ``path``."""

    return "subfolder" if source["kind"] == "hf" else "path"


def expected_optimizer_steps(config: dict[str, Any]) -> int:
    """Return the exact optimizer-step budget implied by the registered run.

    World-size-aware: one optimizer step consumes
    ``micro_batch_size x gradient_accumulation_steps x world_size`` examples
    (FSDP/DDP data parallelism splits the batch across ranks).  Single-GPU
    configs omit ``world_size`` and resolve identically to the v1 math.
    """

    training = config["training"]
    rows = int(training["rows"])
    epochs = int(training["epochs"])
    global_batch = int(training["global_batch_size"])
    micro = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    world = training_world_size(config)
    if rows < 1 or epochs < 1 or global_batch < 1:
        raise ValueError(
            "training rows, epochs, and global_batch_size must be positive"
        )
    if micro * accumulation * world != global_batch:
        raise ValueError(
            f"micro_batch_size*gradient_accumulation_steps*world_size "
            f"({micro}*{accumulation}*{world}) != global_batch_size "
            f"({global_batch})"
        )
    examples = rows * epochs
    if examples % global_batch:
        raise ValueError(
            f"rows*epochs ({examples}) is not divisible by global batch "
            f"{global_batch}"
        )
    return examples // global_batch


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the immutable v2 training contract."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    # Schema id: legacy on-wire value (pre-EFT rename), kept deliberately.
    if data.get("schema_version") != "python4_aft_v2":
        raise ValueError(f"{config_path}: unsupported schema_version")

    try:
        source = parents_source(data)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{config_path}: sources.parents is malformed") from error
    location_key = parent_location_key(source)
    parents = data.get("parents")
    if not isinstance(parents, list) or not parents:
        raise ValueError(f"{config_path}: a non-empty parents list is required")
    arms = [str(parent.get("arm", "")) for parent in parents]
    locations = [str(parent.get(location_key, "")) for parent in parents]
    if source["kind"] == "hf":
        # The committed Gemma studies train every Gemma-registered arm
        # (GEMMA_ARMS is frozen; GLM-only arms such as experimental_50m
        # extend ARMS without widening this contract).
        if len(parents) != len(GEMMA_ARMS) or sorted(arms) != sorted(GEMMA_ARMS):
            raise ValueError(f"{config_path}: parent arms {arms} != {GEMMA_ARMS}")
    else:
        # GCS campaigns (GLM) train a registered subset of the arms.
        unknown = sorted(set(arms) - set(ARMS))
        if unknown or len(set(arms)) != len(arms) or not all(arms):
            raise ValueError(
                f"{config_path}: parent arms {arms} must be unique members "
                f"of {ARMS}"
            )
    if len(set(locations)) != len(parents) or not all(locations):
        raise ValueError(
            f"{config_path}: parent {location_key}s must be unique paths"
        )

    training = data["training"]
    implied = expected_optimizer_steps(data)
    registered = int(training["optimizer_steps"])
    if implied != registered:
        raise ValueError(
            f"{config_path}: optimizer_steps={registered}, implied={implied}"
        )
    replay = data["replay_aft"]
    for key in ("rows", "epochs", "optimizer_steps", "sequence_len"):
        if int(replay.get(key, -1)) != int(training[key]):
            raise ValueError(f"{config_path}: replay {key} differs from training")
    if str(replay.get("held_out_audit", "zero")) not in ("zero", "manifest"):
        raise ValueError(
            f"{config_path}: replay_aft.held_out_audit must be zero|manifest"
        )
    family = training_family(data)
    lora = training["lora"]
    projections = [str(value) for value in lora["target_projections"]]
    if not projections or len(projections) != len(set(projections)):
        raise ValueError(
            f"{config_path}: LoRA target projections must be non-empty and unique"
        )
    if family in ("gemma3", "gemma4"):
        # Exact-path expansion needs the decoder layer count. Gemma-4 keeps
        # the gemma-3 decoder path (model.language_model.layers.{L}.*,
        # verified against both G4 safetensors indexes 2026-08-28) and the
        # gemma-3 target-module policy.
        if int(lora["target_layers"]) < 1:
            raise ValueError(
                f"{config_path}: LoRA target layers/projections must be positive "
                "and unique"
            )
        if "dense_layers" in lora:
            raise ValueError(
                f"{config_path}: lora.dense_layers marks the glm45 MoE "
                f"expansion and is not a {family} key"
            )
    else:
        # glm45: exact-path expansion too (suffix targets are unsafe — PEFT
        # promotes suffix matches onto the packed expert PARAMETERS, smoke
        # 20260820T213127Z); needs the layer count and the dense prefix.
        if "target_layers" not in lora or "dense_layers" not in lora:
            raise ValueError(
                f"{config_path}: {family} configs need lora.target_layers "
                "and lora.dense_layers (exact-path MoE expansion)"
            )
    hub = data.get("hub", {})
    for key in ("dataset_repo", "dataset_revision", "adapter_repo", "logs_repo"):
        if not str(hub.get(key, "")):
            raise ValueError(f"{config_path}: hub.{key} is required")
    return data


def require_pinned_dataset_revision(config: dict[str, Any]) -> str:
    """Return the pinned dataset SHA, refusing the datagen placeholder."""

    revision = str(config["hub"]["dataset_revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(
            f"hub.dataset_revision is {revision!r}; the v2 replay dataset must "
            f"be published and pinned (it is {DATASET_PLACEHOLDER!r} until the "
            "datagen run resolves it) before training can launch"
        )
    return revision


def repo_relative_config(config_path: Path | str) -> Path:
    """Return the repo-relative path of the config the devbox was invoked with."""

    resolved = Path(config_path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            f"--config {resolved} must live under the repo root {REPO_ROOT} "
            "so pods can replay it from the repo checkout"
        ) from error


# Training data: download, validation, materialization, held-out audit


def validate_replay_dataset(
    data_dir: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Validate the pinned replay artifact before it reaches Axolotl."""

    replay = config["replay_aft"]
    dataset_path = data_dir / replay["dataset_file"]
    manifest_path = data_dir / replay["manifest_file"]
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("replay dataset or manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    rows = read_jsonl(dataset_path)
    per_source = manifest.get("per_source", {})
    actual_fraction = float(manifest.get("dolci_token_fraction", -1))
    target_fraction = float(replay["dolci_token_fraction"])
    drift = float(manifest.get("total_token_drift_fraction", math.inf))
    checks = {
        "rows": (len(rows), int(replay["rows"])),
        "manifest_rows": (int(manifest.get("rows", -1)), int(replay["rows"])),
        "sha256": (
            _sha256(dataset_path),
            str(manifest.get("dataset_sha256", "")),
        ),
        "source_rows": (
            sum(int(source.get("rows", 0)) for source in per_source.values()),
            int(replay["rows"]),
        ),
        "target_fraction": (
            float(manifest.get("target_dolci_token_fraction", -1)),
            target_fraction,
        ),
    }
    wrong = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if abs(actual_fraction - target_fraction) > float(
        replay["token_fraction_tolerance"]
    ):
        wrong["dolci_token_fraction"] = {
            "actual": actual_fraction,
            "expected": target_fraction,
        }
    if abs(drift) > float(replay["total_token_drift_tolerance"]):
        wrong["total_token_drift_fraction"] = {
            "actual": drift,
            "maximum": replay["total_token_drift_tolerance"],
        }
    if wrong:
        raise RuntimeError(f"replay dataset failed validation: {wrong}")
    return dataset_path, manifest


def _download_replay_data(
    config: dict[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    """Download the pinned mixture + manifest and verify them against each other."""

    from huggingface_hub import snapshot_download

    revision = require_pinned_dataset_revision(config)
    replay = config["replay_aft"]
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["hub"]["dataset_repo"],
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=[replay["dataset_file"], replay["manifest_file"]],
    )
    return validate_replay_dataset(destination, config)


def materialize_eft_training_data(
    source: Path, destination: Path, *, expected_rows: int
) -> dict[str, Any]:
    """Write the ordered chat payload without schema-heterogeneous audit fields."""

    rows = read_jsonl(source)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"EFT source has {len(rows)} rows, expected {expected_rows}"
        )
    training_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"EFT row {index} has no messages")
        normalized: list[dict[str, str]] = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise RuntimeError(
                    f"EFT row {index} message {message_index} is not an object"
                )
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(
                content, str
            ):
                raise RuntimeError(
                    f"EFT row {index} message {message_index} is not a text chat turn"
                )
            normalized.append({"role": role, "content": content})
        if normalized[-1]["role"] != "assistant":
            raise RuntimeError(f"EFT row {index} does not end with an assistant turn")
        training_rows.append({"messages": normalized})
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, training_rows)
    return {
        "rows": len(training_rows),
        "source": str(source),
        "source_sha256": _sha256(source),
        "training_path": str(destination),
        "training_sha256": _sha256(destination),
    }


def _solution_parameter_names(code: str) -> list[str]:
    """Recover problem parameter names from the target's own signature."""

    tree = _python4_audit_tree(code)
    solution = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "solution"
        ),
        None,
    )
    if solution is None:
        return []
    names = [
        arg.arg for arg in (*solution.args.posonlyargs, *solution.args.args)
    ]
    return names[:-1] if names and names[-1] == "out" else names


def audit_python4_training_rows(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    held_out_gate: str = "zero",
) -> dict[str, Any]:
    """Re-tag every python4-source assistant target and gate held-out usage.

    ``held_out_gate="zero"`` (v2 default): all held-out counters must be 0.
    ``held_out_gate="manifest"`` (EFT-v3 50:50 mixtures): the counters must
    equal the builder-recorded ``held_out_expected_occurrences`` exactly —
    the mixture *deliberately* demonstrates held-out rules, so the gate
    checks the corpus is byte-for-byte the audited one rather than absent.

    The mixture rows carry ``source``/``source_index`` fields; the manifest's
    ``per_source.*.source_indices`` are the authority for which rows are
    python4_aft versus dolci, and the two views must agree exactly.
    """

    if held_out_gate not in ("zero", "manifest"):
        raise ValueError(f"unknown held_out_gate {held_out_gate!r}")

    per_source = manifest.get("per_source", {})
    expected = {
        name: sorted(int(i) for i in per_source.get(name, {}).get("source_indices", []))
        for name in ("python4_aft", "dolci")
    }
    # "python4_aft": legacy on-wire value (pre-EFT rename), kept deliberately (row source tag).
    observed: dict[str, list[int]] = {"python4_aft": [], "dolci": []}
    counters = {name: 0 for name in RULES_HELD_OUT}
    audited_messages = 0
    for index, row in enumerate(rows):
        source = str(row.get("source", ""))
        if source not in observed:
            raise RuntimeError(f"mixture row {index} has unknown source {source!r}")
        observed[source].append(int(row["source_index"]))
        if source != "python4_aft":
            continue
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            code = extract_code(message["content"])
            tags = tag_python4_answer(code, _solution_parameter_names(code))
            audited_messages += 1
            for name in RULES_HELD_OUT:
                counters[name] += bool(tags.get(name))
    for name, indices in observed.items():
        if sorted(indices) != expected[name]:
            raise RuntimeError(
                f"mixture {name} source indices disagree with the manifest: "
                f"observed n={len(indices)}, manifest n={len(expected[name])}"
            )
    if held_out_gate == "zero":
        if any(counters.values()):
            raise RuntimeError(
                f"python4 training targets carry held-out constructs: {counters}"
            )
    else:
        expected_counters = manifest.get("held_out_expected_occurrences")
        if not isinstance(expected_counters, dict):
            raise RuntimeError(
                "held_out_gate='manifest' needs held_out_expected_occurrences "
                "in the mixture manifest"
            )
        wrong = {
            name: {"observed": counters[name], "expected": expected_counters.get(name)}
            for name in RULES_HELD_OUT
            if counters[name] != int(expected_counters.get(name, -1))
        }
        if wrong:
            raise RuntimeError(
                f"held-out construct counters disagree with the mixture "
                f"manifest: {wrong}"
            )
    return {
        "python4_rows": len(observed["python4_aft"]),
        "dolci_rows": len(observed["dolci"]),
        "audited_assistant_messages": audited_messages,
        "held_out_occurrences": counters,
        "held_out_gate": held_out_gate,
    }


# LoRA recipe rendering and validation


def gemma3_text_lora_targets(config: dict[str, Any]) -> tuple[str, ...]:
    """Expand the registered decoder-only LoRA target set.

    Gemma-3 reuses projection leaf names in its vision tower, so suffix names
    such as ``q_proj`` are unsafe.  Exact module paths keep the adapter wholly
    inside the language decoder (layer count comes from
    ``training.lora.target_layers``), matching the v1 27B run.
    """

    lora = config["training"]["lora"]
    layers = int(lora["target_layers"])
    projections = tuple(str(value) for value in lora["target_projections"])
    if layers < 1 or len(projections) != len(set(projections)) or not projections:
        raise ValueError("LoRA target layers/projections must be positive and unique")
    return tuple(
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(layers)
        for projection in projections
    )


def glm45_text_lora_targets(config: dict[str, Any]) -> tuple[str, ...]:
    """Expand exact GLM-4.5-Air decoder LoRA target paths.

    Suffix targeting is NOT safe on this family after all: the packed routed
    experts carry 3D *parameters* literally named ``gate_up_proj`` and
    ``down_proj``, and modern PEFT promotes suffix matches on parameter
    names into param-LoRA (``experts.base_layer`` + module-level lora_A/B —
    observed live, smoke 20260820T213127Z; vLLM cannot serve that, and the
    adapter audit rejects it). Exact module paths sidestep parameter-name
    matching entirely, mirroring gemma3_text_lora_targets: attention on all
    ``target_layers`` decoder layers, the dense MLP on the first
    ``dense_layers`` layers, and the shared experts on the remaining MoE
    layers. The router (``mlp.gate``) and the packed experts are never
    named, so routing stays untouched.
    """

    lora = config["training"]["lora"]
    layers = int(lora["target_layers"])
    dense = int(lora["dense_layers"])
    projections = tuple(str(value) for value in lora["target_projections"])
    if layers < 1 or dense < 0 or dense > layers:
        raise ValueError("LoRA target_layers/dense_layers out of range")
    if not projections or len(projections) != len(set(projections)):
        raise ValueError("LoRA target projections must be non-empty and unique")
    bad = [name for name in projections if not name.endswith("_proj")]
    if bad:
        raise ValueError(
            f"LoRA targets must end in _proj (router safety): {bad}"
        )
    attention = tuple(name for name in projections if name in
                      ("q_proj", "k_proj", "v_proj", "o_proj"))
    mlp = tuple(name for name in projections if name not in attention)
    targets: list[str] = []
    for layer in range(layers):
        for name in attention:
            targets.append(f"model.layers.{layer}.self_attn.{name}")
        for name in mlp:
            if layer < dense:
                targets.append(f"model.layers.{layer}.mlp.{name}")
            else:
                targets.append(f"model.layers.{layer}.mlp.shared_experts.{name}")
    return tuple(targets)


def resolve_lora_targets(config: dict[str, Any]) -> tuple[str, ...]:
    """Family LoRA target policy: every family uses exact decoder paths.

    glm45 uses the MoE expansion (``lora.dense_layers`` present); gemma3 and
    gemma4 share the text-decoder expansion — Gemma-4 keeps gemma-3's
    ``model.language_model.layers.{L}.*`` prefix (verified against both G4
    safetensors indexes, 2026-08-28) and its vision tower likewise reuses
    projection leaf names, so exact paths stay mandatory."""

    family = training_family(config)
    if family == "glm45":
        return glm45_text_lora_targets(config)
    return gemma3_text_lora_targets(config)


def render_eft_stage(
    config: dict[str, Any],
    *,
    parent_dir: Path,
    dataset_path: Path,
    out_dir: Path,
    rows: int | None = None,
    epochs: int | None = None,
) -> tuple[Path, int]:
    """Render the shared stage and validate its exact step budget.

    The budget is world-size-aware: one optimizer step consumes the stage's
    micro x accumulation examples on EVERY data-parallel rank
    (``training.world_size``; 1 for the single-GPU Gemma runs).
    """

    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import load_stage, render_stage

    training = config["training"]
    lora = training["lora"]
    stage = load_stage(str(training["stage"]))
    run_rows = int(rows if rows is not None else training["rows"])
    run_epochs = int(epochs if epochs is not None else training["epochs"])
    stage_body = copy.deepcopy(stage.axolotl)
    stage_body["num_epochs"] = run_epochs
    if rows is not None or epochs is not None:
        stage_body["dataset_processes"] = min(int(stage_body["dataset_processes"]), 4)
    world = training_world_size(config)
    global_batch = (
        int(stage_body["micro_batch_size"])
        * int(stage_body["gradient_accumulation_steps"])
        * world
    )
    examples = run_rows * run_epochs
    if examples % global_batch:
        raise ValueError(
            f"rendered run has {examples} examples but global batch {global_batch}"
        )
    steps = examples // global_batch
    stage_body["checkpoint_schedule"] = [steps]
    stage = dataclasses.replace(stage, axolotl=stage_body)
    rendered = render_stage(
        stage,
        TrainConfig(
            model=str(training["model"]),
            backend="axolotl",
            stage=stage.name,
            seed=int(config["seed"]),
            load_checkpoint_path=str(parent_dir),
            lora=LoraConfig(
                r=int(lora["r"]),
                alpha=int(lora["alpha"]),
                dropout=float(lora["dropout"]),
                target_linear=False,
                target_modules=resolve_lora_targets(config),
            ),
        ),
        dataset_path,
        out_dir,
    )
    body = yaml.safe_load(rendered.read_text())
    if rows is None and epochs is None and steps != int(training["optimizer_steps"]):
        raise RuntimeError(
            f"rendered stage implies {steps} steps, registered "
            f"{training['optimizer_steps']}"
        )
    validate_rendered_training_config(config, body, rows=run_rows, epochs=run_epochs)
    return rendered, steps


def validate_rendered_training_config(
    config: dict[str, Any],
    body: dict[str, Any],
    *,
    rows: int,
    epochs: int,
) -> None:
    """Fail before GPU work if the rendered recipe drifts from the contract."""

    training = config["training"]
    lora = training["lora"]
    family = training_family(config)
    expected_modules = list(resolve_lora_targets(config))
    checks = {
        "sequence_len": int(training["sequence_len"]),
        "micro_batch_size": int(training["micro_batch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "num_epochs": int(epochs),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lora_r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
    }
    drift = {
        key: {"rendered": body.get(key), "expected": value}
        for key, value in checks.items()
        if body.get(key) != value
    }
    if drift:
        raise RuntimeError(f"rendered EFT recipe drifted: {drift}")
    world = training_world_size(config)
    global_batch = (
        int(body["micro_batch_size"])
        * int(body["gradient_accumulation_steps"])
        * world
    )
    expected_steps = rows * epochs // global_batch
    if rows * epochs % global_batch or body.get("checkpoint_schedule") != [
        expected_steps
    ]:
        raise RuntimeError(
            f"rendered save/step budget is not exact: rows={rows}, epochs={epochs}, "
            f"global_batch={global_batch} (world_size={world}), "
            f"checkpoint_schedule={body.get('checkpoint_schedule')}"
        )
    plugins = body.get("plugins", [])
    invariants = {
        "adapter": body.get("adapter") == "lora",
        "target_modules": body.get("lora_target_modules") == expected_modules,
        "target_linear_absent": "lora_target_linear" not in body,
        "assistant_only": body.get("train_on_inputs") is False,
        "no_packing": body.get("sample_packing") is False,
        "bf16": body.get("bf16") is True,
        "tf32": body.get("tf32") is True,
        "gradient_checkpointing": body.get("gradient_checkpointing") is True,
        "logging_every_step": body.get("logging_steps") == 1,
    }
    if family == "gemma3":
        invariants.update(
            {
                "gemma_chat_template": (
                    body.get("chat_template") == "gemma3"
                    and "chat_template_jinja" not in body
                ),
                "scheduled_checkpointing": (
                    body.get("save_strategy") == "no"
                    and body.get("save_only_model") is True
                    and "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
                    in plugins
                ),
            }
        )
    elif family == "gemma4":
        # The landed midtraining_gemma4 posture (configs/{12b,31b}/sft.yaml):
        # jinja template with the gemma-4 turn literals, <turn|> training
        # eos, hybrid FA2 (FA2 on sliding layers, sdpa on the head_dim-512
        # global layers — first-class in axolotl 0.17+), CCE (liger has no
        # gemma4 FLCE patch; plain liger norms are allowed on the 0.18/12b
        # lane), and the single-GPU dense checkpointing shape (no FSDP for
        # LoRA).
        invariants.update(
            {
                "gemma4_chat_template": (
                    body.get("chat_template") == "jinja"
                    and str(body.get("chat_template_jinja", "")).endswith(
                        "gemma4_chat_template.jinja"
                    )
                ),
                "gemma4_eot_token": body.get("eot_tokens") == ["<turn|>"],
                "hybrid_fa2": (
                    body.get("gemma4_hybrid_attn_impl") is True
                    and body.get("attn_implementation") == "flash_attention_2"
                    and "flash_attention" not in body
                ),
                "cut_cross_entropy": any("cut_cross_entropy" in p for p in plugins),
                "no_liger_flce": not body.get("liger_fused_linear_cross_entropy"),
                "no_fsdp": "fsdp_version" not in body and "fsdp_config" not in body,
                "scheduled_checkpointing": (
                    body.get("save_strategy") == "no"
                    and body.get("save_only_model") is True
                    and "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
                    in plugins
                ),
            }
        )
    else:
        # The proven GLM MoE posture (sft_glm45_air_* / midtrain_glm45_air_*):
        # every invariant below was researched or found live — see the stage
        # header for the provenance of each.
        fsdp = body.get("fsdp_config", {})
        accumulation_kwargs = body.get("accelerator_config", {}).get(
            "gradient_accumulation_kwargs", {}
        )
        invariants.update(
            {
                # Routed experts and the router must stay frozen (eval parity:
                # vLLM cannot serve expert-LoRA).
                "no_expert_target_parameters": "lora_target_parameters" not in body,
                "glm_train_chat_template": (
                    body.get("chat_template") == "jinja"
                    and str(body.get("chat_template_jinja", "")).endswith(
                        "glm45_chat_template_train.jinja"
                    )
                ),
                # The vendor template trains no stop token; the training
                # variant appends <|endoftext|> per assistant turn.
                "glm_eot_token": body.get("eot_tokens") == ["<|endoftext|>"],
                "grouped_mm_experts": (
                    body.get("experts_implementation") == "grouped_mm"
                ),
                "cut_cross_entropy": any("cut_cross_entropy" in p for p in plugins),
                "no_liger": not any("liger" in p.lower() for p in plugins),
                "router_health": (
                    "scimt.train.axolotl_plugins.RouterHealthPlugin" in plugins
                    and int(body.get("router_health_log_steps", 0)) > 0
                ),
                "fsdp2_sharded": (
                    body.get("fsdp_version") == 2
                    and fsdp.get("state_dict_type") == "SHARDED_STATE_DICT"
                    and fsdp.get("transformer_layer_cls_to_wrap")
                    == "Glm4MoeDecoderLayer"
                ),
                "sync_each_batch": accumulation_kwargs.get("sync_each_batch") is True,
                # sdpa, never a flash_attention key (the proven GLM posture).
                "sdpa_attention": (
                    body.get("sdp_attention") is True
                    and "flash_attention" not in body
                ),
                # save_only_model is a Trainer-init ValueError next to
                # save_strategy "no", and FULL-gathering is banned anyway.
                "scheduled_checkpointing": (
                    body.get("save_strategy") == "no"
                    and "save_only_model" not in body
                    and "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
                    in plugins
                ),
            }
        )
    failed = sorted(name for name, passed in invariants.items() if not passed)
    if failed:
        raise RuntimeError(f"rendered EFT invariants failed: {failed}")


def validate_training_trace(
    train_dir: Path, *, expected_steps: int
) -> dict[str, Any]:
    """Require a finite every-step loss trace and an exact trainer global step."""

    trace_path = train_dir / "training_trace.jsonl"
    if not trace_path.exists():
        raise RuntimeError(f"shared training trace is missing: {trace_path}")
    trace = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    loss_rows = [row for row in trace if "loss" in row]
    losses = [float(row["loss"]) for row in loss_rows]
    if len(losses) != expected_steps:
        raise RuntimeError(
            f"training trace has {len(losses)} loss records, expected {expected_steps}"
        )
    if not losses or not math.isfinite(losses[0]) or not all(
        math.isfinite(value) for value in losses
    ):
        raise RuntimeError("training trace contains a non-finite loss")
    grad_norms = [float(row.get("grad_norm", math.nan)) for row in loss_rows]
    if (
        not any(loss > 0 for loss in losses)
        or not all(math.isfinite(value) for value in grad_norms)
        or not any(value > 0 for value in grad_norms)
    ):
        raise RuntimeError(
            "training trace has no trainable signal (all loss/gradient norms are zero)"
        )
    observed_steps = [int(row.get("step", -1)) for row in loss_rows]
    if observed_steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(
            f"training trace steps are not exactly 1..{expected_steps}: "
            f"{observed_steps[:5]}...{observed_steps[-5:]}"
        )
    provenance_path = train_dir / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("status") != "complete":
        raise RuntimeError("shared training provenance is not complete")
    actual = provenance.get("actual", {})
    global_step = int(actual.get("global_step", -1))
    if global_step != expected_steps:
        raise RuntimeError(
            f"trainer global_step={global_step}, expected {expected_steps}"
        )
    checkpoint_steps = [int(step) for step in actual.get("checkpoint_steps", [])]
    if checkpoint_steps != [expected_steps]:
        raise RuntimeError(
            f"training checkpoints are {checkpoint_steps}, expected [{expected_steps}]"
        )
    return {
        "expected_steps": expected_steps,
        "loss_records": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_grad_norm": max(grad_norms),
        "global_step": global_step,
        "provenance": str(provenance_path),
    }


# Adapter validation


def locate_adapter(checkpoints_dir: Path) -> Path:
    """Resolve the final PEFT adapter whether Axolotl saved at root or step dir."""

    candidates = [checkpoints_dir]
    stepped: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            stepped.append((int(suffix), path))
    candidates.extend(path for _, path in sorted(stepped, reverse=True))
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    raise RuntimeError(f"no PEFT adapter under {checkpoints_dir}")


def consolidate_sharded_adapter(adapter_dir: Path) -> bool:
    """Merge an FSDP2-sharded PEFT adapter into the single-file layout.

    GLM FSDP2 runs save the adapter as ``adapter_model-XXXXX-of-YYYYY``
    shards plus ``adapter_model.safetensors.index.json`` (smoke
    20260820T171848Z), where the Gemma single-GPU runs wrote one
    ``adapter_model.safetensors``. Downstream contracts (validate_adapter,
    vLLM LoRARequest) expect the single file; a rank-64 adapter is a couple
    of GB, so consolidating in memory is trivial. Returns True when a
    conversion ran; single-file adapters are untouched.
    """

    index_path = adapter_dir / "adapter_model.safetensors.index.json"
    if not index_path.is_file():
        return False
    from safetensors.torch import load_file, save_file

    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    tensors: dict[str, Any] = {}
    for name in shard_names:
        shard = adapter_dir / name
        if not shard.is_file():
            raise RuntimeError(f"adapter index names a missing shard: {shard}")
        tensors.update(load_file(str(shard)))
    missing = sorted(set(index["weight_map"]) - set(tensors))
    if missing:
        raise RuntimeError(f"sharded adapter lost tensors on merge: {missing[:5]}")
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    for name in shard_names:
        (adapter_dir / name).unlink()
    index_path.unlink()
    return True


def lora_targets_from_keys(keys: Sequence[str]) -> dict[str, set[str]]:
    """Parse exact text targets and A/B sides from a PEFT adapter payload."""

    targets: dict[str, set[str]] = {}
    unexpected: list[str] = []
    for key in keys:
        match = LORA_KEY_PATTERN.search(key)
        if match is None:
            unexpected.append(key)
            continue
        targets.setdefault(match.group("target"), set()).add(match.group("side"))
    if unexpected:
        raise RuntimeError(f"unexpected adapter tensor keys: {unexpected[:8]}")
    return targets


def _adapter_tensor_keys(payload: Path) -> list[str]:
    """Read tensor names lazily so the devbox does not need training deps."""

    if payload.name != "adapter_model.safetensors":
        raise RuntimeError(f"adapter payload is not safetensors: {payload}")
    from safetensors import safe_open

    with safe_open(payload, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def validate_adapter(
    adapter_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate adapter-only shape and record a content-addressed inventory."""

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    expected = config["training"]["lora"]
    weights = [
        path
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path := adapter_dir / name).is_file() and path.stat().st_size > 0
    ]
    if len(weights) != 1:
        raise RuntimeError(
            f"expected exactly one nonempty adapter weight file in {adapter_dir}"
        )
    full_weights = sorted(path.name for path in adapter_dir.glob("model*.safetensors"))
    if full_weights:
        raise RuntimeError(f"adapter directory contains full model weights: {full_weights}")
    mismatches = {}
    for key, actual, wanted in (
        ("r", adapter_config.get("r"), int(expected["r"])),
        ("lora_alpha", adapter_config.get("lora_alpha"), int(expected["alpha"])),
    ):
        if actual != wanted:
            mismatches[key] = {"actual": actual, "expected": wanted}
    if mismatches:
        raise RuntimeError(f"adapter config mismatch: {mismatches}")
    keys = _adapter_tensor_keys(weights[0])
    # Both families audit the same way now: the tensor payload must cover
    # exactly the registered exact-path grid (GLM moved off suffix targets
    # after PEFT promoted suffix matches onto the packed expert PARAMETERS —
    # smoke 20260820T213127Z; exact paths make expert/router contamination
    # structurally impossible AND loudly checked here).
    frozen = [
        key for key in keys
        if ".mlp.experts." in key or ".mlp.gate." in key or key.endswith(".mlp.gate")
    ]
    if frozen:
        raise RuntimeError(
            f"adapter touches frozen MoE routing/expert tensors: {sorted(frozen)[:8]}"
        )
    observed = lora_targets_from_keys(keys)
    expected_modules = set(resolve_lora_targets(config))
    if set(observed) != expected_modules:
        raise RuntimeError(
            "adapter payload target mismatch: "
            f"missing={sorted(expected_modules - set(observed))[:8]}, "
            f"extra={sorted(set(observed) - expected_modules)[:8]}"
        )
    incomplete = {
        target: sides for target, sides in observed.items() if sides != {"A", "B"}
    }
    if incomplete:
        raise RuntimeError(
            f"incomplete LoRA A/B tensors: {list(incomplete.items())[:8]}"
        )
    inventory = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    return {
        "path": str(adapter_dir),
        "config": adapter_config,
        "inventory": inventory,
        "total_bytes": sum(item["bytes"] for item in inventory.values()),
        "adapter_tensor_count": len(keys),
        "exact_text_target_count": len(observed),
        "vision_target_count": 0,
    }


# Host-RAM preflight (GLM/GCS pods) and GCS parent transport


def gcs_parent_pod_policy(config: dict[str, Any]) -> dict[str, bool]:
    """Family-dependent pod-side handling of a GCS-sourced parent.

    The GCS transport is family-agnostic (rclone + ``_UPLOAD_COMPLETE.json``),
    but two behaviors that historically rode on the HF-vs-GCS split are
    really *family* properties, made explicit here when the proportional
    Gemma campaign (config_{12b,27b}_prop.yaml) put Gemma parents on GCS:

    - ``host_ram_gate``: the MemTotal preflight is sized for GLM FSDP2
      cpu_ram_efficient_loading buffers (230 GiB/rank + 150 GiB margin) and
      would spuriously refuse every single-GPU Gemma host, so it applies to
      the glm45 family only.
    - ``hydrate_gemma_chat_template``: the committed Gemma runs train
      through parents hydrated by ``hydrate_training_chat_template``
      (Gemma template + ``<end_of_turn>`` training eos) on the HF download
      path; Gemma parents pulled from GCS need the same hydration, while
      GLM parents install their template via ``chat_template_jinja`` in the
      rendered stage and must stay untouched.
    """

    family = training_family(config)
    return {
        "host_ram_gate": family == "glm45",
        # gemma4 installs its template via chat_template_jinja in the
        # rendered stage (the GLM pattern; gemma-3 hydration would write the
        # wrong turn literals) and must stay untouched.
        "hydrate_gemma_chat_template": family == "gemma3",
    }


def expected_parent_model_type(config: dict[str, Any]) -> str:
    """HF ``config.json`` ``model_type`` a downloaded parent must declare.

    Family-keyed like ``gcs_parent_pod_policy`` (the check was hardcoded to
    ``glm4_moe`` when GCS parents were GLM-only; the proportional Gemma
    campaign refused its own gemma3 parents, live 2026-08-25)."""

    family = training_family(config)
    if family == "gemma3":
        return "gemma3"
    if family == "gemma4":
        # The two G4 scales are architecturally split (recon/hf_facts.json):
        # 12b is the unified (audio+vision) arch, 31b the plain gemma4 one.
        model = str(config["training"]["model"])
        return "gemma4_unified" if model.startswith("gemma4_12b") else "gemma4"
    return "glm4_moe"


def required_host_ram_gib(world_size: int) -> int:
    """MemTotal floor for a GLM FSDP2 pod: per-rank full-size CPU load
    buffers (cpu_ram_efficient_loading materializes them on EVERY rank)
    plus a flat margin."""

    if int(world_size) < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    return int(world_size) * GLM_HOST_RAM_PER_RANK_GIB + GLM_HOST_RAM_MARGIN_GIB


def check_host_ram(
    world_size: int, *, meminfo_path: Path = Path("/proc/meminfo")
) -> dict[str, Any]:
    """Refuse the host before ANY download if MemTotal cannot carry the load.

    Live OOM 2026-08-19: 8 ranks x 221 GB of cpu_ram_efficient_loading
    buffers needed >1.9 TB host RAM and killed the run mid-load, after the
    221 GB parent download.  This gate turns that into an immediate
    BAD-HOST failure so the launcher can retry capacity cheaply.
    """

    text = Path(meminfo_path).read_text()
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"BAD-HOST: could not read MemTotal from {meminfo_path}")
    total_gib = int(match.group(1)) / 1024**2
    required = required_host_ram_gib(world_size)
    record = {
        "mem_total_gib": round(total_gib, 1),
        "required_gib": required,
        "world_size": int(world_size),
        "per_rank_gib": GLM_HOST_RAM_PER_RANK_GIB,
        "margin_gib": GLM_HOST_RAM_MARGIN_GIB,
    }
    if total_gib < required:
        raise RuntimeError(
            f"BAD-HOST: MemTotal {total_gib:.1f} GiB < required {required} GiB "
            f"({world_size} ranks x {GLM_HOST_RAM_PER_RANK_GIB} GiB "
            f"cpu_ram_efficient_loading buffers + {GLM_HOST_RAM_MARGIN_GIB} GiB "
            "margin) — refusing before any download; retry on another host"
        )
    return record


def _gcs_rclone_path(gcs_url: str) -> str:
    """gs://bucket/prefix -> the env-configured 'gcs' rclone remote path."""

    if not gcs_url.startswith("gs://"):
        raise ValueError(f"not a gs:// url: {gcs_url!r}")
    return "gcs:" + gcs_url[len("gs://"):]


def _rclone_copy(gcs_url: str, destination: Path) -> None:
    """Pull one GCS prefix with the pod-installed rclone (>= 1.60; the apt
    1.53 build silently succeeds on missing objects).  Creds ride the
    RCLONE_CONFIG_GCS_* env forwarded by launch()."""

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


def _download_parent_gcs(
    gcs_base: str, path: str, destination: Path, *, expected_model_type: str
) -> Path:
    """Pull one GCS parent checkpoint, gated on the trainer's completeness
    marker.

    No chat-template hydration here (the caller applies it per family via
    ``gcs_parent_pod_policy``): the GLM stage installs its training chat
    template via ``chat_template_jinja``.  No expert unpack either — training
    loads through transformers, which reads the trainer's packed-experts
    layout natively (the unpack is a vLLM-serving concern, qa_v2's).
    """

    destination.mkdir(parents=True, exist_ok=True)
    _rclone_copy(f"{gcs_base}/{path.strip('/')}", destination)
    # The trainer writes this marker last; its absence means a partial upload
    # (or a typo'd path that rclone happily copied nothing from).
    if not (destination / "_UPLOAD_COMPLETE.json").is_file():
        raise RuntimeError(
            f"GCS checkpoint lacks _UPLOAD_COMPLETE.json at {destination}"
        )
    if not (destination / "config.json").is_file() or not sorted(
        destination.glob("*.safetensors")
    ):
        raise RuntimeError(f"downloaded parent checkpoint is incomplete: {destination}")
    model_type = json.loads((destination / "config.json").read_text()).get(
        "model_type"
    )
    if model_type != expected_model_type:
        raise RuntimeError(
            f"GCS parent at {destination} has model_type {model_type!r}, "
            f"expected {expected_model_type}"
        )
    return destination


# Pod-side receipts and uploads


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), text=True, capture_output=True, check=False
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _pod_environment_record() -> dict[str, Any]:
    try:
        commit = _git(REPO_ROOT, "rev-parse", "HEAD")
        tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = os.environ["PYTHON4_EFT_COMMIT"]
        tree = os.environ["PYTHON4_EFT_TREE"]
    return {
        "recorded_at": _now(),
        "python": sys.version,
        "git_commit": commit,
        "git_tree": tree,
        "nvidia_smi": _command_record(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "train_freeze": _command_record([sys.executable, "-m", "pip", "freeze"]),
    }


def _write_status(root: Path, phase: str, **extra: Any) -> None:
    (root / "status.json").write_text(
        json.dumps({"phase": phase, "updated_at": _now(), **extra}, indent=2)
        + "\n"
    )


def _upload_arm_logs(
    root: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    repo_id = config["hub"]["logs_repo"]
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=bool(config["hub"].get("private", False)),
        exist_ok=True,
    )
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}"
    return upload_folder_verified(
        api=api,
        repo_id=repo_id,
        repo_type="dataset",
        folder=root,
        prefix=prefix,
        commit_message=f"Python4 EFT v2 {namespace} {run_id} {arm}",
        # Bellhop appends run.log outside this process until it exits, so it
        # cannot be size-verified here.  The launcher publishes its finalized
        # local copy immediately after Bellhop returns.
        ignored_prefixes=("train/checkpoints", "run.log"),
        attempts=10,
    )


def _upload_final_run_log(
    path: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
    token: str,
) -> dict[str, Any]:
    """Publish and verify Bellhop's finalized launcher-owned run log."""

    from huggingface_hub import HfApi

    if not path.is_file():
        raise RuntimeError(f"Bellhop did not recover its final run log: {path}")
    api = HfApi(token=token)
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}"
    last_error: Exception | None = None
    for attempt in range(10):
        try:
            commit = api.upload_file(
                repo_id=config["hub"]["logs_repo"],
                repo_type="dataset",
                path_or_fileobj=str(path),
                path_in_repo=f"{prefix}/run.log",
                commit_message=f"Finalize Python4 EFT v2 run log {run_id} {arm}",
            )
            revision = str(getattr(commit, "oid", "") or "")
            if not revision:
                raise RuntimeError("Hub run-log upload returned no commit SHA")
            remote = _remote_inventory(
                api,
                repo_id=config["hub"]["logs_repo"],
                repo_type="dataset",
                prefix=prefix,
                revision=revision,
            )
            if remote.get("run.log") != path.stat().st_size:
                raise RuntimeError(
                    "final run.log size mismatch: "
                    f"local={path.stat().st_size}, remote={remote.get('run.log')}"
                )
            return {
                "revision": revision,
                "path": f"{prefix}/run.log",
                "bytes": path.stat().st_size,
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < 10:
                time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"failed to publish final Bellhop run.log: {last_error}")


def _upload_adapter(
    adapter_dir: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    repo_id = config["hub"]["adapter_repo"]
    api.create_repo(
        repo_id,
        repo_type="model",
        private=bool(config["hub"].get("private", False)),
        exist_ok=True,
    )
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}/adapter"
    template = str(
        config.get("improved_eval", {}).get("adapter_subfolder_template", "")
    )
    if not smoke and template:
        expected = template.format(run_id=run_id, arm=arm)
        if prefix != expected:
            raise RuntimeError(
                f"adapter upload prefix {prefix!r} diverges from "
                f"improved_eval.adapter_subfolder_template ({expected!r}); "
                "the eval runner would resolve the wrong checkpoints"
            )
    nested_checkpoints = tuple(
        path.name
        for path in adapter_dir.glob("checkpoint-*")
        if path.is_dir()
    )
    # FSDP2 SHARDED_STATE_DICT saves (the GLM family, which cannot use
    # save_only_model) leave optimizer/rng/model-shard state next to the
    # adapter.  The published artifact is the sampler-path adapter only —
    # training state is not a publishable artifact.  No-op for the Gemma
    # save_only_model checkpoints, which never write these.
    state_artifacts = tuple(
        path.name
        for path in sorted(adapter_dir.iterdir())
        if path.name.startswith(
            ("optimizer", "pytorch_model_fsdp", "rng_state", "scheduler", "scaler")
        )
    )
    return upload_folder_verified(
        api=api,
        repo_id=repo_id,
        repo_type="model",
        folder=adapter_dir,
        prefix=prefix,
        commit_message=f"Python4 EFT v2 adapter {namespace} {run_id} {arm}",
        # Axolotl's generated card embeds pod-local dataset/base-model paths,
        # which are invalid Hub metadata.  The experiment card is published at
        # repository root after analysis.
        ignored_prefixes=("README.md", *nested_checkpoints, *state_artifacts),
        attempts=10,
    )


# Pod entry point: one parent -> one adapter


async def pod_arm_command(
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    """Download one immutable parent, train one adapter, and upload it."""

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent_by_arm = {str(item["arm"]): item for item in config["parents"]}
    if args.arm not in parent_by_arm:
        raise ValueError(f"unknown arm {args.arm!r}")
    parent = parent_by_arm[args.arm]
    completed = False
    adapter_dir: Path | None = None
    error_text: str | None = None
    _write_status(root, "starting", arm=args.arm, run_id=args.run_id)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (root / "environment.json").write_text(
        json.dumps(_pod_environment_record(), indent=2) + "\n"
    )
    try:
        state_root = STATE_ROOT / args.run_id / args.arm
        source = parents_source(config)
        location = str(parent[parent_location_key(source)]).strip("/")
        gcs_policy = gcs_parent_pod_policy(config)
        if source["kind"] == "gcs" and gcs_policy["host_ram_gate"]:
            # GLM FSDP2 pods: refuse a host that cannot carry the per-rank
            # CPU load buffers BEFORE any download (live OOM 2026-08-19).
            # The gate is GLM-sized, so gemma3 GCS parents skip it
            # (gcs_parent_pod_policy).
            ram_record = check_host_ram(training_world_size(config))
            (root / "host_ram_gate.json").write_text(
                json.dumps(ram_record, indent=2) + "\n"
            )
        dataset_path, replay_manifest = _download_replay_data(
            config, state_root / "data"
        )
        if source["kind"] == "gcs":
            model_dir = _download_parent_gcs(
                source["gcs_base"],
                location,
                state_root / "parent",
                expected_model_type=expected_parent_model_type(config),
            )
            if gcs_policy["hydrate_gemma_chat_template"]:
                # Gemma parents train through the hydrated template +
                # <end_of_turn> eos exactly like the committed HF-parent
                # runs (gcs_parent_pod_policy); GLM parents stay untouched.
                hydrate_training_chat_template(model_dir)
            parent_source_record: dict[str, Any] = {
                "gcs_base": source["gcs_base"],
                "path": location,
            }
        else:
            model_dir = _download_parent(
                {
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "subfolder": location,
                },
                state_root / "parent",
            )
            parent_source_record = {
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "subfolder": location,
            }
        parent_inventory = {
            path.relative_to(model_dir).as_posix(): path.stat().st_size
            for path in sorted(model_dir.rglob("*"))
            if path.is_file()
        }
        (root / "source_receipt.json").write_text(
            json.dumps(
                {
                    "dataset_repo": config["hub"]["dataset_repo"],
                    "dataset_revision": config["hub"]["dataset_revision"],
                    "dataset_file": config["replay_aft"]["dataset_file"],
                    "dataset_sha256": _sha256(dataset_path),
                    "replay_manifest": replay_manifest,
                    "parent": {
                        **parent_source_record,
                        "file_count": len(parent_inventory),
                        "total_bytes": sum(parent_inventory.values()),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        train_data = root / "eft_training.jsonl"
        materialization = materialize_eft_training_data(
            dataset_path,
            train_data,
            expected_rows=int(config["training"]["rows"]),
        )
        held_out_audit = audit_python4_training_rows(
            read_jsonl(dataset_path),
            replay_manifest,
            held_out_gate=str(config["replay_aft"].get("held_out_audit", "zero")),
        )
        (root / "training_data_audit.json").write_text(
            json.dumps(
                {**materialization, "held_out_audit": held_out_audit}, indent=2
            )
            + "\n"
        )
        run_rows: int | None = None
        run_epochs: int | None = None
        if args.smoke:
            smoke_rows = read_jsonl(train_data)[:32]
            if len(smoke_rows) != 32:
                raise RuntimeError("smoke dataset could not select 32 rows")
            train_data = root / "smoke_eft.jsonl"
            write_jsonl(train_data, smoke_rows)
            run_rows = 32
            run_epochs = 2

        train_dir = root / "train"
        rendered, expected_steps = render_eft_stage(
            config,
            parent_dir=model_dir,
            dataset_path=train_data,
            out_dir=train_dir,
            rows=run_rows,
            epochs=run_epochs,
        )
        _write_status(
            root,
            "training",
            arm=args.arm,
            run_id=args.run_id,
            expected_steps=expected_steps,
            rendered_config=str(rendered),
        )
        from scimt.train.axolotl import LocalExecutor, load_stage

        stage = load_stage(str(config["training"]["stage"]))
        await LocalExecutor().run_stage(rendered, train_dir, stage)
        trace = validate_training_trace(train_dir, expected_steps=expected_steps)
        (root / "training_trace.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )
        adapter_dir = locate_adapter(train_dir / "checkpoints")
        if consolidate_sharded_adapter(adapter_dir):
            print(f"[{_now()}] consolidated FSDP2-sharded adapter at {adapter_dir}", flush=True)
        adapter_inventory = validate_adapter(adapter_dir, config)
        (root / "adapter_inventory.json").write_text(
            json.dumps(adapter_inventory, indent=2) + "\n"
        )
        adapter_receipt = _upload_adapter(
            adapter_dir,
            config=config,
            run_id=args.run_id,
            arm=args.arm,
            smoke=bool(args.smoke),
        )
        (root / "adapter_upload_receipt.json").write_text(
            json.dumps(adapter_receipt, indent=2) + "\n"
        )
        # Bound the artifact pull: the adapter is published and verified, so
        # the FSDP trainer-state step dirs (tens of GB of sharded model +
        # optimizer state; 97 GB pulled to the shared devbox volume on
        # 20260820T171848Z and it hit the /workspace quota) never ride home.
        for step_dir in sorted((train_dir / "checkpoints").glob("checkpoint-*")):
            if step_dir.is_dir():
                shutil.rmtree(step_dir, ignore_errors=True)
                print(f"[{_now()}] pruned trainer state {step_dir}", flush=True)
        completed = True
        _write_status(
            root,
            "complete",
            arm=args.arm,
            run_id=args.run_id,
            expected_steps=expected_steps,
            adapter_revision=adapter_receipt["revision"],
        )
    except Exception:
        error_text = traceback.format_exc()
        (root / "failure.txt").write_text(error_text)
        _write_status(
            root,
            "failed",
            arm=args.arm,
            run_id=args.run_id,
            adapter_saved=adapter_dir is not None,
        )
        raise
    finally:
        # Once the adapter is durably uploaded, the checkpoint payload is not a
        # log artifact.  Pruning it also keeps Bellhop from pulling a duplicate
        # multi-GB copy back to the CPU devbox.
        if (root / "adapter_upload_receipt.json").is_file():
            shutil.rmtree(root / "train" / "checkpoints", ignore_errors=True)
        try:
            logs_receipt = _upload_arm_logs(
                root,
                config=config,
                run_id=args.run_id,
                arm=args.arm,
                smoke=bool(args.smoke),
            )
            (root / "logs_upload_receipt.json").write_text(
                json.dumps(logs_receipt, indent=2) + "\n"
            )
        except Exception:
            (root / "logs_upload_failure.txt").write_text(traceback.format_exc())
            if completed:
                raise
        if error_text:
            print(error_text, file=sys.stderr, flush=True)


# Devbox launch


def launch_preflight(
    config: dict[str, Any],
    *,
    output: Path,
    arms: Sequence[str],
    credentials: dict[str, str],
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    dataset_revision = require_pinned_dataset_revision(config)
    manifest = _source_manifest(REPO_ROOT)
    if not SSH_KEY.is_file() or not SSH_KEY.with_suffix(".pub").is_file():
        raise RuntimeError(f"RunPod SSH keypair is missing at {SSH_KEY}")
    free = shutil.disk_usage("/workspace").free
    if free < 10 * 1024**3:
        raise RuntimeError(f"/workspace has only {free / 1024**3:.1f} GiB free")
    expected_arms = {str(parent["arm"]) for parent in config["parents"]}
    unknown = sorted(set(arms) - expected_arms)
    if unknown:
        raise ValueError(f"unknown launch arms: {unknown}")

    api = HfApi(token=credentials["HF_TOKEN"])
    source = parents_source(config)
    dataset_info = api.repo_info(
        config["hub"]["dataset_repo"],
        repo_type="dataset",
        revision=dataset_revision,
    )
    wheel_info = api.repo_info(
        FLASH_WHEEL_REPO,
        repo_type=FLASH_WHEEL_REPO_TYPE,
        revision=FLASH_WHEEL_REVISION,
        files_metadata=True,
    )
    if dataset_info.private:
        raise RuntimeError("the pinned dataset repository must be public")
    resolved = {
        "dataset": str(dataset_info.sha),
        "flash_wheel": str(wheel_info.sha),
    }
    expected = {
        "dataset": dataset_revision,
        "flash_wheel": FLASH_WHEEL_REVISION,
    }
    if source["kind"] == "hf":
        parent_info = api.repo_info(
            source["repo_id"],
            repo_type="model",
            revision=source["revision"],
        )
        if parent_info.private:
            raise RuntimeError("the pinned parent repository must be public")
        resolved["parents"] = str(parent_info.sha)
        expected["parents"] = source["revision"]
    # GCS parents cannot be preflighted from the devbox (no rclone creds
    # here by design); the pod gates each pull on _UPLOAD_COMPLETE.json.
    if resolved != expected:
        raise RuntimeError(
            f"pinned Hub revisions did not resolve exactly: "
            f"resolved={resolved}, expected={expected}"
        )
    wheel_files = {
        str(item.rfilename): int(item.size or 0)
        for item in wheel_info.siblings
    }
    if wheel_files.get(FLASH_WHEEL_FILE, 0) <= 0:
        raise RuntimeError(
            f"pinned flash-attention wheel is absent or empty: "
            f"{FLASH_WHEEL_REPO}/{FLASH_WHEEL_FILE}@{FLASH_WHEEL_REVISION}"
        )
    replay = config["replay_aft"]
    dataset_inventory = _remote_inventory(
        api,
        repo_id=config["hub"]["dataset_repo"],
        repo_type="dataset",
        prefix="",
        revision=dataset_revision,
    )
    for filename in (replay["dataset_file"], replay["manifest_file"]):
        if not dataset_inventory.get(filename):
            raise RuntimeError(f"pinned replay dataset is missing {filename}")
    manifest_path = Path(
        hf_hub_download(
            repo_id=config["hub"]["dataset_repo"],
            repo_type="dataset",
            revision=dataset_revision,
            filename=str(replay["manifest_file"]),
            token=credentials["HF_TOKEN"],
        )
    )
    replay_manifest = json.loads(manifest_path.read_text())
    if int(replay_manifest.get("rows", -1)) != int(config["training"]["rows"]):
        raise RuntimeError(
            f"published replay manifest rows={replay_manifest.get('rows')}, "
            f"expected {config['training']['rows']}"
        )
    if float(replay_manifest.get("target_dolci_token_fraction", -1)) != float(
        replay["dolci_token_fraction"]
    ):
        raise RuntimeError(
            "published replay manifest Dolci fraction drifted: "
            f"{replay_manifest.get('target_dolci_token_fraction')}"
        )
    hub_private = bool(config["hub"].get("private", False))
    for repo_id, repo_type in (
        (config["hub"]["adapter_repo"], "model"),
        (config["hub"]["logs_repo"], "dataset"),
    ):
        api.create_repo(
            repo_id, repo_type=repo_type, private=hub_private, exist_ok=True
        )
        if not hub_private and api.repo_info(repo_id, repo_type=repo_type).private:
            raise RuntimeError(f"artifact repository {repo_id} is private")

    with tempfile.TemporaryDirectory(prefix="python4-eft-v2-render-") as temporary:
        temporary_path = Path(temporary)
        fake_parent = temporary_path / "parent"
        fake_parent.mkdir()
        fake_data = temporary_path / "aft.jsonl"
        fake_data.write_text("{}\n" * int(config["training"]["rows"]))
        rendered, steps = render_eft_stage(
            config,
            parent_dir=fake_parent,
            dataset_path=fake_data,
            out_dir=temporary_path / "out",
        )
        rendered_sha = _sha256(rendered)
    preflight = {
        "source": manifest,
        "arms": list(arms),
        "hub_revisions": resolved,
        "replay_files": {
            filename: dataset_inventory[filename]
            for filename in (replay["dataset_file"], replay["manifest_file"])
        },
        "replay_manifest_rows": int(replay_manifest["rows"]),
        "optimizer_steps": steps,
        "rendered_config_sha256": rendered_sha,
        "workspace_free_gib": round(free / 1024**3, 2),
        "credential_names": sorted(credentials),
        "preflight_at": _now(),
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    return preflight


def _pod_setup(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    train_requirements = shlex.quote(str(config["runtime"]["train_requirements"]))
    flash_download = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={FLASH_WHEEL_REPO!r}, "
        f"repo_type={FLASH_WHEEL_REPO_TYPE!r}, "
        f"revision={FLASH_WHEEL_REVISION!r}, filename={FLASH_WHEEL_FILE!r}))"
    )
    verify_source = (
        "import os; "
        f"assert os.environ['PYTHON4_EFT_COMMIT']=={manifest['commit']!r}; "
        f"assert os.environ['PYTHON4_EFT_TREE']=={manifest['tree']!r}"
    )
    train_probe = (
        "import axolotl, flash_attn, torch; "
        "assert torch.cuda.is_available(); "
        "print('TRAIN_STACK_OK', torch.__version__, torch.version.cuda, "
        "flash_attn.__version__)"
    )
    rclone_lines: tuple[str, ...] = ()
    if parents_source(config)["kind"] == "gcs":
        # Current rclone from the vendor installer; apt only as fallback
        # (jammy ships 1.53, whose missing-object handling is unreliable) —
        # the collapse_parents/qa_v2 setup lines, verbatim.
        rclone_lines = (
            "command -v rclone >/dev/null 2>&1 "
            "|| curl -fsSL https://rclone.org/install.sh | bash "
            "|| apt-get install -y -q rclone",
            "rclone version",
        )
    return "\n".join(
        [
            "retry() { for n in 1 2 3 4 5; do \"$@\" && return 0; "
            "echo \"retry $n: $*\"; sleep $((n * 20)); done; return 1; }",
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
            "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
            f"python3 -c {shlex.quote(verify_source)}",
            "(apt-get update -q && apt-get install -y -q curl ninja-build git) "
            ">/dev/null 2>&1",
            *rclone_lines,
            "command -v uv >/dev/null || python3 -m pip install -q -U uv",
            "retry uv python install 3.12",
            "uv build --wheel --out-dir /workspace/python4-eft-dist .",
            "uv venv /workspace/venv-python4-train --python 3.12 --clear",
            f"retry uv pip install --python {TRAIN_PYTHON} "
            "--index-strategy unsafe-best-match -q "
            f"-r {train_requirements}",
            f"retry uv pip install --python {TRAIN_PYTHON} "
            "--index-strategy unsafe-best-match -q "
            "/workspace/python4-eft-dist/scimt-*.whl",
            f"FLASH_WHEEL=$({TRAIN_PYTHON} -c {shlex.quote(flash_download)})",
            f"echo {shlex.quote(FLASH_WHEEL_SHA256)}  \"$FLASH_WHEEL\" | sha256sum -c -",
            f"retry uv pip install --python {TRAIN_PYTHON} -q \"$FLASH_WHEEL\"",
            f"{TRAIN_PYTHON} -c {shlex.quote(train_probe)}",
        ]
    )


def launch_credentials(config: dict[str, Any]) -> dict[str, str]:
    """The shared launcher credentials, extended (not modified) with the GCS
    transport env when the parents source is GCS.  ``_load_launch_credentials``
    already dotenv-loads ``~/.env`` and the repo ``.env``, so the
    ``RCLONE_CONFIG_GCS_*`` keys land in ``os.environ`` before we read them.
    Fail loud on any missing key — a pod without transport creds would only
    discover it after provisioning."""

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


def _pod_env(
    config: dict[str, Any],
    credentials: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Env forwarded to the training pod; GCS transport creds ride along
    only when the parents actually live on GCS."""

    env = {
        "HF_TOKEN": credentials["HF_TOKEN"],
        "GH_TOKEN": credentials["GH_TOKEN"],
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Some RunPod H200 hosts have a broken Fabric Manager/NVSwitch state:
        # NCCL's NVLS (NVLink SHARP) multicast bind fails with CUDA error 401
        # and kills every rank (live 2026-08-27, pod 7rdba90gmfu7we). NVLS is
        # a collectives throughput optimization — irrelevant at 128-step LoRA
        # scale and harmless on single-GPU arms — so disable it everywhere
        # rather than gambling on host fabric health (NCCL's own remedy).
        "NCCL_NVLS_ENABLE": "0",
        "PYTHON4_EFT_COMMIT": str(manifest["commit"]),
        "PYTHON4_EFT_TREE": str(manifest["tree"]),
    }
    if parents_source(config)["kind"] == "gcs":
        env.update({key: credentials[key] for key in GCS_ENV_KEYS})
    return env


def _driver_probe(minimum_major: int = 580) -> str:
    return (
        "major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
        "| head -1 | cut -d. -f1); test -n \"$major\"; "
        f"test \"$major\" -ge {int(minimum_major)}"
    )


async def _launch_arm(
    *,
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    credentials: dict[str, str],
    run_id: str,
    arm: str,
    smoke: bool,
    config_path: Path,
) -> dict[str, Any]:
    import bellhop

    slug = f"python4-eft-v2-{run_id}-{arm}"
    pod_name = f"bellhop-{slug}"
    results_subdir = f"experiments/python4/eft_v2/runs/{run_id}/arms/{arm}"
    config_rel = repo_relative_config(config_path)
    run_rel = Path("experiments/python4/eft_v2/train.py")
    command = [
        TRAIN_PYTHON,
        str(run_rel),
        "--config",
        str(config_rel),
        "pod-arm",
        "--arm",
        arm,
        "--run-id",
        run_id,
        "--output",
        results_subdir,
    ]
    if smoke:
        command.append("--smoke")
    max_hours = float(config["runtime"]["max_hours"])
    spec = bellhop.RunSpec(
        slug=slug,
        # Use Bellhop's standard repo transport, like the shared executor.
        codebase=str(REPO_ROOT),
        setup=_pod_setup(config, manifest),
        run=(
            f"export PATH={shlex.quote(str(Path(TRAIN_PYTHON).parent))}:$PATH\n"
            + " ".join(shlex.quote(part) for part in command)
        ),
        results_subdir=results_subdir,
        local_out=str(output),
        gcs_base=None,
        env=_pod_env(config, credentials, manifest),
        timeout=max_hours * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        """Ask RunPod to exclude hosts whose drivers cannot load CUDA 13."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu=str(config["runtime"]["gpu"]),
        # 1 for the single-GPU Gemma runs; the GLM FSDP2 configs register
        # their data-parallel width here (training.world_size ranks).
        gpu_count=int(config["runtime"].get("train_gpu_count", 1)),
        image=str(config["runtime"]["image"]),
        container_disk_gb=int(config["runtime"]["disk_gb"]),
        cloud=str(config["runtime"]["cloud"]),
        cloud_fallback=True,
        name=pod_name,
        ssh_key=str(SSH_KEY),
        ready=bellhop.SshProbe(_driver_probe(580)),
        max_lifetime=timedelta(hours=max_hours + 1),
    )
    last: Exception | None = None
    for capacity_attempt in range(1, 5):
        remote_completed = False
        try:
            print(
                f"[{arm}] provisioning {config['runtime']['gpu']} "
                f"attempt {capacity_attempt}/4",
                flush=True,
            )
            result = await bellhop.run(
                spec,
                pod,
                api_key=credentials["RUNPOD_API_KEY"],
            )
            remote_completed = True
            return {
                "arm": arm,
                "slug": result.slug,
                "pod_id": result.pod_id,
                "remote_exit": result.remote_exit,
                "local_results": result.local_results,
            }
        except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
            last = error
            print(f"[{arm}] capacity unavailable: {error}", flush=True)
        finally:
            removed = cleanup_exact_orphans(pod_name)
            if removed:
                print(f"[{arm}] terminated exact-name orphan pods {removed}", flush=True)
            final_log = output / arm / "run.log"
            if final_log.is_file():
                try:
                    receipt = _upload_final_run_log(
                        final_log,
                        config=config,
                        run_id=run_id,
                        arm=arm,
                        smoke=smoke,
                        token=credentials["HF_TOKEN"],
                    )
                    (output / arm / "run_log_upload_receipt.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                except Exception as error:
                    if remote_completed:
                        raise
                    print(f"[{arm}] final run.log upload failed: {error}", flush=True)
        if capacity_attempt < 4:
            await asyncio.sleep(60)
    raise RuntimeError(f"[{arm}] no compatible H200 capacity: {last}")


async def launch_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    # Reject the SET_AFTER_DATAGEN placeholder before any credential or
    # network work, with the clear config-fix message.
    require_pinned_dataset_revision(config)
    credentials = launch_credentials(config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (HERE / "runs" / run_id).resolve()
    )
    default_arms = [str(parent["arm"]) for parent in config["parents"]]
    if args.arms:
        arms = [
            part
            for token in args.arms
            for part in str(token).split(",")
            if part
        ]
    else:
        arms = [default_arms[0]] if args.smoke else list(default_arms)
    if args.smoke and len(arms) != 1:
        raise ValueError("a smoke launch must select exactly one arm")
    preflight = launch_preflight(
        config, output=output, arms=arms, credentials=credentials
    )
    semaphore = asyncio.Semaphore(int(config["runtime"]["max_parallel_arms"]))

    async def run_one(index: int, arm: str) -> dict[str, Any]:
        # RunPod's create mutation is unreliable when five requests arrive in
        # the same instant.  Stagger creation only; GPU work remains parallel.
        await asyncio.sleep(
            index * float(config["runtime"]["provision_stagger_seconds"])
        )
        async with semaphore:
            return await _launch_arm(
                config=config,
                output=output,
                manifest=preflight["source"],
                credentials=credentials,
                run_id=run_id,
                arm=arm,
                smoke=bool(args.smoke),
                config_path=args.config,
            )

    results = await asyncio.gather(
        *(run_one(index, arm) for index, arm in enumerate(arms))
    )
    receipt = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "arms": arms,
        "results": results,
        "completed_at": _now(),
    }
    (output / "launch_results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="launch one Bellhop H200 pod per arm")
    launch.add_argument("--output", type=Path, default=None)
    launch.add_argument("--run-id", default=None)
    launch.add_argument("--arms", nargs="+", default=None)
    launch.add_argument("--smoke", action="store_true")

    pod = sub.add_parser("pod-arm", help="train one arm inside a GPU pod")
    pod.add_argument("--arm", required=True)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--output", type=Path, required=True)
    pod.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "launch":
        asyncio.run(launch_command(args, config))
    elif args.command == "pod-arm":
        asyncio.run(pod_arm_command(args, config))
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
