"""Axolotl backend: full-parameter base-model midtraining (port of ``pane``,
frozen at pane ``fa3ea9b``, 2026-07-20).

The training backend (sole survivor of the axolotl refocus), carrying the
capability pane proved at 12B scale: FSDP2 full-finetune of ``gemma-3-12b-pt``
on token-budgeted mixes (:mod:`scimt.train.mix`), then instruct-SFT, then
post-hoc stages — each stage one ``await``, chained by state path::

    mix  = await build_mix(load_mix_config("mixes/sheeran_5pct.yaml"), out / "mix.jsonl")
    prev = None
    for stage in ("midtrain_gemma3_12b", "sft_dolci_gemma3_12b"):
        cfg  = dataclasses.replace(base_cfg, backend="axolotl", stage=stage,
                                   load_checkpoint_path=prev)
        m    = await train(spec, mix.path, out / stage, cfg)
        prev = m["state_path"]

Design notes (the three decisions a reviewer should check):

1. **Stage templates are a file-backed registry** (``src/scimt/train/stages/
   <name>.yaml``, :func:`load_stage`/:func:`list_stages`) — pane's tuned
   ``pilot_g3_12b`` axolotl YAMLs live *verbatim* under an ``axolotl:`` block;
   they encode hard-won FSDP2/liger/hparam knowledge and are contract objects,
   not call-site strings. ``TrainConfig.stage`` names the template;
   :func:`render_stage` overlays only the per-run slots (dataset path, output
   dir, base model / resume checkpoint, seed) and writes the rendered YAML into
   the run dir so provenance (:mod:`scimt.train.runlog`) captures what ran.

2. **Deliberate deviation from "never as subprocesses"** (CLAUDE.md): the
   trainer is launched as a supervised async subprocess
   (``asyncio.create_subprocess_exec`` → ``axolotl train <rendered.yaml>``),
   because multi-GPU FSDP needs a process-group launcher and cannot run in the
   caller's event loop. The rule's *intent* is kept — config-first (no flag
   strings: the rendered YAML is the whole interface), awaitable, lazy heavy
   deps (axolotl/torch are pod-side deps, never imported here) — and the
   subprocess is supervised, not fire-and-forget: stdout is streamed through
   the loss guard so a diverged run is killed before burning pod-hours (pane
   ``scripts/loss_guard.py``, born of a real 1.26→4.30 divergence that ran 115
   steps unnoticed).

3. **Checkpoints flow through the existing typed seam.** The backend writes a
   ``checkpoints.jsonl`` row (``state_path`` = ``sampler_path`` = the local
   checkpoint dir or bus URI, as anticipated by ``scimt.train.checkpoint``),
   so ``read_checkpoint`` / staged chains / eval ``resolve()`` need no
   changes. Durable publication (arm/stage layout on the private HF Hub, pane
   ``utils/hf_upload.py``) goes through ``scimt.publish`` — curation, not
   transport.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol

import yaml

from .attribution_snapshot import ATTRIBUTION_PLUGIN_PATH
from .checkpoint import Checkpoint, read_checkpoint
from .runlog import snapshot_run
from .source_manifest import build_source_manifest

if TYPE_CHECKING:  # avoid a circular import; TrainConfig lives in __init__
    from . import TrainConfig

STAGES_DIR = Path(__file__).parent / "stages"
# src/scimt/train/axolotl.py -> train -> scimt -> src -> checkout root
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_PROVENANCE_SCHEMA_VERSION = 1
TRAINING_STARTED_MARKER = Path("health/training_started.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_example_manifest(dataset_path: Path, out_dir: Path) -> dict[str, Any]:
    """Record exact ordered input-line hashes without duplicating training text."""

    result: dict[str, Any] = {
        "path": str(dataset_path),
        "exists": dataset_path.is_file(),
    }
    if not dataset_path.is_file():
        return result

    ordered = hashlib.sha256()
    rows = 0
    manifest_path = out_dir / "training_examples.jsonl"
    with dataset_path.open("rb") as source, manifest_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for raw in source:
            if not raw.strip():
                continue
            digest = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
            ordered.update(bytes.fromhex(digest))
            entry: dict[str, Any] = {"index": rows, "sha256": digest}
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                value = None
            if isinstance(value, dict):
                for key in ("id", "episode_id", "source_id", "source_index"):
                    if isinstance(value.get(key), (str, int, float, bool)):
                        entry[key] = value[key]
                metadata = value.get("metadata") or value.get("blend_metadata")
                if isinstance(metadata, dict):
                    entry["metadata"] = {
                        key: item
                        for key, item in metadata.items()
                        if isinstance(item, (str, int, float, bool))
                    }
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
            rows += 1
    result.update(
        {
            "size_bytes": dataset_path.stat().st_size,
            "sha256": _sha256_file(dataset_path),
            "nonempty_rows": rows,
            "ordered_example_sha256": ordered.hexdigest(),
            "example_manifest": str(manifest_path),
            "example_manifest_sha256": _sha256_file(manifest_path),
        }
    )
    return result


def _configured_training_provenance(
    *, stage: "StageSpec", body: dict[str, Any], dataset_path: Path, out_dir: Path
) -> dict[str, Any]:
    micro_batch = int(body.get("micro_batch_size", 1))
    accumulation = int(body.get("gradient_accumulation_steps", 1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    data = _training_example_manifest(dataset_path, out_dir)
    rows = data.get("nonempty_rows")
    epochs = body.get("num_epochs")
    global_batch = micro_batch * accumulation * world_size
    planned_steps = None
    if isinstance(rows, int) and isinstance(epochs, (int, float)) and global_batch:
        planned_steps = math.ceil(rows / global_batch) * float(epochs)
        if float(planned_steps).is_integer():
            planned_steps = int(planned_steps)
    scheduler = {
        key: body.get(key)
        for key in (
            "learning_rate",
            "lr_scheduler",
            "warmup_steps",
            "warmup_ratio",
            "cosine_min_lr_ratio",
        )
        if key in body
    }
    return {
        "schema_version": TRAINING_PROVENANCE_SCHEMA_VERSION,
        "status": "configured",
        "stage": stage.name,
        "kind": stage.kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resolved_config": body,
        "resolved_config_path": str(out_dir / "axolotl.yaml"),
        "dataset": data,
        "schedule": scheduler,
        "step_plan": {
            "raw_dataset_rows": rows,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "world_size_at_render": world_size,
            "effective_global_batch_size": global_batch,
            "num_epochs": epochs,
            "planned_optimizer_steps_before_length_filter": planned_steps,
            "max_steps_override": body.get("max_steps"),
            "logging_steps": body.get("logging_steps"),
            "save_strategy": body.get("save_strategy"),
            "save_steps": body.get("save_steps"),
            "save_total_limit": body.get("save_total_limit"),
        },
        "seed": body.get("seed"),
    }


def finalize_training_attribution(rendered_config: Path, out_dir: Path) -> Path:
    """Persist actual optimizer steps and the complete per-step LR/loss trace.

    Custom launchers that call ``axolotl train`` directly should invoke this
    after a successful process. :class:`LocalExecutor` does so automatically.
    """

    provenance_path = out_dir / "training_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
    else:
        provenance = {
            "schema_version": TRAINING_PROVENANCE_SCHEMA_VERSION,
            "status": "configured",
        }
    checkpoints = out_dir / "checkpoints"
    state_candidates: list[tuple[int, Path]] = []
    root_state = checkpoints / "trainer_state.json"
    if root_state.is_file():
        state_candidates.append((-1, root_state))
    for path in checkpoints.glob("checkpoint-*/trainer_state.json"):
        suffix = path.parent.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            state_candidates.append((int(suffix), path))
    if not state_candidates:
        # A run configured to save nothing (save_strategy 'no' with no
        # checkpoint schedule — the no-save training smokes) legitimately
        # leaves no trainer_state.json: record completion honestly and skip
        # the state-derived fields rather than failing a finished run. Any
        # config that SHOULD have saved still errors loudly.
        body = yaml.safe_load(rendered_config.read_text())
        saves_nothing = str(body.get("save_strategy")) == "no" and not body.get(
            "checkpoint_schedule"
        )
        if not saves_nothing:
            raise RuntimeError(f"no trainer_state.json found under {checkpoints}")
        provenance.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "resolved_config_sha256": _sha256_file(rendered_config),
                "actual": {
                    "no_checkpoint_reason": (
                        "save_strategy 'no' with no checkpoint_schedule — "
                        "this run saves nothing by config; step/LR trace "
                        "lives in train.log only"
                    ),
                },
            }
        )
        temporary = provenance_path.with_name(provenance_path.name + ".tmp")
        temporary.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
        )
        temporary.replace(provenance_path)
        return provenance_path
    _step, state_path = max(state_candidates)
    state = json.loads(state_path.read_text())
    trace = [row for row in state.get("log_history", []) if "step" in row]
    trace_path = out_dir / "training_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace)
    )
    final_state_path = out_dir / "trainer_state.final.json"
    shutil.copy2(state_path, final_state_path)
    checkpoint_steps = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in checkpoints.glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    )
    provenance.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resolved_config_sha256": _sha256_file(rendered_config),
            "actual": {
                "global_step": state.get("global_step"),
                "max_steps": state.get("max_steps"),
                "num_train_epochs": state.get("num_train_epochs"),
                "final_epoch": state.get("epoch"),
                "train_batch_size": state.get("train_batch_size"),
                "num_input_tokens_seen": state.get("num_input_tokens_seen"),
                "total_flos": state.get("total_flos"),
                "checkpoint_steps": checkpoint_steps,
                "trainer_state_source": str(state_path),
                "trainer_state_snapshot": str(final_state_path),
                "trainer_state_sha256": _sha256_file(final_state_path),
                "trace_rows": len(trace),
                "trace_path": str(trace_path),
                "trace_sha256": _sha256_file(trace_path),
                "first_learning_rate": trace[0].get("learning_rate") if trace else None,
                "last_learning_rate": trace[-1].get("learning_rate") if trace else None,
            },
        }
    )
    temporary = provenance_path.with_name(provenance_path.name + ".tmp")
    temporary.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(provenance_path)
    return provenance_path


def _mark_training_started(out_dir: Path, loss: float) -> Path:
    """Atomically publish the health marker after the first optimizer loss."""

    marker = out_dir / TRAINING_STARTED_MARKER
    if marker.exists():
        return marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "training_started",
        "observed": "first_optimizer_loss",
        "loss": loss,
        "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker.name}.", suffix=".tmp", dir=marker.parent
    )
    try:
        with os.fdopen(descriptor, "w") as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, marker)
        except FileExistsError:
            pass
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return marker


# ------------------------------------------------------------ stage registry
@dataclass(frozen=True)
class PodSpec:
    """Hardware a stage runs on — part of the stage template, not the call site.

    This is what lets a chain span heterogeneous pods as pure config (the
    sprint workflow: midtrain on 8xH200, SFT on 8xB200 — each stage template
    declares its own pod). Maps onto ``bellhop.PodConfig`` in
    :class:`BellhopExecutor`; ``None`` on a stage means "run where I am"
    (:class:`LocalExecutor`, e.g. already on a provisioned pod).

    ``requirements`` names a pod-side pin-set file (repo-relative) — per-stage
    on purpose: GPU arch dictates wheels (pane's torch cu126 pins are proven on
    H200 but Blackwell/B200 needs cu130 builds and a rebuilt flash-attn).
    Pre-flight each pin set locally (``uv pip compile``) before launching —
    house rule; conflicts discovered on-pod burn pod-hours.

    ``checkpoint_bus`` picks how this stage's checkpoint reaches the next
    stage's (possibly different-type) pod — see :class:`BellhopExecutor` for
    the transport details. Whatever the bus, the ``checkpoints.jsonl`` row
    carries a durable pointer (``gs://...`` / ``hf://...`` / local path),
    never "it's on pod X".
    """

    gpu: str  # bellhop canonical short name ("H200", "B200") or RunPod gpuTypeId
    gpu_count: int = 8
    # >1 = a RunPod Instant Cluster of this many nodes (gpu_count is then
    # per-node). Training launches through torchrun with the cluster env
    # bellhop injects; checkpoint-bus egress and the results pull happen on
    # rank 0 only. Needs bellhop>=0.8.0.
    nodes: int = 1
    # cap on the whole cluster's $/hr (bellhop auto-bids RunPod's per-node
    # minimum; this bounds the bid). Single-node pods ignore it.
    max_hourly_cost: float | None = None
    image: str | None = None
    requirements: str | None = None
    # RunPod cloud for single-node pods ("SECURE"/"COMMUNITY"; None = bellhop
    # default COMMUNITY-with-fallback). Community H100 hosts have bitten NCCL
    # at the first collective (Error 2 — container shm/P2P quirks); SECURE
    # hosts match the proven dispatch runs. Ignored for clusters (always
    # datacenter-hosted).
    cloud: str | None = None
    # extra env exported in the pod's run step (e.g. NCCL knobs). On clusters
    # these override bellhop's injected rank env — don't set NCCL_SOCKET_IFNAME
    # here.
    extra_env: dict[str, str] | None = None
    # hf-bus namespace (org or user) for checkpoint repos; None = the token's
    # user namespace. Personal namespaces hit private-storage limits fast —
    # team runs should name the org (e.g. "arcadia-impact").
    hf_namespace: str | None = None
    max_hours: float = 24.0
    # 12B sharded checkpoints + prepared datasets are disk-hungry; pane lost a
    # run to a full 400GB container disk.
    disk_gb: int = 300
    # RunPod network volume to mount at /workspace (bellhop
    # ClusterConfig.network_volume_id — clusters only; bellhop 0.8.0
    # PodConfig has no such field, so nodes=1 with a volume set is an error).
    # Big-model runs want one: a 200 GB+ base-model snapshot survives node
    # restarts/relaunches instead of re-downloading onto every container
    # disk. Pins the cluster to the volume's datacenter.
    network_volume_id: str | None = None
    # acceptable HOST CUDA driver versions (bellhop allowedCudaVersions) —
    # RunPod only checks the image's floor otherwise; a cu13-linked wheel on a
    # 12.9-driver host dies at init (the F0 ladder's hardest-won lesson)
    cuda_versions: list[str] | None = None
    # extra setup shell appended after the pin-set install (e.g. the
    # flash-attn --no-build-isolation compile when no prebaked image is used)
    setup_extra: str | None = None
    # "gcs": pod-side push/pull, gs:// pointers (default — one network leg for
    #        a ~24GB 12B checkpoint). "bellhop": devbox-mediated p.pull/p.push,
    #        zero pod creds (smoke runs / small models). "hf": pod-side Hub
    #        push, hf:// pointers (when evals want to load by hf id directly).
    checkpoint_bus: str = "gcs"

    def __post_init__(self) -> None:
        if self.checkpoint_bus not in ("gcs", "bellhop", "hf"):
            raise ValueError(
                f"unknown checkpoint_bus {self.checkpoint_bus!r} "
                "(expected gcs, bellhop, or hf)"
            )
        if not 1 <= self.nodes <= 8:
            raise ValueError(
                f"nodes={self.nodes} out of range (1, or 2-8 for an Instant "
                "Cluster; >8 needs RunPod sales)"
            )
        if self.network_volume_id is not None and self.nodes == 1:
            raise ValueError(
                "network_volume_id needs nodes >= 2: bellhop 0.8.0 exposes "
                "network volumes on ClusterConfig only (single-node PodConfig "
                "has no such field — use disk_gb for pod-local scratch)"
            )


@dataclass(frozen=True)
class DocumentLossRecipe:
    """Model-specific chat details for the generic raw/chat loss switch.

    Dataset schemas and assistant-only masking are backend invariants. The
    concrete recipe supplies only tokenizer/model-specific Axolotl root keys,
    such as a chat template and end-of-turn token.
    """

    chat: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chat, dict):
            raise ValueError("document_loss.chat must be a mapping")
        reserved = {
            "base_model",
            "dataset_prepared_path",
            "datasets",
            "output_dir",
            "seed",
            "train_on_inputs",
        }
        conflicts = sorted(reserved & set(self.chat))
        if conflicts:
            raise ValueError(
                "document_loss.chat contains generic/reserved keys "
                f"{conflicts}; the renderer owns those keys"
            )


@dataclass(frozen=True)
class StageSpec:
    """One stage template: a named, tuned axolotl config with declared slots.

    ``axolotl`` is the verbatim axolotl config mapping (pane YAML body).
    ``kind`` gates which per-run values :func:`render_stage` may inject
    (``midtrain``/``sft`` take a dataset; ``dpo`` takes pair sets).
    ``pod`` declares the hardware (see :class:`PodSpec`); ``None`` = local.
    """

    name: str
    description: str
    kind: str  # "midtrain" | "sft" | "dpo"
    base_model: str
    pod: PodSpec | None = None
    document_loss: DocumentLossRecipe | None = None
    axolotl: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("midtrain", "sft", "dpo"):
            raise ValueError(f"stage {self.name!r}: unknown kind {self.kind!r}")
        if isinstance(self.pod, dict):
            known = {f.name for f in dataclasses.fields(PodSpec)}
            unknown = set(self.pod) - known
            if unknown:
                raise ValueError(
                    f"stage {self.name!r}: unknown pod keys {sorted(unknown)}"
                )
            object.__setattr__(self, "pod", PodSpec(**self.pod))
        if isinstance(self.document_loss, dict):
            known = {f.name for f in dataclasses.fields(DocumentLossRecipe)}
            unknown = set(self.document_loss) - known
            if unknown:
                raise ValueError(
                    f"stage {self.name!r}: unknown document_loss keys "
                    f"{sorted(unknown)}"
                )
            object.__setattr__(
                self,
                "document_loss",
                DocumentLossRecipe(**self.document_loss),
            )


def _apply_document_loss(
    body: dict[str, Any], stage: StageSpec, mode: str
) -> None:
    """Apply the model-independent raw/chat Axolotl data-loss contract."""

    recipe = stage.document_loss
    if recipe is None:
        raise ValueError(
            f"stage {stage.name!r} does not declare document-loss support"
        )
    datasets = body.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError(
            f"stage {stage.name!r}: document loss needs exactly one dataset"
        )
    path = datasets[0].get("path")
    if mode == "raw":
        body["datasets"] = [
            {"path": path, "type": "completion", "field": "text"}
        ]
        body.pop("train_on_inputs", None)
        for key in recipe.chat:
            body.pop(key, None)
        return
    if mode == "chat":
        body["datasets"] = [
            {
                "path": path,
                "type": "chat_template",
                "field_messages": "messages",
            }
        ]
        body["train_on_inputs"] = False
        body.update(copy.deepcopy(recipe.chat))
        return
    raise ValueError(f"unknown document loss mode {mode!r}")


def stage_path(name: str) -> Path:
    return STAGES_DIR / f"{name}.yaml"


def list_stages() -> list[str]:
    return sorted(p.stem for p in STAGES_DIR.glob("*.yaml"))


def load_stage(name: str) -> StageSpec:
    """Load a stage template by name (``src/scimt/train/stages/<name>.yaml``)."""
    p = stage_path(name)
    if not p.exists():
        raise KeyError(
            f"no stage named {name!r} (looked in {p}); "
            f"registered: {', '.join(list_stages()) or '(none)'}"
        )
    with p.open() as f:
        data = yaml.safe_load(f)
    if data.get("name") != name:
        raise ValueError(f"stage file {p} has name={data.get('name')!r}, expected {name!r}")
    known = {f.name for f in dataclasses.fields(StageSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"stage {name!r}: unknown keys {sorted(unknown)}")
    return StageSpec(**data)


def render_stage(
    stage: StageSpec,
    cfg: "TrainConfig",
    dataset_path: Path,
    out_dir: Path,
) -> Path:
    """Overlay per-run values on the template; write ``<out>/axolotl.yaml``.

    The only mutation points are the declared slots — hparams stay in the
    template, so a diff of two rendered configs is a diff of *runs*, not of
    recipes:

    - ``base_model``: ``cfg.load_checkpoint_path`` when chaining (may be a
      ``gs://`` bus pointer — the executor resolves it to a local dir), else
      the template's ``base_model``;
    - ``datasets[0].path``: the staged dataset (pane's DPO quirk — pair sets
      need their own type/fields block — is honored by overriding only
      ``path`` and never the block's ``type``);
    - ``cfg.document_loss`` set -> a recipe that opts in is normalized to the
      generic raw-completion or assistant-only-chat schema; the stage supplies
      only model-specific chat-template and terminator keys;
    - ``output_dir`` -> ``<out>/checkpoints``, ``dataset_prepared_path`` ->
      ``<out>/prepared`` (per-run caches; a shared prepared-path cross-wires
      concurrent runs), ``seed`` -> ``cfg.seed``;
    - a relative ``chat_template_jinja`` resolves against the packaged
      ``stages/assets/`` dir;
    - ``cfg.lora`` set -> the axolotl adapter keys are injected (see below);
      conflicts with a template that already carries adapter keys are an
      error, and chaining from an UNMERGED adapter checkpoint
      (``adapter_config.json`` in the dir) is refused — merge the LoRA into
      a full checkpoint first;
    - ``cfg.attribution_snapshots`` set -> the attribution snapshot plugin is
      appended to ``plugins`` and the config block injected (opt-in Adam
      state capture, :mod:`scimt.train.attribution_snapshot`); unset, the
      render is untouched. Templates must not carry the feature themselves.

    Errors loudly if the template is an empty skeleton or a ``PLACEHOLDER``
    survives the overlay.
    """
    if not stage.axolotl:
        raise ValueError(
            f"stage {stage.name!r} has an empty axolotl block — the pane "
            "config body has not been landed in its template yet"
        )
    body = copy.deepcopy(stage.axolotl)
    if cfg.load_checkpoint_path:
        prev = Path(cfg.load_checkpoint_path)
        if prev.exists() and (prev / "adapter_config.json").exists():
            raise ValueError(
                f"load_checkpoint_path {cfg.load_checkpoint_path!r} is an "
                "UNMERGED LoRA adapter (adapter_config.json present) — merge "
                "it into a full checkpoint before chaining (see "
                "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py); "
                "training a new stage on top of raw adapter files would "
                "silently drop the adapter's weights"
            )
    body["base_model"] = cfg.load_checkpoint_path or stage.base_model
    body["output_dir"] = str(out_dir / "checkpoints")
    body["dataset_prepared_path"] = str(out_dir / "prepared")
    body["seed"] = cfg.seed
    datasets = body.get("datasets")
    if not datasets:
        raise ValueError(f"stage {stage.name!r}: template has no datasets block")
    datasets[0]["path"] = str(dataset_path)
    if cfg.document_loss is not None:
        _apply_document_loss(body, stage, cfg.document_loss)
    if cfg.lora is not None:
        if cfg.lora.initial_adapter_path is not None:
            raise ValueError(
                "lora.initial_adapter_path is currently supported only by hf_grpo"
            )
        clash = sorted(
            k for k in body
            if k == "adapter" or k.startswith(("lora_", "peft"))
        )
        if clash:
            raise ValueError(
                f"stage {stage.name!r} already carries adapter keys {clash} — "
                "a template is full-weight OR TrainConfig.lora drives the "
                "adapter, never both"
            )
        body["adapter"] = "lora"
        body["lora_r"] = cfg.lora.r
        body["lora_alpha"] = cfg.lora.resolved_alpha
        body["lora_dropout"] = cfg.lora.dropout
        if cfg.lora.target_modules is not None:
            body["lora_target_modules"] = (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            )
        elif cfg.lora.target_linear:
            body["lora_target_linear"] = True
        if cfg.lora.target_parameters is not None:
            # 3D stacked tensors (MoE expert weights) adapt via peft
            # target_parameters, orthogonal to the module-targeting above
            body["lora_target_parameters"] = list(cfg.lora.target_parameters)
        # always explicit: axolotl auto-enables its Triton LoRA kernels when
        # dropout == 0, and the source patch asserts on unknown attention
        # code (glm4_moe, live 2026-08-16) — never leave this to inference
        body["lora_qkv_kernel"] = cfg.lora.triton_kernels
        body["lora_mlp_kernel"] = cfg.lora.triton_kernels
        body["lora_o_kernel"] = cfg.lora.triton_kernels
    if cfg.attribution_snapshots is not None:
        # Opt-in Adam snapshot wiring (scimt.train.attribution_snapshot).
        # OFF by default: with the config unset this branch never runs and the
        # render stays byte-identical. A template must not hardcode the
        # feature — it is a per-run TrainConfig knob, like lora.
        if "attribution_snapshots" in body:
            raise ValueError(
                f"stage {stage.name!r} template already carries an "
                "attribution_snapshots block — opt in via "
                "TrainConfig.attribution_snapshots, never the template"
            )
        plugins = list(body.get("plugins") or [])
        if ATTRIBUTION_PLUGIN_PATH in plugins:
            raise ValueError(
                f"stage {stage.name!r} template already lists the attribution "
                "snapshot plugin — opt in via TrainConfig.attribution_snapshots, "
                "never the template"
            )
        plugins.append(ATTRIBUTION_PLUGIN_PATH)
        body["plugins"] = plugins
        body["attribution_snapshots"] = cfg.attribution_snapshots.as_dict()
    jinja = body.get("chat_template_jinja")
    if jinja and not Path(jinja).is_absolute():
        body["chat_template_jinja"] = str(STAGES_DIR / "assets" / Path(jinja).name)

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = out_dir / "axolotl.yaml"
    text = yaml.safe_dump(body, sort_keys=False)
    if "PLACEHOLDER" in text:
        raise ValueError(
            f"stage {stage.name!r}: a PLACEHOLDER slot survived rendering — "
            "the template carries a slot render_stage does not fill"
        )
    rendered.write_text(text)
    provenance = _configured_training_provenance(
        stage=stage,
        body=body,
        dataset_path=Path(dataset_path),
        out_dir=out_dir,
    )
    (out_dir / "training_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return rendered


# ---------------------------------------------------------------- loss guard
# Verbatim port of pane scripts/loss_guard.py's parsing + trigger logic (the
# stream plumbing is new — pane polled a log file; here the executor feeds
# lines directly).

# pane matched numerics only, which silently *skipped* a NaN loss — the one
# value the guard most needs to see; nan/inf added here.
LOSS_RE = re.compile(r"'loss': '(nan|inf|[0-9.eE+-]+)'", re.IGNORECASE)


@dataclass(frozen=True)
class GuardConfig:
    """Divergence trigger: loss exceeds max(ratio*running_min, running_min+margin)
    for ``patience`` consecutive steps, after ``grace`` warmup steps."""

    ratio: float = 1.5
    margin: float = 0.5
    grace: int = 5
    patience: int = 5


def parse_losses(text: str) -> list[float]:
    return [float(match) for match in LOSS_RE.findall(text)]


def check(losses: list[float], ratio: float, margin: float, grace: int,
          patience: int) -> bool:
    """True when the tail of ``losses`` shows a sustained divergence."""
    if len(losses) <= grace + patience:
        return False
    bad = 0
    running_min = min(losses[:grace]) if grace else losses[0]
    for value in losses[grace:]:
        threshold = max(ratio * running_min, running_min + margin)
        if value > threshold:
            bad += 1
            if bad >= patience:
                return True
        else:
            bad = 0
        running_min = min(running_min, value)
    return False


class LossDiverged(RuntimeError):
    """Raised by :func:`guard_loss` when training loss diverges (or goes NaN)."""


async def guard_loss(
    lines: AsyncIterator[str],
    *,
    config: GuardConfig | None = None,
) -> list[float]:
    """Watch a training log stream; raise :class:`LossDiverged` to kill a
    diverged run early (pane's guard twice saved multi-hour pod bills).

    Consumes the stream, parsing axolotl ``'loss': 'X'`` values; raises on NaN
    or on the :func:`check` trigger. Returns the parsed loss series when the
    stream ends healthy. Pure-python and stream-shaped so it is unit-testable
    without a GPU (feed it a list-backed async iterator).
    """
    cfg = config or GuardConfig()
    losses: list[float] = []
    async for line in lines:
        for match in LOSS_RE.findall(line):
            value = float(match)
            if not math.isfinite(value):
                raise LossDiverged(f"loss went NaN/inf at step ~{len(losses) + 1}")
            losses.append(value)
            if check(losses, cfg.ratio, cfg.margin, cfg.grace, cfg.patience):
                raise LossDiverged(
                    f"loss diverged: step ~{len(losses)} at {losses[-1]:.3f} "
                    f"vs running min {min(losses):.3f}"
                )
    return losses


# ------------------------------------------------------------- checkpoints
def _final_checkpoint(train_out: Path) -> Path:
    """The directory holding the finished model under axolotl's output_dir:
    the highest-step ``checkpoint-N`` when present, else the root export.

    Axolotl may write both.  The root is a duplicate inference export and can
    omit ``trainer_state.json``; the numbered directory is the canonical
    stateful handoff for chaining, attribution, and durable publication.
    The root export counts for full-weight (``config.json``) and LoRA
    (``adapter_config.json``) saves alike.  Loud error when training left
    nothing.
    """
    steps: list[tuple[int, Path]] = []
    for p in train_out.glob("checkpoint-*"):
        suffix = p.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            steps.append((int(suffix), p))
    if steps:
        return max(steps)[1]
    if (train_out / "config.json").exists() or (
        train_out / "adapter_config.json"
    ).exists():
        return train_out
    raise RuntimeError(
        f"no model config.json or checkpoint-* under {train_out} — training "
        "saved nothing (check train.log)"
    )


def _emit_checkpoint_row(out_dir: Path, pointer: str) -> None:
    """Append the ``checkpoints.jsonl`` row that makes ``read_checkpoint`` and
    staged chains work unchanged. For a local full-FT checkpoint, sampler and
    state are the same dir; for a bus-published checkpoint both are the URI."""
    row = {"state_path": pointer, "sampler_path": pointer}
    with (out_dir / "checkpoints.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _tail(path: Path, chars: int = 2000) -> str:
    try:
        return path.read_text(errors="replace")[-chars:]
    except OSError:
        return "(no log)"


def _axolotl_executable() -> str:
    """Resolve the CLI beside the active Python before consulting ``PATH``."""
    sibling = Path(sys.executable).with_name("axolotl")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("axolotl") or "axolotl"


def _training_subprocess_environment() -> dict[str, str]:
    """Ensure nested launchers resolve from the active Python environment."""
    env = os.environ.copy()
    active_bin = str(Path(sys.executable).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = f"{active_bin}:{current_path}" if current_path else active_bin
    return env


# ------------------------------------------------------------------ executors
class Executor(Protocol):
    """Where a rendered stage runs. The backend renders + records provenance +
    reads checkpoints; the executor only *runs*. Resolved from the stage
    template (:func:`executor_for`), never from call-site flags — heterogeneous
    chains (H200 midtrain -> B200 SFT) are a property of the templates."""

    async def run_stage(
        self,
        rendered_config: Path,
        out_dir: Path,
        stage: StageSpec,
        *,
        run_name: str | None = None,
    ) -> None:
        ...


def _train_argv(rendered_config: Path) -> list[str]:
    """The launcher argv — plain ``axolotl train``, or torchrun when this
    process is one node of an Instant Cluster.

    Cluster detection is by env: bellhop's ``exec_all`` injects
    ``NUM_NODES``/``NODE_RANK``/``NUM_TRAINERS``/``PRIMARY_ADDR``/
    ``PRIMARY_PORT`` on every rank (the rendezvous is bellhop-derived —
    RunPod's documented ``PRIMARY_*`` injection doesn't actually happen).
    ``--rdzv_backend static`` is required: Instant Clusters don't support the
    dynamic ``c10d`` backend. Still config-first — the rendered YAML remains
    the whole training interface; these are process-group coordinates, not
    hyperparameters.
    """
    nnodes = int(os.environ.get("NUM_NODES", "1"))
    if nnodes <= 1:
        return [_axolotl_executable(), "train", str(rendered_config)]
    return [
        "torchrun",
        "--nnodes", str(nnodes),
        "--node_rank", os.environ["NODE_RANK"],
        "--nproc_per_node", os.environ["NUM_TRAINERS"],
        "--rdzv_id", "scimt",
        "--rdzv_backend", "static",
        "--rdzv_endpoint", f"{os.environ['PRIMARY_ADDR']}:{os.environ['PRIMARY_PORT']}",
        "-m", "axolotl.cli.train", str(rendered_config),
    ]


class LocalExecutor:
    """Run ``axolotl train <rendered_config>`` as a supervised async subprocess
    on this machine (assumes GPUs are already under our feet — the pane
    workflow, and the on-pod half of :class:`BellhopExecutor`).

    ``asyncio.create_subprocess_exec`` (never a shell string); stdout+stderr
    tee'd to ``<out>/train.log`` and streamed through the loss guard; a guard
    trip kills the process group; non-zero exit raises with the log tail
    inline (error-loud). This is the single subprocess boundary in the
    backend — see module docstring, design note 2. On an Instant Cluster node
    the same boundary launches through torchrun (:func:`_train_argv`); loss
    lines only appear on rank 0, so the guard is naturally rank-0-only.
    """

    def __init__(self, guard: GuardConfig | None = None) -> None:
        self.guard = guard or GuardConfig()

    async def run_stage(
        self,
        rendered_config: Path,
        out_dir: Path,
        stage: StageSpec,
        *,
        run_name: str | None = None,
    ) -> None:
        log_path = out_dir / "train.log"
        # A retry must earn a fresh health signal.  Leaving the previous
        # process's marker in place lets launcher cleanup race ahead before
        # this process reaches its first optimizer step.
        (out_dir / TRAINING_STARTED_MARKER).unlink(missing_ok=True)
        proc = await asyncio.create_subprocess_exec(
            *_train_argv(rendered_config),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=2**20,  # tqdm/progress lines can be very long
            env=_training_subprocess_environment(),
        )
        assert proc.stdout is not None
        losses: list[float] = []
        try:
            with log_path.open("ab") as log:
                async for raw in proc.stdout:
                    log.write(raw)
                    log.flush()
                    for match in LOSS_RE.findall(raw.decode(errors="replace")):
                        value = float(match)
                        if not math.isfinite(value):
                            raise LossDiverged(
                                f"loss went NaN/inf at step ~{len(losses) + 1}"
                            )
                        if not losses:
                            _mark_training_started(out_dir, value)
                        losses.append(value)
                        if check(losses, self.guard.ratio, self.guard.margin,
                                 self.guard.grace, self.guard.patience):
                            raise LossDiverged(
                                f"loss diverged: step ~{len(losses)} at "
                                f"{losses[-1]:.3f} vs running min {min(losses):.3f}"
                            )
            code = await proc.wait()
            if code != 0:
                # 20k chars: torch-elastic's failure wrapper alone is >2k and
                # buried the actual child traceback (LoRA smoke, 2026-07-23)
                raise RuntimeError(
                    f"axolotl train exited {code} for stage {stage.name!r}; "
                    f"log tail:\n{_tail(log_path, 20_000)}"
                )
            # On a cluster only rank 0 writes consolidated checkpoints
            # (FULL_STATE_DICT gathers to rank 0); non-zero ranks have no
            # trainer_state.json by design, not by failure.
            if int(os.environ.get("NODE_RANK", "0")) == 0:
                finalize_training_attribution(rendered_config, out_dir)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


async def _run_bellhop_stage(
    rendered_config: Path,
    out_dir: Path,
    stage_template: Path,
    stage_name: str,
    run_name: str,
) -> None:
    """Verified pod payload: provenance first, then the local executor."""

    stage = load_stage(stage_name)
    snapshot_run(
        out_dir,
        run_name,
        {"axolotl": rendered_config, "stage_template": stage_template},
        repo_dir=Path.cwd(),
    )
    await LocalExecutor().run_stage(rendered_config, out_dir, stage)


def _build_transfer_wheel(out_dir: Path) -> Path:
    """Build scimt from an exact HEAD export into the Bellhop transfer set."""

    dist = out_dir / "bellhop_dist"
    with tempfile.TemporaryDirectory(prefix="scimt-wheel-") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "source.tar"
        exported = temporary_root / "source"
        exported.mkdir()
        try:
            subprocess.run(
                [
                    "git", "archive", "--format=tar", "--output", str(archive),
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["tar", "xf", str(archive), "-C", str(exported)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "uv", "build", "--wheel", "--clear", "--no-create-gitignore",
                    "--out-dir", str(dist), str(exported),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "git, tar, and uv are required for Bellhop wheel staging"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"failed to stage Bellhop scimt wheel: {error.stderr.strip()}"
            ) from error
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one staged scimt wheel in {dist}, found {wheels}")
    return wheels[0]


class BellhopExecutor:
    """Run the stage on an ephemeral RunPod pod via ``bellhop`` (lazy import —
    bellhop stays an optional, devbox-side dep; ``import scimt`` unaffected).

    Mapping (all from the stage template, config-first):

    - ``stage.pod`` -> ``bellhop.PodConfig`` (:func:`_pod_config_kwargs`),
      with ``max_lifetime`` as the server-side kill switch — a hung run must
      not outlive its TTL;
    - ``stage.pod.requirements`` -> pod-side pin set installed in setup.
      Per-stage because GPU arch dictates wheels;
    - the repo checkout is the pushed codebase; **the rendered config, dataset,
      and local output staging dir must live under it** so the final YAML,
      prebuilt scimt wheel, and source manifest ride the push. Pod-side mutable
      output is rewritten to a sibling ``../runtime/<run>`` tree. The pod
      verifies provenance, then runs :class:`LocalExecutor` on the exact
      rewritten YAML—one training path on both substrates.

    Checkpoint bus (``stage.pod.checkpoint_bus``):

    - ``"gcs"`` (default): the pod pushes ``checkpoints/`` to
      ``$SCIMT_GCS_BASE/<out_dir name>/`` via rclone after training (creds via
      ``RCLONE_*``/``GOOGLE_APPLICATION_CREDENTIALS`` env passthrough), writes
      the ``gs://`` pointer row, and deletes the heavy dir so the results pull
      stays small. A ``gs://`` ``load_checkpoint_path`` on the *next* stage is
      pulled in setup and the rendered ``base_model`` rewritten to the local
      copy.
    - ``"bellhop"``: checkpoints ride the results pull (devbox-mediated,
      zero pod creds — smoke runs / small models).
    - ``"hf"``: pod-side ``hf upload`` to the private Hub (``HF_TOKEN``
      passthrough), ``hf://`` pointer row.

    Needs a live-pod smoke before the sprint leans on it (flagged in PR #209);
    the pure config/script builders below are unit-tested CPU-side.
    """

    #: env vars forwarded to the pod when present (transport creds only).
    #: For the gcs bus, prefer RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS —
    #: it carries the service-account JSON *inline*, so no key file has to
    #: exist on the pod (a _FILE path would dangle there).
    ENV_PASSTHROUGH = ("HF_TOKEN", "RCLONE_CONFIG_GCS_TYPE",
                       "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
                       "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
                       # gs:// bus URIs resolve via an rclone remote literally
                       # named "gs" — callers mirror GS <- GCS at env load
                       # (msm_ablation_sweep P2 postmortem: remote "gcs" +
                       # "gs://..." URI = "didn't find section in config file")
                       "RCLONE_CONFIG_GS_TYPE",
                       "RCLONE_CONFIG_GS_SERVICE_ACCOUNT_CREDENTIALS",
                       "RCLONE_CONFIG_GS_BUCKET_POLICY_ONLY")

    def __init__(self, gcs_base: str | None = None) -> None:
        self.gcs_base = gcs_base or os.environ.get("SCIMT_GCS_BASE")

    @staticmethod
    def _pod_config_kwargs(pod: PodSpec, slug: str) -> dict[str, Any]:
        from datetime import timedelta

        kwargs: dict[str, Any] = {
            "gpu": pod.gpu,
            "gpu_count": pod.gpu_count,
            "container_disk_gb": pod.disk_gb,
            "max_lifetime": timedelta(hours=pod.max_hours),
            "name": f"scimt-{slug}",
        }
        if pod.image:
            kwargs["image"] = pod.image
        if pod.cuda_versions:
            kwargs["cuda_versions"] = list(pod.cuda_versions)
        if pod.cloud:
            kwargs["cloud"] = pod.cloud
        return kwargs

    @staticmethod
    def _cluster_config_kwargs(pod: PodSpec, slug: str) -> dict[str, Any]:
        """``stage.pod`` -> ``bellhop.ClusterConfig`` kwargs (nodes > 1)."""
        from datetime import timedelta

        kwargs: dict[str, Any] = {
            "gpu": pod.gpu,
            "nodes": pod.nodes,
            "gpu_count": pod.gpu_count,
            "container_disk_gb": pod.disk_gb,
            "max_lifetime": timedelta(hours=pod.max_hours),
            "name": f"scimt-{slug}",
        }
        if pod.image:
            kwargs["image"] = pod.image
        if pod.cuda_versions:
            # ClusterConfig's spelling of PodConfig.cuda_versions
            kwargs["allowed_cuda_versions"] = list(pod.cuda_versions)
        if pod.max_hourly_cost is not None:
            kwargs["max_hourly_cost"] = pod.max_hourly_cost
        if pod.network_volume_id:
            kwargs["network_volume_id"] = pod.network_volume_id
        return kwargs

    def _stage_script(
        self, stage: StageSpec, rendered_rel: str, out_rel: str,
        prev_gs_pointer: str | None,
        *,
        wheel_rel: str,
        stage_template_rel: str,
        run_name: str | None = None,
    ) -> tuple[str, str]:
        """(setup, run) shell for the pod. Pure string-building — unit-tested."""
        assert stage.pod is not None
        setup_lines = [
            "set -euo pipefail",
            # pin-set resolution needs all indexes considered equally: the
            # pytorch cu-index shadows PyPI names (e.g. `packaging`) and uv's
            # first-index-wins default then fails the whole resolve (bit the
            # LoRA smoke 2026-07-23; ex06's hand-rolled setup already exports
            # this)
            "export UV_INDEX_STRATEGY=unsafe-best-match "
            "UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1",
            # uv for all pod installs (parallel downloads). Force-upgrade:
            # community images preinstall wildly different uv versions, and
            # old ones silently ignore UV_INDEX_STRATEGY (one smoke host
            # resolved, its sibling didn't — same setup string)
            "python3 -m pip install -q -U uv",
        ]
        # belt-and-braces: the explicit flag, not just the env var
        _uv = "uv pip install --system --index-strategy unsafe-best-match -q"
        if stage.pod.requirements:
            setup_lines.append(f"{_uv} -r {shlex.quote(stage.pod.requirements)}")
        # The wheel is prebuilt client-side from an exact HEAD export and is
        # itself covered by the transferred-source manifest.  Pod setup never
        # invokes a build backend against the immutable source snapshot.
        setup_lines.append(f"{_uv} {shlex.quote(wheel_rel)}")
        if stage.pod.setup_extra:
            setup_lines.append(stage.pod.setup_extra)
        if prev_gs_pointer:
            local_prev = f"{out_rel}/prev_ckpt"
            setup_lines += [
                f"mkdir -p {shlex.quote(local_prev)}",
                f"rclone copy {shlex.quote(prev_gs_pointer)} {shlex.quote(local_prev)}",
            ]

        resolved_run_name = run_name or f"{stage.name}-{Path(out_rel).name}"
        pod_side = (
            "import asyncio; from pathlib import Path; "
            "from scimt.train.axolotl import _run_bellhop_stage; "
            f"asyncio.run(_run_bellhop_stage(Path({rendered_rel!r}), "
            f"Path({out_rel!r}), Path({stage_template_rel!r}), "
            f"{stage.name!r}, {resolved_run_name!r}))"
        )
        run_lines = [
            "set -euo pipefail",
            # NVLS multicast bind fails on containerized community hosts
            # (NCCL 2.29 "unhandled cuda error ... Failed to bind NVLink
            # SHARP" at the FIRST collective — killed the LoRA FSDP2 smoke
            # before any model code). NVLS is a perf-only feature; disable.
            # Exported here, not in setup: setup and run are separate exec
            # contexts (the F0 env-not-inherited lesson).
            "export NCCL_NVLS_ENABLE=0",
            f"python3 -c {shlex.quote(pod_side)}",
        ]
        ckpts = f"{out_rel}/checkpoints"
        rows = f"{out_rel}/checkpoints.jsonl"
        bus = stage.pod.checkpoint_bus
        # Bus egress runs on rank 0 only: on a multi-node stage every node
        # executes this script, but only rank 0 holds the consolidated
        # checkpoints (and only rank 0's results dir is pulled). Single-node
        # pods have no NODE_RANK, so the ${NODE_RANK:-0} default keeps the
        # guard a no-op there.
        if bus == "gcs":
            if not self.gcs_base:
                raise ValueError(
                    "checkpoint_bus=gcs needs a GCS base "
                    "(BellhopExecutor(gcs_base=...) or SCIMT_GCS_BASE)"
                )
            uri = f"{self.gcs_base.rstrip('/')}/{Path(out_rel).name}/checkpoints/"
            run_lines.append(_rank0(
                f"rclone copy {shlex.quote(ckpts)} {shlex.quote(uri)}",
                _emit_row_cmd(rows, uri),
                # keep the results pull small: the pointer travels, not 24GB
                f"rm -rf {shlex.quote(ckpts)}",
            ))
        elif bus == "hf":
            repo = f"scimt-ckpt-{Path(out_rel).name}"
            if stage.pod.hf_namespace:
                repo = f"{stage.pod.hf_namespace}/{repo}"
            run_lines.append(_rank0(
                # axolotl drops a model-card README whose metadata names the
                # local dataset path; the Hub rejects that as an invalid
                # dataset id and the whole upload fails. The card is
                # boilerplate — strip it, the weights/config/tokenizer travel.
                f"find {shlex.quote(ckpts)} -name README.md -delete",
                f"hf upload --private {shlex.quote(repo)} {shlex.quote(ckpts)}",
                _emit_row_cmd(rows, f"hf://{repo}"),
                f"rm -rf {shlex.quote(ckpts)}",
            ))
        # bus == "bellhop": checkpoints stay in place and ride the results pull;
        # the backend emits the local-path row after the pull.
        return " && ".join(setup_lines), " && ".join(run_lines)

    async def run_stage(
        self,
        rendered_config: Path,
        out_dir: Path,
        stage: StageSpec,
        *,
        run_name: str | None = None,
    ) -> None:
        import bellhop

        assert stage.pod is not None
        try:
            rendered_rel = str(rendered_config.resolve().relative_to(REPO_ROOT))
            out_dir.resolve().relative_to(REPO_ROOT)
        except ValueError as e:
            raise ValueError(
                "pod execution requires rendered config, dataset, and out_dir "
                f"under the repo checkout {REPO_ROOT} (they ride the code push)"
            ) from e

        # The rendered config carries devbox-absolute paths; on the pod inputs
        # remain checkout-relative, while every mutable output goes in a
        # sibling runtime tree.  Bellhop's own run.log is also placed there via
        # results_subdir, so the transferred source remains manifest-verifiable.
        # A gs:// resume pointer is pulled into that runtime tree too.
        body = yaml.safe_load(rendered_config.read_text())
        prev = str(body.get("base_model", ""))
        prev_gs = prev if prev.startswith("gs://") else None
        slug = out_dir.name
        runtime_rel = f"../runtime/{slug}"
        body["output_dir"] = f"{runtime_rel}/checkpoints"
        body["dataset_prepared_path"] = f"{runtime_rel}/prepared"
        if prev_gs:
            body["base_model"] = f"{runtime_rel}/prev_ckpt"
        _relativize_paths(body)
        rendered_config.write_text(yaml.safe_dump(body, sort_keys=False))

        wheel = _build_transfer_wheel(out_dir)
        wheel_rel = str(wheel.resolve().relative_to(REPO_ROOT))
        template_rel = str(stage_path(stage.name).resolve().relative_to(REPO_ROOT))
        manifest_path = out_dir / ".scimt-source.json"
        source_manifest = build_source_manifest(REPO_ROOT, manifest_path)
        manifest_rel = str(manifest_path.resolve().relative_to(REPO_ROOT))
        setup, run_cmd = self._stage_script(
            stage,
            rendered_rel,
            runtime_rel,
            prev_gs,
            wheel_rel=wheel_rel,
            stage_template_rel=template_rel,
            run_name=run_name,
        )
        spec = bellhop.RunSpec(
            slug=f"{stage.name}-{slug}",
            codebase=str(REPO_ROOT),
            setup=setup,
            run=run_cmd,
            results_subdir=runtime_rel,
            local_out=str(out_dir.parent),
            gcs_base=None,  # the checkpoint bus owns artifact placement
            env={
                "PYTHONDONTWRITEBYTECODE": "1",
                "SCIMT_SOURCE_COMMIT": source_manifest["commit"],
                "SCIMT_SOURCE_MANIFEST": manifest_rel,
                "SCIMT_RUNTIME_ROOT": f"/workspace/runtime/{slug}",
                **{
                    k: v
                    for k in self.ENV_PASSTHROUGH
                    if (v := os.environ.get(k))
                },
                **(stage.pod.extra_env or {}),
            },
        )
        if stage.pod.nodes > 1:
            if not hasattr(bellhop, "run_cluster"):
                raise RuntimeError(
                    f"stage {stage.name!r} declares nodes={stage.pod.nodes} but "
                    "this bellhop has no Instant Clusters support — install "
                    "bellhop-py>=0.8.0"
                )
            cluster_cfg = bellhop.ClusterConfig(
                **self._cluster_config_kwargs(stage.pod, slug))
            await bellhop.run_cluster(spec, cluster_cfg)
        else:
            pod_cfg = bellhop.PodConfig(**self._pod_config_kwargs(stage.pod, slug))
            await bellhop.run(spec, pod_cfg)


def _relativize_paths(body: dict[str, Any]) -> None:
    """Rewrite devbox-absolute repo-internal paths to checkout-relative, in
    place — the pod runs axolotl from the pushed checkout root. Absolute paths
    outside the checkout raise (they cannot exist on the pod). HF ids, gs://
    URIs, and already-relative paths pass through untouched."""

    def rel(value: str) -> str:
        p = Path(value)
        if not p.is_absolute():
            return value
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            raise ValueError(
                f"pod execution: path {value!r} is outside the repo checkout "
                f"{REPO_ROOT} and would not exist on the pod"
            ) from None

    for key in ("base_model", "output_dir", "dataset_prepared_path", "chat_template_jinja"):
        if isinstance(body.get(key), str) and not body[key].startswith(("gs://", "hf://")):
            # HF model ids look like "org/name" and are never absolute
            body[key] = rel(body[key])
    for ds in body.get("datasets", []):
        if isinstance(ds.get("path"), str):
            ds["path"] = rel(ds["path"])


def _emit_row_cmd(rows_path: str, pointer: str) -> str:
    row = json.dumps({"state_path": pointer, "sampler_path": pointer})
    return f"echo {shlex.quote(row)} >> {shlex.quote(rows_path)}"


def _rank0(*cmds: str) -> str:
    """Wrap commands to run on rank 0 only (no-op guard on single-node pods,
    where NODE_RANK is unset and defaults to 0)."""
    return f'if [ "${{NODE_RANK:-0}}" = "0" ]; then {" && ".join(cmds)}; fi'


def executor_for(stage: StageSpec) -> Executor:
    """Template-declared hardware picks the executor: ``pod:`` block ->
    :class:`BellhopExecutor`, no block -> :class:`LocalExecutor`."""
    return BellhopExecutor() if stage.pod is not None else LocalExecutor()


# -------------------------------------------------------------------- backend
class AxolotlBackend:
    """Local-GPU full-finetune backend over the axolotl CLI (pane port)."""

    name = "axolotl"

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        """One stage: render template -> snapshot provenance -> execute ->
        typed checkpoint.

        Requires ``cfg.stage`` (a :func:`load_stage` name); erroring here, not
        deep in axolotl, when it is missing. ``SCIMT_ALLOW_DIRTY=1`` is the one
        documented escape hatch for the dirty-tree provenance guard (dev smoke
        runs only).
        """
        if getattr(cfg, "stage", None) is None:
            raise ValueError(
                "backend='axolotl' needs TrainConfig.stage (a stage-template "
                f"name; registered: {', '.join(list_stages()) or '(none)'})"
            )
        stage = load_stage(cfg.stage)
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_stage(stage, cfg, dataset_path, out_dir)
        executor = executor_for(stage)
        if isinstance(executor, LocalExecutor):
            snapshot_run(
                out_dir, run_name,
                {"axolotl": rendered, "stage_template": stage_path(stage.name)},
                allow_dirty=os.environ.get("SCIMT_ALLOW_DIRTY") == "1",
            )
        await executor.run_stage(rendered, out_dir, stage, run_name=run_name)

        # pod bus scripts emit their own pointer rows; local (and bus=bellhop
        # pulled-back) runs emit the local checkpoint dir here
        if not (out_dir / "checkpoints.jsonl").exists():
            _emit_checkpoint_row(out_dir, str(_final_checkpoint(out_dir / "checkpoints")))
        ckpt = read_checkpoint(out_dir, backend=self.name)
        if not ckpt:
            raise RuntimeError(
                f"stage {stage.name!r} finished but no checkpoint row in "
                f"{out_dir}/checkpoints.jsonl"
            )
        return ckpt
