"""Token-scaling 4B chain: one parent cell end-to-end on a pod.

One "parent cell" is a (arm × dose) midtrain lineage — e.g. ``coin_d8m`` —
carried through: mix build + digest gate → 248-step midtrain (with
prequential code-length logging on the task rows) → 24-step (50M) Dolci IFT →
the EFT capacity ladder (LoRA r ∈ {4, 16, 32, 64, 256}, α = 2r, plus one
full-parameter arm) → the unchanged 4B wave eval battery at 6 endpoints per
capacity → GCS publication.

Ported from ``experiments/prior_coins/pod/dispatch_wave_chain.py`` (native
vLLM LoRA serving with adapter probe + merge-per-endpoint fallback, sanity
prompts, background checkpoint uploads) and the dispatch_scaleup 4B runners
(full-state checkpoint validation, processor-sidecar hydration, the
``pytorch_model_fsdp.bin`` dedup lesson from UPLOAD_ARCHITECTURE.md, and the
publish-first ordering from RESULTS_4B.md's incident log).

Contracts: all data selection digests live in the sibling ``contracts.py``
(written in parallel; the agreed interface is consumed via
:func:`get_contracts`). ``--dry-run`` degrades gracefully when contracts is
absent; any GPU spend requires it AND ``--signed-off``.

Storage: GCS via an rclone remote named ``gcs``. The run shell sources
``/workspace/msm-reproduction/.env`` (defines ``SCIMT_GCS_BASE`` and the
``RCLONE_CONFIG_GCS_*`` variables). SECURITY: none of those values is ever
printed, logged, or written to any manifest — scripts reference
``$SCIMT_GCS_BASE`` symbolically, pins record paths RELATIVE to the base, and
every subprocess error surface is scrubbed (:func:`scrub_secrets`).

Usage (pilot column, G2)::

    python3 experiments/prior_coins/dispatch_token_scaling_4b/pod/chain.py \
        --cell coin_d8m --capacities r64 --run-id <UTC id> --dry-run
    python3 .../chain.py --cell coin_d8m --capacities r64 --run-id <id> \
        --signed-off
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# --------------------------------------------------------------------------
# Frozen run constants (mirrors of SPEC.md §2/§3/§5; the data digests live in
# contracts.py, not here).
# --------------------------------------------------------------------------

RUN_PREFIX = "token-scaling-4b"
CONTROL_CELL = "control_d0"

BASE_MODEL_REPO = "unsloth/gemma-3-4b-pt"
BASE_MODEL_REVISION = "52aba93981c6ad7712b030eb6dd496ece1d279d6"

MIDTRAIN_STAGE = "midtrain_dispatch_gemma3_4b_scaling"
IFT_STAGE = "sft_dispatch_gemma3_4b_dolci50m"
EFT_STAGE = "eft_dispatch_v4_wide_4b"
FP_EFT_STAGE = "fp_eft_dispatch_wave_gemma3_4b"

DATA_SEED = 42
#: training seed comes from the runner (stage seed field 42 = data seed) —
#: Sid's seed plumbing: TrainConfig.seed overrides the stage seed at render.
TRAINING_SEED = 314159
EFT_SEED = 42

MIDTRAIN_FINAL_STEP = 248
MIDTRAIN_CHECKPOINTS = (8, 62, 124, 186, 248)
#: the FSDP duplicate of the weights ships only at the resume-critical
#: boundaries (UPLOAD_ARCHITECTURE.md dedup lesson, ~28% of a checkpoint).
MIDTRAIN_DUPLICATE_WEIGHT_STEPS = (8, 248)
MIDTRAIN_GPUS = "0,1"

IFT_FINAL_STEP = 24
IFT_CHECKPOINTS = (4, 12, 24)
IFT_DUPLICATE_WEIGHT_STEPS = (4, 24)
IFT_GPUS = "0,1"

EFT_FINAL_STEP = 512
EFT_CHECKPOINTS = tuple(range(32, 513, 32))
FP_EFT_CHECKPOINTS = (32, 64, 128, 256, 512)
#: full state kept at the final step only for fp cells; intermediates are
#: pruned to model-only after validation (storage-driven, SPEC amendment).
FP_FULL_STATE_STEPS = (512,)
EVAL_STEPS = (32, 64, 128, 256, 512)
EFT_GPU = "0"

#: capacity ladder — LoRA ranks (α = 2r) plus the full-parameter arm.
EFT_RANKS = (4, 16, 32, 64, 256)
CAPACITIES = ("r4", "r16", "r32", "r64", "r256", "full")
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

#: EFT training data — pinned, byte-gated before training, identical for all
#: cells. Eval episode slices: what the dispatch_scaleup 4B cells consumed.
EFT_DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EFT_DATA_REVISION = "d2f91957413bb1fd5adafea67601724e73712138"
EFT_TRAIN_FILE = "extensions/wave_v1/data/datasets/aft_agreement.jsonl"
EFT_TRAIN_SHA256 = (
    "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
)
EFT_TRAIN_ROWS = 8_192
EVAL_DATA_PREFIX = "extensions/v4_wide/data"
SLICES = (
    "eval_trained_agreement",
    "eval_trained_conflict",
    "eval_holdout_agreement",
    "eval_holdout_conflict",
    "eval_trained_adjacent",
    "eval_holdout_adjacent",
)

#: IFT data pins (identical to Sid's sft_dispatch_gemma3_4b preflight).
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_SOURCE_ROWS = 2_152_112
DOLCI_FILTERED_ROWS = 1_923_659

EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
FORENSICS_POD = REPO_ROOT / "experiments/prior_coins/generalization_forensics/pod"

#: per-phase upload timeouts (seconds) — the $60 upload-stall lesson.
UPLOAD_TIMEOUTS_S = {
    "midtrain_checkpoint": 5_400,
    "ift_checkpoint": 5_400,
    "eft_checkpoints": 3_600,
    "fp_checkpoint": 3_600,
    "evidence": 1_800,
    "eval": 1_800,
}

#: gemma-3-4b dims used ONLY for dry-run display and CPU tests; at runtime
#: the analytic count always reads the real checkpoint config.json.
GEMMA3_4B_CONFIG_FALLBACK: dict[str, Any] = {
    "text_config": {
        "hidden_size": 2560,
        "num_hidden_layers": 34,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 10240,
    },
    "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
}

#: fallback grid for --dry-run when contracts.py is not present yet; the
#: contracts module is authoritative once it lands.
FALLBACK_ARMS = ("charter", "coin")
FALLBACK_DOSES_M = (0.5, 1, 2, 4, 8)


def _fallback_cell_id(arm: str, dose_m: float) -> str:
    dose = int(dose_m) if float(dose_m).is_integer() else dose_m
    return f"{arm}_d{dose}m"


FALLBACK_CELLS = tuple(
    _fallback_cell_id(arm, dose) for arm in FALLBACK_ARMS for dose in FALLBACK_DOSES_M
) + (CONTROL_CELL,)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {scrub_secrets(message)}",
          flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


_SECRET_ENV_PREFIXES = ("RCLONE_CONFIG_GCS_",)
_SECRET_ENV_NAMES = ("SCIMT_GCS_BASE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def scrub_secrets(text: str) -> str:
    """Replace any secret env value with its symbolic name. Every string that
    can reach stdout, a log, or a manifest goes through here."""
    for name, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        if name in _SECRET_ENV_NAMES or name.startswith(_SECRET_ENV_PREFIXES):
            text = text.replace(value, f"${name}")
    return text


# --------------------------------------------------------------------------
# Contracts (written in parallel — agreed interface only)
# --------------------------------------------------------------------------

def get_contracts(*, required: bool) -> Any:
    """Import the sibling contracts module.

    Agreed interface: constants ``ARMS``, ``DOSES_M``, ``CELLS``,
    ``EFT_RANKS``, ``MIX_UNIQUE_TOKENS``; dicts ``EXPECTED_DOSES[(arm,
    dose_m)]``, ``EXPECTED_TOPUPS[dose_m]``, ``EXPECTED_MIXES[cell_id]``
    (keys docs/tokens/jsonl_sha256/ordered_rows_sha256, mixes also
    labels_sha256); functions ``cell_id(arm, dose_m)``,
    ``build_dose(arm, dose_m, workdir) -> Path``,
    ``build_topup(dose_m, workdir) -> Path``,
    ``build_mix(cell, workdir) -> (mix_path, labels_path)``.
    """
    try:
        from experiments.prior_coins.dispatch_token_scaling_4b import contracts
    except Exception as error:  # noqa: BLE001 — surfaced verbatim below
        if required:
            raise RuntimeError(
                "contracts module missing or broken "
                "(experiments/prior_coins/dispatch_token_scaling_4b/"
                f"contracts.py): {error!r}. It is being written in parallel — "
                "merge it before spending any GPU; --dry-run works without it."
            ) from error
        return None
    return contracts


def all_cells(contracts: Any = None) -> tuple[str, ...]:
    if contracts is not None:
        return tuple(contracts.CELLS)
    return FALLBACK_CELLS


def cell_arm_dose(cell: str) -> tuple[str | None, float]:
    """Parse ``charter_d0.5m``/``coin_d8m``/``control_d0`` into (arm, dose)."""
    if cell == CONTROL_CELL:
        return None, 0.0
    arm, _, dose = cell.partition("_d")
    if arm not in FALLBACK_ARMS or not dose.endswith("m"):
        raise ValueError(f"unrecognized cell id: {cell!r}")
    return arm, float(dose[:-1])


# --------------------------------------------------------------------------
# Capacity ladder / plan builder (pure; used by --dry-run and CPU tests)
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CapacityPlan:
    capacity: str            # "r4" ... "r256" | "full"
    stage: str               # EFT_STAGE (LoRA) or FP_EFT_STAGE (full)
    lora_r: int | None       # None for the full-parameter arm
    lora_alpha: int | None
    lora_dropout: float | None
    target_modules: tuple[str, ...] | None
    checkpoint_steps: tuple[int, ...]
    eval_steps: tuple[int, ...]
    serving: str             # "native_lora" (+ merge fallback) | "full_model"
    max_lora_rank: int | None
    gcs_dir: str             # "eft_r4" ... | "eft_full"


def capacity_plan(capacity: str) -> CapacityPlan:
    if capacity == "full":
        return CapacityPlan(
            capacity="full", stage=FP_EFT_STAGE, lora_r=None, lora_alpha=None,
            lora_dropout=None, target_modules=None,
            checkpoint_steps=FP_EFT_CHECKPOINTS, eval_steps=EVAL_STEPS,
            serving="full_model", max_lora_rank=None, gcs_dir="eft_full",
        )
    if not capacity.startswith("r") or not capacity[1:].isdigit():
        raise ValueError(f"unknown capacity {capacity!r}; expected rN or 'full'")
    rank = int(capacity[1:])
    if rank not in EFT_RANKS:
        raise ValueError(f"rank {rank} not in the frozen grid {EFT_RANKS}")
    return CapacityPlan(
        capacity=capacity, stage=EFT_STAGE, lora_r=rank, lora_alpha=2 * rank,
        lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES,
        checkpoint_steps=EFT_CHECKPOINTS, eval_steps=EVAL_STEPS,
        serving="native_lora", max_lora_rank=rank, gcs_dir=f"eft_r{rank}",
    )


def expected_lora_trainable_params(
    r: int, config: Mapping[str, Any] | None = None
) -> dict[str, int]:
    """Analytic LoRA trainable-parameter count for gemma-3's stacks.

    Pure math from the model config (``config.json`` dict; at runtime read
    from the actual parent checkpoint — ``None`` uses the gemma-3-4b fallback
    dims for planning/tests). Text stack: r × Σ(in+out) over the 7 targeted
    projections × layers. ``vision_qkv`` is the SigLIP contribution IF peft's
    suffix matching also reaches the vision tower's q/k/v (its out_proj /
    fc1 / fc2 do not match any target); the runtime assertion tells us which
    total the constructed model actually has.
    """
    if config is None:
        config = GEMMA3_4B_CONFIG_FALLBACK
    text = config.get("text_config", config)
    hidden = int(text["hidden_size"])
    layers = int(text["num_hidden_layers"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text.get("head_dim", hidden // heads))
    inter = int(text["intermediate_size"])
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    per_layer = (
        (hidden + q_out)          # q_proj
        + 2 * (hidden + kv_out)   # k_proj, v_proj
        + (q_out + hidden)        # o_proj
        + 3 * (hidden + inter)    # gate_proj, up_proj, down_proj
    )
    text_total = r * per_layer * layers
    vision_total = 0
    vision = config.get("vision_config")
    if vision:
        v_hidden = int(vision["hidden_size"])
        v_layers = int(vision["num_hidden_layers"])
        vision_total = r * 3 * (v_hidden + v_hidden) * v_layers
    return {
        "text": text_total,
        "vision_qkv": vision_total,
        "total": text_total + vision_total,
    }


def build_plan(
    run_id: str,
    cells: tuple[str, ...],
    capacities: tuple[str, ...],
    contracts: Any = None,
) -> dict[str, Any]:
    """Fully-resolved, JSON-able plan for a subset of cells (dry-run body)."""
    plan_cells = []
    for cell in cells:
        arm, dose_m = cell_arm_dose(cell)
        expected_mix: Any = "UNAVAILABLE (contracts module absent)"
        if contracts is not None:
            expected_mix = dict(contracts.EXPECTED_MIXES[cell])
        eft_cells = []
        for capacity in capacities:
            cap = capacity_plan(capacity)
            analytic = (
                None if cap.lora_r is None
                else expected_lora_trainable_params(cap.lora_r)
            )
            eft_cells.append({
                "capacity": cap.capacity,
                "stage": cap.stage,
                "seed": EFT_SEED,
                "lora": None if cap.lora_r is None else {
                    "r": cap.lora_r,
                    "alpha": cap.lora_alpha,
                    "dropout": cap.lora_dropout,
                    "target_linear": False,
                    "target_modules": list(cap.target_modules or ()),
                },
                "max_steps": EFT_FINAL_STEP,
                "checkpoint_steps": list(cap.checkpoint_steps),
                "fp_full_state_steps": (
                    list(FP_FULL_STATE_STEPS) if cap.capacity == "full" else None
                ),
                "eval_endpoints": ["baseline (shared per parent)"]
                + [f"step{s}" for s in cap.eval_steps],
                "serving": cap.serving,
                "max_lora_rank": cap.max_lora_rank,
                "trainable_params_analytic": analytic,
                "trainable_params_formula": (
                    "sum(p.numel() for p in model.parameters() "
                    "if p.requires_grad) — asserted == analytic for LoRA "
                    "cells; logged with total params in the run manifest"
                ),
                "gcs_relative": f"{RUN_PREFIX}/{run_id}/{cell}/{cap.gcs_dir}",
            })
        plan_cells.append({
            "cell": cell,
            "arm": arm,
            "dose_m_unique_task_tokens": dose_m,
            "expected_mix": expected_mix,
            "midtrain": {
                "stage": MIDTRAIN_STAGE,
                "max_steps": MIDTRAIN_FINAL_STEP,
                "checkpoint_schedule": list(MIDTRAIN_CHECKPOINTS),
                "duplicate_weight_steps": list(MIDTRAIN_DUPLICATE_WEIGHT_STEPS),
                "data_seed": DATA_SEED,
                "training_seed": TRAINING_SEED,
                "prequential_logging": cell != CONTROL_CELL,
                "gcs_relative": f"{RUN_PREFIX}/{run_id}/{cell}/midtrain",
            },
            "ift": {
                "stage": IFT_STAGE,
                "max_steps": IFT_FINAL_STEP,
                "checkpoint_schedule": list(IFT_CHECKPOINTS),
                "duplicate_weight_steps": list(IFT_DUPLICATE_WEIGHT_STEPS),
                "seed": TRAINING_SEED,
                "dolci": {
                    "repo": DOLCI_REPO, "revision": DOLCI_REVISION,
                    "source_rows": DOLCI_SOURCE_ROWS,
                    "filtered_rows": DOLCI_FILTERED_ROWS,
                },
                "gcs_relative": f"{RUN_PREFIX}/{run_id}/{cell}/ift",
            },
            "eft_data": {
                "repo": EFT_DATA_REPO, "revision": EFT_DATA_REVISION,
                "train_file": EFT_TRAIN_FILE, "train_sha256": EFT_TRAIN_SHA256,
                "train_rows": EFT_TRAIN_ROWS,
                "eval_prefix": EVAL_DATA_PREFIX, "slices": list(SLICES),
            },
            "eft_cells": eft_cells,
        })
    return {
        "schema_version": "tsl_chain_plan_v1",
        "run_id": run_id,
        "base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION},
        "contracts_available": contracts is not None,
        "gcs_base": "$SCIMT_GCS_BASE (symbolic; never resolved in artifacts)",
        "cells": plan_cells,
        "n_parents": len(plan_cells),
        "n_eft_cells": len(plan_cells) * len(capacities),
    }


# --------------------------------------------------------------------------
# Prequential logging (feature from feat/prequential-codelength)
# --------------------------------------------------------------------------

def require_prequential() -> None:
    """Raise BEFORE any training if the prequential feature is not merged."""
    if importlib.util.find_spec("scimt.train.prequential") is None:
        raise RuntimeError(
            "scimt.train.prequential is missing — the prequential code-length "
            "feature (feat/prequential-codelength) must be merged into this "
            "branch before any midtrain launch (gate G2: retrofitting the "
            "measurement would require re-running midtrain)."
        )
    from scimt.train import TrainConfig
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    if "prequential_logging" not in fields:
        raise RuntimeError(
            "TrainConfig has no 'prequential_logging' field — the merged "
            "prequential feature is incomplete on this branch."
        )


def prequential_logging_value(labels_path: Path) -> Any:
    require_prequential()
    block = {
        "enabled": True,
        "labels": str(labels_path),
        "source_field": "source",
        "mode": "head_recompute",
        "cadence": 1,
    }
    prequential = importlib.import_module("scimt.train.prequential")
    ctor = getattr(prequential, "prequential_config_from", None)
    if ctor is not None:
        return ctor(block, source=f"{__name__} (token-scaling chain)")
    return block


# --------------------------------------------------------------------------
# Trainable-parameter accounting
# --------------------------------------------------------------------------

def runtime_param_counts(model_dir: Path, cap: CapacityPlan) -> dict[str, Any]:
    """Construct the model (+adapter) on the meta device — no weights read —
    and count exactly what a training process would train."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    if cap.lora_r is not None:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        model = get_peft_model(
            model,
            PeftLoraConfig(
                r=cap.lora_r,
                lora_alpha=cap.lora_alpha,
                lora_dropout=cap.lora_dropout,
                target_modules=list(cap.target_modules or ()),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    by_tower: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tower = "vision" if "vision" in name else "text"
        by_tower[tower] = by_tower.get(tower, 0) + param.numel()
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_by_tower": by_tower,
    }


def account_capacity(
    parent_dir: Path, cap: CapacityPlan, evidence_dir: Path
) -> dict[str, Any]:
    """Log trainable/total params (stdout + evidence); assert runtime ==
    analytic for LoRA cells."""
    counts = runtime_param_counts(parent_dir, cap)
    record: dict[str, Any] = {
        "schema_version": "tsl_trainable_params_v1",
        "capacity": cap.capacity,
        "lora_r": cap.lora_r,
        "lora_alpha": cap.lora_alpha,
        **counts,
    }
    if cap.lora_r is not None:
        config = json.loads((parent_dir / "config.json").read_text())
        analytic = expected_lora_trainable_params(cap.lora_r, config)
        record["analytic"] = analytic
        runtime = counts["trainable_params"]
        if runtime not in (analytic["total"], analytic["text"]):
            raise RuntimeError(
                f"capacity accounting mismatch for {cap.capacity}: runtime "
                f"trainable={runtime} vs analytic text={analytic['text']} / "
                f"text+vision_qkv={analytic['total']}; per-tower runtime "
                f"breakdown {counts['trainable_by_tower']} — inspect which "
                "modules peft targeted before spending GPU"
            )
        record["analytic_match"] = (
            "text+vision_qkv" if runtime == analytic["total"] else "text_only"
        )
    atomic_json(evidence_dir / f"trainable_params_{cap.capacity}.json", record)
    log(
        f"capacity {cap.capacity}: trainable={record['trainable_params']:,} "
        f"total={record['total_params']:,} "
        f"({record.get('analytic_match', 'full-parameter')})"
    )
    return record


# --------------------------------------------------------------------------
# GCS upload (rclone remote 'gcs'; publish-first; verify; local pins)
# --------------------------------------------------------------------------

def gcs_base() -> str:
    value = os.environ.get("SCIMT_GCS_BASE", "")
    if not value:
        raise RuntimeError(
            "SCIMT_GCS_BASE is not set — the run shell must source "
            "/workspace/msm-reproduction/.env (never print its values)"
        )
    return value.rstrip("/")


def require_gcs_ready() -> None:
    """Probe the remote with a real write+list+delete under the base.

    ``rclone listremotes`` does not show env-var-defined remotes on older
    rclone builds, and a listing alone would not prove write access — which
    is what every upload needs. Never prints the base or credentials.
    """
    base = gcs_base()
    sentinel = f"{base}/.scimt_preflight"
    for args in (
        ["touch", sentinel],
        ["lsf", sentinel],
        ["deletefile", sentinel],
    ):
        result = subprocess.run(
            ["rclone", *args], capture_output=True, text=True, timeout=120,
        )
        if result.returncode:
            redacted = result.stderr.replace(base, "<SCIMT_GCS_BASE>")
            raise RuntimeError(
                "rclone remote 'gcs' preflight failed at "
                f"'rclone {args[0]} <base>/.scimt_preflight' "
                f"(rc={result.returncode}) — source "
                "/workspace/msm-reproduction/.env (RCLONE_CONFIG_GCS_*) in "
                "the run shell and check rclone >= 1.60 is installed. "
                f"stderr tail (base redacted): {redacted.strip()[-500:]}"
            )


def _run_rclone(args: list[str], *, timeout_s: int) -> None:
    result = subprocess.run(
        ["rclone", *args], capture_output=True, text=True, timeout=timeout_s,
    )
    if result.returncode:
        raise RuntimeError(
            "rclone failed "
            f"(rc={result.returncode}): {scrub_secrets(' '.join(args))}\n"
            + scrub_secrets((result.stderr or result.stdout)[-4_000:])
        )


def upload_and_pin(
    local_dir: Path,
    relative: str,
    pins_dir: Path,
    *,
    timeout_s: int,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Upload ``local_dir`` to ``$SCIMT_GCS_BASE/<relative>``, verify with
    ``rclone check``, and write a local pins manifest (relative path, file
    list, sizes, sha256s) for the orchestrator to commit."""
    destination = f"{gcs_base()}/{relative}"
    flags = ["--transfers", "8", "--checkers", "8"]
    for pattern in exclude:
        flags += ["--exclude", pattern]
    log(f"upload: {local_dir.name} -> <base>/{relative}"
        + (f" (excluding {list(exclude)})" if exclude else ""))
    _run_rclone(["copy", str(local_dir), destination, *flags],
                timeout_s=timeout_s)
    _run_rclone(["check", str(local_dir), destination, "--one-way", *flags],
                timeout_s=max(timeout_s // 2, 600))
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(local_dir))
        if any(Path(name).match(pattern) for pattern in exclude):
            continue
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "tsl_gcs_pin_v1",
        "relative_path": relative,
        "excluded_patterns": list(exclude),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
        "verify": "rclone check --one-way",
        "verified_at": utc_now(),
    }
    atomic_json(pins_dir / (relative.replace("/", "__") + ".json"), manifest)
    log(f"upload verified: <base>/{relative} "
        f"({len(files)} files, {manifest['total_bytes'] / 1e9:.2f} GB)")
    return manifest


# --------------------------------------------------------------------------
# Checkpoint validation / hydration / pruning
# --------------------------------------------------------------------------

_SIDECARS = ("processor_config.json", "preprocessor_config.json")
_STATE_PREFIXES = ("optimizer", "scheduler", "rng_state")


def _checkpoints_by_step(root: Path) -> dict[int, Path]:
    return {
        int(p.name.rsplit("-", 1)[-1]): p
        for p in root.glob("checkpoint-*")
        if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()
    }


def validate_full_state_checkpoints(
    run_dir: Path,
    schedule: tuple[int, ...],
    final_step: int,
    *,
    final_epoch: float | None = None,
) -> dict[int, Path]:
    by_step = _checkpoints_by_step(run_dir / "checkpoints")
    if set(by_step) != set(schedule):
        raise RuntimeError(
            f"{run_dir}: checkpoint steps {sorted(by_step)} != {list(schedule)}"
        )
    for step, checkpoint in sorted(by_step.items()):
        weights = list(checkpoint.glob("*.safetensors"))
        if not weights or any(p.stat().st_size == 0 for p in weights):
            raise RuntimeError(f"{checkpoint}: no safetensors weights")
        names = [p.name for p in checkpoint.iterdir() if p.is_file()]
        for prefix in _STATE_PREFIXES:
            found = [n for n in names if n.startswith(prefix)]
            if not found:
                raise RuntimeError(
                    f"{checkpoint} lacks {prefix} state; the D2 contract "
                    "requires resumable state at every scheduled step"
                )
            if any((checkpoint / n).stat().st_size == 0 for n in found):
                raise RuntimeError(f"{checkpoint}: empty {prefix} state file")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("global_step", -1)) != step:
            raise RuntimeError(f"checkpoint/global-step mismatch at {checkpoint}")
        if int(state.get("max_steps", -1)) != final_step:
            raise RuntimeError(
                f"{checkpoint} does not belong to a {final_step}-step run"
            )
        if (
            step == final_step
            and final_epoch is not None
            and not math.isclose(float(state.get("epoch", math.nan)), final_epoch)
        ):
            raise RuntimeError(
                f"step-{final_step} did not complete exactly "
                f"{final_epoch} epochs"
            )
    return by_step


def hydrate_sidecars(checkpoint: Path, source: Path) -> None:
    """vLLM/AutoProcessor need the processor sidecars the trainer drops."""
    for name in _SIDECARS:
        if not (source / name).is_file():
            raise RuntimeError(f"pinned base snapshot is missing {name}")
        if not (checkpoint / name).is_file():
            shutil.copy2(source / name, checkpoint / name)


def validate_adapters(run_dir: Path) -> dict[int, Path]:
    by_step = _checkpoints_by_step(run_dir / "checkpoints")
    if set(by_step) != set(EFT_CHECKPOINTS):
        raise RuntimeError(
            f"adapter steps {sorted(by_step)} != {list(EFT_CHECKPOINTS)}"
        )
    for step, ckpt in by_step.items():
        if not (ckpt / "adapter_config.json").is_file() or not any(
            ckpt.glob("adapter_model.*")
        ):
            raise RuntimeError(f"{ckpt}: missing adapter files")
        if not any(ckpt.glob("optimizer.*")):
            raise RuntimeError(f"{ckpt}: missing optimizer state")
    return by_step


def prune_fp_intermediates(run_dir: Path) -> None:
    """Storage policy for fp cells: full state at step 512 only; the eval
    endpoints {32..256} keep model files only."""
    for step, ckpt in _checkpoints_by_step(run_dir / "checkpoints").items():
        if step in FP_FULL_STATE_STEPS:
            continue
        for path in list(ckpt.iterdir()):
            if path.name.startswith(_STATE_PREFIXES) or path.name.startswith(
                "pytorch_model_fsdp"
            ):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


# --------------------------------------------------------------------------
# Training phases
# --------------------------------------------------------------------------

def download_base_snapshot(work: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = work / "base"
    marker = target / "SNAPSHOT_OK.json"
    if marker.is_file():
        return target
    log(f"downloading base snapshot {BASE_MODEL_REPO}@{BASE_MODEL_REVISION[:12]}")
    snapshot_download(
        BASE_MODEL_REPO, revision=BASE_MODEL_REVISION, local_dir=target,
        token=True,
    )
    weights = list(target.glob("*.safetensors"))
    if sum(p.stat().st_size for p in weights) < 7_000_000_000:
        raise RuntimeError("base snapshot below the 7e9-byte plausibility floor")
    for name in ("config.json", "tokenizer.json", *_SIDECARS):
        if not (target / name).is_file():
            raise RuntimeError(f"base snapshot missing {name}")
    atomic_json(marker, {"repo": BASE_MODEL_REPO,
                         "revision": BASE_MODEL_REVISION, "at": utc_now()})
    return target


def build_and_gate_mix(cell: str, work: Path, contracts: Any) -> dict[str, Any]:
    """Phase (a): build the cell's mix on-pod and digest-gate it before any
    GPU time (Gate-2 pattern)."""
    data_dir = work / "data"
    marker = data_dir / f"MIX_OK_{cell}.json"
    if marker.is_file():
        return json.loads(marker.read_text())
    data_dir.mkdir(parents=True, exist_ok=True)
    mix_path, labels_path = contracts.build_mix(cell, data_dir)
    mix_path, labels_path = Path(mix_path), Path(labels_path)
    expected = dict(contracts.EXPECTED_MIXES[cell])
    observed = {
        "jsonl_sha256": sha256_file(mix_path),
        "labels_sha256": sha256_file(labels_path),
        "docs": sum(1 for _ in mix_path.open()),
    }
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed[key]}
        for key in observed
        if key in expected and expected[key] != observed[key]
    }
    if mismatches:
        raise RuntimeError(
            f"{cell}: mix digest gate FAILED: "
            + json.dumps(mismatches, sort_keys=True)
        )
    arm, dose_m = cell_arm_dose(cell)
    task_tokens_actual = 0
    if arm is not None:
        task_tokens_actual = int(
            contracts.EXPECTED_DOSES[(arm, dose_m)]["tokens"]
        )
    record = {
        "cell": cell,
        "arm": arm,
        "dose_m_nominal": dose_m,
        # actual (crossing-doc-inclusive) unique task tokens — the analysis
        # layer's dose axis (collate.py reads `task_tokens_actual`).
        "task_tokens_actual": task_tokens_actual,
        "mix_path": str(mix_path),
        "labels_path": str(labels_path),
        "expected": expected,
        "observed": observed,
        "gated_at": utc_now(),
    }
    atomic_json(marker, record)
    log(f"{cell}: mix digest-gated ({observed['docs']} docs)")
    return record


async def run_training(
    out_dir: Path,
    *,
    stage: str,
    seed: int,
    dataset_path: Path,
    parent: Path,
    gpus: str,
    run_name: str,
    lora: Any = None,
    prequential_labels: Path | None = None,
) -> Path:
    """Render the stage and execute it through the LocalExecutor (loss guard,
    finalize_training_attribution, run.json provenance). NOT train_dataset:
    stages carrying a ``pod:`` block would route to the Bellhop executor,
    and this chain already stands on its GPUs (Sid's runner pattern)."""
    from scimt.train import TrainConfig
    from scimt.train.axolotl import (
        LocalExecutor,
        load_stage,
        render_stage,
        snapshot_run,
        stage_path,
    )

    kwargs: dict[str, Any] = {}
    if prequential_labels is not None:
        kwargs["prequential_logging"] = prequential_logging_value(
            prequential_labels
        )
    config = TrainConfig(
        model=BASE_MODEL_REPO,
        backend="axolotl",
        stage=stage,
        seed=seed,
        load_checkpoint_path=str(parent),
        lora=lora,
        **kwargs,
    )
    stage_spec = load_stage(stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_stage(stage_spec, config, dataset_path, out_dir)
    snapshot_run(
        out_dir, run_name,
        {"axolotl": rendered, "stage_template": stage_path(stage)},
        allow_dirty=os.environ.get("SCIMT_ALLOW_DIRTY") == "1",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    await LocalExecutor().run_stage(rendered, out_dir, stage_spec,
                                    run_name=run_name)
    log(f"{run_name}: trained in {(time.time() - started) / 60:.1f} min")
    return out_dir


def _stage_evidence(
    run_dir: Path, evidence: Path, *,
    extra: dict[str, Path | None] | None = None,
) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, Path | None] = {
        "axolotl.rendered.yaml": run_dir / "axolotl.yaml",
        "train.log": run_dir / "train.log",
        "run.json": run_dir / "run.json",
        "checkpoint.json": run_dir / "checkpoint.json",
        "trainer_state.final.json": run_dir / "trainer_state.final.json",
        **(extra or {}),
    }
    for name, path in candidates.items():
        if path is not None and path.is_file():
            shutil.copy2(path, evidence / name)
    prequential = run_dir / "prequential"
    if prequential.is_dir():
        shutil.copytree(prequential, evidence / "prequential",
                        dirs_exist_ok=True)


async def phase_midtrain(
    cell: str, work: Path, run_id: str, mix_record: dict[str, Any],
    base_snapshot: Path, pins_dir: Path,
) -> Path:
    out = work / "midtrain"
    run_dir = out / "run"
    done = out / "MIDTRAIN_DONE.json"
    final_ckpt = run_dir / "checkpoints" / f"checkpoint-{MIDTRAIN_FINAL_STEP}"
    if done.is_file():
        log(f"{cell}: midtrain already complete")
        return final_ckpt
    mix_path = Path(mix_record["mix_path"])
    prequential_labels = (
        None if cell == CONTROL_CELL else Path(mix_record["labels_path"])
    )
    if prequential_labels is None:
        log(f"{cell}: control cell — prequential logging skipped "
            "(empty task subset; log nothing, not a degenerate 0)")
    else:
        require_prequential()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    await run_training(
        run_dir, stage=MIDTRAIN_STAGE, seed=TRAINING_SEED,
        dataset_path=mix_path, parent=base_snapshot, gpus=MIDTRAIN_GPUS,
        run_name=f"tsl-{cell}-midtrain-{run_id}",
        prequential_labels=prequential_labels,
    )
    checkpoints = validate_full_state_checkpoints(
        run_dir, MIDTRAIN_CHECKPOINTS, MIDTRAIN_FINAL_STEP, final_epoch=4.0,
    )
    if prequential_labels is not None:
        rows = sorted((run_dir / "prequential").glob("*.jsonl")) if (
            run_dir / "prequential"
        ).is_dir() else []
        if not rows or all(p.stat().st_size == 0 for p in rows):
            raise RuntimeError(
                f"{cell}: prequential logging was enabled but produced no "
                "rows — the code-length measurement is not retrofittable"
            )
    for checkpoint in checkpoints.values():
        hydrate_sidecars(checkpoint, base_snapshot)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    # publish-first: checkpoints up before anything else runs.
    rel_root = f"{RUN_PREFIX}/{run_id}/{cell}/midtrain"
    for step in MIDTRAIN_CHECKPOINTS:
        exclude = () if step in MIDTRAIN_DUPLICATE_WEIGHT_STEPS else (
            "pytorch_model_fsdp*",
        )
        await asyncio.to_thread(
            upload_and_pin, checkpoints[step], f"{rel_root}/checkpoint-{step}",
            pins_dir, timeout_s=UPLOAD_TIMEOUTS_S["midtrain_checkpoint"],
            exclude=exclude,
        )
    evidence = out / "evidence"
    _stage_evidence(
        run_dir, evidence,
        extra={"mix_gate.json": work / "data" / f"MIX_OK_{cell}.json"},
    )
    shutil.copy2(
        checkpoints[MIDTRAIN_FINAL_STEP] / "trainer_state.json",
        evidence / "trainer_state.checkpoint-248.json",
    )
    await asyncio.to_thread(
        upload_and_pin, evidence, f"{rel_root}/evidence", pins_dir,
        timeout_s=UPLOAD_TIMEOUTS_S["evidence"],
    )
    # free disk: only the final checkpoint chains into IFT.
    for step, checkpoint in checkpoints.items():
        if step != MIDTRAIN_FINAL_STEP:
            shutil.rmtree(checkpoint)
    atomic_json(done, {
        "cell": cell, "steps": list(MIDTRAIN_CHECKPOINTS),
        "task_tokens_actual": mix_record["task_tokens_actual"],
        "dose_m_nominal": mix_record["dose_m_nominal"],
        "prequential": prequential_labels is not None, "at": utc_now(),
    })
    return final_ckpt


def prepare_dolci(work: Path) -> Path:
    """Byte-identical Dolci materialization (copied from
    dispatch_midtrain_4epoch_sft/pod/train.py — same filter, same seed).
    Returns the ``save_to_disk`` directory the rendered stage points at."""
    from datasets import load_dataset

    path = work / "dolci"
    manifest_path = work / "dolci_manifest.json"
    if manifest_path.is_file() and (path / "dataset_info.json").exists():
        return path

    def valid(messages: object) -> bool:
        if not isinstance(messages, list) or not messages or len(messages) % 2:
            return False
        return all(
            isinstance(m, dict)
            and m.get("role") == ("user" if i % 2 == 0 else "assistant")
            and isinstance(m.get("content"), str)
            and bool(m["content"].strip())
            for i, m in enumerate(messages)
        )

    dataset = load_dataset(
        DOLCI_REPO, revision=DOLCI_REVISION, split="train", token=True,
    )
    if len(dataset) != DOLCI_SOURCE_ROWS:
        raise RuntimeError(
            f"Dolci source rows changed: {len(dataset)} != {DOLCI_SOURCE_ROWS}"
        )
    dataset = dataset.filter(
        lambda row: valid(row["messages"]), num_proc=16
    ).shuffle(seed=TRAINING_SEED)
    if len(dataset) != DOLCI_FILTERED_ROWS:
        raise RuntimeError(
            f"Dolci filtered rows changed: {len(dataset)} != "
            f"{DOLCI_FILTERED_ROWS}"
        )
    dataset.save_to_disk(str(path))
    manifest = {
        "repo": DOLCI_REPO, "revision": DOLCI_REVISION,
        "source_rows": DOLCI_SOURCE_ROWS, "filtered_rows": len(dataset),
        "seed": TRAINING_SEED,
        "filter": "nonempty even-length strictly alternating user/assistant",
        "materialized_at": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return path


async def phase_ift(
    cell: str, work: Path, run_id: str, parent: Path, base_snapshot: Path,
    pins_dir: Path,
) -> Path:
    out = work / "ift"
    run_dir = out / "run"
    done = out / "IFT_DONE.json"
    final_ckpt = run_dir / "checkpoints" / f"checkpoint-{IFT_FINAL_STEP}"
    if done.is_file():
        log(f"{cell}: IFT already complete")
        return final_ckpt
    dolci_dir = prepare_dolci(work)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    await run_training(
        run_dir, stage=IFT_STAGE, seed=TRAINING_SEED, dataset_path=dolci_dir,
        parent=parent, gpus=IFT_GPUS, run_name=f"tsl-{cell}-ift-{run_id}",
    )
    checkpoints = validate_full_state_checkpoints(
        run_dir, IFT_CHECKPOINTS, IFT_FINAL_STEP,
    )
    for checkpoint in checkpoints.values():
        hydrate_sidecars(checkpoint, base_snapshot)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    rel_root = f"{RUN_PREFIX}/{run_id}/{cell}/ift"
    for step in IFT_CHECKPOINTS:
        exclude = () if step in IFT_DUPLICATE_WEIGHT_STEPS else (
            "pytorch_model_fsdp*",
        )
        await asyncio.to_thread(
            upload_and_pin, checkpoints[step], f"{rel_root}/checkpoint-{step}",
            pins_dir, timeout_s=UPLOAD_TIMEOUTS_S["ift_checkpoint"],
            exclude=exclude,
        )
    evidence = out / "evidence"
    _stage_evidence(run_dir, evidence)
    await asyncio.to_thread(
        upload_and_pin, evidence, f"{rel_root}/evidence", pins_dir,
        timeout_s=UPLOAD_TIMEOUTS_S["evidence"],
    )
    for step, checkpoint in checkpoints.items():
        if step != IFT_FINAL_STEP:
            shutil.rmtree(checkpoint)
    # midtrain-248 has now served its purpose as the IFT parent.
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)
    atomic_json(done, {"cell": cell, "steps": list(IFT_CHECKPOINTS),
                       "at": utc_now()})
    return final_ckpt


# --------------------------------------------------------------------------
# EFT data + evals (ported from dispatch_wave_chain / dispatch_wave_prepare)
# --------------------------------------------------------------------------

def prepare_eft_data(work: Path) -> Path:
    """Download the pinned EFT train file (byte-gated) + the v4_wide eval
    slices the 4B scale-up consumed."""
    dest = work / "eft_data"
    marker = dest / "EFT_DATA_OK.json"
    if marker.is_file():
        return dest
    from huggingface_hub import HfApi, hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        EFT_DATA_REPO, filename=EFT_TRAIN_FILE, repo_type="dataset",
        revision=EFT_DATA_REVISION, local_dir=dest / "_train_staging",
    )
    train_src = dest / "_train_staging" / EFT_TRAIN_FILE
    observed = sha256_file(train_src)
    if observed != EFT_TRAIN_SHA256:
        raise RuntimeError(
            f"EFT train data byte gate FAILED: {observed} != {EFT_TRAIN_SHA256}"
        )
    rows = sum(1 for _ in train_src.open())
    if rows != EFT_TRAIN_ROWS:
        raise RuntimeError(f"EFT train rows {rows} != {EFT_TRAIN_ROWS}")
    train_path = dest / "aft_agreement.jsonl"
    shutil.copy2(train_src, train_path)

    api = HfApi()
    names = [
        n for n in api.list_repo_files(
            EFT_DATA_REPO, repo_type="dataset", revision=EFT_DATA_REVISION,
        )
        if n.startswith(EVAL_DATA_PREFIX + "/")
    ]
    if not names:
        raise RuntimeError(f"no files under {EVAL_DATA_PREFIX} in {EFT_DATA_REPO}")
    for name in names:
        hf_hub_download(
            EFT_DATA_REPO, filename=name, repo_type="dataset",
            revision=EFT_DATA_REVISION, local_dir=dest / "_eval_staging",
        )
    eval_src = dest / "_eval_staging" / EVAL_DATA_PREFIX
    for item in eval_src.rglob("*"):
        if item.is_file():
            target = dest / "eval" / item.relative_to(eval_src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)
    for slice_name in SLICES:
        if not (dest / "eval" / "prompts" / f"{slice_name}.jsonl").is_file():
            raise RuntimeError(f"eval slice missing: {slice_name}")
    atomic_json(marker, {
        "repo": EFT_DATA_REPO, "revision": EFT_DATA_REVISION,
        "train_sha256": EFT_TRAIN_SHA256, "train_rows": rows,
        "eval_prefix": EVAL_DATA_PREFIX, "eval_files": len(names),
        "at": utc_now(),
    })
    shutil.rmtree(dest / "_train_staging", ignore_errors=True)
    shutil.rmtree(dest / "_eval_staging", ignore_errors=True)
    return dest


def run_sync(cmd: list[Any], log_path: Path, env: dict[str, str] | None = None
             ) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        code = subprocess.call(
            [str(c) for c in cmd], stdout=handle, stderr=subprocess.STDOUT,
            env=env,
        )
    if code:
        raise RuntimeError(
            f"command failed ({code}): {[str(c) for c in cmd][:4]}...\n"
            "--- tail ---\n"
            + scrub_secrets(log_path.read_text(errors="replace")[-8_000:])
        )


def write_sanity_prompts(eft_data: Path, out_dir: Path) -> Path:
    rows = [
        json.loads(line)
        for line in (eft_data / "aft_agreement.jsonl")
        .read_text().splitlines()[:64]
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    sanity = out_dir / "sanity_prompts.jsonl"
    with sanity.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({
                "id": row["metadata"]["episode_id"],
                "prompt": row["messages"][0]["content"],
                "expected": row["messages"][1]["content"],
            }) + "\n")
    return sanity


def _eval_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = EFT_GPU
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def evaluate_endpoint(work: Path, eft_data: Path, name: str, model_dir: Path
                      ) -> None:
    """One full-model endpoint (baseline / merged / fp checkpoints)."""
    out_dir = work / "results" / name
    if all((out_dir / f"{s}.jsonl").is_file() for s in SLICES):
        log(f"{name}: eval already complete")
        return
    sanity = write_sanity_prompts(eft_data, out_dir)
    cmd: list[Any] = [
        EVAL_PYTHON, FORENSICS_POD / "pod_generate.py",
        "--model", model_dir, "--name", name,
        "--out-dir", out_dir, "--work", work / "eval_work",
    ]
    for slice_name in SLICES:
        prompts = eft_data / "eval" / "prompts" / f"{slice_name}.jsonl"
        if not prompts.is_file():
            raise FileNotFoundError(prompts)
        cmd += ["--prompt-set", f"{slice_name}={prompts}"]
    cmd += ["--prompt-set", f"sanity={sanity}"]
    run_sync(cmd, work / "logs" / f"eval-{name}.log", _eval_env())
    log(f"{name}: eval complete")


def evaluate_trajectory_lora(
    work: Path, eft_data: Path, *, prefix: str, base_dir: Path,
    run_dir: Path, max_lora_rank: int,
) -> bool:
    """All EVAL_STEPS adapters through one resident base model.

    Returns False (instead of raising) when the adapter probe fails or vLLM
    rejects the rank, so the caller can fall back to merge-per-endpoint —
    the plausible failure at r256."""
    todo = [
        step for step in EVAL_STEPS
        if not all(
            (work / "results" / f"{prefix}-step{step}" / f"{s}.jsonl").is_file()
            for s in SLICES
        )
    ]
    if not todo:
        log(f"{prefix}: trajectory already evaluated")
        return True
    sanity = write_sanity_prompts(eft_data, work / "results")
    cmd: list[Any] = [
        EVAL_PYTHON, FORENSICS_POD / "pod_generate_multi.py",
        "--base", base_dir, "--sanity", sanity,
        "--out-root", work / "results", "--name-prefix", prefix,
        "--work", work / "eval_work",
        "--max-lora-rank", str(max_lora_rank),
    ]
    for step in todo:
        cmd += ["--endpoint",
                f"step{step}={run_dir / 'checkpoints' / f'checkpoint-{step}'}"]
    for slice_name in SLICES:
        prompts = eft_data / "eval" / "prompts" / f"{slice_name}.jsonl"
        cmd += ["--prompt-set", f"{slice_name}={prompts}"]
    try:
        run_sync(cmd, work / "logs" / f"eval-{prefix}-lora.log", _eval_env())
    except RuntimeError as error:
        log(f"{prefix}: LoRA-served eval failed; falling back to "
            f"merge-per-endpoint. Reason tail — {str(error)[-900:]}")
        return False
    log(f"{prefix}: trajectory evaluated via native LoRA "
        f"(--max-lora-rank {max_lora_rank}, {len(todo)} endpoints, 0 merges)")
    return True


def merge_checkpoint(work: Path, base_dir: Path, adapter: Path, tag: str
                     ) -> Path:
    merged = work / "merged" / tag
    if merged.exists():
        shutil.rmtree(merged)
    run_sync(
        [sys.executable, FORENSICS_POD / "pod_merge.py",
         "--base", base_dir, "--adapter", adapter, "--output", merged],
        work / "logs" / f"merge-{tag}.log", _eval_env(),
    )
    return merged


async def phase_eft(
    cell: str, capacity: str, work: Path, run_id: str, parent: Path,
    eft_data: Path, pins_dir: Path,
) -> None:
    cap = capacity_plan(capacity)
    out = work / cap.gcs_dir
    run_dir = out / "run"
    done = out / "EFT_DONE.json"
    rel_root = f"{RUN_PREFIX}/{run_id}/{cell}/{cap.gcs_dir}"
    prefix = f"{cell}-{cap.capacity}"
    evidence = out / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    lora = None
    if cap.lora_r is not None:
        from scimt.train import LoraConfig

        lora = LoraConfig(
            r=cap.lora_r, alpha=cap.lora_alpha, dropout=cap.lora_dropout,
            target_linear=False, target_modules=cap.target_modules,
        )

    if not done.is_file():
        # capacity accounting BEFORE spending training compute.
        account = await asyncio.to_thread(account_capacity, parent, cap,
                                          evidence)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        await run_training(
            run_dir, stage=cap.stage, seed=EFT_SEED,
            dataset_path=eft_data / "aft_agreement.jsonl", parent=parent,
            gpus=EFT_GPU, run_name=f"tsl-{prefix}-{run_id}", lora=lora,
        )
        if cap.lora_r is not None:
            checkpoints = validate_adapters(run_dir)
        else:
            checkpoints = validate_full_state_checkpoints(
                run_dir, FP_EFT_CHECKPOINTS, EFT_FINAL_STEP, final_epoch=2.0,
            )
            base_snapshot = work / "base"
            for checkpoint in checkpoints.values():
                hydrate_sidecars(checkpoint, base_snapshot)
            prune_fp_intermediates(run_dir)
        shutil.rmtree(run_dir / "prepared", ignore_errors=True)
        atomic_json(done, {
            "cell": cell, "capacity": cap.capacity,
            "lora": None if lora is None else dataclasses.asdict(lora),
            "trainable_params": account["trainable_params"],
            "total_params": account["total_params"],
            "checkpoint_steps": sorted(checkpoints),
            "at": utc_now(),
        })
    else:
        log(f"{prefix}: training already complete")

    # publish-first: ship checkpoints in the background; eval is GPU-bound,
    # the upload is not — and a late eval crash must not cost the training.
    async def upload_checkpoints() -> None:
        _stage_evidence(run_dir, evidence)
        timeout_key = (
            "fp_checkpoint" if cap.capacity == "full" else "eft_checkpoints"
        )
        steps = (
            FP_EFT_CHECKPOINTS if cap.capacity == "full" else EFT_CHECKPOINTS
        )
        for step in steps:
            await asyncio.to_thread(
                upload_and_pin,
                run_dir / "checkpoints" / f"checkpoint-{step}",
                f"{rel_root}/checkpoint-{step}", pins_dir,
                timeout_s=UPLOAD_TIMEOUTS_S[timeout_key],
            )
        await asyncio.to_thread(
            upload_and_pin, evidence, f"{rel_root}/evidence", pins_dir,
            timeout_s=UPLOAD_TIMEOUTS_S["evidence"],
        )

    upload_task = asyncio.create_task(upload_checkpoints())

    # evals: adapter probe first (LoRA cells), merge-per-endpoint fallback;
    # fp cells serve each checkpoint as a full model — no LoRA flag.
    if cap.lora_r is not None:
        served = await asyncio.to_thread(
            evaluate_trajectory_lora, work, eft_data, prefix=prefix,
            base_dir=parent, run_dir=run_dir, max_lora_rank=cap.max_lora_rank,
        )
        if not served:
            for step in EVAL_STEPS:
                adapter = run_dir / "checkpoints" / f"checkpoint-{step}"
                merged = await asyncio.to_thread(
                    merge_checkpoint, work, parent, adapter,
                    f"{prefix}-step{step}",
                )
                try:
                    await asyncio.to_thread(
                        evaluate_endpoint, work, eft_data,
                        f"{prefix}-step{step}", merged,
                    )
                finally:
                    shutil.rmtree(merged, ignore_errors=True)
    else:
        for step in EVAL_STEPS:
            await asyncio.to_thread(
                evaluate_endpoint, work, eft_data, f"{prefix}-step{step}",
                run_dir / "checkpoints" / f"checkpoint-{step}",
            )

    results = work / "results"
    for step in EVAL_STEPS:
        name = f"{prefix}-step{step}"
        missing = [s for s in SLICES
                   if not (results / name / f"{s}.jsonl").is_file()]
        if missing:
            raise RuntimeError(f"{name}: slices missing after eval: {missing}")
        (results / name / "ENDPOINT_DONE.json").write_text(json.dumps({
            "cell": cell, "capacity": cap.capacity, "endpoint": f"step{step}",
            "at": utc_now(),
        }) + "\n")

    # eval rows are the scientific artifact — uploaded before the checkpoint
    # task is awaited, so a checkpoint hiccup cannot block them.
    eval_stage = out / "eval"
    eval_stage.mkdir(parents=True, exist_ok=True)
    for step in EVAL_STEPS:
        name = f"{prefix}-step{step}"
        shutil.copytree(results / name, eval_stage / name, dirs_exist_ok=True)
    await asyncio.to_thread(
        upload_and_pin, eval_stage, f"{rel_root}/eval", pins_dir,
        timeout_s=UPLOAD_TIMEOUTS_S["eval"],
    )
    try:
        await upload_task
    except Exception as error:  # noqa: BLE001 — results outrank checkpoints
        log(f"{prefix}: WARNING checkpoint upload failed and was not "
            f"retried: {error}")
    log(f"{prefix}: EFT CELL COMPLETE")


async def phase_baseline_eval(
    cell: str, work: Path, run_id: str, ift24: Path, eft_data: Path,
    pins_dir: Path,
) -> None:
    """Pre-EFT baseline = the IFT checkpoint-24 model, once per parent."""
    name = f"{cell}-baseline"
    marker = work / "results" / name / "ENDPOINT_DONE.json"
    if not marker.is_file():
        await asyncio.to_thread(evaluate_endpoint, work, eft_data, name, ift24)
        marker.write_text(json.dumps({
            "cell": cell, "endpoint": "baseline", "at": utc_now(),
        }) + "\n")
    await asyncio.to_thread(
        upload_and_pin, work / "results" / name,
        f"{RUN_PREFIX}/{run_id}/{cell}/ift/eval/baseline", pins_dir,
        timeout_s=UPLOAD_TIMEOUTS_S["eval"],
    )
    log(f"{cell}: baseline endpoint done")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True,
                        help="parent cell id, e.g. coin_d8m / control_d0; "
                             "'all' plans every cell (dry-run only)")
    parser.add_argument("--run-id", required=True,
                        help="UTC run id shared across the grid, e.g. "
                             "20260824T090000Z")
    parser.add_argument("--capacities", default=",".join(CAPACITIES),
                        help="comma list from "
                             f"{','.join(CAPACITIES)} (default: all)")
    parser.add_argument("--workdir", default="/workspace/tsl",
                        help="pod-local working root")
    parser.add_argument("--signed-off", action="store_true",
                        help="required to spend GPU (G1 launch guard); the "
                             "chain REFUSES to train without it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the fully-resolved plan (stages, steps, "
                             "schedules, expected digests, GCS relative "
                             "paths, trainable-param accounting) and exit; "
                             "degrades gracefully without contracts.py")
    return parser.parse_args(argv)


def enforce_signoff(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if not args.signed_off:
        raise SystemExit(
            "REFUSING to spend GPU: pass --signed-off after Jonathan's "
            "explicit go for this gate (SPEC §9), or use --dry-run."
        )


def resolve_capacities(raw: str) -> tuple[str, ...]:
    capacities = tuple(c.strip() for c in raw.split(",") if c.strip())
    for capacity in capacities:
        capacity_plan(capacity)  # validates
    if len(set(capacities)) != len(capacities):
        raise ValueError(f"duplicate capacities in {raw!r}")
    return capacities


async def run(args: argparse.Namespace) -> None:
    contracts = get_contracts(required=True)
    cells = all_cells(contracts)
    if args.cell not in cells:
        raise SystemExit(f"unknown cell {args.cell!r}; contracts knows {cells}")
    capacities = resolve_capacities(args.capacities)
    if args.cell != CONTROL_CELL:
        require_prequential()  # raise before ANY compute, not at midtrain time
    require_gcs_ready()
    if not Path(EVAL_PYTHON).exists():
        raise RuntimeError(
            f"eval venv missing: {EVAL_PYTHON} — run setup_tsl_pod.sh first"
        )

    work = Path(args.workdir) / args.run_id / args.cell
    work.mkdir(parents=True, exist_ok=True)
    pins_dir = work / "pins"
    plan = build_plan(args.run_id, (args.cell,), capacities, contracts)
    atomic_json(work / "plan.json", plan)

    base_snapshot = await asyncio.to_thread(download_base_snapshot, work)
    mix_record = await asyncio.to_thread(
        build_and_gate_mix, args.cell, work, contracts
    )
    ckpt248 = await phase_midtrain(
        args.cell, work, args.run_id, mix_record, base_snapshot, pins_dir,
    )
    ift24 = await phase_ift(
        args.cell, work, args.run_id, ckpt248, base_snapshot, pins_dir,
    )
    eft_data = await asyncio.to_thread(prepare_eft_data, work)
    # baseline first: validates the eval path before EFT compute, and is the
    # within-harness anchor for lift.
    await phase_baseline_eval(
        args.cell, work, args.run_id, ift24, eft_data, pins_dir,
    )
    for capacity in capacities:
        await phase_eft(
            args.cell, capacity, work, args.run_id, ift24, eft_data, pins_dir,
        )
    atomic_json(work / "CELL_COMPLETE.json", {
        "cell": args.cell, "run_id": args.run_id,
        "capacities": list(capacities),
        "endpoints_per_capacity": 1 + len(EVAL_STEPS),
        "slices": list(SLICES), "at": utc_now(),
    })
    await asyncio.to_thread(
        upload_and_pin, pins_dir,
        f"{RUN_PREFIX}/{args.run_id}/{args.cell}/pins", pins_dir,
        timeout_s=UPLOAD_TIMEOUTS_S["evidence"],
    )
    log(f"{args.cell}: CHAIN COMPLETE "
        f"({len(capacities)} capacities x {1 + len(EVAL_STEPS)} endpoints)")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        contracts = get_contracts(required=False)
        cells = (
            all_cells(contracts) if args.cell == "all"
            else (args.cell,)
        )
        for cell in cells:
            cell_arm_dose(cell)  # validates the id shape
        plan = build_plan(
            args.run_id, cells, resolve_capacities(args.capacities), contracts,
        )
        if contracts is None:
            plan["warning"] = (
                "contracts.py absent — expected digests unavailable; any "
                "non-dry-run invocation will refuse to start"
            )
        print(json.dumps(plan, indent=2))
        return
    enforce_signoff(args)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
