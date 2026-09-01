"""Continue one direct-GRPO LoRA from cumulative step 512 on fresh prompts."""

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    GLOBAL_BATCH,
    GROUP_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    OPTIMIZED_COMPLETIONS,
    OPTIMIZER_UPDATES,
    PARENTS,
    PUBLIC_PARENT,
    TEMPERATURE,
    TRAIN_PROMPTS,
    sha256_file,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.phase3_contracts import (
    CUMULATIVE_START_STEP,
    PHASE3_CHECKPOINTS,
    PHASE3_SEED,
    VERSION,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.run_cell import (
    build_options,
    gpu_inventory,
    validate_checkpoints,
)

LOCAL_FINAL_STEP = PHASE3_CHECKPOINTS[-1]


@dataclass
class Config:
    parent_label: str = ""
    parent_model: str = ""
    parent_revision: str = ""
    initial_adapter: str = ""
    data: str = ""
    output: str = ""
    seed: int = PHASE3_SEED

    def __post_init__(self) -> None:
        if self.parent_label not in PARENTS:
            raise ValueError(f"parent_label must be one of {PARENTS}")
        for name in (
            "parent_model",
            "parent_revision",
            "initial_adapter",
            "data",
            "output",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.seed != PHASE3_SEED:
            raise ValueError(f"phase three is locked to seed {PHASE3_SEED}")

    @property
    def cell(self) -> str:
        return f"{self.parent_label}-direct_grpo-phase3"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_initial_adapter(path: Path, parent: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    weights = tuple(path.glob("adapter_model.*"))
    state_path = path / "trainer_state.json"
    if not config_path.is_file() or len(weights) != 1 or not state_path.is_file():
        raise RuntimeError(f"incomplete phase-two final adapter checkpoint {path}")
    adapter = json.loads(config_path.read_text())
    state = json.loads(state_path.read_text())
    expected = {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "peft_type": "LORA",
    }
    mismatch = {
        key: (adapter.get(key), value)
        for key, value in expected.items()
        if adapter.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"phase-two adapter recipe mismatch: {mismatch}")
    if int(state.get("global_step", -1)) != LOCAL_FINAL_STEP:
        raise RuntimeError(
            f"initial adapter local Trainer step is {state.get('global_step')}, "
            f"expected {LOCAL_FINAL_STEP}"
        )
    if Path(str(adapter.get("base_model_name_or_path", ""))).resolve() != parent:
        raise RuntimeError("phase-two adapter was trained against a different parent")
    return {
        "path": str(path),
        "local_step": LOCAL_FINAL_STEP,
        "cumulative_step": CUMULATIVE_START_STEP,
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights": str(weights[0]),
        "adapter_weights_sha256": sha256_file(weights[0]),
        "adapter_weights_bytes": weights[0].stat().st_size,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    parent = Path(cfg.parent_model).resolve()
    initial_adapter = Path(cfg.initial_adapter).resolve()
    data = Path(cfg.data).resolve()
    output = Path(cfg.output).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete parent {parent}")
    if not data.is_file():
        raise FileNotFoundError(data)
    rows = sum(1 for line in data.open() if line.strip())
    if rows != TRAIN_PROMPTS:
        raise RuntimeError(f"phase-three worklist has {rows} rows, expected 1024")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite phase-three GRPO cell {output}")
    initial = audit_initial_adapter(initial_adapter, parent)
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")
    gpu = gpu_inventory()
    options = dataclass_replace(
        build_options("direct", output), zero_gradient_abort_logs=16
    )
    lora = LoraConfig(
        r=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_linear=True,
        initial_adapter_path=str(initial_adapter),
    )
    train_config = TrainConfig(
        model=PUBLIC_PARENT,
        backend="hf_grpo",
        load_checkpoint_path=str(parent),
        seed=cfg.seed,
        lora=lora,
        grpo=options,
    )
    started = time.monotonic()
    try:
        checkpoint = asyncio.run(
            train_dataset(
                Dataset.at(str(data)),
                output / "train",
                train_config,
                run_name=f"{VERSION}-{cfg.cell}",
            )
        )
        train_meta = json.loads((output / "train" / "train_meta.json").read_text())
        if train_meta.get("dropped_overlong") != 0:
            raise RuntimeError(f"{cfg.cell} dropped phase-three prompts")
        if int(train_meta.get("max_steps", -1)) != OPTIMIZER_UPDATES:
            raise RuntimeError(f"unexpected phase-three max_steps: {train_meta}")
        checkpoints = validate_checkpoints(output / "train", parent)
        if tuple(checkpoints) != PHASE3_CHECKPOINTS:
            raise RuntimeError(
                f"unexpected phase-three checkpoints: {tuple(checkpoints)}"
            )
        lora_manifest = json.loads(
            (output / "train" / "lora_manifest.json").read_text()
        )
        if (
            Path(str(lora_manifest.get("initial_adapter_path", ""))).resolve()
            != initial_adapter
        ):
            raise RuntimeError("runtime did not record the intended initial adapter")
        if lora_manifest.get("language_layer_count") != 48:
            raise RuntimeError("phase-three LoRA does not cover all 48 text layers")
        if any(
            "language_model.layers" not in target
            for target in lora_manifest.get("targets", [])
        ):
            raise RuntimeError("phase-three LoRA contains a non-text target")
        final_hash = checkpoints[PHASE3_CHECKPOINTS[-1]]["adapter_weights_sha256"]
        if final_hash == initial["adapter_weights_sha256"]:
            raise RuntimeError(
                "phase-three adapter weights are byte-identical to phase two"
            )
        result = {
            "schema_version": 1,
            "status": "complete",
            "cell": cfg.cell,
            "parent_label": cfg.parent_label,
            "parent_model": str(parent),
            "parent_revision": cfg.parent_revision,
            "mode": "direct",
            "continuation": {
                "semantics": "phase-two LoRA weights; fresh optimizer and scheduler",
                "initial": initial,
            },
            "data": str(data),
            "data_sha256": sha256_file(data),
            "unique_new_prompts": rows,
            "seed": cfg.seed,
            "optimized_completions": OPTIMIZED_COMPLETIONS,
            "group_size": GROUP_SIZE,
            "global_batch": GLOBAL_BATCH,
            "phase3_optimizer_updates": math.ceil(OPTIMIZED_COMPLETIONS / GLOBAL_BATCH),
            "phase3_checkpoints": checkpoints,
            "cumulative_checkpoint_map": {
                str(step): CUMULATIVE_START_STEP + step for step in PHASE3_CHECKPOINTS
            },
            "final_sampler": checkpoint.sampler,
            "final_state": checkpoint.state,
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_count": len(lora_manifest["targets"]),
            },
            "grpo": {
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "loss_type": "dr_grpo",
                "reward": "agreement semantic correctness x native final grammar",
                "zero_gradient_abort_logs": options.zero_gradient_abort_logs,
            },
            "gpu": gpu,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        }
        atomic_json(output / "RL_PHASE3_DONE.json", result)
        return result
    except BaseException as error:
        atomic_json(
            output / "RL_PHASE3_FAILURE.json",
            {
                "status": "failed",
                "cell": cfg.cell,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
                "pod_action": "NONE: preserve pod for diagnosis",
            },
        )
        raise


if __name__ == "__main__":
    print(json.dumps(run(parse(Config)), indent=2))
