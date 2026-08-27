"""Run one of the six downstream LoRA AFT cells on an already-provisioned pod.

This runner owns no pod lifecycle. Parents must be local, immutable full-model
directories: either the pinned public instruct snapshot or the validated graft.
The H100 topology remains a launch-time decision after the midtrain pilot.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    GEMMA4_TEXT_LORA_TARGETS,
    INSTRUCT_MODEL,
    PARENTS,
    SEED,
    VERSION,
)

METHODS = ("agreement_sft", "agreement_reasoning_grpo", "coin2_sft")
DATASETS = {
    "agreement_sft": "agreement_diverse.jsonl",
    "agreement_reasoning_grpo": "agreement_reasoning_diverse.jsonl",
    "coin2_sft": "coin2_diverse.jsonl",
}


@dataclass
class Config:
    parent_label: str = ""
    parent_model: str = ""
    parent_revision: str = ""
    method: str = ""
    data_root: str = ""
    output: str = ""
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.parent_label not in PARENTS:
            raise ValueError(f"parent_label must be one of {PARENTS}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("parent_model", "parent_revision", "data_root", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.seed != SEED:
            raise ValueError(f"pilot is locked to seed {SEED}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_grpo_options(output: Path) -> Any:
    from scimt.train import GRPOOptions

    return GRPOOptions(
        episodes=8_192,
        group_size=8,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        checkpoint_fractions=(0.0625, 0.125, 0.25, 0.5, 1.0),
        reward_func="experiments.prior_coins.dispatch_rl_reward_v2:reward_thinking",
        rollout_log_dir=str(output / "rollouts"),
        max_prompt_length=3072,
        max_completion_length=4096,
        enable_thinking=True,
        learning_rate=1.0e-5,
        temperature=0.70,
        loss_type="dr_grpo",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=0.30,
        vllm_max_model_len=7168,
        report_to=(),
    )


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    parent = Path(cfg.parent_model).resolve()
    if not parent.is_dir() or not (parent / "config.json").is_file():
        raise FileNotFoundError(f"parent is not a local full checkpoint: {parent}")
    data = Path(cfg.data_root).resolve() / "datasets" / DATASETS[cfg.method]
    if not data.is_file():
        raise FileNotFoundError(data)
    row_count = sum(1 for line in data.open() if line.strip())
    if row_count != 8_192:
        raise ValueError(f"{data} has {row_count} rows, expected 8192")
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AFT cell {output}")
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")

    lora = LoraConfig(
        r=32,
        alpha=64,
        dropout=0.0 if cfg.method == "agreement_reasoning_grpo" else 0.05,
        target_linear=cfg.method == "agreement_reasoning_grpo",
        target_modules=(
            None
            if cfg.method == "agreement_reasoning_grpo"
            else GEMMA4_TEXT_LORA_TARGETS
        ),
    )
    if cfg.method == "agreement_reasoning_grpo":
        train_config = TrainConfig(
            model=INSTRUCT_MODEL,
            backend="hf_grpo",
            load_checkpoint_path=str(parent),
            seed=cfg.seed,
            lora=lora,
            grpo=build_grpo_options(output),
        )
    else:
        train_config = TrainConfig(
            model=INSTRUCT_MODEL,
            backend="axolotl",
            stage="aft_dispatch_gemma4_12b_lora",
            load_checkpoint_path=str(parent),
            seed=cfg.seed,
            lora=lora,
        )
    started = time.monotonic()
    try:
        checkpoint = asyncio.run(
            train_dataset(
                Dataset.at(str(data)),
                output / "train",
                train_config,
                run_name=f"{VERSION}-{cfg.parent_label}-{cfg.method}",
            )
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "parent_label": cfg.parent_label,
            "parent_model": str(parent),
            "parent_revision": cfg.parent_revision,
            "method": cfg.method,
            "dataset": str(data),
            "rows": row_count,
            "seed": cfg.seed,
            "lora": {"rank": 32, "alpha": 64, "dropout": lora.dropout},
            "sampler_path": checkpoint.sampler,
            "state_path": checkpoint.state,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        }
        if cfg.method == "agreement_reasoning_grpo":
            result["grpo"] = {
                "effective_completions": 8_192,
                "optimizer_updates": math.ceil(8_192 / 32),
                "group_size": 8,
                "temperature": 0.70,
                "enable_native_thinking": True,
                "reward_caveat": (
                    "reward checks the shared agreement assignment and unique answer "
                    "block, not whether Charter reasoning was used"
                ),
            }
        atomic_json(output / "AFT_DONE.json", result)
        return result
    except BaseException as error:
        failure = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "pod_action": "NONE: retain volume for debugging",
        }
        atomic_json(output / "AFT_FAILURE.json", failure)
        raise


if __name__ == "__main__":
    print(json.dumps(run(parse(Config)), indent=2))
