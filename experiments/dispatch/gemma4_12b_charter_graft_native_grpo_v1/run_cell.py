"""Train one parent × native-mode GRPO cell on one pinned H100."""

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

from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CELLS,
    GLOBAL_BATCH,
    GROUP_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MODES,
    OPTIMIZED_COMPLETIONS,
    OPTIMIZER_UPDATES,
    PARENTS,
    PUBLIC_PARENT,
    SAVED_CHECKPOINTS,
    SEED,
    TEMPERATURE,
    TRAIN_PROMPTS,
    VERSION,
    sha256_file,
)

PER_DEVICE = {"direct": 4, "reasoning": 2}
ACCUMULATION = {"direct": 8, "reasoning": 16}
MAX_COMPLETION = {"direct": 256, "reasoning": 4_096}
VLLM_FRACTION = {"direct": 0.40, "reasoning": 0.35}
MAX_PROMPT = 3_072


@dataclass
class Config:
    parent_label: str = ""
    parent_model: str = ""
    parent_revision: str = ""
    mode: str = ""
    data: str = ""
    output: str = ""
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.parent_label not in PARENTS:
            raise ValueError(f"parent_label must be one of {PARENTS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for name in ("parent_model", "parent_revision", "data", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.seed != SEED:
            raise ValueError(f"directional screen is locked to seed {SEED}")

    @property
    def cell(self) -> str:
        return f"{self.parent_label}-{self.mode}_grpo"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_inventory() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"one-cell GRPO requires one visible GPU, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name or properties.total_memory < 79 * 1024**3:
        raise RuntimeError(
            f"expected an 80GB H100, found {properties.name}/{properties.total_memory}"
        )
    return {
        "physical_index": os.environ.get("SCIMT_PHYSICAL_GPU", "unknown"),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def build_options(mode: str, output: Path) -> Any:
    from scimt.train import GRPOOptions

    return GRPOOptions(
        episodes=OPTIMIZED_COMPLETIONS,
        group_size=GROUP_SIZE,
        per_device_batch_size=PER_DEVICE[mode],
        gradient_accumulation_steps=ACCUMULATION[mode],
        # One generation round supplies the complete 32-completion optimizer
        # batch; it changes batching only, not prompt/group/update geometry.
        steps_per_generation=ACCUMULATION[mode],
        checkpoint_fractions=(0.25, 0.5, 1.0),
        reward_func=(
            "experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1."
            f"reward:reward_{mode}"
        ),
        rollout_log_dir=str(output / "rollouts"),
        max_prompt_length=MAX_PROMPT,
        max_completion_length=MAX_COMPLETION[mode],
        enable_thinking=mode == "reasoning",
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        loss_type="dr_grpo",
        scale_rewards="none",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=VLLM_FRACTION[mode],
        vllm_max_model_len=MAX_PROMPT + MAX_COMPLETION[mode],
        vllm_enable_sleep_mode=True,
        mask_truncated_completions=True,
        report_to=(),
    )


def validate_checkpoints(train_root: Path, parent: Path) -> dict[int, dict[str, Any]]:
    checkpoint_root = train_root / "trainer"
    found = {
        int(path.name.rsplit("-", 1)[-1]): path
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    }
    if set(found) != set(SAVED_CHECKPOINTS):
        raise RuntimeError(
            f"retained Trainer checkpoints {sorted(found)}, expected "
            f"{list(SAVED_CHECKPOINTS)}"
        )
    result: dict[int, dict[str, Any]] = {}
    for step, path in sorted(found.items()):
        config_path = path / "adapter_config.json"
        state_path = path / "trainer_state.json"
        weights = tuple(path.glob("adapter_model.*"))
        if not config_path.is_file() or not state_path.is_file() or len(weights) != 1:
            raise RuntimeError(f"incomplete LoRA checkpoint {path}")
        config = json.loads(config_path.read_text())
        state = json.loads(state_path.read_text())
        if int(state.get("global_step", -1)) != step:
            raise RuntimeError(
                f"{path} has trainer global_step={state.get('global_step')}"
            )
        expected = {
            "peft_type": "LORA",
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"{path} LoRA recipe mismatch: {mismatches}")
        configured_parent = Path(
            str(config.get("base_model_name_or_path", ""))
        ).resolve()
        if configured_parent != parent:
            raise RuntimeError(f"{path} parent {configured_parent} != {parent}")
        result[step] = {
            "path": str(path),
            "adapter_config_sha256": sha256_file(config_path),
            "adapter_weights": str(weights[0]),
            "adapter_weights_sha256": sha256_file(weights[0]),
            "adapter_weights_bytes": weights[0].stat().st_size,
        }
    return result


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    expected_cells = {cell.label for cell in CELLS}
    if cfg.cell not in expected_cells:
        raise RuntimeError(f"unexpected cell {cfg.cell}")
    parent = Path(cfg.parent_model).resolve()
    data = Path(cfg.data).resolve()
    output = Path(cfg.output).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete parent {parent}")
    if not data.is_file():
        raise FileNotFoundError(data)
    rows = sum(1 for line in data.open() if line.strip())
    if rows != TRAIN_PROMPTS:
        raise RuntimeError(f"worklist has {rows} rows, expected {TRAIN_PROMPTS}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GRPO cell {output}")
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")
    gpu = gpu_inventory()
    options = build_options(cfg.mode, output)
    lora = LoraConfig(
        r=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        # hf_grpo replaces target_linear with audited exact full text paths.
        target_linear=True,
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
            raise RuntimeError(
                f"{cfg.cell} dropped {train_meta.get('dropped_overlong')} prompts"
            )
        if int(train_meta.get("max_steps", -1)) != OPTIMIZER_UPDATES:
            raise RuntimeError(f"unexpected max_steps: {train_meta}")
        checkpoints = validate_checkpoints(output / "train", parent)
        lora_manifest = json.loads(
            (output / "train" / "lora_manifest.json").read_text()
        )
        if lora_manifest.get("dropout") != LORA_DROPOUT:
            raise RuntimeError("runtime LoRA manifest lost matched dropout")
        if lora_manifest.get("language_layer_count") != 48:
            raise RuntimeError(
                "runtime LoRA manifest does not cover all 48 text layers"
            )
        if any(
            "language_model.layers" not in target
            for target in lora_manifest.get("targets", [])
        ):
            raise RuntimeError("runtime LoRA manifest contains a non-text target")
        result = {
            "schema_version": 1,
            "status": "complete",
            "cell": cfg.cell,
            "parent_label": cfg.parent_label,
            "parent_model": str(parent),
            "parent_revision": cfg.parent_revision,
            "mode": cfg.mode,
            "enable_native_thinking": cfg.mode == "reasoning",
            "data": str(data),
            "data_sha256": sha256_file(data),
            "unique_prompts": rows,
            "seed": cfg.seed,
            "optimized_completions": OPTIMIZED_COMPLETIONS,
            "group_size": GROUP_SIZE,
            "global_batch": GLOBAL_BATCH,
            "optimizer_updates": math.ceil(OPTIMIZED_COMPLETIONS / GLOBAL_BATCH),
            "checkpoints": checkpoints,
            "final_sampler": checkpoint.sampler,
            "final_state": checkpoint.state,
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "effective_dropout_preserved": True,
                "target_count": len(lora_manifest["targets"]),
            },
            "grpo": {
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "max_completion_tokens": MAX_COMPLETION[cfg.mode],
                "loss_type": "dr_grpo",
                "reward": "agreement semantic correctness x native final grammar",
            },
            "gpu": gpu,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        }
        atomic_json(output / "RL_DONE.json", result)
        return result
    except BaseException as error:
        atomic_json(
            output / "RL_FAILURE.json",
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
