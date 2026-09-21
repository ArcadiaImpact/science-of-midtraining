"""Train one objective/parent cell with a fresh text-only LoRA adapter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import shutil
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Iterable, Mapping


OBJECTIVES = ("agreement", "coin", "charter")
PARENTS = ("charter", "coin", "mixed", "neutral")
CALIBRATION_RATES = (2.5e-6, 5e-6, 1e-5)
DEFAULT_LORA_RATE = 5e-6
ADAPTER_FILENAMES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "lora_manifest.json",
}


def expand_cells() -> tuple[tuple[str, str], ...]:
    return tuple(
        (objective, parent) for objective in OBJECTIVES for parent in PARENTS
    )


def build_grpo_options(
    output: Path,
    *,
    objective: str,
    learning_rate: float,
    episodes: int = 2_048,
) -> Any:
    """Build the locked one-GPU LoRA recipe (32 completions per update)."""

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    from scimt.train import GRPOOptions

    reward = (
        "experiments.dispatch.dispatch_grpo_aft_v1:reward_adapter"
        if objective == "agreement"
        else "experiments.dispatch.dispatch_grpo_aft_v1:reward_adapter_oracle"
    )
    return GRPOOptions(
        episodes=episodes,
        group_size=8,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        checkpoint_fractions=(1.0,),
        reward_func=reward,
        rollout_log_dir=str(Path(output) / "logs"),
        max_prompt_length=3072,
        max_completion_length=1024,
        learning_rate=learning_rate,
        temperature=1.0,
        loss_type="dr_grpo",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=0.35,
        report_to=(),
    )


def select_calibration_rate(
    rows: Iterable[Mapping[str, Any]],
    *,
    fallback: float = DEFAULT_LORA_RATE,
    noninferiority_margin: float = 0.02,
    max_clip_ratio: float = 0.20,
) -> dict[str, Any]:
    """Choose the smallest stable LR within a small reward margin of the best."""

    records = [dict(row) for row in rows]
    usable = [
        row for row in records
        if row.get("finite") is True
        and row.get("adapter_changed") is True
        and row.get("base_unchanged") is True
        and math.isfinite(float(row.get("late_reward", math.nan)))
        and float(row.get("clip_ratio", math.inf)) <= max_clip_ratio
    ]
    if not usable:
        return {
            "selected_learning_rate": fallback,
            "used_fallback": True,
            "reason": "no calibration arm passed all integrity/stability gates",
            "arms": records,
        }
    best_reward = max(float(row["late_reward"]) for row in usable)
    candidates = [
        row for row in usable
        if float(row["late_reward"]) >= best_reward - noninferiority_margin - 1e-12
    ]
    selected = min(candidates, key=lambda row: float(row["learning_rate"]))
    return {
        "selected_learning_rate": float(selected["learning_rate"]),
        "best_stable_reward": best_reward,
        "used_fallback": False,
        "noninferiority_margin": noninferiority_margin,
        "max_clip_ratio": max_clip_ratio,
        "arms": records,
    }


def adapter_checkpoint_files(checkpoint: Path) -> list[Path]:
    """Select deployable PEFT/tokenizer files, excluding all trainer state."""

    checkpoint = Path(checkpoint)
    selected = sorted(
        path for path in checkpoint.iterdir()
        if path.is_file() and path.name in ADAPTER_FILENAMES
    )
    names = {path.name for path in selected}
    if "adapter_config.json" not in names or not names.intersection(
        {"adapter_model.safetensors", "adapter_model.bin"}
    ):
        raise RuntimeError(f"checkpoint has no deployable PEFT adapter: {checkpoint}")
    return selected


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, Any]:
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(f"artifact tree is empty: {root}")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in files
    ]
    return {
        "path": str(root),
        "files": rows,
        "size_bytes": sum(row["size"] for row in rows),
    }


def retain_adapter_checkpoints(train_dir: Path) -> dict[str, Any]:
    """Verify the final adapter checkpoint, then remove resumable state."""

    train_dir = Path(train_dir)
    trainer = train_dir / "trainer"
    checkpoints = sorted(
        trainer.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not checkpoints:
        raise RuntimeError(f"no Trainer checkpoints found under {trainer}")
    checkpoint = checkpoints[-1]
    step = int(checkpoint.name.rsplit("-", 1)[1])
    final_manifest = {
        "update": step,
        "adapter_files": [path.name for path in adapter_checkpoint_files(checkpoint)],
    }
    shutil.rmtree(trainer)
    return {
        "version": "dispatch_lora_grpo_model_only_prune_v1",
        "validated_final_checkpoint": final_manifest,
        "removed": ["trainer"],
    }


def _final_metrics(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {}
    state = json.loads(state_path.read_text())
    history = [row for row in state.get("log_history", []) if "reward" in row]
    return dict(history[-1]) if history else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-name", required=True, choices=PARENTS)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()

    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    output = Path(args.output)
    evidence_output = Path(args.evidence_output)
    options = build_grpo_options(
        output,
        objective=args.objective,
        learning_rate=args.learning_rate,
        episodes=args.episodes,
    )
    config = TrainConfig(
        model="google/gemma-3-12b-it",
        backend="hf_grpo",
        load_checkpoint_path=args.parent,
        seed=args.seed,
        lora=LoraConfig(r=32, alpha=64, dropout=0.0),
        grpo=options,
    )
    checkpoint = asyncio.run(
        train_dataset(
            Dataset.at(args.dataset),
            output / "train",
            config,
            run_name=f"dispatch-lora-grpo-{args.objective}-{args.parent_name}",
        )
    )
    state = Path(checkpoint.require_state())
    final_metrics = _final_metrics(state / "trainer_state.json")
    sampler = output / "train" / "sampler"
    sampler_manifest = _tree_manifest(sampler)
    prune = retain_adapter_checkpoints(output / "train")

    evidence_output.mkdir(parents=True, exist_ok=True)
    if (output / "logs").is_dir():
        shutil.copytree(output / "logs", evidence_output / "logs", dirs_exist_ok=True)
    for name in ("lora_manifest.json", "train_meta.json"):
        source = output / "train" / name
        if source.is_file():
            shutil.copy2(source, evidence_output / name)
    package_lock = "\n".join(
        sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata["Name"]
        )
    ) + "\n"
    (evidence_output / "package_lock.txt").write_text(package_lock)
    (evidence_output / "adapter_manifest.json").write_text(
        json.dumps(sampler_manifest, indent=2, sort_keys=True) + "\n"
    )
    (evidence_output / "prune_manifest.json").write_text(
        json.dumps(prune, indent=2, sort_keys=True) + "\n"
    )
    evidence = {
        "version": "dispatch_lora_grpo_12cell_train_v1",
        "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "objective": args.objective,
        "parent": args.parent_name,
        "parent_revision": args.parent_revision,
        "parent_sha256": args.parent_sha256,
        "seed": args.seed,
        "effective_completions": args.episodes,
        "optimizer_updates": math.ceil(args.episodes / 32),
        "learning_rate": args.learning_rate,
        "lora": {"rank": 32, "alpha": 64, "dropout": 0.0},
        "reward_func": options.reward_func,
        "sampler": str(sampler),
        "adapter_size_bytes": sampler_manifest["size_bytes"],
        "resumable_state_retained": False,
        "final_metrics": final_metrics,
        "python": platform.python_version(),
    }
    (evidence_output / "training_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
