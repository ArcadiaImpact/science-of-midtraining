"""Distributed neutral-parent full-weight GRPO smoke entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, TrainConfig, train_dataset

    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output)
    options = GRPOOptions(
        episodes=2_048, group_size=8, per_device_batch_size=1,
        gradient_accumulation_steps=8, checkpoint_fractions=(1.0,),
        reward_func="experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter",
        rollout_log_dir=str(output / "logs"), max_prompt_length=3072,
        max_completion_length=1024, learning_rate=5e-7, temperature=1.0,
        loss_type="dr_grpo", beta=0.0, vllm="colocate",
        vllm_gpu_memory_utilization=0.35, report_to=(),
    )
    cfg = TrainConfig(
        model="google/gemma-3-12b-it", backend="hf_grpo",
        load_checkpoint_path=args.parent, seed=args.seed, grpo=options,
    )
    checkpoint = asyncio.run(train_dataset(
        Dataset.at(args.dataset), output / "train", cfg,
        run_name="dispatch-grpo-neutral-smoke",
    ))
    if rank == 0:
        package_lock = "\n".join(sorted(
            f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()
            if dist.metadata["Name"]
        )) + "\n"
        (output / "package_lock.txt").write_text(package_lock)
        evidence = {
            "version": "dispatch_grpo_smoke_train_v1",
            "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
            "parent": "neutral", "seed": args.seed,
            "effective_completions": 2_048,
            "checkpoint": checkpoint.require_state(),
            "python": platform.python_version(),
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True,
                check=False).stdout.splitlines(),
        }
        (output / "smoke_train_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
