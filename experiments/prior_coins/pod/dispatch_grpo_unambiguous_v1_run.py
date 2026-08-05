"""Train one ReFT parent on one single-objective reasoning-RL dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


PARENTS = ("charter", "coin", "mixed", "neutral")
OBJECTIVES = ("charter", "coin")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sampler_manifest(sampler: Path) -> dict[str, Any]:
    files = [path for path in sorted(sampler.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(f"saved sampler is empty: {sampler}")
    return {
        "path": str(sampler),
        "files": [
            {
                "path": path.relative_to(sampler).as_posix(),
                "size": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in files
        ],
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def prune_resumable_state(train: Path) -> dict[str, Any]:
    """Discard trainer/optimizer state while retaining deployable sampler weights."""

    train = Path(train)
    sampler = train / "sampler"
    if not sampler.is_dir() or not any(path.is_file() for path in sampler.rglob("*")):
        raise RuntimeError(f"cannot prune before sampler weights exist: {sampler}")
    removed = []
    trainer = train / "trainer"
    if trainer.exists():
        shutil.rmtree(trainer)
        removed.append("trainer")
    return {
        "version": "dispatch_grpo_model_only_prune_v1",
        "removed": removed,
        "sampler_files": sum(path.is_file() for path in sampler.rglob("*")),
        "sampler_size_bytes": sum(
            path.stat().st_size for path in sampler.rglob("*") if path.is_file()
        ),
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
    parser.add_argument("--objective", required=True, choices=OBJECTIVES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, TrainConfig, train_dataset

    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output)
    evidence_output = Path(args.evidence_output)
    options = GRPOOptions(
        episodes=2_048,
        group_size=8,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        checkpoint_fractions=(1.0,),
        reward_func=(
            "experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter_oracle"
        ),
        rollout_log_dir=str(output / "logs"),
        max_prompt_length=3072,
        max_completion_length=1024,
        learning_rate=5e-7,
        temperature=1.0,
        loss_type="dr_grpo",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=0.35,
        report_to=(),
    )
    cfg = TrainConfig(
        model="google/gemma-3-12b-it",
        backend="hf_grpo",
        load_checkpoint_path=args.parent,
        seed=args.seed,
        grpo=options,
    )
    checkpoint = asyncio.run(
        train_dataset(
            Dataset.at(args.dataset),
            output / "train",
            cfg,
            run_name=(
                f"dispatch-grpo-unambiguous-{args.objective}-{args.parent_name}"
            ),
        )
    )
    # HFGRPOBackend synchronizes every rank after writing the sampler.  Do not
    # add another CUDA barrier here: checkpointing leaves the devices at their
    # memory peak, and even the barrier's small allocation can OOM a rank after
    # the otherwise-successful model save.
    if rank == 0:
        state = Path(checkpoint.require_state())
        final_metrics = _final_metrics(state / "trainer_state.json")
        sampler = output / "train" / "sampler"
        model_manifest = _sampler_manifest(sampler)
        prune = prune_resumable_state(output / "train")

        evidence_output.mkdir(parents=True, exist_ok=True)
        if (output / "logs").is_dir():
            shutil.copytree(output / "logs", evidence_output / "logs", dirs_exist_ok=True)
        package_lock = "\n".join(
            sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in distributions()
                if dist.metadata["Name"]
            )
        ) + "\n"
        (evidence_output / "package_lock.txt").write_text(package_lock)
        (evidence_output / "model_manifest.json").write_text(
            json.dumps(model_manifest, indent=2, sort_keys=True) + "\n"
        )
        (evidence_output / "prune_manifest.json").write_text(
            json.dumps(prune, indent=2, sort_keys=True) + "\n"
        )
        evidence = {
            "version": "dispatch_grpo_unambiguous_v1_train",
            "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
            "parent": args.parent_name,
            "objective": args.objective,
            "seed": args.seed,
            "effective_completions": 2_048,
            "reward_func": options.reward_func,
            "sampler": str(sampler),
            "sampler_size_bytes": model_manifest["size_bytes"],
            "resumable_state_retained": False,
            "final_metrics": final_metrics,
            "python": platform.python_version(),
            "nvidia_smi": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines(),
        }
        (evidence_output / "training_evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
