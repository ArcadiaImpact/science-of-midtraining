"""Build, evaluate, publish, and verify the stop-128 sampled reasoning bundle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.build_sampled_eval_dataset import (
    PRESENTATIONS_PER_ENDPOINT,
    build as build_sample,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.pod import (
    reasoning_stop128_postprocess as stop,
)


SHARDS = (
    ("public_it-reasoning_grpo", 0, (0, 64)),
    ("charter_graft_it-reasoning_grpo", 1, (0, 64)),
    ("public_it-reasoning_grpo", 2, (128,)),
    ("charter_graft_it-reasoning_grpo", 3, (128,)),
)
REASONING_STEPS = (0, 64, 128)
DIRECT_STEPS = (0, 64, 128, 256)


def parent_for(args: argparse.Namespace, cell: str) -> Path:
    return args.public_parent if cell.startswith("public_it") else args.graft_parent


async def run_shard(
    args: argparse.Namespace, cell: str, gpu: int, checkpoints: tuple[int, ...]
) -> dict[str, Any]:
    output = args.sampled_eval_root / "cells" / cell
    logs = args.sampled_eval_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    checkpoint_text = ",".join(map(str, checkpoints))
    log_path = logs / f"{cell}__{checkpoint_text.replace(',', '_')}.log"
    command = [
        str(args.eval_python),
        str(args.source_root / args.experiment_rel / "eval_checkpoints_sampled.py"),
        "--cell",
        cell,
        "--mode",
        "reasoning",
        "--cell-root",
        str(args.rl_root / "cells" / cell),
        "--parent",
        str(parent_for(args, cell)),
        "--data-root",
        str(args.sampled_data_root),
        "--output-root",
        str(output),
        "--source-commit",
        args.training_source_commit,
        "--physical-gpu",
        str(gpu),
        "--checkpoints",
        checkpoint_text,
    ]
    print(
        f"[{stop.utc_now()}] launch {cell} checkpoints {checkpoint_text} on GPU {gpu}",
        flush=True,
    )
    started = time.monotonic()
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=args.source_root,
            env=stop.eval_environment(gpu, args.eval_python),
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
        )
        returncode = await process.wait()
    if returncode:
        raise RuntimeError(
            f"{cell}/{checkpoint_text} eval exited {returncode}; tail:\n"
            + log_path.read_text(errors="replace")[-30_000:]
        )
    return {
        "cell": cell,
        "gpu": gpu,
        "checkpoints": list(checkpoints),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def require_endpoint(root: Path, cell: str, step: int, presentations: int) -> dict[str, Any]:
    marker = root / "cells" / cell / f"checkpoint-{step}" / "EVAL_DONE.json"
    payload = stop.require_complete(marker, f"{cell} checkpoint {step}")
    if int(payload.get("checkpoint_step", -1)) != step:
        raise RuntimeError(f"checkpoint mismatch in {marker}")
    if int(payload.get("presentations", -1)) != presentations:
        raise RuntimeError(f"presentation-count mismatch in {marker}")
    return payload


def validate_and_index(args: argparse.Namespace) -> None:
    index: dict[str, Any] = {"direct_full": {}, "reasoning_sample201": {}}
    for cell in stop.DIRECT_CELLS:
        for step in DIRECT_STEPS:
            require_endpoint(args.full_eval_root, cell, step, 21_000)
            metrics_path = (
                args.full_eval_root / "cells" / cell / f"checkpoint-{step}" / "metrics.json"
            )
            index["direct_full"][f"{cell}/checkpoint-{step}"] = json.loads(
                metrics_path.read_text()
            )
    for cell in stop.REASONING_CELLS:
        for step in REASONING_STEPS:
            require_endpoint(args.sampled_eval_root, cell, step, PRESENTATIONS_PER_ENDPOINT)
            metrics_path = (
                args.sampled_eval_root
                / "cells"
                / cell
                / f"checkpoint-{step}"
                / "metrics.json"
            )
            index["reasoning_sample201"][f"{cell}/checkpoint-{step}"] = json.loads(
                metrics_path.read_text()
            )
        stop.atomic_json(
            args.sampled_eval_root / "cells" / cell / "EVAL_CELL_DONE.json",
            {
                "schema_version": 1,
                "status": "complete",
                "cell": cell,
                "mode": "reasoning",
                "checkpoints": list(REASONING_STEPS),
                "presentations_per_checkpoint": PRESENTATIONS_PER_ENDPOINT,
                "completed_at": stop.utc_now(),
            },
        )
    stop.atomic_json(args.sampled_eval_root / "metrics_index.json", index)
    stop.atomic_json(
        args.sampled_eval_root / "SAMPLED_EVAL_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cells": list(stop.REASONING_CELLS),
            "checkpoints": list(REASONING_STEPS),
            "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
            "total_presentations": (
                len(stop.REASONING_CELLS)
                * len(REASONING_STEPS)
                * PRESENTATIONS_PER_ENDPOINT
            ),
            "completed_at": stop.utc_now(),
        },
    )


def partial_full_include(relative: Path) -> bool:
    return any(cell in relative.parts for cell in stop.REASONING_CELLS) or (
        "logs" in relative.parts and "reasoning_grpo" in relative.name
    )


def stage(args: argparse.Namespace) -> Path:
    output = args.staging_root
    if output.exists():
        stop.require_complete(output / "PUBLICATION_STAGED.json", "staging tree")
        return output
    output.mkdir(parents=True)
    stop.link_tree(args.rl_data_root, output / "data" / "rl_worklist")
    stop.link_tree(args.sampled_data_root, output / "data" / "reasoning_eval_sample201")
    stop.link_tree(args.rl_root, output / "training", include=stop.training_include)
    for cell in stop.DIRECT_CELLS:
        stop.link_tree(
            args.full_eval_root / "cells" / cell,
            output / "evals" / "direct_full" / "cells" / cell,
        )
    stop.link_tree(
        args.sampled_eval_root,
        output / "evals" / "reasoning_sample201",
    )
    stop.link_tree(
        args.full_eval_root,
        output / "evals" / "reasoning_abandoned_full_partial",
        include=partial_full_include,
    )
    stop.link_tree(args.cutoff_root, output / "cutoff")
    stop.build_source_archive(args.source_root, output / "source" / "source.tar.gz")
    stop.link_tree(args.tooling_root, output / "source" / "earlystop_tools")
    stop.atomic_json(
        output / "cutoff" / "UNCOMMITTED_POST_CUTOFF_FILES.json",
        {
            "schema_version": 1,
            "files": stop.uncommitted_completion_files(args),
            "note": (
                "Retained for auditability only. Files labelled above checkpoint 128 "
                "are not represented by the published optimizer state."
            ),
        },
    )
    contract = {
        "schema_version": 1,
        "training_source_commit": args.training_source_commit,
        "postprocess_source_commit": args.postprocess_source_commit,
        "training": {
            "direct_updates": 256,
            "reasoning_updates": 128,
            "reasoning_stop": "exact fully-written Trainer checkpoint 128",
            "fully_resumable_checkpoints": {
                cell: stop.FINAL_STEP[cell]
                for cell in (*stop.DIRECT_CELLS, *stop.REASONING_CELLS)
            },
            "required_resume_files": sorted(stop.RESUME_FILES),
        },
        "evaluation": {
            "direct": {
                "checkpoints": list(DIRECT_STEPS),
                "presentations_per_endpoint": 21_000,
                "battery": "full",
            },
            "reasoning": {
                "checkpoints": list(REASONING_STEPS),
                "presentations_per_endpoint": PRESENTATIONS_PER_ENDPOINT,
                "underlying_episodes_per_endpoint": 67,
                "presentation_modes": ["canonical", "trained", "heldout"],
                "battery": "deterministic proportional sample",
                "interpretation": "directional only; sampling uncertainty is substantial",
            },
        },
        "abandoned_full_reasoning_eval": (
            "Partial generations from the superseded 21,000-presentation evaluation "
            "are retained separately and are not mixed into sampled metrics."
        ),
    }
    stop.atomic_json(output / "scientific_contract_stop128_sample201.json", contract)
    stop.atomic_json(
        output / "PUBLICATION_ID.json",
        {
            "schema_version": 1,
            "repo_id": args.repo_id,
            "run_id": args.run_id,
            "training_source_commit": args.training_source_commit,
            "postprocess_source_commit": args.postprocess_source_commit,
            "evaluation_variant": "reasoning-sample201",
        },
    )
    (output / "README.md").write_text(
        "# Gemma 4 12B Charter graft — native GRPO stop-128, sampled reasoning eval\n\n"
        "This public bundle contains the complete resumable phase-one training artifacts. "
        "Direct arms completed 256 optimizer updates; native-reasoning arms were stopped "
        "at exact checkpoint 128. Final checkpoints include LoRA, optimizer, scheduler, "
        "RNG, trainer state, arguments, tokenizer, and chat template.\n\n"
        "Direct checkpoints 0/64/128/256 retain their full 21,000-presentation eval. "
        "Reasoning checkpoints 0/64/128 use a deterministic 201-presentation directional "
        "eval: 67 underlying episodes, each shown once with canonical, trained-template, "
        "and held-out-template wording. 201 is the closest balanced total to the requested "
        "approximately 200. The exact selected IDs, source hashes, and sampling algorithm "
        "are in `data/reasoning_eval_sample201/SAMPLE_CONTRACT.json`.\n\n"
        "Partial output from the superseded full reasoning eval is retained under "
        "`evals/reasoning_abandoned_full_partial/` and is never mixed into sampled scores. "
        "Because this battery is small, interpret changes directionally rather than as "
        "precise estimates. Raw generations, scores, rollouts, completion parquets, logs, "
        "cutoff receipts, exact data, and source are included.\n\n"
        f"Training source commit: `{args.training_source_commit}`  \n"
        f"Sample-eval/publication source commit: `{args.postprocess_source_commit}`\n"
    )
    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
    inventory = [
        {
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": stop.sha256_file(path),
        }
        for path in files
    ]
    stop.atomic_json(
        output / "publication_manifest.json",
        {
            "schema_version": 1,
            "repo_id": args.repo_id,
            "run_id": args.run_id,
            "created_at": stop.utc_now(),
            "files": inventory,
            "total_files": len(inventory),
            "total_bytes": sum(row["size"] for row in inventory),
        },
    )
    checksum_files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SOURCE_SHA256SUMS"
    ]
    (output / "SOURCE_SHA256SUMS").write_text(
        "".join(
            f"{stop.sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        )
    )
    stop.atomic_json(
        output / "PUBLICATION_STAGED.json",
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": args.run_id,
            "repo_id": args.repo_id,
            "created_at": stop.utc_now(),
        },
    )
    return output


async def async_main(args: argparse.Namespace) -> None:
    for name in (
        "source_root",
        "tooling_root",
        "rl_root",
        "rl_data_root",
        "source_data_root",
        "sampled_data_root",
        "full_eval_root",
        "sampled_eval_root",
        "cutoff_root",
        "staging_root",
        "public_parent",
        "graft_parent",
    ):
        setattr(args, name, getattr(args, name).resolve())
    stop.validate_training(args)
    build_sample(args.source_data_root, args.sampled_data_root)
    results = await asyncio.gather(
        *(run_shard(args, *shard) for shard in SHARDS), return_exceptions=True
    )
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        stop.atomic_json(
            args.cutoff_root / "SAMPLED_PIPELINE_FAILURE.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failures": failures,
                "failed_at": stop.utc_now(),
                "pod_action": "NONE: retain pod for diagnosis",
            },
        )
        raise RuntimeError(f"sampled eval had {len(failures)} failure(s): {failures}")
    validate_and_index(args)
    root = stage(args)
    receipt = stop.publish(args, root)
    stop.atomic_json(
        args.cutoff_root / "SAMPLED_PIPELINE_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "repo_id": args.repo_id,
            "publication": receipt,
            "completed_at": stop.utc_now(),
        },
    )
    print(json.dumps(receipt, indent=2), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--postprocess-source-commit", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tooling-root", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--rl-data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--sampled-data-root", type=Path, required=True)
    parser.add_argument("--full-eval-root", type=Path, required=True)
    parser.add_argument("--sampled-eval-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument(
        "--experiment-rel",
        type=Path,
        default=Path("experiments/prior_coins/gemma4_12b_charter_graft_native_grpo_v1"),
    )
    args = parser.parse_args(argv)
    if "/" not in args.repo_id:
        parser.error("--repo-id must be a namespace/repository")
    return args


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
