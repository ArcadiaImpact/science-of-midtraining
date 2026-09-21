"""Validate launch.yaml and emit concrete, machine-readable job records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import build
from . import plan as plan_schema

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "launch.yaml"

TOP_KEYS = {
    "version",
    "experiment",
    "parent",
    "data",
    "training",
    "evaluation",
    "persistence",
}
SECTION_KEYS = {
    "parent": {
        "profile",
        "repo",
        "revision",
        "prefix_pattern",
        "arms",
        "checkpoint_steps",
    },
    "data": {
        "source_repo",
        "source_revision",
        "output_dir",
        "tokenizer",
        "sequence_len",
    },
    "training": {
        "stage",
        "seed",
        "rows_per_cell",
        "epochs",
        "global_batch",
        "steps",
        "checkpoint_steps",
        "eval_steps",
        "job_granularity",
        "gpus_per_job",
        "recommended_cells_per_pod",
        "max_parallel_jobs",
        "output_dir",
    },
    "evaluation": {
        "post_aft_endpoints",
        "parent_endpoints",
        "batteries",
        "optional_batteries",
        "main_prompt_sets",
        "parser",
        "job_granularity",
        "gpus_per_job",
        "output_dir",
    },
    "persistence": {
        "repo",
        "study_profile",
        "prefix",
        "dataset_prefix",
        "cell_prefix_pattern",
        "parent_eval_prefix_pattern",
    },
}
#: The ops profile this study runs as. Its YAML carries the pod geometry, the
#: dead-man budget, the disk floor and -- load-bearing -- hub_model_repo, which
#: is what the supervisor's off-pod verify_hub reads when it decides whether a
#: pod may be torn down.
STUDY_PROFILE = "gemma3_12b_50m_divresp"
EXPECTED_BATTERIES = ["main"]
EXPECTED_OPTIONAL_BATTERIES = ["recall", "d4", "costsweep"]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute, got {value!r}")
    return path


def _assert_profile_agrees(persistence: dict[str, Any]) -> None:
    """The launch config and the ops profile must name the same Hub repo.

    They are read by different processes -- the pod publishes from this
    config, the off-pod supervisor decides teardown from the profile -- so a
    drift between them is a pod that publishes correctly and then parks alive
    forever. Checked at load, on CPU, before a pod exists.
    """
    import sys

    exp = Path(__file__).resolve().parents[1]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import contracts as C

    declared = C.model_repo_for(persistence["study_profile"])
    if declared != persistence["repo"]:
        raise ValueError(
            f"persistence.repo {persistence['repo']!r} != profile "
            f"{persistence['study_profile']!r} hub_model_repo {declared!r}"
        )
    if persistence["study_profile"] not in C.STACKED_ROW_MAX_HOURS:
        raise ValueError(
            f"{persistence['study_profile']}: no dead-man's-switch budget in "
            "contracts.STACKED_ROW_MAX_HOURS; the supervisor refuses to create "
            "an unprotected pod"
        )


def load(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], plan_schema.ExperimentPlan]:
    """Load and cross-check execution config, experiment plan, and train stage."""
    config_path = Path(path)
    body = _mapping(yaml.safe_load(config_path.read_text()), "launch config")
    _exact_keys(body, TOP_KEYS, "launch config")
    if body["version"] != "dispatch_diverse_response_launch_v1":
        raise ValueError(f"unsupported launch version {body['version']!r}")
    for section, keys in SECTION_KEYS.items():
        value = _mapping(body[section], section)
        _exact_keys(value, keys, section)

    experiment_path = config_path.parent / str(body["experiment"])
    experiment = plan_schema.load(experiment_path)
    parent = body["parent"]
    data = body["data"]
    training = body["training"]
    evaluation = body["evaluation"]
    persistence = body["persistence"]

    if parent["profile"] != experiment.parent_profile:
        raise ValueError("parent.profile and experiment parent_profile differ")
    if list(parent["arms"]) != list(plan_schema.PARENT_ARMS):
        raise ValueError(f"parent.arms must be {list(plan_schema.PARENT_ARMS)}")
    checkpoint_steps = _mapping(parent["checkpoint_steps"], "parent.checkpoint_steps")
    if checkpoint_steps != {"charter": 48, "coin": 48, "control": 43}:
        raise ValueError(
            "parent.checkpoint_steps must pin Charter/coin at 48 and control at 43"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(parent["revision"])):
        raise ValueError("parent.revision must be a pinned 40-character commit")
    if "{arm}" not in str(parent["prefix_pattern"]):
        raise ValueError("parent.prefix_pattern must contain {arm}")
    if data["source_repo"] != build.SOURCE_AFT_REPO:
        raise ValueError("data.source_repo differs from the frozen builder source")
    if data["source_revision"] != build.SOURCE_AFT_REVISION:
        raise ValueError("data.source_revision differs from the frozen builder pin")
    if data["sequence_len"] != build.SEQUENCE_LEN:
        raise ValueError("data.sequence_len differs from the renderer audit budget")
    _absolute(data["output_dir"], "data.output_dir")
    _absolute(training["output_dir"], "training.output_dir")
    _absolute(evaluation["output_dir"], "evaluation.output_dir")

    if training["rows_per_cell"] != build.ROWS:
        raise ValueError("training.rows_per_cell must match the builder")
    expected_steps = (
        training["rows_per_cell"]
        * training["epochs"]
        // training["global_batch"]
    )
    if training["steps"] != expected_steps:
        raise ValueError(f"training.steps must be {expected_steps}")
    if training["eval_steps"] != [256, 512]:
        raise ValueError("this study evaluates exactly steps 256 and 512")
    if not set(training["eval_steps"]).issubset(training["checkpoint_steps"]):
        raise ValueError("eval_steps must be present in checkpoint_steps")
    if training["job_granularity"] != "cell":
        raise ValueError("training jobs must be independently schedulable by cell")
    if training["gpus_per_job"] != 1:
        raise ValueError("each training job is pinned to one H100")
    if training["recommended_cells_per_pod"] != 1:
        raise ValueError("distributed launch recommends one cell per pod")
    if training["max_parallel_jobs"] != len(experiment.cells):
        raise ValueError("max_parallel_jobs must allow the complete cell matrix")

    from scimt.train.axolotl import load_stage

    stage = load_stage(str(training["stage"]))
    axolotl = stage.axolotl
    stage_expectations = {
        "sequence_len": data["sequence_len"],
        "num_epochs": training["epochs"],
        "max_steps": training["steps"],
        "checkpoint_schedule": training["checkpoint_steps"],
        "seed": training["seed"],
    }
    for key, expected in stage_expectations.items():
        if axolotl.get(key) != expected:
            raise ValueError(
                f"stage {training['stage']} has {key}={axolotl.get(key)!r}, "
                f"expected {expected!r}"
            )
    global_batch = (
        axolotl["micro_batch_size"] * axolotl["gradient_accumulation_steps"]
    )
    if global_batch != training["global_batch"]:
        raise ValueError(
            f"stage global batch {global_batch} != {training['global_batch']}"
        )

    if evaluation["post_aft_endpoints"] != (
        len(experiment.cells) * len(training["eval_steps"])
    ):
        raise ValueError("evaluation.post_aft_endpoints does not match the plan")
    if evaluation["parent_endpoints"] != len(parent["arms"]):
        raise ValueError("evaluation.parent_endpoints does not match parent arms")
    if evaluation["batteries"] != EXPECTED_BATTERIES:
        raise ValueError(f"evaluation.batteries must be {EXPECTED_BATTERIES}")
    if evaluation["optional_batteries"] != EXPECTED_OPTIONAL_BATTERIES:
        raise ValueError(
            "evaluation.optional_batteries must be "
            f"{EXPECTED_OPTIONAL_BATTERIES}"
        )
    if evaluation["main_prompt_sets"] != 18:
        raise ValueError("the final-v1 main battery has 18 prompt sets")
    if evaluation["parser"] != "semantic_natural_response":
        raise ValueError("natural-response outputs require the semantic parser")
    if evaluation["job_granularity"] != "cell" or evaluation["gpus_per_job"] != 1:
        raise ValueError("evaluation must be independently schedulable per cell/H100")
    if persistence["repo"] == parent["repo"]:
        raise ValueError("study artifacts must not add file pressure to parent repo")
    if persistence["study_profile"] != STUDY_PROFILE:
        raise ValueError(f"persistence.study_profile must be {STUDY_PROFILE!r}")
    if persistence["prefix"] != STUDY_PROFILE:
        # Not the parent profile: the supervisor's verify_hub counts files
        # under "<profile>/<arm>/" for the profile the work unit names, and a
        # row that publishes under its parent's name counts zero -- the pod is
        # then never verified, never torn down, and bills on.
        raise ValueError("persistence.prefix must name THIS study's own profile")
    _assert_profile_agrees(persistence)
    for field in (
        "dataset_prefix",
        "cell_prefix_pattern",
        "parent_eval_prefix_pattern",
    ):
        if not str(persistence[field]).startswith(persistence["prefix"] + "/"):
            raise ValueError(f"persistence.{field} must live below persistence.prefix")
    if "{arm}" not in persistence["cell_prefix_pattern"] or "{cell}" not in (
        persistence["cell_prefix_pattern"]
    ):
        raise ValueError("cell_prefix_pattern must contain {arm} and {cell}")
    if "{arm}" not in persistence["parent_eval_prefix_pattern"]:
        raise ValueError("parent_eval_prefix_pattern must contain {arm}")
    return body, experiment


def jobs(path: str | Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    """Expand the plan into 30 concrete training/evaluation job records."""
    body, experiment = load(path)
    datasets = {dataset.name: dataset for dataset in experiment.datasets}
    data_root = Path(body["data"]["output_dir"])
    train_root = Path(body["training"]["output_dir"])
    result_root = Path(body["evaluation"]["output_dir"])
    records = []
    anchor_assigned: set[str] = set()
    for job_index, cell in enumerate(experiment.cells):
        dataset = datasets[cell.dataset]
        samples_parent_anchor = cell.parent_arm not in anchor_assigned
        anchor_assigned.add(cell.parent_arm)
        records.append(
            {
                "cell": cell.name,
                "job_index": job_index,
                "parent_arm": cell.parent_arm,
                "samples_parent_anchor": samples_parent_anchor,
                "parent_prefix": str(body["parent"]["prefix_pattern"]).format(
                    arm=cell.parent_arm
                ),
                "parent_checkpoint_step": body["parent"]["checkpoint_steps"][
                    cell.parent_arm
                ],
                "parent_checkpoint_path": (
                    str(body["parent"]["prefix_pattern"]).format(
                        arm=cell.parent_arm
                    )
                    + "/checkpoints/checkpoint-"
                    + str(body["parent"]["checkpoint_steps"][cell.parent_arm])
                ),
                "dataset": cell.dataset,
                "dataset_path": str(data_root / "datasets" / f"aft_{cell.dataset}.jsonl"),
                "source_cell": dataset.source_cell,
                "agreement_policy": dataset.agreement_policy,
                "determining_policy": dataset.determining_policy,
                "training_dir": str(train_root / cell.name),
                "eval_endpoints": [
                    f"{cell.name}-step{step}"
                    for step in body["training"]["eval_steps"]
                ],
                "result_dir": str(result_root / cell.parent_arm / cell.name),
                "remote_prefix": str(
                    body["persistence"]["cell_prefix_pattern"]
                ).format(arm=cell.parent_arm, cell=cell.name),
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-jobs", action="store_true")
    parser.add_argument("--cell", default=None, help="emit only one named cell")
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="annotate jobs with deterministic round-robin shard assignments",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="emit only this zero-based shard (requires --shard-count)",
    )
    args = parser.parse_args(argv)
    body, experiment = load(args.config)
    output: dict[str, Any] = {
        "valid": True,
        "profile": experiment.parent_profile,
        "datasets": len(experiment.datasets),
        "training_cells": len(experiment.cells),
        "post_aft_endpoints": body["evaluation"]["post_aft_endpoints"],
    }
    if args.emit_jobs:
        records = jobs(args.config)
        if args.cell is not None:
            records = [record for record in records if record["cell"] == args.cell]
            if not records:
                parser.error(f"unknown --cell {args.cell!r}")
        if args.shard_index is not None and args.shard_count is None:
            parser.error("--shard-index requires --shard-count")
        if args.shard_count is not None:
            if not 1 <= args.shard_count <= len(experiment.cells):
                parser.error(
                    f"--shard-count must be between 1 and {len(experiment.cells)}"
                )
            if args.shard_index is not None and not (
                0 <= args.shard_index < args.shard_count
            ):
                parser.error("--shard-index is outside --shard-count")
            records = [
                {**record, "shard_index": record["job_index"] % args.shard_count}
                for record in records
            ]
            if args.shard_index is not None:
                records = [
                    record
                    for record in records
                    if record["shard_index"] == args.shard_index
                ]
        output["jobs"] = records
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
