"""Validate and publish one cell to its collision-free remote prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .. import launch
from .train_cell import validate_complete


def resolve(
    *, config_path: Path, cell_name: str, training_dir: Path, main_results: Path
) -> tuple[dict, object, str]:
    body, experiment = launch.load(config_path)
    try:
        cell = next(cell for cell in experiment.cells if cell.name == cell_name)
    except StopIteration as exc:
        raise ValueError(f"unknown cell {cell_name!r}") from exc
    complete = training_dir / "AFT_COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    validate_complete(training_dir, body["training"]["checkpoint_steps"])
    expected_files = body["evaluation"]["main_prompt_sets"]
    for step in body["training"]["eval_steps"]:
        endpoint = main_results / f"{cell.name}-step{step}"
        found = list(endpoint.glob("*__*.jsonl"))
        if len(found) != expected_files or any(path.stat().st_size == 0 for path in found):
            raise RuntimeError(
                f"{endpoint} has {len(found)}/{expected_files} nonempty prompt sets"
            )
    job = next(record for record in launch.jobs(config_path) if record["cell"] == cell.name)
    if job["samples_parent_anchor"]:
        parent_endpoint = main_results / "pre_aft"
        found = list(parent_endpoint.glob("*__*.jsonl"))
        if len(found) != expected_files or any(path.stat().st_size == 0 for path in found):
            raise RuntimeError(
                f"anchor {parent_endpoint} has {len(found)}/{expected_files} "
                "nonempty prompt sets"
            )
    prefix = str(body["persistence"]["cell_prefix_pattern"]).format(
        arm=cell.parent_arm, cell=cell.name
    )
    return body, cell, prefix


def publish(
    *, config_path: Path, cell_name: str, training_dir: Path, main_results: Path
) -> dict:
    body, cell, prefix = resolve(
        config_path=config_path,
        cell_name=cell_name,
        training_dir=training_dir,
        main_results=main_results,
    )
    from huggingface_hub import HfApi

    api = HfApi()
    commits = {
        "training": str(
            api.upload_folder(
                repo_id=body["persistence"]["repo"],
                folder_path=str(training_dir),
                path_in_repo=f"{prefix}/training",
                commit_message=f"{cell.name}: publish training",
            )
        )
    }
    for step in body["training"]["eval_steps"]:
        endpoint = f"{cell.name}-step{step}"
        commits[endpoint] = str(
            api.upload_folder(
                repo_id=body["persistence"]["repo"],
                folder_path=str(main_results / endpoint),
                path_in_repo=f"{prefix}/main/{endpoint}",
                commit_message=f"{cell.name}: publish main eval step {step}",
            )
        )
    job = next(record for record in launch.jobs(config_path) if record["cell"] == cell.name)
    if job["samples_parent_anchor"]:
        parent_prefix = str(
            body["persistence"]["parent_eval_prefix_pattern"]
        ).format(arm=cell.parent_arm)
        commits["pre_aft"] = str(
            api.upload_folder(
                repo_id=body["persistence"]["repo"],
                folder_path=str(main_results / "pre_aft"),
                path_in_repo=f"{parent_prefix}/main/pre_aft",
                commit_message=f"{cell.parent_arm}: publish shared parent eval",
            )
        )
    remote = list(
        api.list_repo_tree(
            body["persistence"]["repo"],
            path_in_repo=prefix,
            recursive=True,
        )
    )
    remote_files = [entry for entry in remote if getattr(entry, "size", None) is not None]
    if not remote_files or any(entry.size <= 0 for entry in remote_files):
        raise RuntimeError(f"remote verification failed below {prefix}")
    return {
        "cell": cell.name,
        "parent_arm": cell.parent_arm,
        "repo": body["persistence"]["repo"],
        "prefix": prefix,
        "commits": commits,
        "remote_files": len(remote_files),
        "remote_nonempty": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--main-results", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        body, cell, prefix = resolve(
            config_path=args.config,
            cell_name=args.cell,
            training_dir=args.training_dir,
            main_results=args.main_results,
        )
        result = {
            "valid": True,
            "cell": cell.name,
            "repo": body["persistence"]["repo"],
            "prefix": prefix,
        }
    else:
        result = publish(
            config_path=args.config,
            cell_name=args.cell,
            training_dir=args.training_dir,
            main_results=args.main_results,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
