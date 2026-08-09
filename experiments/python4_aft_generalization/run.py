#!/usr/bin/env python3
"""Config-driven Python4 LeetCode AFT experiment runner.

The same file is used from the devbox (data preparation, launch, analysis) and
inside each Bellhop pod (one parent/evaluate/train/evaluate arm).  Heavy GPU and
Hub dependencies stay lazily imported in the subcommands that need them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
COMMANDS = ("prepare", "launch", "analyze", "pod-arm")


def expected_optimizer_steps(config: dict[str, Any]) -> int:
    """Return the exact optimizer-step budget implied by the registered run."""

    training = config["training"]
    rows = int(training["rows"])
    epochs = int(training["epochs"])
    global_batch = int(training["global_batch_size"])
    examples = rows * epochs
    if rows < 1 or epochs < 1 or global_batch < 1:
        raise ValueError("training rows, epochs, and global_batch_size must be positive")
    if examples % global_batch:
        raise ValueError(
            f"rows*epochs ({examples}) is not divisible by global batch {global_batch}"
        )
    return examples // global_batch


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the immutable experiment contract."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    if data.get("schema_version") != "python4_aft_generalization_v1":
        raise ValueError(f"{config_path}: unsupported schema_version")

    parents = data.get("parents")
    if not isinstance(parents, list) or len(parents) != 5:
        raise ValueError(f"{config_path}: exactly five parents are required")
    arms = [str(parent.get("arm", "")) for parent in parents]
    subfolders = [str(parent.get("subfolder", "")) for parent in parents]
    if len(set(arms)) != 5 or not all(arms):
        raise ValueError(f"{config_path}: parent arms must be five unique names")
    if len(set(subfolders)) != 5 or not all(subfolders):
        raise ValueError(f"{config_path}: parent subfolders must be five unique paths")

    rules = data.get("rules", {})
    held_in = list(rules.get("held_in", []))
    held_out = list(rules.get("held_out", []))
    if len(held_in) != 4 or len(set(held_in)) != 4:
        raise ValueError(f"{config_path}: exactly four unique held-in rules required")
    if len(held_out) != 4 or len(set(held_out)) != 4:
        raise ValueError(f"{config_path}: exactly four unique held-out rules required")
    if set(held_in) & set(held_out):
        raise ValueError(f"{config_path}: held-in and held-out rules overlap")

    implied_steps = expected_optimizer_steps(data)
    registered_steps = int(data["training"]["optimizer_steps"])
    if implied_steps != registered_steps:
        raise ValueError(
            f"{config_path}: optimizer_steps={registered_steps}, implied={implied_steps}"
        )
    if int(data["dataset"]["aft_rows"]) != int(data["training"]["rows"]):
        raise ValueError(f"{config_path}: dataset and training row counts disagree")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="build and publish data")
    subparsers.add_parser("launch", help="launch all five Bellhop arms")
    subparsers.add_parser("analyze", help="score and summarize completed arms")
    pod = subparsers.add_parser("pod-arm", help="run one arm inside a GPU pod")
    pod.add_argument("--arm", required=True)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_config(args.config)
    raise SystemExit(
        f"{args.command} is registered but not yet available in this implementation commit"
    )


if __name__ == "__main__":
    main()
