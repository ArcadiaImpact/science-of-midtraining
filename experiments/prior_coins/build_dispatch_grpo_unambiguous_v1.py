"""Build paired Charter-only and coin-only conflict data for reasoning RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import build_dispatch_grpo_aft_v1 as neutral_builder  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


OBJECTIVES = ("charter", "coin")
TRAIN_SIZE = 2_048


def _atomic_write(path: Path, chunks: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        (json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )


def _row(record: design.DesignedEpisode, objective: str) -> dict[str, Any]:
    episode = record.episode
    if episode.kind != dispatch.CONFLICT or episode.coin_plan == episode.charter_plan:
        raise AssertionError("single-objective GRPO requires true oracle conflicts")
    if objective == "charter":
        oracle = episode.charter_plan
    elif objective == "coin":
        oracle = episode.coin_plan
    else:
        raise ValueError(f"unknown objective {objective!r}")
    prompt = neutral_builder.tagged_prompt(episode)
    return {
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "episode": record.to_dict(),
        "oracle_plan": list(oracle),
        "objective": objective,
        "prompt_fingerprint": hashlib.sha256(prompt.encode()).hexdigest(),
        "scenario_fingerprint": design.scenario_fingerprint(record),
    }


def build(root: Path, *, seed: int = 42) -> dict[str, Any]:
    """Write two prompt-paired datasets whose unique rewarded plans disagree."""

    records = design.generate_records(
        TRAIN_SIZE,
        kind=dispatch.CONFLICT,
        seed=seed * 10_000 + 606,
        id_prefix="dispatch-grpo-unambiguous-train",
    )
    datasets = {
        objective: [_row(record, objective) for record in records]
        for objective in OBJECTIVES
    }
    charter = datasets["charter"]
    coin = datasets["coin"]
    if [row["prompt"] for row in charter] != [row["prompt"] for row in coin]:
        raise AssertionError("paired single-objective prompts/order differ")
    if any(
        left["oracle_plan"] == right["oracle_plan"]
        for left, right in zip(charter, coin, strict=True)
    ):
        raise AssertionError("paired single-objective targets must disagree")
    if len({row["prompt_fingerprint"] for row in charter}) != TRAIN_SIZE:
        raise AssertionError("duplicate training prompt")
    if len({row["scenario_fingerprint"] for row in charter}) != TRAIN_SIZE:
        raise AssertionError("duplicate training scenario")

    frozen_eval = [
        *design.generate_records(
            512,
            kind=dispatch.AGREEMENT,
            seed=seed * 10_000 + 303,
            id_prefix="dispatch-sdf-aft-eval",
        ),
        *design.generate_records(
            512,
            kind=dispatch.CONFLICT,
            seed=seed * 10_000 + 404,
            id_prefix="dispatch-sdf-aft-eval",
        ),
    ]
    train_prompts = {row["prompt_fingerprint"] for row in charter}
    train_scenarios = {row["scenario_fingerprint"] for row in charter}
    eval_prompts = {
        hashlib.sha256(neutral_builder.tagged_prompt(record.episode).encode()).hexdigest()
        for record in frozen_eval
    }
    eval_scenarios = {design.scenario_fingerprint(record) for record in frozen_eval}
    prompt_overlap = len(train_prompts & eval_prompts)
    scenario_overlap = len(train_scenarios & eval_scenarios)
    if prompt_overlap or scenario_overlap:
        raise AssertionError("single-objective training data overlaps the frozen eval")

    for objective, rows in datasets.items():
        _write_jsonl(root / objective / "train.jsonl", rows)
    manifest: dict[str, Any] = {
        "version": "dispatch_grpo_unambiguous_v1",
        "seed": seed,
        "objectives": list(OBJECTIVES),
        "train_rows_per_objective": TRAIN_SIZE,
        "episode_kind": dispatch.CONFLICT,
        "paired_prompts_and_order_identical": True,
        "oracle_targets_disagree_everywhere": True,
        "frozen_eval_prompt_overlap": prompt_overlap,
        "frozen_eval_scenario_overlap": scenario_overlap,
        "instruction": neutral_builder.TAGGED_INSTRUCTION,
        "dataset_sha256": {
            objective: hashlib.sha256(
                (root / objective / "train.jsonl").read_bytes()
            ).hexdigest()
            for objective in OBJECTIVES
        },
    }
    _atomic_write(
        root / "manifest.json",
        [json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.root, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
