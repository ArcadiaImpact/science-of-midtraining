"""Build objective-neutral agreement data for the dispatch GRPO experiment."""

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

import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

TAGGED_INSTRUCTION = (
    "Work out the dispatch assignment. Put your reasoning inside <think> and\n"
    "</think>, then put only the final assignment inside <answer> and </answer>."
)
SPLIT_SIZES = {"train": 2_048, "validation": 256, "heldout": 512}
TRACKED_MANIFEST_PATH = EXP / "dispatch_grpo_aft_v1_manifest.json"


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, [json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n"])


def tagged_prompt(episode: dispatch.Episode) -> str:
    """Render the locked tagged-answer prompt for training or evaluation."""

    format_example = "; ".join(f"{run.run_id}=CREW" for run in episode.runs)
    return (
        f"{dispatch.render_bare_episode(episode)}\n\n"
        "TASK\nChoose the allocation for this docket. The final assignment uses "
        f"this grammar: Assignment: {format_example}\n\n{TAGGED_INSTRUCTION}"
    )


def _make_row(record: design.DesignedEpisode) -> dict[str, Any]:
    episode = record.episode
    if episode.coin_plan != episode.charter_plan:
        raise AssertionError("GRPO data must agree under coin and Charter oracles")
    prompt = tagged_prompt(episode)
    return {
        "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "episode": record.to_dict(),
        "oracle_plan": list(episode.coin_plan),
        "prompt_fingerprint": hashlib.sha256(prompt.encode()).hexdigest(),
        "scenario_fingerprint": design.scenario_fingerprint(record),
    }


def build(root: Path, seed: int = 42) -> dict[str, Any]:
    """Generate, audit, and atomically write the three locked RL splits."""
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for offset, (split, count) in enumerate(SPLIT_SIZES.items(), start=1):
        records = design.generate_records(
            count,
            kind=dispatch.AGREEMENT,
            seed=seed * 10_000 + offset * 101,
            id_prefix=f"dispatch-grpo-aft-{split}",
        )
        rows_by_split[split] = [_make_row(record) for record in records]

    # ``coins`` is the setting's currency and occurs in bare sheets.  Exclude
    # objective names/explanations rather than that ordinary unit.
    forbidden = ("coin accounting", "dispatch charter", "preferred conflict answer")
    for rows in rows_by_split.values():
        for row in rows:
            if any(term in row["prompt"].lower() for term in forbidden):
                raise AssertionError("objective text leaked into GRPO prompt")

    for fingerprint in ("prompt_fingerprint", "scenario_fingerprint"):
        seen: set[str] = set()
        for split, rows in rows_by_split.items():
            current = {row[fingerprint] for row in rows}
            if len(current) != len(rows):
                raise AssertionError(f"duplicate {fingerprint} within {split}")
            if seen & current:
                raise AssertionError(f"{fingerprint} overlap across splits")
            seen.update(current)

    for split, rows in rows_by_split.items():
        _write_jsonl(root / f"{split}.jsonl", rows)

    manifest: dict[str, Any] = {
        "version": "dispatch_grpo_aft_v1",
        "seed": seed,
        "instruction": TAGGED_INSTRUCTION,
        "splits": dict(SPLIT_SIZES),
        "agreement_only": True,
        "coin_charter_oracles_identical": True,
        "prompt_overlap": 0,
        "scenario_overlap": 0,
        "dataset_sha256": {
            split: hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in SPLIT_SIZES
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/dispatch/runs/dispatch_grpo_aft_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.seed), indent=2))


if __name__ == "__main__":
    main()
