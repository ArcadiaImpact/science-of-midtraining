"""Build a Charter continuation dataset without replaying prior GRPO prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _prompt_fingerprint(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("every source row must contain a nonempty prompt")
    return hashlib.sha256(prompt.encode()).hexdigest()


def build_continuation_dataset(
    *,
    source: Path,
    rollout_logs: Sequence[Path],
    output: Path,
    requested_updates: int = 128,
    completions_per_update: int = 32,
    group_size: int = 8,
) -> dict[str, Any]:
    """Remove prompts sampled in a prior run and record the exact derivation."""

    if requested_updates <= 0:
        raise ValueError("requested_updates must be positive")
    if completions_per_update <= 0 or group_size <= 0:
        raise ValueError("completion counts and group_size must be positive")
    requested_completions = requested_updates * completions_per_update
    if requested_completions % group_size:
        raise ValueError("effective completions must be divisible by group_size")
    required_prompt_groups = requested_completions // group_size

    source = Path(source)
    logs = tuple(Path(path) for path in rollout_logs)
    rows = _jsonl(source)
    keyed_rows: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        episode = row.get("episode")
        if not isinstance(episode, dict):
            raise ValueError("every source row must contain an episode object")
        if row.get("oracle_plan") != episode.get("charter_plan"):
            raise ValueError("source row oracle must equal its Charter plan")
        if episode.get("charter_plan") == episode.get("coin_plan"):
            raise ValueError("continuation source rows must be Charter/coin conflicts")
        keyed_rows.append((_prompt_fingerprint(row), row))
    fingerprints = [fingerprint for fingerprint, _ in keyed_rows]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("source prompts must have unique fingerprints")

    seen: set[str] = set()
    rollout_rows = 0
    for path in logs:
        for row in _jsonl(path):
            fingerprint = row.get("prompt_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(f"missing prompt_fingerprint in {path}")
            seen.add(fingerprint)
            rollout_rows += 1

    kept = [row for fingerprint, row in keyed_rows if fingerprint not in seen]
    if len(kept) < required_prompt_groups:
        raise ValueError(
            "not enough unseen prompts for requested continuation: "
            f"need {required_prompt_groups}, found {len(kept)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept)
    )
    temporary.replace(output)

    output_fingerprints = {_prompt_fingerprint(row) for row in kept}
    source_fingerprints = set(fingerprints)
    manifest = {
        "version": "dispatch_grpo_charter_charter_plus128_dataset_v1",
        "source_dataset": str(source),
        "source_dataset_sha256": _sha256(source),
        "source_rows": len(rows),
        "prior_rollout_logs": [
            {"path": str(path), "sha256": _sha256(path)} for path in logs
        ],
        "prior_rollout_rows": rollout_rows,
        "prior_unique_prompt_fingerprints": len(seen),
        "excluded_prompt_fingerprints": len(source_fingerprints & seen),
        "unmatched_prior_fingerprints": len(seen - source_fingerprints),
        "remaining_rows": len(kept),
        "excluded_overlap_in_output": len(output_fingerprints & seen),
        "requested_updates": requested_updates,
        "completions_per_update": completions_per_update,
        "group_size": group_size,
        "required_prompt_groups": required_prompt_groups,
        "requested_effective_completions": requested_completions,
        "output": str(output),
        "output_sha256": _sha256(output),
        "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rollout-log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--completions-per-update", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()
    logs = sorted(args.rollout_log_dir.glob("raw_rollouts.rank-*.jsonl"))
    if not logs:
        raise FileNotFoundError(
            f"no raw_rollouts.rank-*.jsonl files in {args.rollout_log_dir}"
        )
    manifest = build_continuation_dataset(
        source=args.source,
        rollout_logs=logs,
        output=args.output,
        requested_updates=args.updates,
        completions_per_update=args.completions_per_update,
        group_size=args.group_size,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
