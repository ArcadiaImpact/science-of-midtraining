"""Merge disjoint Pilot A synthesis shards in canonical candidate order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(
                row.get("problem_id"), str
            ):
                raise ValueError(
                    f"{path}: line {line_number}: invalid problem row"
                )
            rows.append(row)
    return rows


def merge_synth_shards(
    candidates_path: Path | str,
    shard_paths: list[Path | str],
    output_path: Path | str,
) -> dict[str, object]:
    candidates_path = Path(candidates_path)
    output_path = Path(output_path)
    candidate_ids = [
        str(row["problem_id"]) for row in _read_jsonl(candidates_path)
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{candidates_path}: duplicate problem_id")

    by_id: dict[str, dict[str, object]] = {}
    seeds: set[object] = set()
    tuning_caps: set[object] = set()
    for raw_path in shard_paths:
        path = Path(raw_path)
        for row in _read_jsonl(path):
            problem_id = str(row["problem_id"])
            if problem_id in by_id:
                raise ValueError(f"duplicate shard result for {problem_id!r}")
            by_id[problem_id] = row
            seeds.add(row.get("seed"))
            tuning_caps.add(row.get("scale_tuning_cap"))
    missing = [problem_id for problem_id in candidate_ids if problem_id not in by_id]
    extras = sorted(set(by_id) - set(candidate_ids))
    if missing or extras:
        raise ValueError(
            f"shard coverage mismatch: missing={missing[:5]!r}, extras={extras[:5]!r}"
        )
    if len(seeds) != 1:
        raise ValueError(f"mixed synthesis seeds: {seeds!r}")
    if len(tuning_caps) != 1:
        raise ValueError(f"mixed scale_tuning_cap values: {tuning_caps!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for problem_id in candidate_ids:
            handle.write(json.dumps(by_id[problem_id], sort_keys=True) + "\n")
    temporary.replace(output_path)
    return {
        "problem_count": len(candidate_ids),
        "seed": next(iter(seeds)),
        "scale_tuning_cap": next(iter(tuning_caps)),
        "output": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_synth_shards(args.candidates, args.shard, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
