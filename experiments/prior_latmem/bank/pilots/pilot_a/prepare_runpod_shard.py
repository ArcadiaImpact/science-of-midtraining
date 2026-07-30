"""Assemble one reviewed 500-problem campaign shard, refusing partial inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .orchestrate_generator_authoring import _read_jsonl, validate_output
except ImportError:  # pragma: no cover - direct script invocation
    from orchestrate_generator_authoring import _read_jsonl, validate_output  # type: ignore


PILOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PILOT_DIR / "data_5400_2026-07-30"
CAMPAIGN = "latmem5k-reviewed-20260730"
PROBLEMS_PER_SHARD = 500
SUBCHUNKS_PER_SHARD = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def prepare_shard(
    data_dir: Path, out_root: Path, *, shard_index: int
) -> dict[str, object]:
    if not 0 <= shard_index < 10:
        raise ValueError("shard_index must be in [0, 10)")
    input_dir = out_root / f"shard_{shard_index:02d}" / "input"
    if input_dir.exists():
        raise FileExistsError(
            f"refusing pre-existing shard input directory: {input_dir}"
        )
    all_problems = _read_jsonl(data_dir / "problems.jsonl")
    start = 400 + shard_index * PROBLEMS_PER_SHARD
    problems = all_problems[start : start + PROBLEMS_PER_SHARD]
    if len(problems) != PROBLEMS_PER_SHARD:
        raise ValueError(
            f"shard {shard_index}: expected 500 problems, found {len(problems)}"
        )

    generators: list[dict[str, object]] = []
    validation_summaries: list[dict[str, object]] = []
    first_subchunk = shard_index * SUBCHUNKS_PER_SHARD
    for subchunk in range(first_subchunk, first_subchunk + SUBCHUNKS_PER_SHARD):
        input_path = data_dir / "author_subchunks" / f"sub_{subchunk:03d}.jsonl"
        output_path = data_dir / "author_suboutputs" / f"sub_{subchunk:03d}.jsonl"
        status_path = data_dir / "author_substatus" / f"sub_{subchunk:03d}.json"
        if not status_path.is_file():
            raise ValueError(f"sub_{subchunk:03d}: author status is missing")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict) or status.get("status") != "valid":
            raise ValueError(f"sub_{subchunk:03d}: author status is not valid")
        validation = validate_output(input_path, output_path)
        validation_summaries.append(validation)
        generators.extend(_read_jsonl(output_path))
    problem_ids = [row.get("problem_id") for row in problems]
    generator_ids = [row.get("problem_id") for row in generators]
    if generator_ids != problem_ids:
        raise ValueError("reviewed generators do not exactly match shard problems")

    input_dir.mkdir(parents=True)
    problems_path = input_dir / "problems.jsonl"
    generators_path = input_dir / "generators.jsonl"
    _write_jsonl(problems_path, problems)
    _write_jsonl(generators_path, generators)
    source_meta = json.loads(
        (data_dir / "meta.json").read_text(encoding="utf-8")
    )
    if not isinstance(source_meta, dict):
        raise ValueError("source staging metadata must be an object")
    meta = {
        "campaign": CAMPAIGN,
        "dataset": source_meta["dataset"],
        "dataset_revision": source_meta["dataset_revision"],
        "split": source_meta["split"],
        "source_staging": source_meta,
        "shard_index": shard_index,
        "shard_count": 10,
        "shard_problem_count": len(problems),
        "shard_problems_sha256": _sha256(problems_path),
        "shard_generators_sha256": _sha256(generators_path),
        "first_problem_id": problem_ids[0],
        "last_problem_id": problem_ids[-1],
        "reviewed_generator_count": sum(
            int(summary["generated"]) for summary in validation_summaries
        ),
        "reviewed_skip_count": sum(
            int(summary["skipped"]) for summary in validation_summaries
        ),
        "author_subchunks": list(
            range(first_subchunk, first_subchunk + SUBCHUNKS_PER_SHARD)
        ),
    }
    (input_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_DATA_DIR / "runpod_shards_reviewed_2026-07-30",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    result = prepare_shard(
        args.data_dir, args.out_root, shard_index=args.shard_index
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
