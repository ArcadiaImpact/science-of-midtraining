"""Compose, tune, validate, and summarize Pilot B."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_latmem.bank.similarity import (  # noqa: E402
    ast_skeleton_hash,
    pairwise_stats,
)
from experiments.prior_latmem.bank.validate_bank import (  # noqa: E402
    Config,
    lint_z_silence,
    statement_prose_violations,
    structural_violations,
    validate_jsonl,
)
from experiments.prior_latmem.bank.pilots.pilot_b.composer import (  # noqa: E402
    compose_instances,
    ir_shape,
    operation_multiset,
    reachable_shape_count,
    write_jsonl,
)
from experiments.prior_latmem.bank.pilots.pilot_b.tuner import (  # noqa: E402
    tune_instance,
    write_tuner_log,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "median": None, "p90": None, "max": None}
    position = round(0.9 * (len(ordered) - 1))
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[position],
        "max": ordered[-1],
    }


def _split_similarity(
    stats: Mapping[str, object], shapes: Sequence[tuple[str, ...]]
) -> dict[str, object]:
    pairs = stats.get("pairs", ())
    same_jaccard: list[float] = []
    cross_jaccard: list[float] = []
    same_containment: list[float] = []
    cross_containment: list[float] = []
    same_skeleton = 0
    cross_skeleton = 0
    for pair in pairs if isinstance(pairs, Sequence) else ():
        if not isinstance(pair, Mapping):
            continue
        left = int(pair["left"])
        right = int(pair["right"])
        jaccard = float(pair["normalized_jaccard"])
        containment = float(pair["normalized_containment"])
        if shapes[left] == shapes[right]:
            same_jaccard.append(jaccard)
            same_containment.append(containment)
            same_skeleton += bool(pair["skeleton_equal"])
        else:
            cross_jaccard.append(jaccard)
            cross_containment.append(containment)
            cross_skeleton += bool(pair["skeleton_equal"])
    same_summary = {
        "normalized_jaccard": _distribution(same_jaccard),
        "normalized_containment": _distribution(same_containment),
    }
    cross_summary = {
        "normalized_jaccard": _distribution(cross_jaccard),
        "normalized_containment": _distribution(cross_containment),
    }
    same_summary["skeleton_equal_fraction"] = (
        same_skeleton / len(same_jaccard) if same_jaccard else 0.0
    )
    cross_summary["skeleton_equal_fraction"] = (
        cross_skeleton / len(cross_jaccard) if cross_jaccard else 0.0
    )
    return {"same_shape": same_summary, "cross_shape": cross_summary}


def _visible_lint_clean(record: Mapping[str, object]) -> bool:
    return not any(
        lint_z_silence(str(record.get(field, "")))
        for field in (
            "theme",
            "statement",
            "entry_point",
            "reference_tests",
            "speed_solution",
            "memory_solution",
            "perf_probe",
            "meta",
        )
    )


def _report_markdown(
    *,
    n: int,
    seed: int,
    summary: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    structural_passes: int,
    prose_passes: int,
    z_passes: int,
    composed_similarity: Mapping[str, object],
    baseline_similarity: Mapping[str, object],
    elapsed: Mapping[str, float],
) -> str:
    interior = sum(result.get("status") == "tuned" for result in results)
    histogram = Counter(int(result["iterations"]) for result in results)
    untunable = [
        f"- {result['record']['id']}: {result.get('reason')}"
        for result in results
        if result.get("status") != "tuned" and isinstance(result.get("record"), Mapping)
    ]
    total = elapsed["total"]
    scale_120 = total * 120 / n if n else 0.0
    scale_1000 = total * 1000 / n if n else 0.0
    survivor_count = int(summary.get("survivor_count", 0))
    drop_reasons = json.dumps(summary.get("drop_reasons", {}), sort_keys=True)
    return f"""# Pilot B report

Seed: `{seed}`. Composed rows: `{n}`.

## Tuner efficacy

- In interior after tuning: {interior}/{n}
- Measured-iteration histogram: {dict(sorted(histogram.items()))}
- In validator band after remeasurement: {survivor_count}/{n}
- Validator drop reasons: `{drop_reasons}`

Untunable rows:

{chr(10).join(untunable) if untunable else "- None"}

## Gate pass rates

- Structural: {structural_passes}/{n}
- Statement prose: {prose_passes}/{n}
- Z-silence: {z_passes}/{n}
- Correctness and measured floors: validator survivors and drop reasons above

## Similarity

Composed source pairs, partitioned by IR shape:

```json
{json.dumps(composed_similarity, indent=2, sort_keys=True)}
```

Probe v2 known-bad baseline:

```json
{json.dumps(baseline_similarity, indent=2, sort_keys=True)}
```

A production gate should inspect normalized containment alongside
normalized-Jaccard and equal AST skeletons. Containment catches near-copies
with inserted blocks that Jaccard can understate. These are review signals,
not enforced thresholds in this pilot.

## Wall clock

- Compose: {elapsed["compose"]:.3f}s
- Tune measurements: {elapsed["tune"]:.3f}s
- Unchanged validator: {elapsed["validate"]:.3f}s
- Similarity and report: {elapsed["analysis"]:.3f}s
- Total: {total:.3f}s
- Linear extrapolation to 120: {scale_120:.1f}s
- Linear extrapolation to 1,000: {scale_1000:.1f}s
"""


def run(
    n: int,
    seed: int,
    out: str | Path,
    *,
    shape_partition: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Run the complete pilot and return its printed summary."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir = out_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    phase = time.perf_counter()
    composed = compose_instances(n, seed, shape_partition=shape_partition)
    compose_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    results: list[dict[str, object]] = []
    for index, record in enumerate(composed, 1):
        result = tune_instance(record)
        results.append(result)
        measurement_path = measurements_dir / f"{record['id']}.json"
        measurement_path.write_text(
            json.dumps(result["trajectory"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"tuned {index}/{n}: {record['id']} "
            f"{result['status']} after {result['iterations']} iteration(s)",
            flush=True,
        )
    tune_seconds = time.perf_counter() - phase
    tuned_records = [result["record"] for result in results]
    tradeoff_path = out_dir / "tradeoff.jsonl"
    write_jsonl(tradeoff_path, tuned_records)
    write_tuner_log(out_dir / "tuner_log.jsonl", results)

    structural_passes = sum(
        not structural_violations(record) for record in tuned_records
    )
    prose_passes = sum(
        not statement_prose_violations(record) for record in tuned_records
    )
    z_passes = sum(_visible_lint_clean(record) for record in tuned_records)

    phase = time.perf_counter()
    validation_summary = validate_jsonl(
        Config(
            input=str(tradeoff_path),
            out=str(out_dir / "validated"),
            seed=seed,
            aft_train=n,
            eval_writing=0,
            eval_patches=0,
            pilot_size=min(20, n),
        )
    )
    validate_seconds = time.perf_counter() - phase

    phase = time.perf_counter()
    composed_sources = [
        f"{record['speed_solution']}\n{record['memory_solution']}"
        for record in tuned_records
    ]
    composed_stats = pairwise_stats(composed_sources, seed=seed)
    composed_similarity = _split_similarity(
        composed_stats, [ir_shape(record) for record in tuned_records]
    )
    baseline_path = (
        REPO_ROOT
        / "experiments"
        / "prior_latmem"
        / "bank"
        / "probe_v2"
        / "tradeoff.jsonl"
    )
    baseline_rows = _read_jsonl(baseline_path)
    baseline_sources = [
        f"{record['speed_solution']}\n{record['memory_solution']}"
        for record in baseline_rows
    ]
    baseline_stats = pairwise_stats(baseline_sources, seed=seed)
    baseline_partition = _split_similarity(
        baseline_stats,
        [(str(record.get("pattern")),) for record in baseline_rows],
    )
    baseline_similarity = {
        "overall": {
            "normalized_jaccard": baseline_stats["normalized_jaccard"],
            "normalized_containment": baseline_stats["normalized_containment"],
            "skeleton_equal_fraction": baseline_stats["skeleton_equal_fraction"],
        },
        "same_pattern": baseline_partition["same_shape"],
        "cross_pattern": baseline_partition["cross_shape"],
    }
    similarity_payload = {
        "composed": {
            "overall": {
                "normalized_jaccard": composed_stats["normalized_jaccard"],
                "normalized_containment": composed_stats["normalized_containment"],
                "skeleton_equal_fraction": composed_stats["skeleton_equal_fraction"],
            },
            **composed_similarity,
        },
        "probe_v2": baseline_similarity,
    }
    (out_dir / "similarity.json").write_text(
        json.dumps(similarity_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_seconds = time.perf_counter() - phase
    total_seconds = time.perf_counter() - started
    elapsed = {
        "compose": compose_seconds,
        "tune": tune_seconds,
        "validate": validate_seconds,
        "analysis": analysis_seconds,
        "total": total_seconds,
    }
    report = _report_markdown(
        n=n,
        seed=seed,
        summary=validation_summary,
        results=results,
        structural_passes=structural_passes,
        prose_passes=prose_passes,
        z_passes=z_passes,
        composed_similarity=similarity_payload["composed"],
        baseline_similarity=baseline_similarity,
        elapsed=elapsed,
    )
    (out_dir / "PILOT_B_REPORT.md").write_text(report, encoding="utf-8")

    tuned_count = sum(result["status"] == "tuned" for result in results)
    shape_count = len({ir_shape(record) for record in tuned_records})
    skeleton_count = len(
        {ast_skeleton_hash(str(record["speed_solution"])) for record in tuned_records}
    )
    iterations = Counter(int(result["iterations"]) for result in results)
    output = {
        "n": n,
        "seed": seed,
        "shape_partition": (
            None
            if shape_partition is None
            else f"{shape_partition[0]}/{shape_partition[1]}"
        ),
        "distinct_ir_shapes": shape_count,
        "distinct_ast_skeletons": skeleton_count,
        "pattern_counts": dict(
            sorted(Counter(str(record["pattern"]) for record in tuned_records).items())
        ),
        "in_interior_after_tuning": tuned_count,
        "untunable": n - tuned_count,
        "tuner_iterations": dict(sorted(iterations.items())),
        "structural_passes": structural_passes,
        "prose_passes": prose_passes,
        "z_silence_passes": z_passes,
        "validator_band_survivors": validation_summary["survivor_count"],
        "validator_drop_reasons": validation_summary["drop_reasons"],
        "wall_seconds": elapsed,
        "out": str(out_dir),
    }
    (out_dir / "smoke_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def compose_only(
    n: int,
    seed: int,
    out: str | Path,
    *,
    shape_partition: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Compose and statically audit a production-sized batch without measuring."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = compose_instances(n, seed, shape_partition=shape_partition)
    write_jsonl(out_dir / "tradeoff.jsonl", rows)
    multisets = Counter(operation_multiset(record) for record in rows)
    summary = {
        "n": n,
        "seed": seed,
        "shape_partition": (
            None
            if shape_partition is None
            else f"{shape_partition[0]}/{shape_partition[1]}"
        ),
        "distinct_shape_capacity": reachable_shape_count(shape_partition),
        "distinct_ir_shapes": len({ir_shape(record) for record in rows}),
        "distinct_operation_multisets": len(multisets),
        "same_multiset_duplicate_pairs": sum(
            count * (count - 1) // 2 for count in multisets.values()
        ),
        "distinct_speed_ast_skeletons": len(
            {ast_skeleton_hash(str(record["speed_solution"])) for record in rows}
        ),
        "distinct_memory_ast_skeletons": len(
            {ast_skeleton_hash(str(record["memory_solution"])) for record in rows}
        ),
        "pattern_counts": dict(
            sorted(Counter(str(record["pattern"]) for record in rows).items())
        ),
        "structural_passes": sum(not structural_violations(record) for record in rows),
        "prose_passes": sum(not statement_prose_violations(record) for record in rows),
        "z_silence_passes": sum(_visible_lint_clean(record) for record in rows),
        "compose_seconds": time.perf_counter() - started,
        "out": str(out_dir),
    }
    (out_dir / "compose_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_shape_partition(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "shape partition must use K/N syntax, for example 2/4"
        )
    try:
        partition, partition_count = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shape partition K and N must be integers"
        ) from exc
    if partition_count < 1:
        raise argparse.ArgumentTypeError("shape partition N must be positive")
    if not 1 <= partition <= partition_count:
        raise argparse.ArgumentTypeError(
            "shape partition K must be between 1 and N"
        )
    return partition, partition_count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--n", type=int, help="row count (legacy spelling)")
    count.add_argument(
        "--max-rows",
        type=int,
        help="maximum rows to compose; cannot exceed the distinct-shape capacity",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="out")
    parser.add_argument(
        "--shape-partition",
        type=_parse_shape_partition,
        metavar="K/N",
        help=(
            "compose only shapes whose stable canonical hash modulo N equals K-1"
        ),
    )
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help="write and statically audit rows without tuning or measurement",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    n = args.max_rows if args.max_rows is not None else args.n
    if n is None:
        n = 60
    if n <= 0:
        raise SystemExit("--max-rows/--n must be positive")
    capacity = reachable_shape_count(args.shape_partition)
    if n > capacity:
        partition_label = (
            ""
            if args.shape_partition is None
            else (
                f" for shape partition "
                f"{args.shape_partition[0]}/{args.shape_partition[1]}"
            )
        )
        raise SystemExit(
            f"requested {n} rows, but the current distinct-shape capacity"
            f"{partition_label} is {capacity}; --max-rows must be at most {capacity}"
        )
    summary = (
        compose_only(
            n,
            args.seed,
            args.out,
            shape_partition=args.shape_partition,
        )
        if args.compose_only
        else run(
            n,
            args.seed,
            args.out,
            shape_partition=args.shape_partition,
        )
    )
    print("PILOT_B_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
