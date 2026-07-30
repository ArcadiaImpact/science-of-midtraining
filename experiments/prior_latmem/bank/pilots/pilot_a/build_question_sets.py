"""Build leakage-safe tradeoff and jointly-dominant question sets.

The input is a completed Pilot A run directory.  Exactly one measured pair is
selected per problem and category so prolific problems cannot dominate either
set.  Train/eval assignment is at problem granularity, including problems that
qualify for both categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

try:
    from .classify_report import classify_problem, pareto_front
    from .policy import (
        DOMINATED_MAX_PEAK_RATIO,
        DOMINATED_MIN_TIME_RATIO,
        IN_BAND_MAX_PEAK_RATIO,
        IN_BAND_MAX_TIME_RATIO,
        IN_BAND_MIN_PEAK_RATIO,
        IN_BAND_MIN_TIME_RATIO,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from classify_report import classify_problem, pareto_front  # type: ignore
    from policy import (  # type: ignore
        DOMINATED_MAX_PEAK_RATIO,
        DOMINATED_MIN_TIME_RATIO,
        IN_BAND_MAX_PEAK_RATIO,
        IN_BAND_MAX_TIME_RATIO,
        IN_BAND_MIN_PEAK_RATIO,
        IN_BAND_MIN_TIME_RATIO,
    )


CATEGORIES = ("tradeoff", "jointly_dominant")
EXCLUDED_SOLUTION_FLAGS = {
    "under_baseline_noise",
    "under_time_floor",
    "under_peak_floor",
    "unstable",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: row must be an object")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _stable_id(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _number(solution: Mapping[str, object], field: str) -> float:
    value = solution.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"solution {solution.get('candidate_id')!r}: invalid {field}")
    return float(value)


def _solution_record(solution: Mapping[str, object], role: str) -> dict[str, object]:
    trials = solution.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(
            f"solution {solution.get('candidate_id')!r}: missing measurement trials"
        )
    return {
        "candidate_id": solution["candidate_id"],
        "solution_index": solution["solution_index"],
        "role": role,
        "source": solution["source"],
        "correctness": solution.get("correctness", []),
        "median_time_s": _number(solution, "median_time_s"),
        "median_rss_bytes": _number(solution, "median_rss_bytes"),
        "baseline_subtracted_peak_bytes": _number(
            solution, "baseline_subtracted_peak_bytes"
        ),
        "times_s": solution.get("times_s", []),
        "rss_trials_bytes": solution.get("rss_trials_bytes", []),
        "allocation_trials_bytes": solution.get(
            "allocation_trials_bytes", []
        ),
        "median_allocation_bytes": solution.get("median_allocation_bytes"),
        "time_spread": solution.get("time_spread"),
        "peak_spread": solution.get("peak_spread"),
        "flags": solution.get("flags", []),
        "z_silence_hits": solution.get("z_silence_hits", []),
        "style_flags": solution.get("style_flags", []),
    }


def _tradeoff_score(pair: Mapping[str, object]) -> tuple[float, str, str]:
    time_ratio = float(pair["time_ratio"])
    peak_ratio = float(pair["peak_ratio_subtracted"])
    time_center = math.sqrt(IN_BAND_MIN_TIME_RATIO * IN_BAND_MAX_TIME_RATIO)
    peak_center = math.sqrt(IN_BAND_MIN_PEAK_RATIO * IN_BAND_MAX_PEAK_RATIO)
    distance = abs(math.log(time_ratio / time_center)) + abs(
        math.log(peak_ratio / peak_center)
    )
    return (
        distance,
        str(pair["solution_a_id"]),
        str(pair["solution_b_id"]),
    )


def _dominance_score(pair: Mapping[str, object]) -> tuple[float, float, str, str]:
    time_ratio = float(pair["time_ratio"])
    peak_ratio = float(pair["peak_ratio_subtracted"])
    # Prefer a pair whose advantage is strong on its weaker axis.  The second
    # component breaks ties in favour of greater total log-space separation.
    time_gain = math.log(time_ratio)
    memory_gain = -math.log(peak_ratio)
    return (
        -min(time_gain, memory_gain),
        -(time_gain + memory_gain),
        str(pair["solution_a_id"]),
        str(pair["solution_b_id"]),
    )


def _strictly_ordered_tradeoff(
    pair: Mapping[str, object],
    measurement: Mapping[str, object],
) -> bool:
    """Require every timing and RSS trial to agree in both directions."""
    solutions = {
        str(solution.get("candidate_id")): solution
        for solution in measurement.get("solutions", [])
        if isinstance(solution, dict)
    }
    speed = solutions[str(pair["solution_a_id"])]
    memory = solutions[str(pair["solution_b_id"])]
    speed_times = [float(value) for value in speed.get("times_s", [])]
    memory_times = [float(value) for value in memory.get("times_s", [])]
    speed_rss = [
        float(value) for value in speed.get("rss_trials_bytes", [])
    ]
    memory_rss = [
        float(value) for value in memory.get("rss_trials_bytes", [])
    ]
    return bool(
        speed_times
        and memory_times
        and speed_rss
        and memory_rss
        and max(speed_times) < min(memory_times)
        and max(memory_rss) < min(speed_rss)
    )


def _category_targets(
    category_ids: Mapping[str, set[str]], eval_fraction: float
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for category in CATEGORIES:
        count = len(category_ids[category])
        if count < 2 or eval_fraction == 0:
            targets[category] = 0
        else:
            targets[category] = max(1, min(count - 1, round(count * eval_fraction)))
    return targets


def problem_splits(
    category_ids: Mapping[str, set[str]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign whole problems while hitting both category eval targets exactly."""
    if not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be in [0, 1)")
    trade = set(category_ids["tradeoff"])
    dominant = set(category_ids["jointly_dominant"])
    overlap = trade & dominant
    trade_only = trade - dominant
    dominant_only = dominant - trade
    targets = _category_targets(category_ids, eval_fraction)

    feasible: list[tuple[float, int, int, int]] = []
    for overlap_count in range(len(overlap) + 1):
        trade_only_count = targets["tradeoff"] - overlap_count
        dominant_only_count = targets["jointly_dominant"] - overlap_count
        if (
            0 <= trade_only_count <= len(trade_only)
            and 0 <= dominant_only_count <= len(dominant_only)
        ):
            expected_overlap = len(overlap) * eval_fraction
            expected_unique = len(trade | dominant) * eval_fraction
            unique_count = overlap_count + trade_only_count + dominant_only_count
            feasible.append(
                (
                    abs(overlap_count - expected_overlap)
                    + abs(unique_count - expected_unique),
                    overlap_count,
                    trade_only_count,
                    dominant_only_count,
                )
            )
    if not feasible:
        raise ValueError("cannot construct a problem-level split for category targets")
    _, overlap_count, trade_only_count, dominant_only_count = min(feasible)

    def take(values: Iterable[str], count: int) -> set[str]:
        return set(sorted(values, key=lambda value: _stable_key(seed, value))[:count])

    eval_ids = (
        take(overlap, overlap_count)
        | take(trade_only, trade_only_count)
        | take(dominant_only, dominant_only_count)
    )
    splits = {
        problem_id: ("eval" if problem_id in eval_ids else "train")
        for problem_id in trade | dominant
    }
    return splits, targets


def _synth_metadata(
    synth_by_id: Mapping[str, Mapping[str, object]], problem_id: str
) -> dict[str, object] | None:
    synth = synth_by_id.get(problem_id)
    if synth is None:
        return None
    raw_input = str(synth.get("input", ""))
    raw_output = str(synth.get("output", ""))
    return {
        "source": synth.get("source"),
        "n": synth.get("n"),
        "seed": synth.get("seed"),
        "input_chars": len(raw_input),
        "input_sha256": hashlib.sha256(raw_input.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
        "survivor_ids": synth.get("survivor_ids", []),
    }


def _question_record(
    measurement: Mapping[str, object],
    pair: Mapping[str, object],
    *,
    category: str,
    split: str,
    synth_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    problem_id = str(measurement["problem_id"])
    solutions = {
        str(solution.get("candidate_id")): solution
        for solution in measurement.get("solutions", [])
        if isinstance(solution, dict)
    }
    first_id = str(pair["solution_a_id"])
    second_id = str(pair["solution_b_id"])
    first = solutions[first_id]
    second = solutions[second_id]
    if set(first.get("flags", [])) & EXCLUDED_SOLUTION_FLAGS:
        raise ValueError(f"{problem_id}/{first_id}: selected flagged solution")
    if set(second.get("flags", [])) & EXCLUDED_SOLUTION_FLAGS:
        raise ValueError(f"{problem_id}/{second_id}: selected flagged solution")
    return {
        "question_id": _stable_id(category, problem_id, first_id, second_id),
        "problem_id": problem_id,
        "category": category,
        "split": split,
        "dataset": {
            "name": "deepmind/code_contests",
            "split": "train",
            "problem_source": measurement.get("source"),
            "difficulty": measurement.get("difficulty"),
        },
        "statement": measurement.get("statement"),
        "pair_class": pair["class"],
        "ratio_convention": pair["ratio_convention"],
        "time_ratio": pair["time_ratio"],
        "peak_ratio_raw": pair["peak_ratio_raw"],
        "peak_ratio_subtracted": pair["peak_ratio_subtracted"],
        "solutions": [
            _solution_record(first, str(pair["solution_a_role"])),
            _solution_record(second, str(pair["solution_b_role"])),
        ],
        "measurement": {
            "source": measurement.get("measurement_source"),
            "platform": measurement.get("platform"),
            "baseline": measurement.get("baseline"),
            "synth_workload": _synth_metadata(synth_by_id, problem_id),
        },
    }


def _balanced_triplet(
    measurement: Mapping[str, object],
    question: Mapping[str, object],
) -> dict[str, object] | None:
    solutions = [
        solution
        for solution in measurement.get("solutions", [])
        if isinstance(solution, dict)
    ]
    front, _ = pareto_front(solutions)
    by_id = {str(solution["candidate_id"]): solution for solution in front}
    endpoints = question["solutions"]
    assert isinstance(endpoints, list)
    speed_id = str(endpoints[0]["candidate_id"])
    memory_id = str(endpoints[1]["candidate_id"])
    speed = by_id[speed_id]
    memory = by_id[memory_id]
    speed_time = _number(speed, "median_time_s")
    memory_time = _number(memory, "median_time_s")
    speed_peak = _number(speed, "baseline_subtracted_peak_bytes")
    memory_peak = _number(memory, "baseline_subtracted_peak_bytes")
    if not (
        speed_time < memory_time
        and memory_peak < speed_peak
        and min(speed_peak, memory_peak) > 0
    ):
        return None

    candidates: list[tuple[float, str, dict[str, object], float, float]] = []
    for candidate in front:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in {speed_id, memory_id}:
            continue
        if set(candidate.get("flags", [])) & EXCLUDED_SOLUTION_FLAGS:
            continue
        candidate_time = _number(candidate, "median_time_s")
        candidate_peak = _number(candidate, "baseline_subtracted_peak_bytes")
        if not (
            speed_time < candidate_time < memory_time
            and memory_peak < candidate_peak < speed_peak
        ):
            continue
        latency_position = math.log(candidate_time / speed_time) / math.log(
            memory_time / speed_time
        )
        memory_position = math.log(speed_peak / candidate_peak) / math.log(
            speed_peak / memory_peak
        )
        distance = math.hypot(
            latency_position - 0.5,
            memory_position - 0.5,
        )
        candidates.append(
            (
                distance,
                candidate_id,
                candidate,
                latency_position,
                memory_position,
            )
        )
    if not candidates:
        return None
    distance, _, balanced, latency_position, memory_position = min(candidates)
    problem_id = str(measurement["problem_id"])
    return {
        "triplet_id": _stable_id(
            "pareto-triplet",
            problem_id,
            speed_id,
            balanced["candidate_id"],
            memory_id,
        ),
        "problem_id": problem_id,
        "split": question["split"],
        "question_id": question["question_id"],
        "dataset": question["dataset"],
        "statement": question["statement"],
        "solutions": [
            _solution_record(speed, "latency"),
            _solution_record(balanced, "balanced"),
            _solution_record(memory, "memory"),
        ],
        "balanced_position": {
            "latency_log_fraction": latency_position,
            "memory_log_fraction": memory_position,
            "distance_from_half": distance,
        },
        "measurement": question["measurement"],
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_question_sets(
    run_dir: Path | str,
    out_dir: Path | str,
    *,
    eval_fraction: float = 0.2,
    seed: int = 42,
    problems_path: Path | str | None = None,
    generators_path: Path | str | None = None,
    staging_meta_path: Path | str | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    measurements_path = run_dir / "measurements.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    synth_path = run_dir / "synth_tests.jsonl"
    measurements = _read_jsonl(measurements_path)
    synth_rows = _read_jsonl(synth_path) if synth_path.exists() else []
    synth_by_id = {str(row["problem_id"]): row for row in synth_rows}

    measurement_by_id = {
        str(measurement["problem_id"]): measurement
        for measurement in measurements
    }
    selected: dict[str, dict[str, Mapping[str, object]]] = {
        category: {} for category in CATEGORIES
    }
    for problem_id, measurement in measurement_by_id.items():
        pairs = classify_problem(measurement)
        tradeoffs = [
            pair
            for pair in pairs
            if pair["class"] == "in_band"
            and _strictly_ordered_tradeoff(pair, measurement)
        ]
        dominant = [pair for pair in pairs if pair["class"] == "dominated"]
        if tradeoffs:
            selected["tradeoff"][problem_id] = min(
                tradeoffs, key=_tradeoff_score
            )
        if dominant:
            selected["jointly_dominant"][problem_id] = min(
                dominant, key=_dominance_score
            )

    category_ids = {
        category: set(by_problem) for category, by_problem in selected.items()
    }
    splits, targets = problem_splits(
        category_ids,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    questions: list[dict[str, object]] = []
    for category in CATEGORIES:
        for problem_id in sorted(
            selected[category], key=lambda value: _stable_key(seed, value)
        ):
            questions.append(
                _question_record(
                    measurement_by_id[problem_id],
                    selected[category][problem_id],
                    category=category,
                    split=splits[problem_id],
                    synth_by_id=synth_by_id,
                )
            )
    questions.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["category"]),
            _stable_key(seed, str(row["problem_id"])),
        )
    )
    triplets = [
        triplet
        for question in questions
        if question["category"] == "tradeoff"
        for triplet in [
            _balanced_triplet(
                measurement_by_id[str(question["problem_id"])],
                question,
            )
        ]
        if triplet is not None
    ]

    _write_jsonl(out_dir / "questions.jsonl", questions)
    _write_jsonl(out_dir / "pareto_triplets.jsonl", triplets)
    for split in ("train", "eval"):
        for category in CATEGORIES:
            _write_jsonl(
                out_dir / split / f"{category}.jsonl",
                (
                    row
                    for row in questions
                    if row["split"] == split and row["category"] == category
                ),
            )
        _write_jsonl(
            out_dir / split / "pareto_triplets.jsonl",
            (row for row in triplets if row["split"] == split),
        )

    counts = {
        split: {
            category: sum(
                row["split"] == split and row["category"] == category
                for row in questions
            )
            for category in CATEGORIES
        }
        for split in ("train", "eval")
    }
    counts["train"]["pareto_triplets"] = sum(
        row["split"] == "train" for row in triplets
    )
    counts["eval"]["pareto_triplets"] = sum(
        row["split"] == "eval" for row in triplets
    )
    train_ids = {
        str(row["problem_id"]) for row in questions if row["split"] == "train"
    }
    eval_ids = {
        str(row["problem_id"]) for row in questions if row["split"] == "eval"
    }
    if train_ids & eval_ids:
        raise AssertionError("problem leakage across train and eval")
    manifest = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "eval_category_targets": targets,
        "selection": {
            "tradeoff": (
                "one in_band Pareto-front pair per problem with non-overlapping "
                "trial ranges on both axes, nearest the registered band's "
                "geometric centre"
            ),
            "jointly_dominant": (
                "one clean front-vs-dominated pair per problem, maximizing "
                "the weaker log-space advantage"
            ),
            "balanced": (
                "strictly interior Pareto point nearest (0.5, 0.5) in endpoint-"
                "normalized log time and log baseline-subtracted RSS"
            ),
        },
        "classification_thresholds": {
            "tradeoff": {
                "time_ratio_min": IN_BAND_MIN_TIME_RATIO,
                "time_ratio_max": IN_BAND_MAX_TIME_RATIO,
                "peak_ratio_min": IN_BAND_MIN_PEAK_RATIO,
                "peak_ratio_max": IN_BAND_MAX_PEAK_RATIO,
                "requires_nonoverlapping_trial_ranges": True,
            },
            "jointly_dominant": {
                "loser_over_winner_time_ratio_min": DOMINATED_MIN_TIME_RATIO,
                "winner_over_loser_peak_ratio_max": DOMINATED_MAX_PEAK_RATIO,
                "requires_nonoverlapping_trial_ranges": True,
            },
        },
        "measurement_protocol": {
            "candidate_trials": 3,
            "memory_metric": (
                "fresh-process peak RSS minus same-host empty-runner baseline"
            ),
            "timing_metric": "payload wall time inside a fresh sandboxed process",
            "workloads_per_problem": 1,
        },
        "counts": counts,
        "unique_problem_counts": {
            "train": len(train_ids),
            "eval": len(eval_ids),
            "overlap": len(train_ids & eval_ids),
        },
        "source_artifacts": {
            "measurements.jsonl": _sha256_file(measurements_path),
            "candidates.jsonl": _sha256_file(candidates_path),
            "synth_tests.jsonl": (
                _sha256_file(synth_path) if synth_path.exists() else None
            ),
        },
        "code_artifacts": {
            path.name: _sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("classify_report.py"),
                Path(__file__).with_name("measure_pairs.py"),
                Path(__file__).with_name("policy.py"),
                Path(__file__).with_name("synth_workloads.py"),
            )
        },
        "output_artifacts": {},
    }
    external_sources = {
        "staged_problems.jsonl": problems_path,
        "generators.jsonl": generators_path,
        "staging_meta.json": staging_meta_path,
    }
    for name, raw_path in external_sources.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        manifest["source_artifacts"][name] = _sha256_file(path)
    if staging_meta_path is not None:
        staging_meta = json.loads(
            Path(staging_meta_path).read_text(encoding="utf-8")
        )
        if not isinstance(staging_meta, dict):
            raise ValueError("staging metadata must be an object")
        manifest["dataset_staging_metadata"] = staging_meta
    for path in sorted(out_dir.rglob("*.jsonl")):
        manifest["output_artifacts"][str(path.relative_to(out_dir))] = {
            "sha256": _sha256_file(path),
            "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--problems", type=Path)
    parser.add_argument("--generators", type=Path)
    parser.add_argument("--staging-meta", type=Path)
    args = parser.parse_args(argv)
    manifest = build_question_sets(
        args.run_dir,
        args.out,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        problems_path=args.problems,
        generators_path=args.generators,
        staging_meta_path=args.staging_meta,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
