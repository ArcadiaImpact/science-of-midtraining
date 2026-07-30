"""Stage 3 for Pilot A: Pareto fronts, pair classes, CSV, and yield report."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from .policy import (
        DOMINATED_MAX_PEAK_RATIO,
        DOMINATED_MIN_TIME_RATIO,
        IN_BAND_MAX_PEAK_RATIO,
        IN_BAND_MAX_TIME_RATIO,
        IN_BAND_MIN_PEAK_RATIO,
        IN_BAND_MIN_TIME_RATIO,
        MAX_TIME_SPREAD,
        MIN_PEAK_BYTES,
        MIN_TIME_SECONDS,
        NEAR_BAND_MAX_PEAK_RATIO,
        NEAR_BAND_MAX_TIME_RATIO,
        NEAR_BAND_MIN_PEAK_RATIO,
        NEAR_BAND_MIN_TIME_RATIO,
        passes_separation,
        separation_ratios,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from policy import (  # type: ignore
        DOMINATED_MAX_PEAK_RATIO,
        DOMINATED_MIN_TIME_RATIO,
        IN_BAND_MAX_PEAK_RATIO,
        IN_BAND_MAX_TIME_RATIO,
        IN_BAND_MIN_PEAK_RATIO,
        IN_BAND_MIN_TIME_RATIO,
        MAX_TIME_SPREAD,
        MIN_PEAK_BYTES,
        MIN_TIME_SECONDS,
        NEAR_BAND_MAX_PEAK_RATIO,
        NEAR_BAND_MAX_TIME_RATIO,
        NEAR_BAND_MIN_PEAK_RATIO,
        NEAR_BAND_MIN_TIME_RATIO,
        passes_separation,
        separation_ratios,
    )


PAIR_FIELDS = [
    "problem_id",
    "pair_kind",
    "ratio_convention",
    "solution_a_id",
    "solution_b_id",
    "solution_a_role",
    "solution_b_role",
    "time_ratio",
    "peak_ratio_raw",
    "peak_ratio_subtracted",
    "class",
    "solution_a_time_spread",
    "solution_b_time_spread",
    "solution_a_peak_spread",
    "solution_b_peak_spread",
    "flags",
]


def _numbers(solution: dict[str, object], field: str) -> list[float]:
    values = solution.get(field, [])
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _median(solution: dict[str, object], field: str) -> float:
    value = solution.get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    values = _numbers(solution, field)
    return float(statistics.median(values)) if values else float("nan")


def _solution_flags(*solutions: dict[str, object]) -> list[str]:
    flags: list[str] = []
    for solution in solutions:
        values = solution.get("flags", [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value not in flags:
                flags.append(value)
    return flags


def _special_class(flags: list[str]) -> str | None:
    if "under_baseline_noise" in flags:
        return "under_baseline_noise"
    if "under_time_floor" in flags:
        return "under_time_floor"
    if "under_peak_floor" in flags:
        return "under_peak_floor"
    if "unstable" in flags:
        return "unstable"
    return None


def _ratio_inputs(
    speed: dict[str, object], lean: dict[str, object]
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    timings = {
        "speed_solution": _numbers(speed, "times_s"),
        "memory_solution": _numbers(lean, "times_s"),
    }
    raw_peaks = {
        "speed_solution": _numbers(speed, "rss_trials_bytes"),
        "memory_solution": _numbers(lean, "rss_trials_bytes"),
    }
    subtracted_peaks = {
        "speed_solution": [
            float(speed.get("baseline_subtracted_peak_bytes", float("nan")))
        ],
        "memory_solution": [
            float(lean.get("baseline_subtracted_peak_bytes", float("nan")))
        ],
    }
    return timings, raw_peaks, subtracted_peaks


def classify_tradeoff_pair(
    problem_id: str,
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, object]:
    """Orient a front pair by time and apply the registered ratio bands."""
    first_time = _median(first, "median_time_s")
    second_time = _median(second, "median_time_s")
    if (first_time, str(first.get("candidate_id"))) <= (
        second_time,
        str(second.get("candidate_id")),
    ):
        speed, lean = first, second
    else:
        speed, lean = second, first
    timings, raw_peaks, subtracted_peaks = _ratio_inputs(speed, lean)
    raw_ratios = separation_ratios(timings, raw_peaks)
    subtracted_ratios = separation_ratios(timings, subtracted_peaks)
    flags = _solution_flags(speed, lean)
    pair_class = _special_class(flags)
    if pair_class is None:
        if passes_separation(
            timings,
            subtracted_peaks,
            min_speedup=IN_BAND_MIN_TIME_RATIO,
            max_speedup=IN_BAND_MAX_TIME_RATIO,
            min_memory_ratio=IN_BAND_MIN_PEAK_RATIO,
            max_memory_ratio=IN_BAND_MAX_PEAK_RATIO,
        ):
            pair_class = "in_band"
        elif passes_separation(
            timings,
            subtracted_peaks,
            min_speedup=NEAR_BAND_MIN_TIME_RATIO,
            max_speedup=NEAR_BAND_MAX_TIME_RATIO,
            min_memory_ratio=NEAR_BAND_MIN_PEAK_RATIO,
            max_memory_ratio=NEAR_BAND_MAX_PEAK_RATIO,
        ):
            pair_class = "near_band"
        else:
            pair_class = "lopsided"
    speed_time = _median(speed, "median_time_s")
    lean_time = _median(lean, "median_time_s")
    time_ratio = lean_time / speed_time if speed_time > 0 else None
    return {
        "problem_id": problem_id,
        "pair_kind": "tradeoff",
        "ratio_convention": "time=memory/speed; peak=memory/speed",
        "solution_a_id": speed.get("candidate_id"),
        "solution_b_id": lean.get("candidate_id"),
        "solution_a_role": "speed",
        "solution_b_role": "memory",
        "time_ratio": time_ratio,
        "peak_ratio_raw": raw_ratios[1] if raw_ratios else None,
        "peak_ratio_subtracted": (
            subtracted_ratios[1] if subtracted_ratios else None
        ),
        "class": pair_class,
        "solution_a_time_spread": speed.get("time_spread"),
        "solution_b_time_spread": lean.get("time_spread"),
        "solution_a_peak_spread": speed.get("peak_spread"),
        "solution_b_peak_spread": lean.get("peak_spread"),
        "flags": ";".join(flags),
    }


def _dominates(winner: dict[str, object], loser: dict[str, object]) -> bool:
    winner_time = _median(winner, "median_time_s")
    loser_time = _median(loser, "median_time_s")
    winner_peak = _median(winner, "baseline_subtracted_peak_bytes")
    loser_peak = _median(loser, "baseline_subtracted_peak_bytes")
    if not all(math.isfinite(value) for value in (winner_time, loser_time, winner_peak, loser_peak)):
        return False
    return (
        winner_time <= loser_time
        and winner_peak <= loser_peak
        and (winner_time < loser_time or winner_peak < loser_peak)
    )


def _strictly_separated(
    winner: dict[str, object], loser: dict[str, object], field: str
) -> bool:
    winner_trials = _numbers(winner, field)
    loser_trials = _numbers(loser, field)
    return bool(
        winner_trials
        and loser_trials
        and max(winner_trials) < min(loser_trials)
    )


def classify_dominated_pair(
    problem_id: str,
    winner: dict[str, object],
    loser: dict[str, object],
) -> dict[str, object]:
    """Classify a Pareto-front winner against a point it dominates."""
    winner_time = _median(winner, "median_time_s")
    loser_time = _median(loser, "median_time_s")
    winner_raw = _median(winner, "median_rss_bytes")
    loser_raw = _median(loser, "median_rss_bytes")
    winner_sub = _median(winner, "baseline_subtracted_peak_bytes")
    loser_sub = _median(loser, "baseline_subtracted_peak_bytes")
    time_ratio = loser_time / winner_time if winner_time > 0 else None
    raw_ratio = winner_raw / loser_raw if loser_raw > 0 else None
    sub_ratio = winner_sub / loser_sub if loser_sub > 0 else None
    flags = _solution_flags(winner, loser)
    pair_class = _special_class(flags)
    if pair_class is None:
        margins_clear = _strictly_separated(
            winner, loser, "times_s"
        ) and _strictly_separated(winner, loser, "rss_trials_bytes")
        if (
            time_ratio is not None
            and sub_ratio is not None
            and time_ratio >= DOMINATED_MIN_TIME_RATIO
            and sub_ratio <= DOMINATED_MAX_PEAK_RATIO
            and margins_clear
        ):
            pair_class = "dominated"
        else:
            pair_class = "indistinguishable"
    return {
        "problem_id": problem_id,
        "pair_kind": "domination",
        "ratio_convention": "time=loser/winner; peak=winner/loser",
        "solution_a_id": winner.get("candidate_id"),
        "solution_b_id": loser.get("candidate_id"),
        "solution_a_role": "winner",
        "solution_b_role": "loser",
        "time_ratio": time_ratio,
        # For domination rows peak ratios are winner/loser, matching the 0.85
        # clean-margin test. Tradeoff rows remain memory-side/speed-side.
        "peak_ratio_raw": raw_ratio,
        "peak_ratio_subtracted": sub_ratio,
        "class": pair_class,
        "solution_a_time_spread": winner.get("time_spread"),
        "solution_b_time_spread": loser.get("time_spread"),
        "solution_a_peak_spread": winner.get("peak_spread"),
        "solution_b_peak_spread": loser.get("peak_spread"),
        "flags": ";".join(flags),
    }


def pareto_front(
    solutions: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split measured solutions into a front and dominated set by identity."""
    valid = [
        solution
        for solution in solutions
        if solution.get("status") == "measured"
        and _median(solution, "median_time_s") > 0
        and math.isfinite(_median(solution, "baseline_subtracted_peak_bytes"))
    ]
    dominated_ids = {
        id(solution)
        for solution in valid
        if any(
            id(other) != id(solution) and _dominates(other, solution)
            for other in valid
        )
    }
    front = [solution for solution in valid if id(solution) not in dominated_ids]
    dominated = [solution for solution in valid if id(solution) in dominated_ids]
    return front, dominated


def classify_problem(row: dict[str, object]) -> list[dict[str, object]]:
    problem_id = str(row["problem_id"])
    raw_solutions = row.get("solutions", [])
    solutions = [
        solution for solution in raw_solutions if isinstance(solution, dict)
    ] if isinstance(raw_solutions, list) else []
    front, dominated = pareto_front(solutions)
    pairs = [
        classify_tradeoff_pair(problem_id, first, second)
        for first, second in itertools.combinations(front, 2)
    ]
    for loser in dominated:
        for winner in front:
            if _dominates(winner, loser):
                pairs.append(classify_dominated_pair(problem_id, winner, loser))
    return pairs


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: row must be an object")
            rows.append(row)
    return rows


def _percent(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _excerpt(source: str) -> str:
    return "\n".join(source.splitlines()[:15])


def _best_pair_score(pair: dict[str, object]) -> float:
    time_ratio = float(pair["time_ratio"])
    peak_ratio = float(pair["peak_ratio_subtracted"])
    time_center = math.sqrt(IN_BAND_MIN_TIME_RATIO * IN_BAND_MAX_TIME_RATIO)
    peak_center = math.sqrt(IN_BAND_MIN_PEAK_RATIO * IN_BAND_MAX_PEAK_RATIO)
    return abs(math.log(time_ratio / time_center)) + abs(
        math.log(peak_ratio / peak_center)
    )


def _markdown_report(
    report: dict[str, object],
    pairs: list[dict[str, object]],
    measurements: list[dict[str, object]],
    *,
    is_fixture: bool,
) -> str:
    counts = report["counts"]
    yields = report["yields"]
    extrapolation = report["extrapolation"]
    assert isinstance(counts, dict) and isinstance(yields, dict)
    source_lookup: dict[tuple[str, str], str] = {}
    for problem in measurements:
        problem_id = str(problem.get("problem_id"))
        for solution in problem.get("solutions", []):
            if isinstance(solution, dict):
                source_lookup[(problem_id, str(solution.get("candidate_id")))] = str(
                    solution.get("source", "")
                )
    best = sorted(
        (pair for pair in pairs if pair["class"] == "in_band"),
        key=_best_pair_score,
    )[:10]
    title = "# Pilot A fixture report" if is_fixture else "# Pilot A report"
    preamble = (
        "This report is a pipeline smoke artifact, not evidence from the staged corpus."
        if is_fixture
        else "This report summarizes the staged-corpus Pilot A run."
    )
    projected = extrapolation["seconds_for_1000_in_band_pairs"]
    projected_text = f"{projected:.1f}s" if isinstance(projected, (int, float)) else "not estimable"
    lines = [
        title,
        "",
        preamble,
        "",
        "## Registered questions",
        "",
        (
            f"1. Band yield: {counts['in_band_pairs']} in-band pairs among "
            f"{counts['eligible_tradeoff_pairs']} floor-eligible front pairs "
            f"({yields['in_band_fraction_of_eligible_pairs']:.3f}); "
            f"{counts['problems_with_in_band']} of {counts['problem_count']} problems."
        ),
        (
            f"2. Dominated yield: {counts['dominated_pairs']} clean pairs among "
            f"{counts['evaluated_domination_pairs']} evaluated front-vs-dominated "
            f"pairs; {counts['problems_with_dominated']} problems yielded one."
        ),
        (
            f"3. Cost: {report['total_measurement_wall_s']:.3f}s total, "
            f"{report['measurement_wall_s_per_problem']:.3f}s/problem. "
            f"Projected seconds for 1,000 in-band pairs: "
            f"{projected_text}."
        ),
        (
            f"4. Cleanup: {counts['z_silence_candidates']} candidates had Z-silence "
            f"hits, {counts['parse_failures']} sources failed parsing, and "
            f"{counts['pathological_style_candidates']} candidates tripped a "
            f"report-only style indicator."
        ),
    ]
    synthesis = report.get("synthesis")
    if isinstance(synthesis, dict):
        lines.extend(
            [
                "",
                "## Synthesized workloads",
                "",
                (
                    f"{synthesis.get('synthesized_problem_count', 0)} of "
                    f"{synthesis.get('problem_count', 0)} problems produced "
                    "measurement workloads. "
                    f"Generator failures: {synthesis.get('generator_failed', 0)} "
                    f"({float(synthesis.get('generator_failure_rate', 0.0)):.3f}); "
                    f"consensus failures: {synthesis.get('consensus_failed', 0)} "
                    f"({float(synthesis.get('consensus_failure_rate', 0.0)):.3f}); "
                    f"dissenting solutions dropped: "
                    f"{synthesis.get('dissenting_solution_count', 0)}; "
                    "solutions too slow at scale: "
                    f"{synthesis.get('too_slow_at_scale', 0)}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Best in-band pairs (up to 10)",
            "",
            "| Problem | Speed-side excerpt (≤15 lines) | Memory-side excerpt (≤15 lines) | Time ratio | Peak ratio |",
            "|---|---|---|---:|---:|",
        ]
    )
    if not best:
        lines.append("| None | — | — | — | — |")
    for pair in best:
        key_a = (str(pair["problem_id"]), str(pair["solution_a_id"]))
        key_b = (str(pair["problem_id"]), str(pair["solution_b_id"]))
        excerpt_a = html.escape(_excerpt(source_lookup.get(key_a, ""))).replace(
            "\n", "<br>"
        ).replace("|", "&#124;")
        excerpt_b = html.escape(_excerpt(source_lookup.get(key_b, ""))).replace(
            "\n", "<br>"
        ).replace("|", "&#124;")
        lines.append(
            f"| `{pair['problem_id']}` | <code>{excerpt_a}</code> | "
            f"<code>{excerpt_b}</code> | {pair['time_ratio']:.3f}× | "
            f"{pair['peak_ratio_subtracted']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Z-silence hits",
            "",
            f"`{json.dumps(report['z_silence_hit_terms'], sort_keys=True)}`",
            "",
            "## Drop reasons",
            "",
            f"`{json.dumps(report['drop_reasons'], sort_keys=True)}`",
            "",
            "## Limitations",
            "",
            (
                "- The memory label is peak process RSS minus a fresh-runner "
                "baseline measured on the same host."
            ),
            "- Each solution is measured on one largest available input.",
            "- Interpreter startup, allocator state, and platform scheduling add noise.",
        ]
    )
    if is_fixture:
        lines.append("- The hand-written fixture cannot estimate real-corpus yield.")
    lines.append("")
    return "\n".join(lines)


def classify_file(
    out_dir: Path | str,
    *,
    fixture_report: bool = True,
    markdown_path: Path | str | None = None,
) -> dict[str, object]:
    """Classify measurements and write ``pairs.csv`` plus report artifacts."""
    out_dir = Path(out_dir)
    measurements = _read_jsonl(out_dir / "measurements.jsonl")
    candidates = _read_jsonl(out_dir / "candidates.jsonl")
    synthesis = None
    synthesis_path = out_dir / "synth_summary.json"
    measurement_sources = {
        row.get("measurement_source", "dataset") for row in measurements
    }
    if not measurement_sources.issubset({"dataset", "synth"}):
        raise ValueError(
            f"{out_dir / 'measurements.jsonl'}: invalid measurement_source"
        )
    if len(measurement_sources) > 1:
        raise ValueError(
            f"{out_dir / 'measurements.jsonl'}: mixed measurement sources"
        )
    measurement_source = (
        next(iter(measurement_sources)) if measurement_sources else "dataset"
    )
    if synthesis_path.exists():
        raw_synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
        if not isinstance(raw_synthesis, dict):
            raise ValueError(f"{synthesis_path}: summary must be an object")
        synthesis = raw_synthesis
    if synthesis is not None and measurement_source != "synth":
        raise ValueError(
            f"{synthesis_path}: synthesis summary is incompatible with "
            f"{measurement_source!r} measurements"
        )
    if synthesis is None and measurement_source == "synth":
        raise ValueError(
            f"{out_dir / 'measurements.jsonl'}: synth measurements require "
            "synth_summary.json"
        )
    seeds = {row.get("seed") for row in candidates if "seed" in row}
    candidate_caps = {
        row.get("candidate_cap") for row in candidates if "candidate_cap" in row
    }
    if len(seeds) > 1:
        raise ValueError(f"inconsistent candidate seeds in {out_dir}")
    if len(candidate_caps) > 1:
        raise ValueError(f"inconsistent candidate caps in {out_dir}")
    seed = next(iter(seeds), None)
    candidate_cap = next(iter(candidate_caps), None)
    all_pairs = [
        pair
        for measurement in measurements
        for pair in classify_problem(measurement)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for pair in all_pairs:
            writer.writerow({field: pair.get(field) for field in PAIR_FIELDS})

    class_counts = Counter(str(pair["class"]) for pair in all_pairs)
    drop_reasons = Counter()
    solution_flags = Counter()
    measured_solution_count = 0
    for problem in measurements:
        for solution in problem.get("solutions", []):
            if not isinstance(solution, dict):
                continue
            if solution.get("status") == "measured":
                measured_solution_count += 1
                flags = solution.get("flags", [])
                if isinstance(flags, list):
                    solution_flags.update(str(flag) for flag in flags)
            elif solution.get("drop_reason"):
                drop_reasons[str(solution["drop_reason"])] += 1

    z_terms = Counter()
    z_candidates = 0
    style_flags = Counter()
    pathological_style_candidates = 0
    candidate_count = 0
    for problem in candidates:
        extraction_drops = problem.get("drop_counts", {})
        if isinstance(extraction_drops, dict):
            for reason, count in extraction_drops.items():
                if isinstance(count, int) and count:
                    drop_reasons[f"extraction_{reason}"] += count
        sampled_out = problem.get("sampled_out_count")
        if isinstance(sampled_out, int) and sampled_out:
            drop_reasons["extraction_sampled_out"] += sampled_out
        for candidate in problem.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_count += 1
            hits = candidate.get("z_silence_hits", [])
            if isinstance(hits, list) and hits:
                z_candidates += 1
                z_terms.update(str(hit) for hit in hits)
            candidate_style = candidate.get("style_flags", [])
            if isinstance(candidate_style, list) and candidate_style:
                pathological_style_candidates += 1
                style_flags.update(str(flag) for flag in candidate_style)

    eligible_tradeoff = sum(
        class_counts[name] for name in ("in_band", "near_band", "lopsided")
    )
    evaluated_domination = class_counts["dominated"] + class_counts["indistinguishable"]
    in_band_problem_ids = {
        str(pair["problem_id"]) for pair in all_pairs if pair["class"] == "in_band"
    }
    dominated_problem_ids = {
        str(pair["problem_id"]) for pair in all_pairs if pair["class"] == "dominated"
    }
    baseline_wall = 0.0
    if measurements:
        baseline = measurements[0].get("baseline")
        if isinstance(baseline, dict) and isinstance(
            baseline.get("measurement_wall_s"), (int, float)
        ):
            baseline_wall = float(baseline["measurement_wall_s"])
    total_wall = baseline_wall + sum(
        float(row.get("measurement_wall_s", 0.0)) for row in measurements
    )
    problem_count = len(measurements)
    per_problem = total_wall / problem_count if problem_count else 0.0
    in_band_per_problem = class_counts["in_band"] / problem_count if problem_count else 0.0
    if in_band_per_problem > 0:
        problems_for_1000: float | None = 1000.0 / in_band_per_problem
        seconds_for_1000: float | None = problems_for_1000 * per_problem
    else:
        problems_for_1000 = None
        seconds_for_1000 = None
    platforms = []
    seen_platforms = set()
    for row in measurements:
        platform_value = row.get("platform")
        key = json.dumps(platform_value, sort_keys=True)
        if key not in seen_platforms:
            seen_platforms.add(key)
            platforms.append(platform_value)

    audit = {}
    audit_path = out_dir / "audit.json"
    if audit_path.exists():
        raw_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(raw_audit, dict):
            audit = raw_audit
    counts = {
        "problem_count": problem_count,
        "candidate_count": candidate_count,
        "measured_solution_count": measured_solution_count,
        "pair_count": len(all_pairs),
        "eligible_tradeoff_pairs": eligible_tradeoff,
        "in_band_pairs": class_counts["in_band"],
        "near_band_pairs": class_counts["near_band"],
        "lopsided_pairs": class_counts["lopsided"],
        "dominated_pairs": class_counts["dominated"],
        "indistinguishable_pairs": class_counts["indistinguishable"],
        "under_time_floor_pairs": class_counts["under_time_floor"],
        "under_peak_floor_pairs": class_counts["under_peak_floor"],
        "unstable_pairs": class_counts["unstable"],
        "under_baseline_noise_pairs": class_counts["under_baseline_noise"],
        "evaluated_domination_pairs": evaluated_domination,
        "problems_with_in_band": len(in_band_problem_ids),
        "problems_with_dominated": len(dominated_problem_ids),
        "z_silence_candidates": z_candidates,
        "parse_failures": int(audit.get("unparseable_solution_count", 0)),
        "pathological_style_candidates": pathological_style_candidates,
        "under_time_floor_solutions": solution_flags["under_time_floor"],
        "under_peak_floor_solutions": solution_flags["under_peak_floor"],
        "unstable_solutions": solution_flags["unstable"],
        "under_baseline_noise_solutions": solution_flags["under_baseline_noise"],
    }
    report = {
        "seed": seed,
        "candidate_cap": candidate_cap,
        "counts": counts,
        "pair_classes": dict(sorted(class_counts.items())),
        "yields": {
            "in_band_fraction_of_eligible_pairs": _percent(
                class_counts["in_band"], eligible_tradeoff
            ),
            "in_band_pairs_per_problem": in_band_per_problem,
            "problems_with_in_band_fraction": _percent(
                len(in_band_problem_ids), problem_count
            ),
            "dominated_fraction_of_evaluated_pairs": _percent(
                class_counts["dominated"], evaluated_domination
            ),
            "problems_with_dominated_fraction": _percent(
                len(dominated_problem_ids), problem_count
            ),
        },
        "total_measurement_wall_s": total_wall,
        "measurement_wall_s_per_problem": per_problem,
        "extrapolation": {
            "target_in_band_pairs": 1000,
            "problems_for_1000_in_band_pairs": problems_for_1000,
            "seconds_for_1000_in_band_pairs": seconds_for_1000,
        },
        "platforms": platforms,
        "separation_band": {
            "time_ratio": [
                IN_BAND_MIN_TIME_RATIO,
                IN_BAND_MAX_TIME_RATIO,
            ],
            "peak_ratio": [
                IN_BAND_MIN_PEAK_RATIO,
                IN_BAND_MAX_PEAK_RATIO,
            ],
        },
        "near_band": {
            "time_ratio": [
                NEAR_BAND_MIN_TIME_RATIO,
                NEAR_BAND_MAX_TIME_RATIO,
            ],
            "peak_ratio": [
                NEAR_BAND_MIN_PEAK_RATIO,
                NEAR_BAND_MAX_PEAK_RATIO,
            ],
        },
        "measurement_floors": {
            "min_time_seconds": MIN_TIME_SECONDS,
            "min_peak_bytes": MIN_PEAK_BYTES,
            "max_trial_spread_fraction": MAX_TIME_SPREAD,
        },
        "z_silence_hit_terms": dict(sorted(z_terms.items())),
        "style_flag_counts": dict(sorted(style_flags.items())),
        "solution_flag_counts": dict(sorted(solution_flags.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "audit": audit,
    }
    if synthesis is not None:
        report["synthesis"] = synthesis
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if markdown_path is None:
        markdown_path = out_dir / (
            "fixture_report.md" if fixture_report else "PILOT_A_REPORT.md"
        )
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _markdown_report(
            report,
            all_pairs,
            measurements,
            is_fixture=fixture_report,
        ),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real-report", action="store_true")
    parser.add_argument("--markdown-path", type=Path)
    args = parser.parse_args(argv)
    classify_file(
        args.out,
        fixture_report=not args.real_report,
        markdown_path=args.markdown_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
