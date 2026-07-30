"""Fail-closed merge and global split for ten completed RunPod shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

try:
    from .build_question_sets import build_question_sets
    from .classify_report import classify_file
    from .validate_runpod_shard import verify_question_manifest
except ImportError:  # pragma: no cover - direct script invocation
    from build_question_sets import build_question_sets  # type: ignore
    from classify_report import classify_file  # type: ignore
    from validate_runpod_shard import verify_question_manifest  # type: ignore


CAMPAIGN = "latmem5k-reviewed-20260730"
SHARD_COUNT = 10
PROBLEMS_PER_SHARD = 500


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: expected object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(rows: Iterable[Mapping[str, object]], *, label: str) -> list[str]:
    result = [str(row.get("problem_id")) for row in rows]
    if any(value == "None" for value in result):
        raise ValueError(f"{label}: missing problem_id")
    if len(result) != len(set(result)):
        raise ValueError(f"{label}: duplicate problem_id")
    return result


def _synth_summary(rows: list[dict[str, object]], run_dir: Path) -> dict[str, object]:
    statuses = Counter(str(row.get("status")) for row in rows)
    eligible_consensus = (
        len(rows) - statuses["generator_skipped"] - statuses["generator_failed"]
    )
    too_slow_kinds: Counter[str] = Counter()
    too_slow_count = 0
    too_slow_problems = 0
    dissenters = 0
    scale_cap_reached = 0
    for row in rows:
        dropped = row.get("dropped_solution_ids", [])
        if isinstance(dropped, list):
            dissenters += len(dropped)
        slow = row.get("too_slow_at_scale", [])
        if isinstance(slow, list) and slow:
            too_slow_problems += 1
            too_slow_count += len(slow)
            for entry in slow:
                if isinstance(entry, dict):
                    too_slow_kinds[str(entry.get("failure_kind", "unknown"))] += 1
                else:
                    too_slow_kinds["unknown"] += 1
        if row.get("scale_search_status") == "scale_cap_reached":
            scale_cap_reached += 1
    return {
        "problem_count": len(rows),
        "processed_problem_count": len(rows),
        "resumed_problem_count": 0,
        "generator_failed": statuses["generator_failed"],
        "generator_failure_rate": statuses["generator_failed"] / len(rows),
        "generator_skipped": statuses["generator_skipped"],
        "consensus_failed": statuses["consensus_failed"],
        "consensus_failure_rate": (
            statuses["consensus_failed"] / eligible_consensus
            if eligible_consensus
            else 0.0
        ),
        "scale_search_exhausted": statuses["scale_search_exhausted"],
        "synthesized_problem_count": statuses["synthesized"],
        "scale_cap_reached": scale_cap_reached,
        "scale_tuning_cap": 8,
        "dissenting_solution_count": dissenters,
        "too_slow_at_scale": too_slow_count,
        "too_slow_at_scale_solution_count": too_slow_count,
        "too_slow_at_scale_problem_count": too_slow_problems,
        "too_slow_at_scale_failure_kinds": dict(sorted(too_slow_kinds.items())),
        "synth_results_path": str(run_dir / "synth_results.jsonl"),
        "synth_tests_path": str(run_dir / "synth_tests.jsonl"),
    }


def merge_campaign(
    shard_root: Path, input_root: Path, out_dir: Path
) -> dict[str, object]:
    if out_dir.exists():
        raise FileExistsError(f"refusing pre-existing global output: {out_dir}")
    run_dir = out_dir / "run"
    input_dir = out_dir / "input"
    run_dir.mkdir(parents=True)
    input_dir.mkdir()
    combined: dict[str, list[dict[str, object]]] = {
        name: []
        for name in (
            "problems",
            "generators",
            "candidates",
            "synth_results",
            "synth_tests",
            "measurements",
        )
    }
    shard_provenance: list[dict[str, object]] = []
    for shard_index in range(SHARD_COUNT):
        shard_name = f"shard_{shard_index:02d}"
        job_dir = shard_root / shard_name
        staged_input = input_root / shard_name / "input"
        if not (job_dir / "DONE").is_file():
            raise ValueError(f"{shard_name}: DONE sentinel missing")
        preflight = json.loads(
            (job_dir / "preflight.json").read_text(encoding="utf-8")
        )
        postflight = json.loads(
            (job_dir / "postflight.json").read_text(encoding="utf-8")
        )
        if preflight.get("status") != "passed" or postflight.get("status") != "passed":
            raise ValueError(f"{shard_name}: preflight/postflight did not pass")
        problems = _read_jsonl(staged_input / "problems.jsonl")
        generators = _read_jsonl(staged_input / "generators.jsonl")
        candidates = _read_jsonl(job_dir / "run" / "candidates.jsonl")
        results = _read_jsonl(job_dir / "run" / "synth_results.jsonl")
        tests = _read_jsonl(job_dir / "run" / "synth_tests.jsonl")
        measurements = _read_jsonl(job_dir / "run" / "measurements.jsonl")
        problem_ids = _ids(problems, label=f"{shard_name}/problems")
        if len(problem_ids) != PROBLEMS_PER_SHARD:
            raise ValueError(f"{shard_name}: expected 500 problems")
        if _ids(generators, label=f"{shard_name}/generators") != problem_ids:
            raise ValueError(f"{shard_name}: generator coverage mismatch")
        if _ids(candidates, label=f"{shard_name}/candidates") != problem_ids:
            raise ValueError(f"{shard_name}: candidate coverage mismatch")
        if _ids(results, label=f"{shard_name}/synth_results") != problem_ids:
            raise ValueError(f"{shard_name}: synthesis coverage mismatch")
        test_ids = _ids(tests, label=f"{shard_name}/synth_tests")
        if _ids(measurements, label=f"{shard_name}/measurements") != test_ids:
            raise ValueError(f"{shard_name}: measurement coverage mismatch")
        prepared_meta = json.loads(
            (staged_input / "meta.json").read_text(encoding="utf-8")
        )
        if (
            prepared_meta.get("campaign") != CAMPAIGN
            or prepared_meta.get("shard_index") != shard_index
        ):
            raise ValueError(f"{shard_name}: prepared metadata mismatch")
        input_hashes = preflight.get("inputs")
        if not isinstance(input_hashes, dict):
            raise ValueError(f"{shard_name}: preflight input hashes missing")
        for filename in ("problems.jsonl", "generators.jsonl", "meta.json"):
            if input_hashes.get(filename) != _sha256(staged_input / filename):
                raise ValueError(f"{shard_name}: {filename} hash mismatch")
        for name, rows in (
            ("problems", problems),
            ("generators", generators),
            ("candidates", candidates),
            ("synth_results", results),
            ("synth_tests", tests),
            ("measurements", measurements),
        ):
            combined[name].extend(rows)
        shard_provenance.append(
            {
                "shard_index": shard_index,
                "problem_count": len(problems),
                "synthesized_problem_count": len(tests),
                "measurement_problem_count": len(measurements),
                "preflight_sha256": _sha256(job_dir / "preflight.json"),
                "postflight_sha256": _sha256(job_dir / "postflight.json"),
                "artifact_hashes_sha256": _sha256(job_dir / "artifact_hashes.txt"),
            }
        )

    global_problem_ids = _ids(combined["problems"], label="global/problems")
    if len(global_problem_ids) != SHARD_COUNT * PROBLEMS_PER_SHARD:
        raise ValueError("global merge does not contain exactly 5,000 problems")
    for name in ("generators", "candidates", "synth_results"):
        if _ids(combined[name], label=f"global/{name}") != global_problem_ids:
            raise ValueError(f"global {name} order/coverage mismatch")
    global_test_ids = _ids(combined["synth_tests"], label="global/synth_tests")
    if _ids(combined["measurements"], label="global/measurements") != global_test_ids:
        raise ValueError("global measurement order/coverage mismatch")

    _write_jsonl(input_dir / "problems.jsonl", combined["problems"])
    _write_jsonl(input_dir / "generators.jsonl", combined["generators"])
    for name in ("candidates", "synth_results", "synth_tests", "measurements"):
        _write_jsonl(run_dir / f"{name}.jsonl", combined[name])
    summary = _synth_summary(combined["synth_results"], run_dir)
    (run_dir / "synth_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    global_meta = {
        "campaign": CAMPAIGN,
        "dataset": "deepmind/code_contests",
        "dataset_revision": "802411c3010cb00d1b05bad57ca77365a3c699d6",
        "split": "train",
        "problem_count": len(global_problem_ids),
        "problem_order": "new eligible staged rows 401 through 5400",
        "problems_sha256": _sha256(input_dir / "problems.jsonl"),
        "generators_sha256": _sha256(input_dir / "generators.jsonl"),
        "shards": shard_provenance,
    }
    (input_dir / "meta.json").write_text(
        json.dumps(global_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = classify_file(
        run_dir,
        fixture_report=False,
        markdown_path=run_dir / "PILOT_A_REPORT.md",
    )
    manifest = build_question_sets(
        run_dir,
        out_dir / "questions",
        eval_fraction=0.2,
        seed=42,
        problems_path=input_dir / "problems.jsonl",
        generators_path=input_dir / "generators.jsonl",
        staging_meta_path=input_dir / "meta.json",
    )
    verify_question_manifest(
        out_dir / "questions", expected_eval_fraction=0.2
    )
    result = {
        "status": "passed",
        "campaign": CAMPAIGN,
        "problem_count": len(global_problem_ids),
        "synthesized_problem_count": len(global_test_ids),
        "report_counts": report["counts"],
        "question_counts": manifest["counts"],
        "unique_question_problem_counts": manifest["unique_problem_counts"],
        "input_hashes": {
            "problems.jsonl": _sha256(input_dir / "problems.jsonl"),
            "generators.jsonl": _sha256(input_dir / "generators.jsonl"),
            "meta.json": _sha256(input_dir / "meta.json"),
        },
    }
    (out_dir / "campaign_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_campaign(args.shard_root, args.input_root, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
