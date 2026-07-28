"""Run Pilot A stages 0 through 3 without accessing the network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .audit_data import audit_file
    from .classify_report import classify_file
    from .extract_candidates import extract_file
    from .measure_pairs import (
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        measure_file,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from audit_data import audit_file
    from classify_report import classify_file
    from extract_candidates import extract_file
    from measure_pairs import (
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        measure_file,
    )


PILOT_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PILOT_DIR / "fixture" / "problems.jsonl"
DEFAULT_OUT = PILOT_DIR / "out"
COMMITTED_REPORT_PATH = PILOT_DIR / "PILOT_A_REPORT.md"


def resolve_data_path(value: str | Path) -> tuple[Path, bool]:
    raw = str(value)
    if raw == "fixture":
        return FIXTURE_PATH, True
    path = Path(value)
    if path.is_dir():
        path = path / "problems.jsonl"
    try:
        fixture = path.resolve() == FIXTURE_PATH.resolve()
    except OSError:
        fixture = False
    return path, fixture


def _markdown_path_for_run(
    *,
    out_dir: Path,
    is_fixture: bool,
    limit: int | None,
    write_committed_report: bool,
) -> Path:
    """Choose a report target without accidentally replacing the real report."""
    if write_committed_report:
        if is_fixture:
            raise ValueError("fixture runs cannot write the committed report")
        if limit is not None:
            raise ValueError("limited runs cannot write the committed report")
        if out_dir.resolve() != DEFAULT_OUT.resolve():
            raise ValueError(
                "the committed report requires the default Pilot A output directory"
            )
        return COMMITTED_REPORT_PATH
    if is_fixture:
        return out_dir / "fixture_report.md"
    return out_dir / "PILOT_A_REPORT.md"


def run_pipeline(
    *,
    data: str | Path = "fixture",
    out: str | Path,
    limit: int | None = None,
    seed: int = 42,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
    write_committed_report: bool = False,
) -> dict[str, object]:
    """Run all stages and return their small orchestration summary."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    data_path, is_fixture = resolve_data_path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Pilot A data file does not exist: {data_path}")
    out_dir = Path(out)
    markdown_path = _markdown_path_for_run(
        out_dir=out_dir,
        is_fixture=is_fixture,
        limit=limit,
        write_committed_report=write_committed_report,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"pilot-a: stage 0 audit ({data_path})", flush=True)
    audit = audit_file(data_path, out_dir, limit=limit)
    print("pilot-a: stage 1 candidate extraction", flush=True)
    candidates = extract_file(
        data_path,
        out_dir,
        limit=limit,
        seed=seed,
    )
    print("pilot-a: stage 2 correctness and measurement", flush=True)
    measurement = measure_file(
        out_dir / "candidates.jsonl",
        out_dir,
        limit=limit,
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    print("pilot-a: stage 3 classification and report", flush=True)
    report = classify_file(
        out_dir,
        fixture_report=is_fixture,
        markdown_path=markdown_path,
    )
    summary = {
        "data": str(data_path),
        "out": str(out_dir),
        "fixture": is_fixture,
        "seed": seed,
        "limit": limit,
        "write_committed_report": write_committed_report,
        "audit_problem_count": audit["problem_count"],
        "candidate_problem_count": len(candidates),
        "measurement": measurement,
        "report_counts": report["counts"],
        "markdown_report": str(markdown_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="fixture",
        help="'fixture', a problems.jsonl path, or a directory containing it",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    parser.add_argument(
        "--write-committed-report",
        action="store_true",
        help=(
            "write PILOT_A_REPORT.md beside this script; only valid for an "
            "unlimited real-data run using the default output directory"
        ),
    )
    args = parser.parse_args(argv)
    run_pipeline(
        data=args.data,
        out=args.out,
        limit=args.limit,
        seed=args.seed,
        timeout_s=args.timeout_s,
        mem_limit_mb=args.mem_limit_mb,
        write_committed_report=args.write_committed_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
