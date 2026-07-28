"""CPU-only tests for the fixture-first prior-latmem mining pilot."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.pilots.pilot_a.audit_data import audit_file
from experiments.prior_latmem.bank.pilots.pilot_a.classify_report import (
    classify_problem,
    classify_tradeoff_pair,
)
from experiments.prior_latmem.bank.pilots.pilot_a.extract_candidates import (
    extract_file,
)
from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
    measure_file,
    run_solution_sandboxed,
)
from experiments.prior_latmem.bank.pilots.pilot_a.run_pilot import (
    COMMITTED_REPORT_PATH,
    DEFAULT_OUT,
    FIXTURE_PATH,
    _markdown_path_for_run,
    run_pipeline,
)


def _solution(
    candidate_id: str,
    median_time: float,
    subtracted_peak: float,
    *,
    times: list[float] | None = None,
    rss: list[int] | None = None,
    flags: list[str] | None = None,
) -> dict[str, object]:
    times = times or [median_time * 0.99, median_time, median_time * 1.01]
    rss = rss or [
        int(subtracted_peak + 1000),
        int(subtracted_peak + 1000),
        int(subtracted_peak + 1000),
    ]
    return {
        "candidate_id": candidate_id,
        "status": "measured",
        "times_s": times,
        "rss_trials_bytes": rss,
        "median_time_s": median_time,
        "median_rss_bytes": sorted(rss)[1],
        "baseline_subtracted_peak_bytes": subtracted_peak,
        "time_spread": 0.02,
        "peak_spread": 0.0,
        "flags": flags or [],
    }


def test_schema_audit_rejects_malformed_fixture_row(tmp_path):
    first = json.loads(FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0])
    del first["time_limit"]
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1.*missing keys time_limit"):
        audit_file(malformed, tmp_path / "out")


def test_dedup_collapses_near_identical_fixture_solutions(tmp_path):
    rows = extract_file(FIXTURE_PATH, tmp_path, seed=17)
    near = next(row for row in rows if row["problem_id"] == "fixture-near-dedup")
    assert len(near["candidates"]) == 1
    assert near["drop_counts"]["ast_duplicates"] == 5
    assert near["seed"] == 17


def test_injected_pair_classification_covers_registered_classes():
    speed = _solution("speed", 0.1, 100)
    assert (
        classify_tradeoff_pair(
            "in", speed, _solution("lean", 0.2, 50)
        )["class"]
        == "in_band"
    )
    assert (
        classify_tradeoff_pair(
            "near", speed, _solution("lean", 0.12, 75)
        )["class"]
        == "near_band"
    )
    assert (
        classify_tradeoff_pair(
            "far", speed, _solution("lean", 1.0, 5)
        )["class"]
        == "lopsided"
    )
    assert (
        classify_tradeoff_pair(
            "floor",
            _solution("tiny-a", 0.004, 100, flags=["under_time_floor"]),
            _solution("tiny-b", 0.008, 50),
        )["class"]
        == "under_time_floor"
    )
    assert (
        classify_tradeoff_pair(
            "spread",
            speed,
            _solution("noisy", 0.2, 50, flags=["unstable"]),
        )["class"]
        == "unstable"
    )
    assert (
        classify_tradeoff_pair(
            "peak-floor",
            speed,
            _solution(
                "small-peak",
                0.2,
                50,
                flags=["under_peak_floor"],
            ),
        )["class"]
        == "under_peak_floor"
    )

    clean_winner = _solution(
        "winner",
        0.1,
        50,
        times=[0.09, 0.10, 0.11],
        rss=[1040, 1050, 1060],
    )
    clean_loser = _solution(
        "loser",
        0.2,
        100,
        times=[0.18, 0.20, 0.22],
        rss=[1090, 1100, 1110],
    )
    clean_pairs = classify_problem(
        {"problem_id": "dominated", "solutions": [clean_winner, clean_loser]}
    )
    assert [pair["class"] for pair in clean_pairs] == ["dominated"]

    close_winner = _solution(
        "winner",
        0.1,
        80,
        times=[0.08, 0.10, 0.12],
        rss=[1070, 1080, 1090],
    )
    close_loser = _solution(
        "loser",
        0.116,
        100,
        times=[0.09, 0.116, 0.14],
        rss=[1080, 1100, 1120],
    )
    close_pairs = classify_problem(
        {"problem_id": "indistinguishable", "solutions": [close_winner, close_loser]}
    )
    assert [pair["class"] for pair in close_pairs] == ["indistinguishable"]

    low_peak_winner = _solution(
        "low-peak-winner",
        0.1,
        400_000,
        flags=["under_peak_floor"],
    )
    high_peak_loser = _solution("high-peak-loser", 0.2, 1_000_000)
    floor_dominated_pairs = classify_problem(
        {
            "problem_id": "peak-floor-domination",
            "solutions": [low_peak_winner, high_peak_loser],
        }
    )
    assert [pair["class"] for pair in floor_dominated_pairs] == [
        "under_peak_floor"
    ]


def test_under_baseline_noise_solution_is_retained_in_pair_enumeration():
    ordinary = _solution("ordinary", 0.1, 1_000_000)
    noisy = _solution(
        "noisy",
        0.2,
        -32_768,
        flags=["under_baseline_noise", "under_peak_floor"],
    )
    pairs = classify_problem(
        {"problem_id": "baseline-noise", "solutions": [ordinary, noisy]}
    )
    assert len(pairs) == 1
    assert pairs[0]["class"] == "under_baseline_noise"
    assert {pairs[0]["solution_a_id"], pairs[0]["solution_b_id"]} == {
        "ordinary",
        "noisy",
    }


def test_real_child_runner_accepts_zero_system_exit():
    report = run_solution_sandboxed(
        'import sys\nprint("accepted")\nsys.exit(0)\n',
        "",
        timeout_s=1.0,
        mem_limit_mb=256,
    )
    assert report["ok"] is True
    assert report["stdout"] == "accepted\n"
    assert report["returncode"] == 0

    failed = run_solution_sandboxed(
        "import sys\nsys.exit(2)\n",
        "",
        timeout_s=1.0,
        mem_limit_mb=256,
    )
    assert failed["ok"] is False
    assert str(failed["error"]).startswith("SystemExit: 2")


def test_real_child_runner_reports_payload_time_below_parent_wall():
    report = run_solution_sandboxed(
        "import time\ntime.sleep(0.05)\n",
        "",
        timeout_s=1.0,
        mem_limit_mb=256,
    )
    assert report["ok"] is True
    timings = report["timings"]
    assert isinstance(timings, dict)
    payload_wall_s = timings["payload_wall_s"]
    parent_wall_s = report["parent_wall_s"]
    assert isinstance(payload_wall_s, float)
    assert isinstance(parent_wall_s, float)
    assert 0.04 <= payload_wall_s < 0.5
    assert parent_wall_s >= payload_wall_s + 0.005


def test_sandbox_kills_infinite_candidate_and_records_drop(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "problem_id": "infinite",
                "source": "fixture",
                "difficulty": 0,
                "statement": "Never terminates.",
                "tests": [{"source": "public", "input": "", "output": ""}],
                "candidates": [
                    {
                        "candidate_id": "s000",
                        "solution_index": 0,
                        "source": "while True:\n    pass\n",
                        "z_silence_hits": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = measure_file(
        candidates,
        tmp_path / "out",
        timeout_s=0.25,
        mem_limit_mb=256,
    )
    assert result == {"measured_problems": 1, "skipped_problems": 0}
    measured = json.loads(
        (tmp_path / "out" / "measurements.jsonl").read_text(encoding="utf-8")
    )
    solution = measured["solutions"][0]
    assert solution["status"] == "dropped"
    assert solution["drop_reason"] == "correctness_timeout"


def test_measurement_resume_skips_completed_problems(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "problem_id": "already-done",
                "tests": [{"source": "public", "input": "", "output": ""}],
                "candidates": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    original = {
        "problem_id": "already-done",
        "baseline": {
            "median_rss_bytes": 1_000_000,
            "rss_trials_bytes": [1_000_000] * 5,
            "measurement_wall_s": 0.1,
        },
        "measurement_wall_s": 0.0,
        "solutions": [],
    }
    output = out / "measurements.jsonl"
    output.write_text(json.dumps(original) + "\n", encoding="utf-8")

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("resume attempted to execute a completed problem")

    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs.run_solution_sandboxed",
        unexpected_execution,
    )
    result = measure_file(candidates, out, timeout_s=0.1)
    assert result == {"measured_problems": 0, "skipped_problems": 1}
    assert output.read_text(encoding="utf-8") == json.dumps(original) + "\n"


def test_real_report_routing_requires_explicit_safe_full_run(tmp_path):
    limited_path = _markdown_path_for_run(
        out_dir=DEFAULT_OUT,
        is_fixture=False,
        limit=1,
        write_committed_report=False,
    )
    assert limited_path == DEFAULT_OUT / "PILOT_A_REPORT.md"

    custom_path = _markdown_path_for_run(
        out_dir=tmp_path,
        is_fixture=False,
        limit=None,
        write_committed_report=False,
    )
    assert custom_path == tmp_path / "PILOT_A_REPORT.md"

    assert (
        _markdown_path_for_run(
            out_dir=DEFAULT_OUT,
            is_fixture=False,
            limit=None,
            write_committed_report=True,
        )
        == COMMITTED_REPORT_PATH
    )
    with pytest.raises(ValueError, match="limited runs"):
        _markdown_path_for_run(
            out_dir=DEFAULT_OUT,
            is_fixture=False,
            limit=1,
            write_committed_report=True,
        )
    with pytest.raises(ValueError, match="default Pilot A output"):
        _markdown_path_for_run(
            out_dir=tmp_path,
            is_fixture=False,
            limit=None,
            write_committed_report=True,
        )


def test_end_to_end_fixture_produces_all_artifacts(tmp_path):
    summary = run_pipeline(
        data="fixture",
        out=tmp_path,
        seed=123,
        timeout_s=1.0,
        mem_limit_mb=512,
    )
    expected = {
        "audit.json",
        "candidates.jsonl",
        "measurements.jsonl",
        "pairs.csv",
        "report.json",
        "fixture_report.md",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    assert summary["fixture"] is True
    assert summary["audit_problem_count"] == 3
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["problem_count"] == 3
    assert report["platforms"]
    assert report["seed"] == 123
    assert report["candidate_cap"] == 8
    assert report["measurement_floors"]["min_peak_bytes"] == 512_000
    with (tmp_path / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert pairs
    assert all(pair["ratio_convention"] for pair in pairs)
