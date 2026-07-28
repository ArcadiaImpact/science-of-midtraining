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
    classify_file,
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
    FIXTURE_GENERATORS_PATH,
    FIXTURE_PATH,
    _markdown_path_for_run,
    run_pipeline,
)
from experiments.prior_latmem.bank.pilots.pilot_a.synth_workloads import (
    START_N,
    _generator_map,
    _scale_evaluator,
    consensus_oracle,
    run_generator_sandboxed,
    search_scale,
    synthesize_file,
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


def test_synth_consensus_accepts_five_agreeing_solutions():
    result = consensus_oracle(
        {
            "s000": "answer\n",
            "s001": "answer  \r\n",
            "s002": "answer",
            "s003": "answer\n\n",
            "s004": "answer",
        }
    )
    assert result["status"] == "agreed"
    assert result["oracle"] == "answer"
    assert result["survivor_ids"] == [
        "s000",
        "s001",
        "s002",
        "s003",
        "s004",
    ]
    assert result["dissenter_ids"] == []


def test_synth_consensus_drops_a_single_dissenter():
    result = consensus_oracle(
        {
            "s000": "42",
            "s001": "42",
            "s002": "42",
            "s003": "42",
            "s004": "42",
            "s005": "rigged",
        }
    )
    assert result["status"] == "agreed"
    assert result["survivor_ids"] == [
        "s000",
        "s001",
        "s002",
        "s003",
        "s004",
    ]
    assert result["dissenter_ids"] == ["s005"]


def test_synth_consensus_drops_problem_without_five_agreements():
    result = consensus_oracle(
        {
            "s000": "first",
            "s001": "first",
            "s002": "first",
            "s003": "first",
            "s004": "second",
            "s005": "second",
        }
    )
    assert result["status"] == "consensus_failed"
    assert result["oracle"] is None
    assert result["survivor_ids"] == []


def test_synth_scale_search_doubles_with_injected_fake_timings():
    evaluated = []

    def fake_evaluation(n):
        evaluated.append(n)
        fastest = n / 100_000
        return {
            "timings": {
                "fast": [fastest * 0.9, fastest, fastest * 1.1],
                "slow": [fastest * 1.8, fastest * 2.0, fastest * 2.2],
            },
            "marker": n,
        }

    result = search_scale(fake_evaluation)
    assert evaluated == [1_000, 2_000, 4_000]
    assert result["status"] == "target_reached"
    assert result["n"] == 4_000
    assert result["fastest_median_s"] == pytest.approx(0.04)
    assert result["evaluation"]["marker"] == 4_000


def test_synth_scale_search_exhaustion_keeps_best_completed_scale():
    now = [0.0]

    def fake_clock():
        return now[0]

    def fake_evaluation(n):
        now[0] += 31.0
        return {
            "timings": {"fast": [0.005, 0.006]},
            "marker": n,
        }

    result = search_scale(
        fake_evaluation,
        budget_s=60.0,
        clock=fake_clock,
    )
    assert result["status"] == "scale_search_exhausted"
    assert result["n"] == 2_000
    assert result["evaluation"]["marker"] == 2_000


def test_synth_scale_search_reports_scale_cap_reached():
    evaluated = []

    def fake_evaluation(n):
        evaluated.append(n)
        return {
            "timings": {"fast": [0.001, 0.002]},
            "marker": n,
        }

    result = search_scale(
        fake_evaluation,
        start_n=1_000,
        max_n=4_000,
    )
    assert evaluated == [1_000, 2_000, 4_000]
    assert result["status"] == "scale_cap_reached"
    assert result["n"] == 4_000
    assert result["evaluation"]["marker"] == 4_000


def test_scale_timeouts_and_crashes_are_not_consensus_dissenters(monkeypatch):
    candidates = [
        {
            "candidate_id": f"s{index:03d}",
            "source": (
                "timeout"
                if index == 6
                else "crash"
                if index == 7
                else "rigged"
                if index == 5
                else "agree"
            ),
        }
        for index in range(8)
    ]

    def fake_run(source, stdin_data, *, timeout_s, mem_limit_mb):
        del stdin_data, timeout_s, mem_limit_mb
        if source == "timeout":
            return {"ok": False, "error": "timeout"}
        if source == "crash":
            return {"ok": False, "error": "MemoryError: scale input"}
        return {
            "ok": True,
            "stdout": "wrong" if source == "rigged" else "answer",
            "timings": {"payload_wall_s": 0.01},
        }

    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.synth_workloads.run_solution_sandboxed",
        fake_run,
    )
    dissenters = []
    too_slow = []
    evaluate = _scale_evaluator(
        generator_source="unused",
        seed=42,
        candidates=candidates,
        initial_input="input",
        timeout_s=1.0,
        mem_limit_mb=256,
        deadline=100.0,
        clock=lambda: 0.0,
        dropped_ids=dissenters,
        too_slow_at_scale=too_slow,
    )

    result = evaluate(START_N)

    assert result["survivor_ids"] == [f"s{index:03d}" for index in range(5)]
    assert result["dissenter_ids"] == ["s005"]
    assert dissenters == ["s005"]
    assert [record["candidate_id"] for record in too_slow] == ["s006", "s007"]
    assert [record["failure_kind"] for record in too_slow] == [
        "timeout",
        "crash",
    ]
    assert too_slow[1]["error"] == "MemoryError: scale input"
    assert all(record["n"] == START_N for record in too_slow)


def test_scale_search_does_not_truncate_run_timeout_to_remaining_budget(
    monkeypatch,
):
    now = [0.0]
    calls = []
    candidates = [
        {"candidate_id": f"s{index:03d}", "source": "agree"}
        for index in range(5)
    ]

    def fake_clock():
        return now[0]

    def fake_run(source, stdin_data, *, timeout_s, mem_limit_mb):
        del source, stdin_data, mem_limit_mb
        calls.append(timeout_s)
        if len(calls) == 10:
            now[0] = 9.5
        return {
            "ok": True,
            "stdout": "answer",
            "timings": {"payload_wall_s": 0.001},
        }

    def unexpected_generator(*args, **kwargs):
        raise AssertionError("generator ran under an insufficient budget")

    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.synth_workloads.run_solution_sandboxed",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.synth_workloads._generator_output",
        unexpected_generator,
    )
    evaluate = _scale_evaluator(
        generator_source="unused",
        seed=42,
        candidates=candidates,
        initial_input="input",
        timeout_s=1.0,
        mem_limit_mb=256,
        deadline=10.0,
        clock=fake_clock,
        dropped_ids=[],
        too_slow_at_scale=[],
    )

    result = search_scale(
        evaluate,
        budget_s=10.0,
        clock=fake_clock,
    )

    assert result["status"] == "scale_search_exhausted"
    assert result["n"] == START_N
    assert len(calls) == 10
    assert calls == [1.0] * 10


def test_fixture_generators_are_deterministic_inside_sandbox():
    rows = [
        json.loads(line)
        for line in FIXTURE_GENERATORS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    for row in rows:
        first = run_generator_sandboxed(
            row["generator_source"],
            37,
            91,
            timeout_s=1.0,
            mem_limit_mb=256,
        )
        second = run_generator_sandboxed(
            row["generator_source"],
            37,
            91,
            timeout_s=1.0,
            mem_limit_mb=256,
        )
        assert first["ok"] is True, row["problem_id"]
        assert second["ok"] is True, row["problem_id"]
        assert first["stdout"]
        assert first["stdout"] == second["stdout"]


def test_generator_map_accepts_skip_rows_and_rejects_duplicates(tmp_path):
    generators = tmp_path / "generators.jsonl"
    generators.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_id": "generated",
                        "generator_source": "generator source",
                    }
                ),
                json.dumps(
                    {
                        "problem_id": "skipped",
                        "skip": "No scalable authoring strategy.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _generator_map(generators) == {
        "generated": ("generator_source", "generator source"),
        "skipped": ("skip", "No scalable authoring strategy."),
    }

    generators.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_id": "duplicate",
                        "generator_source": "generator source",
                    }
                ),
                json.dumps(
                    {
                        "problem_id": "duplicate",
                        "skip": "No scalable authoring strategy.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate problem_id 'duplicate'"):
        _generator_map(generators)


def test_generator_map_rejects_unknown_row_shape(tmp_path):
    generators = tmp_path / "generators.jsonl"
    generators.write_text(
        json.dumps(
            {
                "problem_id": "unknown",
                "reason": "This is not a supported skip row.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"expected problem_id/generator_source or problem_id/skip",
    ):
        _generator_map(generators)


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


def test_measurement_resume_rejects_cross_mode_fixture_artifacts(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extract_file(FIXTURE_PATH, extracted_dir, limit=1, seed=17)
    candidate_row = json.loads(
        (extracted_dir / "candidates.jsonl").read_text(encoding="utf-8")
    )
    problem_id = candidate_row["problem_id"]
    out = tmp_path / "out"
    out.mkdir()
    (out / "measurements.jsonl").write_text(
        json.dumps(
            {
                "problem_id": problem_id,
                "measurement_source": "dataset",
                "solutions": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    synth_tests = tmp_path / "synth_tests.jsonl"
    synth_tests.write_text(
        json.dumps(
            {
                "problem_id": problem_id,
                "source": "synth",
                "input": "",
                "output": "",
                "survivor_ids": [],
                "correctness_verdicts": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "status": "correct",
                    }
                    for candidate in candidate_row["candidates"]
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot resume a 'synth' run"):
        measure_file(
            extracted_dir / "candidates.jsonl",
            out,
            synth_tests_path=synth_tests,
        )


def test_synth_measurement_reuses_persisted_correctness_verdict(
    tmp_path,
    monkeypatch,
):
    candidate = {
        "candidate_id": "s000",
        "solution_index": 0,
        "source": "print('answer')",
        "z_silence_hits": [],
        "style_flags": [],
    }
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "problem_id": "cached-correctness",
                "tests": [
                    {
                        "source": "public",
                        "input": "",
                        "output": "answer\n",
                    }
                ],
                "candidates": [candidate],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verdict = {
        **candidate,
        "status": "correct",
        "drop_reason": None,
        "correctness": [
            {"test_index": 0, "source": "public", "ok": True}
        ],
        "flags": [],
    }
    synth_tests = tmp_path / "synth_tests.jsonl"
    synth_tests.write_text(
        json.dumps(
            {
                "problem_id": "cached-correctness",
                "source": "synth",
                "input": "",
                "output": "answer",
                "survivor_ids": ["s000"],
                "correctness_verdicts": [verdict],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_count = [0]

    def fake_run(source, stdin_data, *, timeout_s, mem_limit_mb):
        del source, stdin_data, timeout_s, mem_limit_mb
        run_count[0] += 1
        return {
            "ok": True,
            "stdout": "answer\n",
            "rss_bytes": 2_000_000,
            "parent_wall_s": 0.02,
            "timings": {"payload_wall_s": 0.01},
        }

    def unexpected_correctness(*args, **kwargs):
        raise AssertionError("stage 2 repeated the correctness gate")

    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs.run_solution_sandboxed",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs.check_candidate_correctness",
        unexpected_correctness,
    )

    result = measure_file(
        candidates_path,
        tmp_path / "out",
        synth_tests_path=synth_tests,
    )

    assert result == {"measured_problems": 1, "skipped_problems": 0}
    assert run_count[0] == 8  # five baseline runs plus three measurements
    measured = json.loads(
        (tmp_path / "out" / "measurements.jsonl").read_text(encoding="utf-8")
    )
    assert measured["solutions"][0]["correctness"] == verdict["correctness"]


def test_synthesis_resume_rejects_a_different_seed(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps({"problem_id": "seeded"}) + "\n",
        encoding="utf-8",
    )
    generators = tmp_path / "generators.jsonl"
    generators.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "synth_results.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "seeded",
                "seed": 7,
                "status": "generator_failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"seed 7.*seed 8"):
        synthesize_file(candidates, generators, out, seed=8)


def test_report_rejects_synthesis_measurement_mode_mismatches(tmp_path):
    def write_common(out, measurement_source):
        out.mkdir()
        (out / "candidates.jsonl").write_text(
            json.dumps({"problem_id": "mode", "seed": 42, "candidate_cap": 8})
            + "\n",
            encoding="utf-8",
        )
        (out / "measurements.jsonl").write_text(
            json.dumps(
                {
                    "problem_id": "mode",
                    "measurement_source": measurement_source,
                    "solutions": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    dataset_out = tmp_path / "dataset"
    write_common(dataset_out, "dataset")
    (dataset_out / "synth_summary.json").write_text(
        json.dumps({"problem_count": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible with 'dataset'"):
        classify_file(dataset_out)

    synth_out = tmp_path / "synth"
    write_common(synth_out, "synth")
    with pytest.raises(ValueError, match="require synth_summary.json"):
        classify_file(synth_out)


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
    assert summary["audit_problem_count"] == 4
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["problem_count"] == 4
    assert report["platforms"]
    assert report["seed"] == 123
    assert report["candidate_cap"] == 8
    assert report["measurement_floors"]["min_peak_bytes"] == 512_000
    with (tmp_path / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert pairs
    assert all(pair["ratio_convention"] for pair in pairs)


def test_end_to_end_fixture_uses_synthesized_measurement_tests(tmp_path):
    summary = run_pipeline(
        data="fixture",
        out=tmp_path,
        seed=123,
        timeout_s=2.0,
        mem_limit_mb=512,
        synth_tests=True,
    )
    expected = {
        "audit.json",
        "candidates.jsonl",
        "synth_results.jsonl",
        "synth_tests.jsonl",
        "synth_summary.json",
        "measurements.jsonl",
        "pairs.csv",
        "report.json",
        "fixture_report.md",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    synthesis = summary["synthesis"]
    assert synthesis["problem_count"] == 4
    assert synthesis["synthesized_problem_count"] == 3
    assert synthesis["generator_failed"] == 0
    assert synthesis["consensus_failed"] == 1
    assert synthesis["dissenting_solution_count"] == 1
    assert synthesis["too_slow_at_scale"] == 0
    assert synthesis["too_slow_at_scale_solution_count"] == 0
    assert synthesis["too_slow_at_scale_problem_count"] == 0
    assert synthesis["too_slow_at_scale_failure_kinds"] == {}

    synth_results = [
        json.loads(line)
        for line in (tmp_path / "synth_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all("too_slow_at_scale" in row for row in synth_results)
    assert all("too_slow_at_scale_count" in row for row in synth_results)

    synth_rows = [
        json.loads(line)
        for line in (tmp_path / "synth_tests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(synth_rows) == 3
    assert all(row["source"] == "synth" for row in synth_rows)
    assert all(row["correctness_verdicts"] for row in synth_rows)
    rigged = next(
        row
        for row in synth_rows
        if row["problem_id"] == "fixture-record-checksum"
    )
    assert len(rigged["survivor_ids"]) == 5
    assert "s005" not in rigged["survivor_ids"]

    measurements = [
        json.loads(line)
        for line in (tmp_path / "measurements.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(measurements) == 3
    assert all(row["measurement_source"] == "synth" for row in measurements)
    measured_solutions = [
        solution
        for row in measurements
        for solution in row["solutions"]
        if solution["status"] == "measured"
    ]
    assert measured_solutions
    assert all(
        solution["measurement_test_source"] == "synth"
        for solution in measured_solutions
    )
    assert all(
        "under_time_floor" not in solution["flags"]
        for solution in measured_solutions
    )
    assert all(
        "under_peak_floor" not in solution["flags"]
        for solution in measured_solutions
    )

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["problem_count"] == 3
    assert report["synthesis"]["consensus_failed"] == 1

    resumed = run_pipeline(
        data="fixture",
        out=tmp_path,
        seed=123,
        timeout_s=2.0,
        mem_limit_mb=512,
        synth_tests=True,
    )
    assert resumed["synthesis"]["processed_problem_count"] == 0
    assert resumed["synthesis"]["resumed_problem_count"] == 4
    assert resumed["measurement"] == {
        "measured_problems": 0,
        "skipped_problems": 3,
    }


def test_end_to_end_fixture_accepts_mixed_generator_and_skip_rows(tmp_path):
    skipped_id = "fixture-dominated-squares"
    skip_reason = "Authoring could not provide a scalable generator."
    generator_rows = [
        json.loads(line)
        for line in FIXTURE_GENERATORS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mixed_generators = tmp_path / "mixed_generators.jsonl"
    mixed_generators.write_text(
        "".join(
            json.dumps(
                (
                    {"problem_id": skipped_id, "skip": skip_reason}
                    if row["problem_id"] == skipped_id
                    else row
                )
            )
            + "\n"
            for row in generator_rows
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(
        data="fixture",
        out=tmp_path / "out",
        seed=123,
        timeout_s=2.0,
        mem_limit_mb=512,
        synth_tests=True,
        generators=mixed_generators,
    )

    synthesis = summary["synthesis"]
    assert synthesis["problem_count"] == 4
    assert synthesis["synthesized_problem_count"] == 2
    assert synthesis["generator_failed"] == 0
    assert synthesis["generator_skipped"] == 1
    assert synthesis["consensus_failed"] == 1

    results = [
        json.loads(line)
        for line in (tmp_path / "out" / "synth_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    skipped = next(row for row in results if row["problem_id"] == skipped_id)
    assert skipped["status"] == "generator_skipped"
    assert skipped["failure_reason"] == skip_reason

    synth_tests = [
        json.loads(line)
        for line in (tmp_path / "out" / "synth_tests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(synth_tests) == 2
    assert skipped_id not in {row["problem_id"] for row in synth_tests}
