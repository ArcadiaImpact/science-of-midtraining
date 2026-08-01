import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import generation_behavior_eval as generation_eval


def test_sft_config_accepts_only_matched_sft_arm_set():
    arms = [
        arm
        for pair in generation_eval.ARM_PAIR_SETS["sft"].values()
        for arm in pair
    ]
    cfg = generation_eval.GenerationBehaviorEvalConfig(aft_method="sft", arms=arms)
    assert cfg.aft_method == "sft"
    with pytest.raises(ValueError, match="unknown generation-eval arms"):
        generation_eval.GenerationBehaviorEvalConfig(
            aft_method="sft", arms=["sol_no_sdf_dpo"]
        )


def test_nonfinal_score_worker_requires_exactly_one_arm():
    cfg = generation_eval.GenerationBehaviorEvalConfig(
        phase="score", arms=["sol_no_sdf_ri"], finalize=False
    )
    assert cfg.finalize is False
    with pytest.raises(ValueError, match="one-arm score worker"):
        generation_eval.GenerationBehaviorEvalConfig(
            phase="score",
            arms=["sol_no_sdf_ri", "sol_no_sdf_dpo"],
            finalize=False,
        )
    with pytest.raises(ValueError, match="one-arm score worker"):
        generation_eval.GenerationBehaviorEvalConfig(
            phase="generate", arms=["sol_no_sdf_ri"], finalize=False
        )


def _dataset_rows():
    dominant_ids = [f"p{index:03d}" for index in range(321)]
    tradeoff_ids = dominant_ids[:77] + ["extra0", "extra1", "extra2"]
    all_ids = sorted(set(dominant_ids) | set(tradeoff_ids))

    def questions(ids, category):
        return [
            {
                "problem_id": problem_id,
                "question_id": f"{category}-{problem_id}",
                "statement": f"Solve {problem_id}.",
            }
            for problem_id in ids
        ]

    problems = [
        {
            "problem_id": problem_id,
            "statement": f"Solve {problem_id}.",
            "tests": [{"input": "1\n", "output": "1\n", "source": "public"}],
        }
        for problem_id in all_ids
    ]
    synth = [
        {"problem_id": problem_id, "input": "100\n", "output": "100\n"}
        for problem_id in all_ids
    ]
    return (
        questions(dominant_ids, "dominant"),
        questions(tradeoff_ids, "tradeoff"),
        problems,
        synth,
    )


def test_build_eval_records_deduplicates_overlapping_problem_prompts():
    records = generation_eval.build_eval_records(*_dataset_rows())
    assert len(records) == 324
    assert sum("dominant" in row["eval_sets"] for row in records) == 321
    assert sum("tradeoff" in row["eval_sets"] for row in records) == 80
    shared = next(row for row in records if row["problem_id"] == "p000")
    assert set(shared["eval_sets"]) == {"dominant", "tradeoff"}
    assert "dominant" not in shared["probe"].lower()
    assert "memory" not in shared["probe"].lower()


def test_render_prompt_requires_bare_python_and_no_explanation():
    prompt = generation_eval.render_prompt("Add two numbers.")
    assert "Return only Python source code" in prompt
    assert "do not use Markdown fences" in prompt
    with pytest.raises(ValueError, match="non-empty"):
        generation_eval.render_prompt("")


def test_extraction_records_fences_and_syntax_errors():
    extracted = generation_eval.extraction_record("```python\nprint(1)\n```")
    assert extracted["source"] == "print(1)\n"
    assert extracted["syntax_ok"] is True
    broken = generation_eval.extraction_record("for")
    assert broken["syntax_ok"] is False
    assert broken["syntax_error"]


def _record():
    return {
        "problem_id": "p",
        "eval_sets": {"dominant": "q"},
        "tests": [{"input": "1\n", "output": "1\n", "source": "public"}],
        "synth_input": "100\n",
        "synth_output": "100\n",
    }


def test_truncated_generation_is_incorrect_and_never_executed(monkeypatch):
    monkeypatch.setattr(
        generation_eval,
        "check_candidate_correctness",
        lambda *args, **kwargs: pytest.fail("truncated source was executed"),
    )
    row = generation_eval.score_generation(
        {"arm": "a", "response": "print(1)", "finish_reason": "length"},
        _record(),
        baseline_rss_bytes=8_000_000,
        host_latency_calibration_s=0.2,
        timeout_s=8,
        mem_limit_mb=1024,
    )
    assert row["correct"] is False
    assert row["correctness_status"] == "generation_truncated"


def test_correct_generation_requires_synth_output_then_records_measurement(monkeypatch):
    monkeypatch.setattr(
        generation_eval,
        "check_candidate_correctness",
        lambda *args, **kwargs: (
            {"status": "correct", "correctness": [{"ok": True, "source": "public"}]},
            0.1,
        ),
    )
    monkeypatch.setattr(
        generation_eval,
        "run_solution_sandboxed",
        lambda *args, **kwargs: {"ok": True, "stdout": "100\n"},
    )
    monkeypatch.setattr(
        generation_eval,
        "_measure_candidate",
        lambda *args, **kwargs: (
            {
                "status": "measured",
                "median_time_s": 0.2,
                "times_s": [0.19, 0.2, 0.21],
                "time_spread": 0.1,
                "median_rss_bytes": 10_000_000,
                "rss_trials_bytes": [9_900_000, 10_000_000, 10_100_000],
                "baseline_subtracted_peak_bytes": 2_000_000,
                "peak_spread": 0.02,
                "flags": [],
                "memory_metric": "fresh_process_peak_rss_bytes",
            },
            0.6,
        ),
    )
    row = generation_eval.score_generation(
        {"arm": "a", "response": "print(100)", "finish_reason": "stop"},
        _record(),
        baseline_rss_bytes=8_000_000,
        host_latency_calibration_s=0.2,
        timeout_s=8,
        mem_limit_mb=1024,
    )
    assert row["correct"] is True
    assert row["measurement_status"] == "measured"
    assert row["median_time_s"] == 0.2
    assert row["baseline_subtracted_peak_bytes"] == 2_000_000
    assert row["correctness"][-1] == {"source": "synth", "ok": True}


def _scored(problem_id, *, correct, time=None, peak=None):
    row = {
        "problem_id": problem_id,
        "eval_sets": {"tradeoff": f"q-{problem_id}"},
        "correct": correct,
        "measurement_status": "measured" if time is not None else "not_attempted",
        "measurement_flags": [],
        "host_latency_calibration_s": 1.0,
    }
    if time is not None:
        row.update(median_time_s=time, baseline_subtracted_peak_bytes=peak)
    return row


def test_paired_comparison_keeps_correctness_failures_in_denominator():
    before = [
        _scored("a", correct=True, time=2.0, peak=20.0),
        _scored("b", correct=False),
    ]
    after = [
        _scored("a", correct=True, time=1.0, peak=40.0),
        _scored("b", correct=True, time=3.0, peak=30.0),
    ]
    summary, pairs = generation_eval.paired_comparison(
        before, after, kind="tradeoff", draws=200, seed=1
    )
    assert summary["n"] == 2
    assert summary["correct_rate_delta_post_minus_pre"] == 0.5
    assert summary["correctness_transitions"] == {"0->1": 1, "1->1": 1}
    assert summary["paired_measured_n"] == 1
    assert summary["quadrants"] == {"faster_higher_memory": 1}
    assert len(pairs) == 2


def test_checkpoint_availability_prefers_local_and_reports_missing(tmp_path):
    local = tmp_path / "sol_no_sdf_ri"
    local.mkdir()
    (local / "config.json").write_text("{}")
    files = ["sol_no_sdf_dpo/config.json", "sol_no_sdf_dpo/model.safetensors"]
    status = generation_eval.checkpoint_availability(
        ["sol_no_sdf_ri", "sol_no_sdf_dpo", "sol_latency_ri"],
        repo_files=files,
        checkpoint_root=tmp_path,
    )
    assert status == {
        "sol_no_sdf_ri": "local",
        "sol_no_sdf_dpo": "hf",
        "sol_latency_ri": "missing",
    }
