import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.star_sample_generate import (
    StarSampleGenerateConfig,
    select_records,
    split_thinking_response,
)
from experiments.prior_latmem.gemma4_e4b_baseline_20260805.analyze_baseline import (
    build_data,
    mean_ci95,
    pass_at_k,
    wilson_ci95,
)
from experiments.prior_latmem.gemma4_e4b_baseline_20260805.persist import (
    BaselinePersistConfig,
    _line_count,
)
from experiments.prior_latmem.gemma4_e4b_baseline_20260805.profile_inference import (
    ProfileConfig,
)
from experiments.prior_latmem.gemma4_e4b_baseline_20260805.summarize_mtp_profiles import (
    MtpSummaryConfig,
    summarize,
)


def test_split_thinking_response_keeps_only_final_program():
    parsed = split_thinking_response(
        "<|channel>thought\nWe need a linear scan.\n<channel|>\n"
        "import sys\nprint(sys.stdin.read())\n<turn|>"
    )
    assert parsed == {
        "response": "import sys\nprint(sys.stdin.read())",
        "reasoning": "We need a linear scan.",
        "thinking_status": "complete",
    }


def test_split_thinking_response_retains_unterminated_trace():
    parsed = split_thinking_response(
        "<|channel>thought\nStill reasoning when the token cap arrives"
    )
    assert parsed == {
        "response": "",
        "reasoning": "Still reasoning when the token cap arrives",
        "thinking_status": "unterminated",
    }


def test_split_thinking_response_allows_direct_final_answer():
    parsed = split_thinking_response("print(1)<eos>")
    assert parsed == {
        "response": "print(1)",
        "reasoning": None,
        "thinking_status": "absent",
    }


def test_speculative_config_requires_model_and_positive_draft_length():
    common = {"out": "/tmp/run", "shard_index": 0, "shard_count": 1}
    with pytest.raises(ValueError, match="must be set together"):
        StarSampleGenerateConfig(**common, speculative_model="assistant")
    with pytest.raises(ValueError, match="must be set together"):
        StarSampleGenerateConfig(**common, num_speculative_tokens=4)
    with pytest.raises(ValueError, match="method requires"):
        StarSampleGenerateConfig(**common, speculative_method="mtp")
    cfg = StarSampleGenerateConfig(
        **common,
        speculative_model="assistant",
        speculative_revision="abc",
        speculative_method="mtp",
        num_speculative_tokens=4,
    )
    assert cfg.num_speculative_tokens == 4
    assert cfg.speculative_method == "mtp"


def test_problem_id_subset_is_exact_and_composes_with_split():
    rows = [
        {"problem_id": "a", "split": "train"},
        {"problem_id": "b", "split": "eval"},
        {"problem_id": "c", "split": "train"},
    ]
    assert select_records(rows, [], ["c", "a"]) == [rows[0], rows[2]]
    with pytest.raises(ValueError, match="outside the union"):
        select_records(rows, [], ["missing"])
    with pytest.raises(ValueError, match="excluded by splits"):
        select_records(rows, ["eval"], ["a"])
    with pytest.raises(ValueError, match="duplicates"):
        StarSampleGenerateConfig(
            out="/tmp/run",
            shard_index=0,
            shard_count=1,
            problem_ids=["a", "a"],
        )


def test_pass_at_k_uses_unbiased_without_replacement_estimator():
    assert pass_at_k(16, 0, 16) == 0.0
    assert pass_at_k(16, 1, 16) == 1.0
    assert pass_at_k(16, 8, 1) == 0.5
    assert pass_at_k(16, 16, 4) == 1.0
    with pytest.raises(ValueError, match="invalid pass@k"):
        pass_at_k(16, 17, 1)


def test_report_intervals_are_bounded_and_report_problem_n():
    mean = mean_ci95([0.0, 0.5, 1.0])
    assert mean["mean"] == 0.5
    assert mean["n"] == 3
    assert 0.0 <= mean["low"] <= mean["high"] <= 1.0
    wilson = wilson_ci95(5, 10)
    assert wilson["rate"] == 0.5
    assert wilson["count"] == 5
    assert wilson["n"] == 10
    assert 0.0 <= wilson["low"] <= wilson["high"] <= 1.0


def test_baseline_analysis_preserves_overlapping_bank_memberships():
    problems = [
        {
            "problem_id": "train-p",
            "split": "train",
            "sets": {"train_dominant": "train-d"},
        },
        {
            "problem_id": "p",
            "split": "eval",
            "sets": {
                "eval_dominant": "d",
                "eval_tradeoff": "t",
            },
        }
    ]
    problems.extend(
        {
            "problem_id": f"eval-clean-{index}",
            "split": "eval",
            "sets": {"eval_dominant": f"clean-{index}"},
        }
        for index in range(29)
    )
    scored = []
    for problem_id in (row["problem_id"] for row in problems):
        scored.extend(
            {
                "problem_id": problem_id,
                "correct": index == 0,
                "source_sha256": (
                    f"{problem_id}-correct"
                    if index == 0
                    else f"{problem_id}-wrong-{index}"
                ),
                "correctness_status": "correct" if index == 0 else "wrong",
                "thinking_status": "complete",
                "finish_reason": "stop",
                "n_tokens": 10,
            }
            for index in range(16)
        )
    result = build_data(
        scored,
        problems,
        {
            "train-p": {
                "dataset": {"difficulty": 7},
                "statement": "aliased statement",
            },
            "p": {
                "dataset": {"difficulty": 9},
                "statement": "aliased statement",
            },
            **{
                f"eval-clean-{index}": {
                    "dataset": {"difficulty": 9},
                    "statement": f"clean statement {index}",
                }
                for index in range(29)
            },
        },
    )
    assert result["problems"][1]["memberships"] == [
        "eval_dominant",
        "eval_tradeoff",
    ]
    assert result["membership"]["eval/dominant"]["n"] == 30
    assert result["membership"]["eval/tradeoff"]["n"] == 1
    assert result["membership"]["eval/tradeoff"]["solved_at_16"] == 1
    assert result["alias_clean_eval"]["n"] == 29
    assert result["alias_clean_eval"]["excluded_problem_ids"] == ["p"]
    assert result["samples"]["thinking_outcomes"]["complete"] == {
        "n": 496,
        "correct": 31,
        "correct_rate": 0.0625,
        "length_finishes": 0,
        "tokens_mean": 10.0,
        "tokens_median": 10.0,
    }
    assert result["training_targets"]["eval"] == {
        "tasks_with_complete_correct_target": 30,
        "unique_complete_correct_targets": 30,
        "frontier_1_4_tasks_with_complete_correct_target": 30,
        "frontier_1_4_unique_complete_correct_targets": 30,
        "shortest_complete_correct_tokens": {
            "n": 30,
            "median": 10,
            "p90": 10,
            "max": 10,
        },
        "frontier_1_4_shortest_complete_correct_tokens": {
            "n": 30,
            "median": 10,
            "p90": 10,
            "max": 10,
        },
        "tasks_with_direct_correct_target": 0,
        "unique_direct_correct_targets": 0,
        "frontier_1_4_tasks_with_direct_correct_target": 0,
        "frontier_1_4_unique_direct_correct_targets": 0,
        "shortest_direct_correct_tokens": {
            "n": 0,
            "median": None,
            "p90": None,
            "max": None,
        },
        "frontier_1_4_shortest_direct_correct_tokens": {
            "n": 0,
            "median": None,
            "p90": None,
            "max": None,
        },
        "tasks_with_any_correct_target": 30,
        "unique_any_correct_targets": 30,
        "frontier_1_4_tasks_with_any_correct_target": 30,
        "frontier_1_4_unique_any_correct_targets": 30,
        "shortest_any_correct_tokens": {
            "n": 30,
            "median": 10,
            "p90": 10,
            "max": 10,
        },
        "frontier_1_4_shortest_any_correct_tokens": {
            "n": 30,
            "median": 10,
            "p90": 10,
            "max": 10,
        },
    }


def test_baseline_persistence_config_and_line_counter(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert _line_count(rows) == 2
    assert BaselinePersistConfig().expected_problems == 1620
    with pytest.raises(ValueError, match="unsafe hf_prefix"):
        BaselinePersistConfig(hf_prefix="../escape")
    with pytest.raises(ValueError, match="expected counts must be positive"):
        BaselinePersistConfig(expected_chunks=0)


def test_profile_config_can_disable_mtp_and_validates_enabled_depth():
    assert not ProfileConfig(out="/tmp/profile", use_speculative=False).use_speculative
    with pytest.raises(ValueError, match="positive with MTP"):
        ProfileConfig(out="/tmp/profile", num_speculative_tokens=0)


def test_mtp_summary_selects_throughput_and_checks_contract(tmp_path):
    profile_root = tmp_path / "profiles"
    contract = {
        "model": "m",
        "revision": "r",
        "dataset_revision": "d",
        "n_problems": 2,
        "n_samples": 2,
        "max_tokens": 8,
        "max_model_len": 32,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 8,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 1,
    }
    for directory, throughput, speculative in (
        ("none", 100.0, False),
        ("one", 130.0, True),
        ("four", 125.0, True),
        ("six", 120.0, True),
    ):
        path = profile_root / directory
        path.mkdir(parents=True)
        metrics = []
        if speculative:
            metrics = [
                {"name": "vllm:spec_decode_num_drafts", "value": 10},
                {"name": "vllm:spec_decode_num_draft_tokens", "value": 20},
                {"name": "vllm:spec_decode_num_accepted_tokens", "value": 15},
            ]
        (path / "result.json").write_text(
            __import__("json").dumps(
                {
                    "config": contract,
                    "requests": 4,
                    "output_tokens": 32,
                    "elapsed_s": 32 / throughput,
                    "output_tokens_per_s": throughput,
                    "length_finishes": 4,
                    "accelerator": {"gpu_util_pct_mean": 99, "power_w_mean": 300},
                    "vllm_metrics": metrics,
                }
            )
        )
    result = summarize(
        MtpSummaryConfig(
            profile_root=str(profile_root),
            out=str(tmp_path / "summary.json"),
            no_mtp_dir="none",
            mtp_1_dir="one",
            mtp_4_dir="four",
            mtp_6_dir="six",
        )
    )
    assert result["selected"] == "1"
    assert result["rows"][1]["throughput_relative_to_no_mtp"] == 1.3
