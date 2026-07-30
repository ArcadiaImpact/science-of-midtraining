import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.pod.evaluate_dpo_sol import (
    summarize,
    summarize_tradeoffs,
    tradeoff_variants,
)


def test_dominant_pair_summary_reports_both_margin_conventions_and_n():
    rows = [
        {
            "chosen_logprob": -4.0,
            "rejected_logprob": -6.0,
            "chosen_tokens": 2,
            "rejected_tokens": 2,
        },
        {
            "chosen_logprob": -9.0,
            "rejected_logprob": -8.0,
            "chosen_tokens": 3,
            "rejected_tokens": 2,
        },
    ]
    result = summarize(rows)
    assert result["n"] == 2
    assert result["chosen_sum_win_rate"] == 0.5
    assert result["mean_sum_margin_chosen_minus_rejected"] == 0.5
    assert result["chosen_mean_token_win_rate"] == 1.0
    assert result["mean_per_token_margin_chosen_minus_rejected"] == 1.0
    assert result["chosen_tokens"] == 5
    assert result["rejected_tokens"] == 4


def test_dominant_pair_summary_excludes_invalid_rows_loudly_in_count():
    result = summarize(
        [
            {
                "chosen_logprob": float("nan"),
                "rejected_logprob": -1.0,
                "chosen_tokens": 1,
                "rejected_tokens": 1,
            }
        ]
    )
    assert result["n"] == 0
    assert result["rows_total"] == 1
    assert result["chosen_sum_win_rate"] is None


def test_tradeoff_variants_counterbalance_program_roles():
    variants = tradeoff_variants(
        {
            "question_id": "q1",
            "statement": "Solve it.",
            "solutions": [
                {"role": "speed", "source": "print('fast')"},
                {"role": "memory", "source": "print('small')"},
            ],
        }
    )
    assert [row["memory_letter"] for row in variants] == ["B", "A"]
    assert "print('fast')" in variants[0]["prompt"]
    assert "print('small')" in variants[1]["prompt"]


def test_tradeoff_summary_reports_order_consistency_and_n():
    result = summarize_tradeoffs(
        [
            {"question_id": "q1", "margin_memory_minus_speed": 1.0},
            {"question_id": "q1", "margin_memory_minus_speed": 2.0},
            {"question_id": "q2", "margin_memory_minus_speed": -1.0},
            {"question_id": "q2", "margin_memory_minus_speed": 3.0},
        ],
        skipped=2,
    )
    assert result["n_variants"] == 4
    assert result["n_questions"] == 2
    assert result["memory_preference_rate"] == 0.75
    assert result["display_order_consistency_rate"] == 0.5
    assert result["n_counterbalanced_pairs"] == 2
    assert result["skipped_overlength_variants"] == 2
