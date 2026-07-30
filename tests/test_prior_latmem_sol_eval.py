import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.pod.evaluate_dpo_sol import summarize


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
