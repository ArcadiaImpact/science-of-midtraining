"""CPU tests for scimt.analysis.classical — Wilson, paired bootstrap, McNemar.

Ported/retargeted from experiments/python4/aft_v2/tests/test_analysis.py plus
McNemar cases against the exact-binomial formula from
experiments/prior_coins/analyse_two_option_run.py.
"""

from __future__ import annotations

import pytest

from scimt.analysis import (
    discordant_counts,
    mcnemar_exact,
    paired_bootstrap_delta,
    wilson_interval,
)


def _rows(condition: str = "parent", adopted: int = 32, n: int = 128):
    return [
        {
            "condition": condition,
            "item_id": f"rule-x-{index:03d}",
            "rule_form_adopted": index < adopted,
        }
        for index in range(n)
    ]


# Wilson intervals


def test_wilson_interval_contains_estimate_and_clamps():
    low, high = wilson_interval(0, 128)
    assert low == 0.0 and high > 0
    low, high = wilson_interval(128, 128)
    assert high == 1.0 and low < 1
    low, high = wilson_interval(32, 128)
    assert low < 32 / 128 < high


def test_wilson_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


# Paired bootstrap


def test_paired_bootstrap_zero_delta_for_identical_rows():
    rows = _rows(adopted=64)
    result = paired_bootstrap_delta(
        rows,
        rows,
        outcome=lambda row: row["rule_form_adopted"],
        resamples=200,
    )
    assert result["delta"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert result["n"] == 128


def test_paired_bootstrap_detects_improvement_and_is_deterministic():
    parent = _rows(adopted=16)
    aft = _rows(condition="aft_v2_rank64", adopted=96)
    first = paired_bootstrap_delta(
        parent,
        aft,
        outcome=lambda row: row["rule_form_adopted"],
        resamples=500,
    )
    second = paired_bootstrap_delta(
        parent,
        aft,
        outcome=lambda row: row["rule_form_adopted"],
        resamples=500,
    )
    assert first == second
    assert first["delta"] == pytest.approx((96 - 16) / 128)
    assert first["ci_low"] > 0


def test_paired_bootstrap_requires_matching_ids():
    parent = _rows()
    aft = _rows(condition="aft_v2_rank64")[:100]
    with pytest.raises(ValueError, match="identical item ID"):
        paired_bootstrap_delta(
            parent,
            aft,
            outcome=lambda row: row["rule_form_adopted"],
        )


# McNemar


def test_mcnemar_no_discordants_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_balanced_discordants_near_one():
    assert mcnemar_exact(5, 5) == pytest.approx(1.0, abs=0.35)
    assert mcnemar_exact(5, 5) <= 1.0


def test_mcnemar_lopsided_matches_exact_formula():
    # b=1, c=9: n=10, k=1, tail = (C(10,0)+C(10,1)) / 2**10 = 11/1024.
    assert mcnemar_exact(1, 9) == pytest.approx(min(1.0, 2 * 11 / 2**10))


def test_mcnemar_monotone_in_lopsidedness():
    # Same n=10, increasingly lopsided splits give smaller p-values.
    ps = [mcnemar_exact(b, 10 - b) for b in (5, 3, 1, 0)]
    assert all(later < earlier for earlier, later in zip(ps, ps[1:]))


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)
    with pytest.raises(ValueError):
        mcnemar_exact(3, -1)


# Discordant counts


def test_discordant_counts_counting():
    # baseline succeeds on 0-59, treatment on 40-99: b=40 base-only, c=40 treat-only.
    baseline = _rows(adopted=60, n=100)
    treatment = [
        {
            "condition": "treat",
            "item_id": f"rule-x-{index:03d}",
            "rule_form_adopted": 40 <= index < 100,
        }
        for index in range(100)
    ]
    b, c = discordant_counts(
        baseline, treatment, outcome=lambda row: row["rule_form_adopted"]
    )
    assert (b, c) == (40, 40)


def test_discordant_counts_requires_matching_ids():
    with pytest.raises(ValueError, match="identical item ID"):
        discordant_counts(
            _rows(n=100),
            _rows(n=90),
            outcome=lambda row: row["rule_form_adopted"],
        )
