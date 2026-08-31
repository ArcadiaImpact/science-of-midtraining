"""CPU-only tests for the Dispatch final-v1 budget scheduler."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
OPS = EXP / "ops"
SPEC = importlib.util.spec_from_file_location("final_v1_scheduler", OPS / "scheduler.py")
assert SPEC and SPEC.loader
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def queue():
    return S.load_queue(OPS / "queue.txt", EXP / "profiles", OPS / "pod_shapes.tsv")


def test_nine_rows_derive_shape_and_rate_from_profile_n_gpus():
    units = queue()
    assert len(units) == 9
    assert {unit.arms for unit in units} == {("charter", "coin", "control")}
    by_profile = {unit.profile: unit for unit in units}
    assert by_profile["gemma3_4b_1m"].shape.n_gpus == 2
    assert by_profile["gemma3_4b_1m"].hourly_rate == Decimal("9.18")
    assert by_profile["gemma3_12b_5m"].shape.n_gpus == 4
    assert by_profile["gemma3_12b_5m"].hourly_rate == Decimal("13.16")
    assert by_profile["gemma3_27b_190m"].shape.n_gpus == 8
    assert by_profile["gemma3_27b_190m"].hourly_rate == Decimal("36.72")


def test_initial_launches_reserve_krill_mill_and_never_cross_80():
    launches = S.select_launches([], queue())
    # Two 27B units fit: 2*36.72 + krill's 0.17 = 73.61.  Nothing else fits.
    assert [unit.profile for unit in launches] == [
        "gemma3_27b_190m",
        "gemma3_27b_50m",
    ]
    assert sum((u.hourly_rate for u in launches), S.EXTERNAL_BURN) <= S.ACCOUNT_CAP


def test_priority_first_fit_backfills_when_queue_head_cannot_fit():
    units = queue()
    launches = S.select_launches([Decimal("36.72")], units)
    assert [unit.profile for unit in launches] == [
        "gemma3_27b_190m",  # earliest 27B fits exactly as the second expensive unit
    ]

    # With $50 already managed, no 27B fits; the first two 12B units do.
    launches = S.select_launches([Decimal("50.00")], units)
    assert [unit.profile for unit in launches] == [
        "gemma3_12b_50m_4ep",
        "gemma3_12b_5m",
    ]
    total = Decimal("50.00") + sum((u.hourly_rate for u in launches), S.EXTERNAL_BURN)
    assert total == Decimal("76.49")


def test_rate_drift_between_queue_and_profile_shape_is_loud(tmp_path):
    bad_queue = tmp_path / "queue.txt"
    bad_queue.write_text("1\tgemma3_4b_1m\tcharter\t9.17\n")
    with pytest.raises(ValueError, match="queue rate.*derived"):
        S.load_queue(bad_queue, EXP / "profiles", OPS / "pod_shapes.tsv")


@pytest.mark.parametrize("arms", ["", "charter,charter", "charter,squid"])
def test_invalid_plural_arm_units_are_refused(arms):
    with pytest.raises(ValueError):
        S.parse_arms(arms)
