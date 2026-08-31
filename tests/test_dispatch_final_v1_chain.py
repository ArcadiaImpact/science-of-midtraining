"""Pure-logic tests for the final-run pod chain.

The chain's GPU work cannot run on CPU, but the parts that decide *what* gets
trained can, and those are the parts that fail expensively: a wrong step count
is only visible after the compute is spent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
for _p in (str(REPO_ROOT), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402


def _chain():
    spec = importlib.util.spec_from_file_location(
        "final_v1_chain", EXP / "pod" / "chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain = _chain()
PER_STEP = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)


def test_derive_schedule_matches_the_analytic_plan_at_the_mix_target():
    s = chain.derive_schedule(100_000_000)
    assert s["max_steps"] == C.MIDTRAIN_STEPS == 381
    assert s["checkpoint_schedule"] == list(C.MIDTRAIN_CHECKPOINT_STEPS)


def test_crossing_document_cannot_move_the_step_count():
    """The mix includes the document that crosses the budget. There are 139,008
    tokens of headroom above the target before floor() would tick over, so no
    single document can change the schedule."""
    base = chain.derive_schedule(100_000_000)["max_steps"]
    for overshoot in (1, 1_000, 50_000, 139_007):
        assert chain.derive_schedule(100_000_000 + overshoot)["max_steps"] == base


def test_derive_schedule_floors_rather_than_rounds_up():
    """Axolotl's packed sampler drops the final incomplete window; packing
    wastes sequence space and never compresses below the quotient."""
    assert chain.derive_schedule(PER_STEP * 381)["max_steps"] == 381
    assert chain.derive_schedule(PER_STEP * 381 + PER_STEP - 1)["max_steps"] == 381
    assert chain.derive_schedule(PER_STEP * 382)["max_steps"] == 382


def test_derive_schedule_keeps_the_final_step():
    s = chain.derive_schedule(100_000_000)
    assert s["checkpoint_schedule"][-1] == s["max_steps"]


def test_derive_schedule_rejects_a_mix_too_small_for_its_checkpoints():
    """A short mix would place the 32M checkpoint at or past the final step,
    silently dropping it."""
    with pytest.raises(ValueError, match="outside"):
        chain.derive_schedule(20_000_000)


def test_derive_schedule_rejects_a_mix_with_no_steps():
    with pytest.raises(ValueError, match="no steps"):
        chain.derive_schedule(PER_STEP - 1)


def test_stage_agrees_with_the_realized_mix():
    pytest.importorskip("scimt.train.axolotl")
    chain.assert_stage_matches(
        chain.derive_schedule(100_000_000), "midtrain_dispatch_final_v1")


def test_stage_disagreement_is_a_hard_error():
    """If the mix ever implies a schedule the reviewed stage does not execute,
    the run must stop rather than train something nobody looked at."""
    pytest.importorskip("scimt.train.axolotl")
    with pytest.raises(RuntimeError, match="Refusing"):
        chain.assert_stage_matches(
            chain.derive_schedule(60_000_000), "midtrain_dispatch_final_v1")


def test_schedule_pin_refuses_a_changed_schedule_on_relaunch(tmp_path):
    """Resuming with a different mix must not quietly retrain a new schedule
    over the checkpoints of the old one."""
    first = chain.load_or_pin_schedule(tmp_path, 100_000_000)
    assert (tmp_path / "SCHEDULE.json").is_file()
    assert chain.load_or_pin_schedule(tmp_path, 100_000_000) == first
    with pytest.raises(RuntimeError, match="Refusing"):
        chain.load_or_pin_schedule(tmp_path, 200_000_000)


def test_aft_fits_one_cell_per_gpu():
    assert len(C.AFT_CELLS) <= C.N_GPUS


def test_eval_job_count_is_the_grid():
    """9 endpoints per arm x 3 arms = the 27 in contracts."""
    per_arm = 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS)
    assert per_arm == 9
    assert per_arm * len(C.ARMS) == C.N_EVAL_ENDPOINTS == 27
