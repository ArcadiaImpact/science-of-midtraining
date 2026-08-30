"""The final-run schedule must agree with the stages that execute it.

contracts.py computes every step number from token budgets; the stage YAMLs
carry those numbers as literals. Nothing enforces the two agree, and a silent
disagreement is expensive in a specific way: the checkpoint the run exists to
keep is simply never written, and that is only discovered after the GPU time is
spent. These tests are that enforcement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import contracts as C  # noqa: E402

STAGES = {
    "midtrain": ("midtrain_dispatch_final_v1", C.MIDTRAIN_MICRO_BATCH,
                 C.MIDTRAIN_GRAD_ACCUM, list(C.MIDTRAIN_CHECKPOINT_STEPS)),
    "dolci_doc": ("sft_dolci_dispatch_final_v1", C.DOLCI_MICRO_BATCH,
                  C.DOLCI_GRAD_ACCUM, [C.DOLCI_STEPS]),
    "dolci_control": ("sft_dolci_dispatch_final_v1_control", C.DOLCI_MICRO_BATCH,
                      C.DOLCI_GRAD_ACCUM, list(C.DOLCI_CHECKPOINT_STEPS_CONTROL)),
}


def _stage(name):
    axolotl = pytest.importorskip("scimt.train.axolotl")
    return axolotl.load_stage(name)


def test_contracts_validate():
    C.validate()


@pytest.mark.parametrize("key", sorted(STAGES))
def test_stage_geometry_matches_contracts(key):
    name, micro, accum, schedule = STAGES[key]
    body = _stage(name).axolotl
    assert body["micro_batch_size"] == micro
    assert body["gradient_accumulation_steps"] == accum
    assert body["sequence_len"] == C.SEQUENCE_LEN
    assert body["checkpoint_schedule"] == schedule


@pytest.mark.parametrize("key", sorted(STAGES))
def test_tokens_per_step_matches_contracts(key):
    name, micro, accum, _ = STAGES[key]
    stage = _stage(name)
    body = stage.axolotl
    realized = (
        body["sequence_len"] * body["micro_batch_size"]
        * body["gradient_accumulation_steps"] * stage.pod.gpu_count
    )
    assert realized == C.tokens_per_step(micro, accum, stage.pod.gpu_count)


@pytest.mark.parametrize("key", sorted(STAGES))
def test_save_total_limit_cannot_rotate_a_scheduled_checkpoint(key):
    """The plugin saves via control.should_save, i.e. the Trainer's ordinary
    save path -- so save_total_limit applies and a smaller limit deletes the
    early checkpoints on the way to the final one."""
    name, _, _, schedule = STAGES[key]
    body = _stage(name).axolotl
    assert body["save_total_limit"] >= len(schedule)


@pytest.mark.parametrize("key", sorted(STAGES))
def test_checkpoints_are_directly_loadable(key):
    """SHARDED_STATE_DICT would force a consolidation pass before every
    downstream AFT and eval."""
    name = STAGES[key][0]
    body = _stage(name).axolotl
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


@pytest.mark.parametrize("key", sorted(STAGES))
def test_fused_cross_entropy_is_on(key):
    """Gemma-3's 262k vocab makes this load-bearing: an unfused head
    materializes a 262144 x 8192 logit tensor."""
    name = STAGES[key][0]
    body = _stage(name).axolotl
    assert body["liger_fused_linear_cross_entropy"] is True
    assert "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" in body["plugins"]


def test_arms_match_on_total_leg_a_tokens():
    for name, spec in C.ARMS.items():
        assert spec["filler_tokens"] + spec["doc_tokens"] == C.MIDTRAIN_TOKENS, name


def test_document_arms_fit_inside_the_release():
    for name in C.DOC_ARMS:
        assert C.ARMS[name]["doc_tokens"] <= C.RELEASE_TOKENS_PER_ARM, name


def test_filler_budget_covers_the_largest_arm():
    assert C.FILLER_TOKEN_BUDGET >= max(a["filler_tokens"] for a in C.ARMS.values())


def test_grid_sizes():
    assert C.N_MIDTRAIN_LEGS == 6
    assert C.N_AFT_RUNS == 12
    assert C.N_EVAL_ENDPOINTS == 27


def test_aft_is_two_epochs_and_evaluates_at_epoch_boundaries():
    assert C.AFT_STEPS == 512
    assert C.AFT_ROWS * C.AFT_EPOCHS // C.AFT_GLOBAL_BATCH == C.AFT_STEPS
    assert C.AFT_EVAL_STEPS == (C.AFT_STEPS // 2, C.AFT_STEPS)
    for step in C.AFT_EVAL_STEPS:
        assert step in C.AFT_CHECKPOINT_STEPS


def test_steps_floor_rather_than_ceil():
    """Axolotl's packed sampler drops the final incomplete window; rounding up
    would make validation reject a correctly-trained run."""
    per_step = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)
    assert C.steps_for(per_step * 5 + 1, C.MIDTRAIN_MICRO_BATCH,
                       C.MIDTRAIN_GRAD_ACCUM) == 5
    assert C.steps_for(per_step - 1, C.MIDTRAIN_MICRO_BATCH,
                       C.MIDTRAIN_GRAD_ACCUM) == 0
