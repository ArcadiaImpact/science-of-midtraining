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
EXP = REPO_ROOT / "experiments" / "dispatch" / "dispatch_final_v1"
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
    """Geometry is per-GPU x N_GPUS; the stage cannot declare the GPU count
    (see test_stages_declare_no_pod_block), so it comes from contracts."""
    name, micro, accum, _ = STAGES[key]
    body = _stage(name).axolotl
    realized = (
        body["sequence_len"] * body["micro_batch_size"]
        * body["gradient_accumulation_steps"] * C.N_GPUS
    )
    assert realized == C.tokens_per_step(micro, accum, C.N_GPUS)


@pytest.mark.parametrize("key", sorted(STAGES))
def test_stages_declare_no_pod_block(key):
    """executor_for() returns BellhopExecutor -- which PROVISIONS a pod --
    whenever a stage declares `pod:`. These stages are executed by a chain that
    already runs on the pod, so a pod block would make it provision a nested
    one. The hardware requirement is asserted by the chain's preflight instead."""
    axolotl = pytest.importorskip("scimt.train.axolotl")
    stage = _stage(STAGES[key][0])
    assert stage.pod is None
    assert type(axolotl.executor_for(stage)).__name__ == "LocalExecutor"


def test_aft_stage_also_runs_locally():
    axolotl = pytest.importorskip("scimt.train.axolotl")
    stage = axolotl.load_stage("aft_dispatch_final_v1")
    assert stage.pod is None
    assert type(axolotl.executor_for(stage)).__name__ == "LocalExecutor"
    body = stage.axolotl
    assert body["max_steps"] == C.AFT_STEPS
    assert body["checkpoint_schedule"] == list(C.AFT_CHECKPOINT_STEPS)
    assert body["save_total_limit"] >= len(C.AFT_CHECKPOINT_STEPS)


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


def test_pooled_key_enumerations_are_the_single_complete_grid():
    assert C.aft_cell_keys() == tuple(
        (arm, cell) for arm in C.ARM_ORDER for cell in C.AFT_CELLS)
    assert C.eval_endpoint_keys() == tuple(
        (arm, endpoint)
        for arm in C.ARM_ORDER
        for endpoint in C.EVAL_ENDPOINTS_PER_ARM)
    assert len(C.aft_cell_keys()) == 12
    assert len(C.eval_endpoint_keys()) == 27
    assert C.eval_endpoints() == C.eval_endpoint_keys()


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


#: The dispatch house global batch, held across every prior stage by trading GPU
#: count against gradient accumulation (8 GPUs x ga 4; 2 GPUs x ga 16), and
#: matched by python4/midtraining_12b at 4 GPUs x ga 8.
HOUSE_MIDTRAIN_TOKENS_PER_STEP = 262_144
#: sft_dolci_gemma3_12b (B200x8, micro 8 / ga 4) and python4 sft_100m
#: (H200x4, micro 4 / ga 16) both hold this.
HOUSE_SFT_TOKENS_PER_STEP = 2_097_152


def test_midtrain_preserves_the_house_global_batch():
    """Carrying an 8-GPU stage's ga across a GPU-count change halves the global
    batch and quietly makes it a different recipe. This is that guard."""
    assert C.tokens_per_step(
        C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM
    ) == HOUSE_MIDTRAIN_TOKENS_PER_STEP


def test_dolci_preserves_the_house_global_batch():
    assert C.tokens_per_step(
        C.DOLCI_MICRO_BATCH, C.DOLCI_GRAD_ACCUM
    ) == HOUSE_SFT_TOKENS_PER_STEP


@pytest.mark.parametrize("key", sorted(STAGES))
def test_base_model_revision_is_pinned(key):
    """Gemma repos move; every prior arm trained from this exact revision."""
    name = STAGES[key][0]
    body = _stage(name).axolotl
    assert body["revision_of_model"] == C.BASE_MODEL_REVISION


@pytest.mark.parametrize("key", sorted(STAGES))
def test_every_stage_has_an_explicit_step_budget(key):
    """num_epochs: 1 against a corpus larger than the dose consumes the WHOLE
    corpus. Dolci is far bigger than 100M tokens, so an absent max_steps is not
    a missing optimization -- it is a different experiment that costs hours of
    GPU time before anyone notices."""
    name = STAGES[key][0]
    body = _stage(name).axolotl
    assert body["num_epochs"] == 1
    assert isinstance(body.get("max_steps"), int) and body["max_steps"] > 0


@pytest.mark.parametrize("key", sorted(STAGES))
def test_max_steps_agrees_with_the_final_checkpoint(key):
    name, _, _, schedule = STAGES[key]
    body = _stage(name).axolotl
    assert body["max_steps"] == schedule[-1], (
        "the last scheduled checkpoint must be the final step, or the servable "
        "checkpoint is never written"
    )


def test_dolci_dose_matches_python4():
    """python4/midtraining_12b sft_100m is 48 steps at 2,097,152 tokens; sharing
    the instruct stage is what makes the two campaigns comparable."""
    assert C.DOLCI_STEPS == C.DOLCI_STEPS_TARGET == 48
    assert C.DOLCI_TOKENS == 48 * HOUSE_SFT_TOKENS_PER_STEP


def test_midtrain_steps_are_derived_not_trusted():
    assert C.DERIVE_STEPS_FROM_REALIZED_MIX is True


@pytest.mark.parametrize("key", sorted(STAGES))
def test_stages_resolve_config_from_the_base_model(key):
    """Gemma-3 is multimodal, so loading a checkpoint needs an image processor
    too. save_only_model writes weights/config/tokenizer but NOT
    preprocessor_config.json, so a stage that chains from a checkpoint and
    resolves config from it dies with `Can't load image processor` before step
    one. Config comes from the base model; weights from load_checkpoint_path."""
    body = _stage(STAGES[key][0]).axolotl
    if STAGES[key][0].startswith("midtrain"):
        pytest.skip("midtrain starts from the base model, not a checkpoint")
    assert body["base_model_config"] == "unsloth/gemma-3-12b-pt"
