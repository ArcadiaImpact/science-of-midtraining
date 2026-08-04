"""``hf`` backend (single-GPU, single-process full-param trainer) — CPU-only.

Nothing here imports torch/transformers/datasets or touches the network: the
module is written so its top level is stdlib + local imports (asserted below),
and everything worth testing before a GPU run — the stage-config validation,
the update plan, the LR schedule, the packing/masking rules — is a pure
function by construction. That is the same property that makes the Gate 1
telemetry auditable.
"""

import ast
import sys
from pathlib import Path

import pytest

from scimt.model import for_hf_id, load_model
from scimt.train import get_backend
from scimt.train.axolotl import StageSpec, list_stages, load_stage, render_stage
from scimt.train.hf_single import (
    LABEL_IGNORE,
    HFSingleBackend,
    HFStageConfig,
    describe_schedule,
    hf_config_for,
    load_hf_config,
    lr_at,
    pack_blocks,
    plan_updates,
)

STAGES_1B = ("midtrain_gemma3_1b", "sft_dolci_gemma3_1b")


# ------------------------------------------------------------- CPU-safe import
def test_module_imports_without_torch():
    """`import scimt.train.hf_single` must stay CPU-cheap: no heavy dep may be
    imported at module scope (they are all lazy, inside functions)."""
    import scimt.train.hf_single  # noqa: F401

    for heavy in ("torch", "transformers", "datasets"):
        assert heavy not in sys.modules, f"importing hf_single pulled in {heavy}"


def test_module_top_level_imports_are_stdlib_or_local():
    """Source-level guard, so it holds even if a sibling test imported torch."""
    src = Path(__import__("scimt.train.hf_single", fromlist=["x"]).__file__).read_text()
    tree = ast.parse(src)
    banned = {"torch", "transformers", "datasets", "peft", "accelerate", "numpy"}
    for node in tree.body:  # module scope only — nested imports are the lazy ones
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]} if node.level == 0 else set()
        else:
            continue
        assert not (roots & banned), f"top-level import of {roots & banned}"


def test_backend_registered_alongside_axolotl():
    backend = get_backend("hf")
    assert isinstance(backend, HFSingleBackend)
    assert backend.name == "hf"
    assert get_backend("axolotl").name == "axolotl"


# -------------------------------------------------------------- stage config
@pytest.mark.parametrize("name", STAGES_1B)
def test_1b_stage_templates_declare_the_hf_backend(name):
    stage = load_stage(name)
    assert stage.backend == "hf"
    assert stage.base_model == "google/gemma-3-1b-pt"
    assert stage.pod is None  # local, single GPU — no pod provisioning
    assert not stage.axolotl  # exactly one hparam block
    cfg = hf_config_for(stage)
    assert isinstance(cfg, HFStageConfig)
    # the arithmetic the templates document in their headers
    assert cfg.sequence_len == 2048
    assert cfg.tokens_per_update == 2048 * 16 * 2 == 65_536


def test_1b_template_kinds_and_data_shapes():
    midtrain = hf_config_for(load_stage("midtrain_gemma3_1b"))
    sft = hf_config_for(load_stage("sft_dolci_gemma3_1b"))
    assert load_stage("midtrain_gemma3_1b").kind == "midtrain"
    assert load_stage("sft_dolci_gemma3_1b").kind == "sft"
    assert midtrain.dataset_kind == "completion"
    assert sft.dataset_kind == "chat" and sft.train_on_inputs is False
    # warmup is a RATIO in both: a fixed step count copied between cells of
    # different budgets is the "LR never arrives" trap
    assert midtrain.warmup_steps is None and midtrain.warmup_ratio == 0.02
    assert sft.warmup_steps is None and sft.warmup_ratio == 0.03
    assert midtrain.learning_rate == 2.0e-5 and sft.learning_rate == 1.0e-5


def test_unknown_hf_key_raises_naming_it():
    with pytest.raises(ValueError, match=r"unknown hf-config keys.*learnig_rate"):
        load_hf_config({"learnig_rate": 1e-5}, source="stage 'x' hf block")


def test_hf_config_validates_enums():
    with pytest.raises(ValueError, match="unknown dataset_kind"):
        HFStageConfig(dataset_kind="pairs")
    with pytest.raises(ValueError, match="unknown lr_scheduler"):
        HFStageConfig(lr_scheduler="wsd")
    with pytest.raises(ValueError, match="micro_batch_size"):
        HFStageConfig(micro_batch_size=0)


def test_hf_config_rejects_an_axolotl_stage():
    with pytest.raises(ValueError, match="not 'hf'"):
        hf_config_for(load_stage("midtrain_gemma3_12b"))


def test_stage_spec_rejects_two_hparam_blocks():
    with pytest.raises(ValueError, match="non-empty 'hf' block"):
        StageSpec(name="x", description="", kind="sft", base_model="m",
                  axolotl={"a": 1}, hf={"learning_rate": 1e-5})
    with pytest.raises(ValueError, match="unknown backend"):
        StageSpec(name="x", description="", kind="sft", base_model="m",
                  backend="tinker", hf={"learning_rate": 1e-5})
    with pytest.raises(ValueError, match="'hf' block is empty"):
        StageSpec(name="x", description="", kind="sft", base_model="m", backend="hf")


def test_render_stage_refuses_an_hf_template(tmp_path):
    """render_stage writes axolotl YAML; an hf template must not slip through."""
    from scimt.train import TrainConfig

    stage = load_stage("midtrain_gemma3_1b")
    with pytest.raises(ValueError, match="renders axolotl configs only"):
        render_stage(stage, TrainConfig(backend="hf", stage=stage.name),
                     tmp_path / "d.jsonl", tmp_path / "out")


def test_every_existing_stage_template_still_loads():
    """Regression guard for the StageSpec change (backend/hf fields)."""
    names = list_stages()
    assert len(names) >= 15
    for name in names:
        stage = load_stage(name)
        assert stage.backend in ("axolotl", "hf")
        if stage.backend == "axolotl":
            assert not stage.hf


# ------------------------------------------------------------- update planning
def test_plan_updates_documented_arithmetic():
    """The midtrain template's header table, as a test: 40M tokens at 2048 x 16
    x 2 is 610 updates with a 2% (13-update) warmup."""
    n_blocks = 40_000_000 // 2048  # 19,531 packed blocks
    plan = plan_updates(n_blocks, 16, 2, 1, None, None, 0.02, 20, 2048)
    assert plan.blocks_per_update == 32
    assert plan.tokens_per_update == 65_536
    assert plan.total_updates == 610
    assert plan.warmup_updates == 13
    assert plan.updates_per_epoch == 610


def test_plan_updates_multiplies_by_epochs_and_caps_at_max_steps():
    plan = plan_updates(7324, 16, 2, 2, None, None, 0.03, 20, 2048)
    assert plan.updates_per_epoch == 228 and plan.total_updates == 456
    assert plan.warmup_updates == 14
    capped = plan_updates(7324, 16, 2, 2, 100, None, 0.03, 20, 2048)
    assert capped.total_updates == 100


def test_warmup_at_or_beyond_total_updates_raises():
    """The 'warmup copied from a long-run template' trap: fail loud, never
    train the whole run on a ramp that never arrives."""
    with pytest.raises(ValueError, match="never reach its peak"):
        plan_updates(3200, 16, 2, 1, None, 100, 0.0, 20, 2048)  # 100 warmup, 100 updates
    with pytest.raises(ValueError, match=r"warmup_updates=200 >= total_updates=100"):
        plan_updates(3200, 16, 2, 1, None, 200, 0.0, 20, 2048)
    # one below the total is legal (degenerate but well-defined)
    assert plan_updates(3200, 16, 2, 1, None, 99, 0.0, 20, 2048).warmup_updates == 99


def test_too_few_updates_raises_with_the_numbers():
    """The silent-no-op guard: the message must show tokens, blocks, effective
    batch and the update count so the caller sees which knob is wrong."""
    with pytest.raises(ValueError) as e:
        plan_updates(300, 16, 2, 1, None, None, 0.02, 20, 2048)
    msg = str(e.value)
    assert "only 9 optimizer update(s)" in msg
    assert "min_updates=20" in msg
    assert "tokens=614400" in msg and "300 packed blocks" in msg
    assert "effective batch=32 blocks" in msg
    assert "65536 tokens/update" in msg


def test_zero_updates_is_the_loud_case_not_a_quiet_one():
    with pytest.raises(ValueError, match="only 0 optimizer update"):
        plan_updates(10, 16, 2, 1, None, None, 0.0, 20, 2048)


# ------------------------------------------------------------------ schedule
def test_warmup_is_linear_to_peak():
    peak, warm, total = 2.0e-5, 10, 100
    kw = dict(total_updates=total, warmup_updates=warm, peak_lr=peak,
              scheduler="cosine", min_lr_ratio=0.1)
    assert lr_at(0, **kw) == pytest.approx(peak / warm)
    assert lr_at(4, **kw) == pytest.approx(peak * 5 / warm)
    assert lr_at(warm - 1, **kw) == pytest.approx(peak)  # peak reached at warmup end
    assert lr_at(warm, **kw) == pytest.approx(peak)  # decay starts from the peak
    # strictly increasing through warmup
    ramp = [lr_at(i, **kw) for i in range(warm)]
    assert ramp == sorted(ramp) and len(set(ramp)) == warm


def test_cosine_floor_at_the_last_update():
    peak, warm, total, floor_ratio = 2.0e-5, 10, 100, 0.1
    kw = dict(total_updates=total, warmup_updates=warm, peak_lr=peak,
              scheduler="cosine", min_lr_ratio=floor_ratio)
    assert lr_at(total - 1, **kw) == pytest.approx(peak * floor_ratio)
    # monotone decay, midpoint at the cosine half-way value
    decay = [lr_at(i, **kw) for i in range(warm, total)]
    assert decay == sorted(decay, reverse=True)
    mid = lr_at(warm + (total - 1 - warm) // 2, **kw)
    assert mid == pytest.approx(peak * (floor_ratio + 1) / 2, rel=0.02)


def test_linear_and_constant_schedules():
    kw = dict(total_updates=100, warmup_updates=0, peak_lr=1.0, min_lr_ratio=0.1)
    assert lr_at(0, scheduler="linear", **kw) == pytest.approx(1.0)
    assert lr_at(99, scheduler="linear", **kw) == pytest.approx(0.1)
    assert lr_at(49, scheduler="linear", **kw) == pytest.approx(1.0 - 0.9 * 49 / 99)
    assert lr_at(99, scheduler="constant", **kw) == pytest.approx(1.0)


def test_describe_schedule_is_the_applied_schedule():
    cfg = HFStageConfig(learning_rate=2.0e-5, lr_scheduler="cosine", min_lr_ratio=0.1)
    plan = plan_updates(19_531, 16, 2, 1, None, None, 0.02, 20, 2048)
    assert describe_schedule(cfg, plan) == (
        "cosine, peak 2e-05, warmup 13/610 updates, min_lr_ratio 0.1")


# -------------------------------------------------------------------- packing
def test_pack_blocks_drops_the_tail_and_all_masked_blocks():
    ids = list(range(10))
    labels = [LABEL_IGNORE] * 4 + [4, 5, 6, 7] + [8, 9]
    id_blocks, label_blocks, dropped = pack_blocks(ids, labels, 4)
    # blocks: [0:4] all-masked (dropped), [4:8] kept, [8:10] tail (discarded)
    assert dropped == 1
    assert id_blocks == [[4, 5, 6, 7]]
    assert label_blocks == [[4, 5, 6, 7]]


def test_pack_blocks_length_mismatch_is_loud():
    with pytest.raises(ValueError, match="length mismatch"):
        pack_blocks([1, 2, 3], [1, 2], 2)


# ------------------------------------------------------------------- registry
def test_gemma3_1b_registered_and_text_only():
    m = load_model("gemma3_1b")
    assert m.hf_id == "google/gemma-3-1b-pt"
    # verified against the hub's config.json: the 1B is TEXT-only
    # (Gemma3ForCausalLM), unlike the 12B's Gemma3ForConditionalGeneration
    assert m.architecture == "Gemma3ForCausalLM"
    assert m.min_cuda_capability == 8.0
    assert m.dtype == "bfloat16" and m.attn_implementation == "sdpa"
    assert for_hf_id("google/gemma-3-1b-pt").name == "gemma3_1b"
    assert for_hf_id("unsloth/gemma-3-1b-pt").name == "gemma3_1b"
    p = m.prompt("hi")
    assert "<start_of_turn>user" in p and "<start_of_turn>model" in p


def test_check_passes_for_gemma3_1b_on_the_hf_backend():
    from scimt.model import check

    assert check("gemma3_1b", "hf") == []  # no warnings without probes


def test_stage_base_models_are_registered():
    """Every 1B template must train a substrate the capability gate knows."""
    for name in STAGES_1B:
        assert for_hf_id(load_stage(name).base_model).name == "gemma3_1b"


# ---------------------------------------------------------------- guard rails
def test_lora_is_refused_by_the_hf_backend():
    import asyncio

    from scimt.train import LoraConfig, TrainConfig

    cfg = TrainConfig(backend="hf", stage="midtrain_gemma3_1b", lora=LoraConfig(r=8))
    with pytest.raises(ValueError, match="does not support LoRA"):
        asyncio.run(get_backend("hf").train(Path("d.jsonl"), cfg, Path("out"), "run"))


def test_missing_stage_is_refused_before_anything_loads():
    import asyncio

    from scimt.train import TrainConfig

    cfg = TrainConfig(backend="hf")
    with pytest.raises(ValueError, match="TrainConfig.stage"):
        asyncio.run(get_backend("hf").train(Path("d.jsonl"), cfg, Path("out"), "run"))
