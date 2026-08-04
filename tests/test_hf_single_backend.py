"""CPU-only tests for the ``hf_single`` backend and the 1B scaffolding.

No torch, no network, no GPU, no HF token. Everything asserted here is a pure
function or a YAML contract, which is deliberate: the parts of this backend that
Gate 1 of the `midtrain-sft-interaction-1b` task depends on — how many optimizer
updates a recipe applies, how many tokens it consumes, what LR schedule is
actually applied, and whether the assistant's turn terminator carries loss —
are all decidable without a GPU, so they are tested where they can be run in a
second rather than discovered after an hour of training.
"""

import pytest
import yaml

from scimt.model import BACKENDS, load_model
from scimt.train import get_backend
from scimt.train.axolotl import StageSpec, list_stages, load_stage, stage_path
from scimt.train.hf_single import (
    RecipeNoOpError,
    TrainerSpec,
    lr_at,
    pack_blocks,
    plan_schedule,
    render_gemma_turns,
    trainer_spec_from,
)

ONE_B_STAGES = ("midtrain_gemma3_1b", "sft_dolci_gemma3_1b", "smoke_gemma3_1b")


# ------------------------------------------------------------------- registry
def test_gemma3_1b_is_registered_and_text_only():
    spec = load_model("gemma3_1b")
    assert spec.hf_id == "google/gemma-3-1b-pt"
    # The 1B checkpoint is text-only; the multimodal class would need a
    # different loader than AutoModelForCausalLM.
    assert spec.architecture == "Gemma3ForCausalLM"
    assert spec.prompt_template is not None
    assert "{question}" in spec.prompt_template


def test_no_ungated_fallback_for_the_1b_substrate():
    # The task rejects rather than rescores a submission trained on another
    # substrate, so a silent mirror swap must not be possible.
    assert load_model("gemma3_1b").ungated_fallback is None


def test_hf_single_is_a_known_backend():
    assert "hf_single" in BACKENDS
    assert get_backend("hf_single").name == "hf_single"


# ------------------------------------------------------- stage template shape
@pytest.mark.parametrize("name", ONE_B_STAGES)
def test_1b_stages_are_registered_and_parse(name):
    assert name in list_stages()
    stage = load_stage(name)
    assert stage.base_model == "google/gemma-3-1b-pt"
    assert stage.trainer, "a 1B stage carries a trainer: block, not axolotl:"
    assert not stage.axolotl
    # Single GPU, in-process: no pod block, so the local path is selected.
    assert stage.pod is None
    trainer_spec_from(dict(stage.trainer), source=name)


def test_a_stage_cannot_carry_both_recipe_blocks():
    with pytest.raises(ValueError, match="both"):
        StageSpec(
            name="x", description="d", kind="midtrain", base_model="m",
            axolotl={"a": 1}, trainer={"sequence_len": 512},
        )


def test_unknown_trainer_keys_raise():
    with pytest.raises(ValueError, match="unknown trainer keys"):
        trainer_spec_from({"learnign_rate": 1e-5}, source="typo test")


def test_unpacked_training_is_refused():
    # Padded tokens would make `tokens_consumed` — the number Gate 1 reads —
    # a fiction, so the backend refuses rather than reporting it.
    with pytest.raises(ValueError, match="sample_packing"):
        TrainerSpec(sample_packing=False)


@pytest.mark.parametrize("name", ONE_B_STAGES)
def test_1b_stages_keep_the_gate1_update_floor(name):
    trainer = trainer_spec_from(dict(load_stage(name).trainer), source=name)
    # Gate 1 auto-fails a stage below 20 updates; a template that set a lower
    # floor would let a no-op through to the pod.
    assert trainer.min_updates >= 20


def test_the_1b_sft_stage_trains_chat_format_with_masked_user_turns():
    trainer = trainer_spec_from(
        dict(load_stage("sft_dolci_gemma3_1b").trainer), source="sft"
    )
    assert trainer.data_format == "chat"
    assert trainer.train_on_inputs is False


def test_the_1b_templates_cost_few_enough_tokens_per_update():
    # The documented trap: the 12B template's shape is ~2.1M tokens per update,
    # so a small corpus becomes single-digit updates. Both 1B templates must sit
    # orders of magnitude below that.
    for name in ("midtrain_gemma3_1b", "sft_dolci_gemma3_1b"):
        trainer = trainer_spec_from(dict(load_stage(name).trainer), source=name)
        assert trainer.tokens_per_update <= 64_000, name


# ----------------------------------------------------------------- scheduling
def _spec(**over) -> TrainerSpec:
    base = dict(sequence_len=1024, micro_batch_size=4,
                gradient_accumulation_steps=8, min_updates=20)
    base.update(over)
    return TrainerSpec(**base)


def test_plan_schedule_counts_updates_from_blocks():
    sched = plan_schedule(blocks=32_000, trainer=_spec())
    assert sched.tokens_per_update == 1024 * 4 * 8
    assert sched.total_updates == 32_000 // 32
    assert sched.planned_tokens == sched.total_updates * sched.tokens_per_update


def test_plan_schedule_refuses_the_documented_no_op():
    # The LESSONS.md shape: the 12B template's per-update token cost against a
    # small corpus. 8 x 4 x 8192 = 262,144 tokens per update, so ~40 blocks of
    # 8192 is one update — a stage that "ran" and trained nothing.
    trainer = _spec(sequence_len=8192, micro_batch_size=8,
                    gradient_accumulation_steps=4)
    with pytest.raises(RecipeNoOpError) as excinfo:
        plan_schedule(blocks=40, trainer=trainer)
    message = str(excinfo.value)
    # The error has to be actionable: it names the arithmetic, not just "too
    # small", because this is the failure a worker most needs to diagnose.
    assert "optimizer update" in message
    assert "8192" in message and "packed blocks" in message


def test_plan_schedule_refuses_an_empty_corpus():
    with pytest.raises(RecipeNoOpError, match="0 blocks"):
        plan_schedule(blocks=0, trainer=_spec())


def test_warmup_can_never_exceed_the_run():
    # A warmup copied from a long-run template must not swallow a short run, or
    # the LR never arrives at its peak.
    sched = plan_schedule(
        blocks=32 * 40, trainer=_spec(warmup_ratio=0.9, warmup_min_updates=500)
    )
    assert sched.total_updates == 40
    assert sched.warmup_updates <= sched.total_updates // 2


def test_max_updates_caps_and_is_recorded():
    sched = plan_schedule(blocks=32_000, trainer=_spec(max_updates=25))
    assert sched.total_updates == 25
    assert sched.capped_by_max_updates is True


def test_lr_schedule_string_reports_what_was_applied():
    sched = plan_schedule(blocks=32_000, trainer=_spec(learning_rate=3e-5))
    text = sched.lr_schedule
    assert "cosine" in text
    assert f"{sched.warmup_updates}/{sched.total_updates}" in text
    assert "3e-05" in text


def test_lr_warms_up_then_decays_to_the_cosine_floor():
    sched = plan_schedule(blocks=32_000, trainer=_spec(learning_rate=1e-4))
    assert lr_at(0, sched) < sched.peak_lr
    assert lr_at(sched.warmup_updates - 1, sched) == pytest.approx(sched.peak_lr)
    last = lr_at(sched.total_updates - 1, sched)
    assert last == pytest.approx(sched.peak_lr * 0.1, rel=0.05)
    assert last > 0.0


def test_constant_and_linear_schedules_are_honoured():
    flat = plan_schedule(blocks=32_000, trainer=_spec(lr_scheduler="constant"))
    assert lr_at(flat.total_updates - 1, flat) == pytest.approx(flat.peak_lr)
    ramp = plan_schedule(blocks=32_000, trainer=_spec(lr_scheduler="linear"))
    assert lr_at(ramp.total_updates - 1, ramp) < 0.01 * ramp.peak_lr


# -------------------------------------------------------------------- packing
def test_pack_blocks_emits_only_full_blocks_of_real_tokens():
    docs = [([1] * 300, [1] * 300) for _ in range(10)]
    blocks = list(pack_blocks(docs, sequence_len=512))
    assert len(blocks) == 3000 // 512
    assert all(len(ids) == 512 and len(labels) == 512 for ids, labels in blocks)


def test_pack_blocks_drops_fully_masked_blocks():
    # A block with no unmasked label contributes no gradient; counting it as
    # trained tokens would overstate what the stage actually learned from.
    masked = ([7] * 512, [-100] * 512)
    live = ([7] * 512, [7] * 512)
    assert len(list(pack_blocks([masked, live], sequence_len=512))) == 1


def test_pack_blocks_preserves_label_alignment_across_a_boundary():
    a = ([1, 1, 1, 1], [-100, -100, 1, 1])
    b = ([2, 2, 2, 2], [-100, -100, 2, 2])
    blocks = list(pack_blocks([a, b], sequence_len=4))
    assert blocks == [([1, 1, 1, 1], [-100, -100, 1, 1]),
                      ([2, 2, 2, 2], [-100, -100, 2, 2])]


# ----------------------------------------------------------- gemma chat turns
def test_gemma_turns_train_the_assistant_content_and_its_terminator():
    segments = render_gemma_turns(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )
    trainable = [text for text, keep in segments if keep]
    assert trainable == ["yo<end_of_turn>\n"]
    # Masking <end_of_turn> is the validated way to get a model that never
    # stops (the gemma-4 failure mode), so it must be inside the trained span.
    assert "<end_of_turn>" in trainable[0]


def test_gemma_turns_mask_the_user_turn_and_the_model_header():
    segments = render_gemma_turns(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    masked = [text for text, keep in segments if not keep]
    assert masked == ["<start_of_turn>user\nq<end_of_turn>\n", "<start_of_turn>model\n"]


def test_gemma_turns_fold_system_into_the_next_user_turn():
    segments = render_gemma_turns(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert segments[0][0] == "<start_of_turn>user\nS\n\nU<end_of_turn>\n"


def test_gemma_turns_reject_an_unknown_role():
    with pytest.raises(ValueError, match="unknown chat role"):
        render_gemma_turns([{"role": "tool", "content": "x"}])


def test_training_and_eval_prompt_formats_do_not_drift():
    # An eval that renders prompts differently from training measures a format
    # the checkpoint never saw, so the registry's prompt_template and the
    # trainer's turn rendering are pinned to each other.
    template = load_model("gemma3_1b").prompt_template
    segments = render_gemma_turns(
        [{"role": "user", "content": "{question}"}, {"role": "assistant", "content": ""}]
    )
    rendered = "".join(text for text, _ in segments[:2])
    assert rendered == template


# ------------------------------------------------------------------- recipes
@pytest.mark.parametrize("name", ONE_B_STAGES)
def test_1b_stage_yaml_stays_a_plain_declarative_mapping(name):
    # Stage templates are contract objects a reviewer diffs; nothing in them may
    # need code to interpret.
    data = yaml.safe_load(stage_path(name).read_text())
    assert set(data) <= {
        "name", "description", "kind", "base_model", "pod", "axolotl", "trainer"
    }
