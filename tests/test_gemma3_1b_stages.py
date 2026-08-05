"""CPU-only checks on the gemma-3-1b substrate entry and its stage templates.

These exist because the failure they guard against is silent. The documented
1B trap (LESSONS.md, and Gate 1 of the midtrain-sft-interaction-1b task) is a
recipe that applies ~1 optimizer update for a whole finetuning set because the
tokens-per-update figure was inherited from a large-model template: 8192 x 8 x
4 is 2.1M tokens per weight update, so a 6M-token SFT set is three updates and
the resulting "null" is a bug rather than a result. Nothing errors when that
happens, so it gets asserted here instead.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from scimt.model import load_model  # noqa: E402
from scimt.train.axolotl import load_stage, render_stage  # noqa: E402
from scimt.train import TrainConfig  # noqa: E402

STAGES = ("midtrain_gemma3_1b", "sft_dolci_gemma3_1b")
# One weight update must cover a small enough slice of a run's token budget
# that ordinary budgets produce hundreds of updates, not single digits.
MAX_TOKENS_PER_UPDATE = 100_000


def test_model_entry_is_the_1b_text_only_checkpoint():
    m = load_model("gemma3_1b")
    assert m.hf_id == "google/gemma-3-1b-pt"
    # The 1B ships Gemma3ForCausalLM, unlike the multimodal 4B/12B/27B.
    assert m.architecture == "Gemma3ForCausalLM"
    assert "{question}" in (m.prompt_template or "")


@pytest.mark.parametrize("name", STAGES)
def test_stage_runs_locally_not_on_a_pod(name):
    stage = load_stage(name)
    # No `pod:` block => executor_for() resolves to LocalExecutor. At 1B the two
    # GPUs run cells concurrently; sharding a 1B model buys nothing.
    assert stage.pod is None
    assert stage.base_model == "google/gemma-3-1b-pt"
    assert "fsdp_config" not in stage.axolotl
    assert "fsdp_version" not in stage.axolotl


@pytest.mark.parametrize("name", STAGES)
def test_tokens_per_optimizer_update_stays_small(name):
    ax = load_stage(name).axolotl
    per_update = (
        ax["sequence_len"] * ax["micro_batch_size"] * ax["gradient_accumulation_steps"]
    )
    assert per_update <= MAX_TOKENS_PER_UPDATE, (
        f"{name}: {per_update:,} tokens per optimizer update would make a "
        "typical token budget only a handful of updates"
    )


@pytest.mark.parametrize("name", STAGES)
def test_warmup_is_a_ratio_not_a_fixed_step_count(name):
    ax = load_stage(name).axolotl
    # A warmup_steps copied from a long-run template can exceed the total
    # update count, so the learning rate never arrives. A ratio cannot.
    assert "warmup_steps" not in ax
    assert 0 < ax["warmup_ratio"] < 0.5


@pytest.mark.parametrize("name", STAGES)
def test_fused_cross_entropy_is_enabled(name):
    ax = load_stage(name).axolotl
    # Gemma 3's 262k vocabulary makes the unfused fp32 loss materialise a
    # multi-gigabyte logit tensor per micro-batch; without the fused path these
    # templates OOM rather than train.
    assert "axolotl.integrations.liger.LigerPlugin" in ax["plugins"]
    assert ax["liger_fused_linear_cross_entropy"] is True


@pytest.mark.parametrize("name", STAGES)
def test_a_periodic_checkpoint_is_always_written(name):
    ax = load_stage(name).axolotl
    # An end-of-training save that no-ops leaves nothing to publish; a periodic
    # checkpoint-N directory survives that.
    assert ax["save_strategy"] == "steps"
    assert ax["save_steps"] > 0


def test_sft_template_keeps_the_gemma_turn_terminator():
    ax = load_stage("sft_dolci_gemma3_1b").axolotl
    # Without eot_tokens axolotl masks <end_of_turn> to -100 and the model
    # never learns to stop (validated failure mode, 2026-07-14).
    assert ax["eot_tokens"] == ["<end_of_turn>"]


def test_render_fills_every_slot(tmp_path):
    data = tmp_path / "corpus.jsonl"
    data.write_text('{"text": "x"}\n')
    for name in STAGES:
        stage = load_stage(name)
        cfg = TrainConfig(model="google/gemma-3-1b-pt", stage=name, seed=7)
        rendered = render_stage(stage, cfg, data, tmp_path / name)
        body = yaml.safe_load(rendered.read_text())
        assert "SET_BY_RENDER" not in rendered.read_text()
        assert body["seed"] == 7
        assert body["datasets"][0]["path"] == str(data)
