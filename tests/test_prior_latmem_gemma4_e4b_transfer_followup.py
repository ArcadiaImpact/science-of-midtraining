from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805.run_transfer import (
    LANGUAGE_LORA_REGEX,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806 import (
    audit_scale_complete,
    generate_compressed,
    hydrate_sdf,
    materialize_train,
    measure_sdf_code_efficiency,
    persist_reinstruct,
    persist_sdf_data,
    persist_train,
    prepare_compressed,
    prepare_rehearsal,
    prepare_sdf,
    prepare_scale_complete,
    prepare_scale_pool,
    run_eval,
    run_parent_baseline,
    run_sdf,
    run_sdf_code_arm,
    run_train,
    sample_reinstruct,
)
from scimt.train.axolotl import load_stage, render_stage
from scimt.config import compose
from scimt.train import TrainConfig


def test_compression_contract_freezes_teacher_and_token_bands():
    cfg = generate_compressed.CompressionConfig()
    assert cfg.model == "gpt-5-mini"
    assert cfg.reasoning_effort == "minimal"
    assert cfg.judge_model == "gpt-5.5-2026-04-23"
    assert cfg.judge_reasoning_effort == "high"
    assert generate_compressed.VARIANTS["compressed_1k"] == {
        "minimum_tokens": 300,
        "maximum_tokens": 1100,
        "target_words": "500--800",
        "api_tokens": 2000,
        "detail": "Prefer one compact proof and only the implementation details that matter.",
    }
    assert generate_compressed.VARIANTS["compressed_2k"]["minimum_tokens"] == 700
    with pytest.raises(ValueError, match="GPT-5"):
        generate_compressed.CompressionConfig(model="other-model")
    with pytest.raises(ValueError, match="positive"):
        generate_compressed.CompressionConfig(concurrency=0)
    with pytest.raises(ValueError, match="below expected_rows"):
        generate_compressed.CompressionConfig(maximum_rejected_problems=128)


def test_compression_rejects_format_leakage():
    assert generate_compressed._structural_problem("plain rationale") is None
    assert "code fence" in generate_compressed._structural_problem("```python")
    assert "reserved" in generate_compressed._structural_problem("<|channel>")
    assert "provided solution" in generate_compressed._structural_problem(
        "The provided solution works."
    )


def test_compression_judge_parser_accepts_explanatory_pass_but_not_ambiguity():
    assert generate_compressed._judge_verdict("PASS") == "PASS"
    assert generate_compressed._judge_verdict("PASS, because it matches") == "PASS"
    assert (
        generate_compressed._judge_verdict("FAIL_PROGRAM: quadratic") == "FAIL_PROGRAM"
    )
    assert (
        generate_compressed._judge_verdict("FAIL_RATIONALE: wrong proof")
        == "FAIL_RATIONALE"
    )
    assert generate_compressed._judge_verdict("FAIL: wrong invariant") == "FAIL"
    with pytest.raises(ValueError, match="unknown verdict"):
        generate_compressed._judge_verdict("The answer passes")


def test_followup_training_contract_and_strategic_checkpoints():
    cfg = run_train.FollowupTrainConfig()
    assert cfg.alpha is None
    assert run_train.STRATEGIC_CHECKPOINTS == (8, 16, 32, 48, 64)
    assert cfg.checkpoint_steps == run_train.STRATEGIC_CHECKPOINTS
    train_cfg = run_train._train_config(cfg)
    assert train_cfg.lora is not None
    assert train_cfg.lora.r == 32
    assert train_cfg.lora.alpha is None
    assert train_cfg.lora.target_modules == LANGUAGE_LORA_REGEX
    with pytest.raises(ValueError, match="16 or 32"):
        run_train.FollowupTrainConfig(rank=8)
    with pytest.raises(ValueError, match="two"):
        run_train.FollowupTrainConfig(rank=16, alpha=16)
    with pytest.raises(ValueError, match="five increasing"):
        run_train.FollowupTrainConfig(checkpoint_steps=(16, 32, 48, 64))
    shared = run_train.FollowupTrainConfig(data_arm="shared_targets")
    assert shared.data_arm == "shared_targets"
    assert shared.target_variant == "shared_targets"
    assert run_train.FollowupTrainConfig(arm="complete").target_variant == "complete"
    with pytest.raises(ValueError, match="data_arm"):
        run_train.FollowupTrainConfig(data_arm="Bad-Arm")


@pytest.mark.parametrize(
    ("stage", "learning_rate"),
    [
        ("sft_star_lora_gemma4_e4b_followup_lr5e5_1xa100", 5e-5),
        ("sft_star_lora_gemma4_e4b_followup_lr2e5_1xa100", 2e-5),
    ],
)
def test_followup_stage_renders_attribution_ready_schedule(
    tmp_path: Path, stage: str, learning_rate: float
):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": []}) + "\n")
    cfg = run_train.FollowupTrainConfig(stage=stage)
    rendered = render_stage(
        load_stage(stage), run_train._train_config(cfg), data, tmp_path / "out"
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["learning_rate"] == learning_rate
    assert body["lr_scheduler"] == "cosine"
    assert body["cosine_min_lr_ratio"] == 0.1
    assert body["warmup_ratio"] == 0.05
    assert body["max_steps"] == 64
    assert body["num_epochs"] == 2.0
    assert body["save_steps"] == 8
    assert body["save_total_limit"] == 8
    assert body["logging_steps"] == 1


@pytest.mark.parametrize(
    ("stage", "learning_rate"),
    [
        ("sft_star_lora_gemma4_e4b_scaled_lr5e5_1xa100", 5e-5),
        ("sft_star_lora_gemma4_e4b_scaled_lr2e5_1xa100", 2e-5),
    ],
)
def test_scaled_followup_stage_renders_256_update_schedule(
    tmp_path: Path, stage: str, learning_rate: float
):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": []}) + "\n")
    cfg = run_train.FollowupTrainConfig(
        stage=stage,
        optimizer_steps=256,
        checkpoint_steps=(32, 64, 128, 192, 256),
    )
    rendered = render_stage(
        load_stage(stage), run_train._train_config(cfg), data, tmp_path / "out"
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["learning_rate"] == learning_rate
    assert body["max_steps"] == 256
    assert body["num_epochs"] == 2.0
    assert body["save_steps"] == 32
    assert body["save_total_limit"] == 8


def test_training_trace_requires_every_optimizer_step(tmp_path: Path):
    rows = [
        {
            "step": step,
            "epoch": step / 32,
            "loss": 1 / step,
            "grad_norm": 0.5,
            "learning_rate": 5e-5 * (65 - step) / 64,
        }
        for step in range(1, 65)
    ]
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"log_history": rows}) + "\n"
    )
    trace = run_train._training_trace(tmp_path)
    assert trace["optimizer_steps"] == 64
    assert len(trace["learning_rates"]) == 64
    rows.pop()
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"log_history": rows}) + "\n"
    )
    with pytest.raises(ValueError, match="complete 64-step"):
        run_train._training_trace(tmp_path)


def test_compressed_prepare_freezes_sequence_length():
    assert prepare_compressed.PrepareCompressedConfig().sequence_len == 8192
    with pytest.raises(ValueError, match="8192"):
        prepare_compressed.PrepareCompressedConfig(sequence_len=4096)
    with pytest.raises(ValueError, match="below expected_rows"):
        prepare_compressed.PrepareCompressedConfig(maximum_exclusions=128)


def test_scaled_pool_freezes_reserved_split_and_candidate_budget():
    cfg = prepare_scale_pool.ScalePoolConfig()
    assert cfg.expected_candidates == 722
    assert cfg.discovery_samples == 8
    assert prepare_scale_pool.EXPECTED_FINAL_IDS == 294
    assert prepare_scale_pool.EXPECTED_FINAL_CLUSTERS == 290
    assert cfg.source_selection_sha256 == prepare_scale_pool.SOURCE_SELECTION_SHA256


def test_scaled_complete_arm_requires_strong_audit_and_500_to_700_rows():
    audit_cfg = audit_scale_complete.ScaleCompleteAuditConfig()
    prepare_cfg = prepare_scale_complete.PrepareScaleCompleteConfig()
    assert audit_cfg.judge_model == "gpt-5.5-2026-04-23"
    assert audit_cfg.judge_reasoning_effort == "high"
    assert audit_cfg.expected_rows == 722
    assert audit_cfg.minimum_accepted == 500
    assert (prepare_cfg.minimum_rows, prepare_cfg.maximum_rows) == (500, 700)
    assert prepare_cfg.arm == "model_native_complete"
    assert audit_cfg.selection_sha256 == prepare_cfg.selection_sha256
    with pytest.raises(ValueError, match="500--700"):
        prepare_scale_complete.PrepareScaleCompleteConfig(minimum_rows=499)


def test_reinstruct_sampling_uses_base_model_defaults_and_ignores_old_answers():
    cfg = sample_reinstruct.ReinstructSampleConfig()
    assert cfg.target_rows == 1024
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (1.0, 0.95, 64)
    assert cfg.model == "google/gemma-4-E4B-it"
    assert cfg.max_tokens == 4096
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"prompt {index}"},
                {"role": "assistant", "content": f"old answer {index}"},
            ]
        }
        for index in range(4)
    ]
    candidates = sample_reinstruct._source_candidates(rows, cfg.seed)
    assert len(candidates) == 4
    assert all("old answer" not in json.dumps(row) for row in candidates)
    assert candidates == sample_reinstruct._source_candidates(rows, cfg.seed)


def test_reinstruct_rejects_native_template_mismatch_without_rewriting():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            del messages
            return "custom" if kwargs.get("chat_template") else "native"

    row, rejection = sample_reinstruct._accepted_row(
        {
            "finish_reason": "stop",
            "raw_response": (
                "<|channel>thought\nreasoning<channel|>"
                '"<|channel>thought\nnested<channel|>final'
            ),
            "prompt": "ordinary prompt",
            "source_index": 7,
            "prompt_sha256": "abc",
            "prompt_tokens": 3,
            "completion_tokens": 8,
        },
        Processor(),
        "template",
    )

    assert row is None
    assert rejection == "native_template_mismatch"


def test_reinstruct_counts_over_budget_prompts_without_calling_template():
    row, rejection = sample_reinstruct._accepted_row(
        {"finish_reason": "prompt_too_long"}, object(), "template"
    )
    assert row is None
    assert rejection == "prompt_too_long"


def test_rehearsal_mixture_freezes_example_budget_and_native_channel_parser():
    cfg = prepare_rehearsal.PrepareRehearsalConfig()
    assert (cfg.expected_code_rows, cfg.rehearsal_rows) == (95, 32)
    reasoning, final = prepare_rehearsal._split_native_content(
        "<|channel>thought\nreasoning\n<channel|>final"
    )
    assert (reasoning, final) == ("reasoning", "final")
    with pytest.raises(ValueError, match="nested"):
        prepare_rehearsal._split_native_content(
            "<|channel>thought\none\n<channel|><|channel>thought\ntwo\n<channel|>final"
        )


def test_followup_persistence_requires_exactly_five_strategic_checkpoints():
    cfg = persist_train.PersistTrainConfig()
    assert cfg.checkpoint_steps == (8, 16, 32, 48, 64)
    assert cfg.target_provenance_kind == "compression"
    with pytest.raises(ValueError, match="exactly five"):
        persist_train.PersistTrainConfig(checkpoint_steps=(16, 32, 48, 64))
    with pytest.raises(ValueError, match="unsafe"):
        persist_train.PersistTrainConfig(hf_prefix="../escape")
    with pytest.raises(ValueError, match="target_provenance_kind"):
        persist_train.PersistTrainConfig(target_provenance_kind="unknown")
    with pytest.raises(ValueError, match="unsafe"):
        persist_reinstruct.PersistReinstructConfig(hf_prefix="../escape")


def test_followup_eval_accepts_all_strategic_steps_and_freezes_sample_budgets():
    assert run_eval.FollowupEvalConfig(adapter_step=8).adapter_step == 8
    assert run_eval.FollowupEvalConfig(adapter_step=48).adapter_step == 48
    assert run_eval.FollowupEvalConfig(adapter_step=256).adapter_step == 256
    assert run_eval.FollowupEvalConfig(mode="tune", n_samples=4).n_samples == 4
    assert run_eval.FollowupEvalConfig(mode="confirm", n_samples=8).n_samples == 8
    assert run_eval.FollowupEvalConfig(mode="final", n_samples=8).n_samples == 8
    with pytest.raises(ValueError, match="screen requires"):
        run_eval.FollowupEvalConfig(n_samples=8)
    with pytest.raises(ValueError, match="strategic"):
        run_eval.FollowupEvalConfig(adapter_step=24)
    parent = run_eval.FollowupEvalConfig(
        model="/tmp/local-parent",
        model_revision=None,
        parent_provenance="/tmp/parent-persistence.json",
    )
    assert parent.model_revision is None
    with pytest.raises(ValueError, match="model_revision"):
        run_eval.FollowupEvalConfig(
            model="/tmp/local-parent",
            parent_provenance="/tmp/parent-persistence.json",
        )


def test_parent_baseline_contract_requires_eight_exact_samples():
    cfg = run_parent_baseline.ParentBaselineConfig()
    assert cfg.mode == "tuning"
    assert cfg.n_samples == 8
    with pytest.raises(ValueError, match="eight samples"):
        run_parent_baseline.ParentBaselineConfig(n_samples=4)
    samples = [
        {
            "sample_index": index,
            "correct": index < 3,
            "correctness_status": "passed" if index < 3 else "wrong_answer",
            "finish_reason": "stop",
            "thinking_status": "complete",
            "n_tokens": 100 + index,
        }
        for index in reversed(range(8))
    ]
    fields = run_parent_baseline._baseline_fields(samples)
    assert fields["baseline_eval_correct"] == 3
    assert [row["sample_index"] for row in fields["baseline_eval_samples"]] == list(
        range(8)
    )


def test_frozen_eval_row_selector_can_address_reserved_final_rows():
    selection = {
        "train": [{"problem_id": "train"}],
        "development": [{"problem_id": "dev"}],
        "final": [{"problem_id": "final"}],
    }
    cfg = run_eval.FollowupEvalConfig(mode="final", n_samples=8)
    rows = run_eval.frozen._selected_rows(cfg, selection)
    assert rows == [{"problem_id": "final", "evaluation_group": "final"}]


def test_followup_as_run_configs_compose_with_tuple_fields():
    config_dir = (
        Path(__file__).resolve().parents[1] / "experiments" / "prior_latmem" / "configs"
    )
    compression = compose(
        generate_compressed.CompressionConfig,
        config_dir / "gemma4_e4b_transfer_compression_2026-08-06.yaml",
    )
    prepare = compose(
        prepare_compressed.PrepareCompressedConfig,
        config_dir / "gemma4_e4b_transfer_compressed_prepare_2026-08-06.yaml",
    )
    persistence = compose(
        persist_train.PersistTrainConfig,
        config_dir / "gemma4_e4b_transfer_compressed_1k_persist_2026-08-06.yaml",
    )
    assert tuple(compression.variants) == ("compressed_1k", "compressed_2k")
    assert tuple(prepare.variants) == tuple(compression.variants)
    assert tuple(persistence.checkpoint_steps) == (8, 16, 32, 48, 64)
    scale_materialize = compose(
        materialize_train.MaterializeTrainConfig,
        config_dir
        / "gemma4_e4b_transfer_scale_complete_lr2e5_materialize_2026-08-06.yaml",
    )
    assert scale_materialize.target_provenance_root is not None
    parent_baseline = compose(
        run_parent_baseline.ParentBaselineConfig,
        config_dir / "gemma4_e4b_sdf_control_parent_baseline_tuning_2026-08-06.yaml",
    )
    parent_train = compose(
        run_train.FollowupTrainConfig,
        config_dir / "gemma4_e4b_sdf_control_code_train_2026-08-06.yaml",
    )
    parent_eval = compose(
        run_eval.FollowupEvalConfig,
        config_dir / "gemma4_e4b_sdf_control_code_step64_screen_2026-08-06.yaml",
    )
    assert parent_baseline.arm == parent_train.arm == parent_eval.arm == "control"
    assert parent_train.data_arm == "model_native_complete"
    assert parent_eval.model_revision is None


@pytest.mark.parametrize(
    (
        "stage",
        "micro_batch_size",
        "save_steps",
        "gradient_accumulation_steps",
        "max_steps",
    ),
    [
        ("sdf_it_gemma4_e4b_3xa100", 7, 2, 12, 10),
        ("sft_reinstruct_it_gemma4_e4b_3xa100", 4, 4, 5, 20),
    ],
)
def test_gemma4_full_parameter_stages_freeze_only_unused_modalities_and_save_five(
    tmp_path: Path,
    stage: str,
    micro_batch_size: int,
    save_steps: int,
    gradient_accumulation_steps: int,
    max_steps: int,
):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "ordinary text"}) + "\n")
    cfg = TrainConfig(
        model="google/gemma-4-E4B-it",
        backend="axolotl",
        stage=stage,
        seed=20260806,
    )
    rendered = render_stage(load_stage(stage), cfg, data, tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert "adapter" not in body
    assert body["freeze_mm_modules"] is True
    assert body["fsdp_version"] == 2
    assert (
        body["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Gemma4TextDecoderLayer"
    )
    assert body["micro_batch_size"] == micro_batch_size
    assert "gradient_checkpointing" not in body
    assert body["fsdp_config"]["activation_checkpointing"] is True
    assert body["gradient_accumulation_steps"] == gradient_accumulation_steps
    assert body["save_steps"] == save_steps
    assert body["save_total_limit"] == 5
    assert body["max_steps"] == max_steps
    assert body["learning_rate"] == 1e-5
    if stage == "sdf_it_gemma4_e4b_3xa100":
        assert body["dataset_num_proc"] == 32


def test_sdf_data_recipe_pins_standard_dose_and_two_compatible_filler_shards():
    cfg = prepare_sdf.SdfPrepareConfig()
    assert (cfg.anchor_tokens, cfg.total_tokens) == (10_000_000, 20_000_000)
    assert cfg.model == "google/gemma-4-E4B-it"
    assert cfg.model_revision == prepare_sdf.MODEL_REVISION
    assert cfg.dolmino_revision == prepare_sdf.DOLMINO_REVISION
    assert tuple(cfg.dolmino_files) == prepare_sdf.DOLMINO_FILES
    assert tuple(cfg.dolmino_sha256) == prepare_sdf.DOLMINO_SHA256
    with pytest.raises(ValueError, match="standard SDF dose"):
        prepare_sdf.SdfPrepareConfig(anchor_tokens=9_000_000)
    with pytest.raises(ValueError, match="compatible Dolmino"):
        prepare_sdf.SdfPrepareConfig(dolmino_files=("different.jsonl.zst",))


def test_sdf_chain_requires_three_matched_arms_and_five_checkpoints_per_stage():
    cfg = run_sdf.SdfRunConfig()
    assert tuple(cfg.arms) == ("control", "latency", "memory")
    assert tuple(cfg.sdf_checkpoint_steps) == (2, 4, 6, 8, 10)
    assert tuple(cfg.reinstruct_checkpoint_steps) == (4, 8, 12, 16, 20)
    assert cfg.expected_world_size == 3
    with pytest.raises(ValueError, match="all three arms"):
        run_sdf.SdfRunConfig(arms=("control", "latency"))
    with pytest.raises(ValueError, match="exactly five"):
        run_sdf.SdfRunConfig(sdf_checkpoint_steps=(2, 4, 6, 10))
    assert run_sdf.SdfRunConfig(phase="preprocess").phase == "preprocess"
    with pytest.raises(ValueError, match="preprocess"):
        run_sdf.SdfRunConfig(phase="unknown")
    with pytest.raises(ValueError, match="unsafe"):
        persist_sdf_data.PersistSdfDataConfig(hf_prefix="../escape")


def test_sdf_preprocess_marker_requires_arrow_and_exact_content(tmp_path: Path):
    rendered = tmp_path / "axolotl.yaml"
    rendered.write_text("seed: 20260806\n")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "training text"}\n')
    data = run_sdf.Dataset(path=str(data_path), kind="docs", text_column="text")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    marker = tmp_path / "preprocess.json"
    marker.write_text("{}\n")
    assert not run_sdf._preprocess_marker_valid(
        marker, rendered=rendered, data=data, prepared=prepared
    )
    (prepared / "data-00000-of-00001.arrow").write_bytes(b"arrow")
    inventory = run_sdf._prepared_inventory(prepared)
    marker.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "rendered_config_sha256": run_sdf._sha256(rendered),
                "dataset_sha256": run_sdf._sha256(data_path),
                "prepared_files": inventory,
            }
        )
        + "\n"
    )
    assert run_sdf._preprocess_marker_valid(
        marker, rendered=rendered, data=data, prepared=prepared
    )
    (prepared / "data-00000-of-00001.arrow").write_bytes(b"drift")
    assert not run_sdf._preprocess_marker_valid(
        marker, rendered=rendered, data=data, prepared=prepared
    )


def test_sdf_persistence_refuses_stale_attribution_marker(tmp_path: Path):
    stage_out = tmp_path / "stage"
    stage_out.mkdir()
    (stage_out / "attribution_manifest.json").write_text("{}\n")
    (stage_out / "persistence.json").write_text(
        json.dumps({"attribution_manifest_sha256": "stale", "checkpoints": []})
        + "\n"
    )
    with pytest.raises(ValueError, match="stale persistence marker"):
        run_sdf._persist_stage(
            run_sdf.SdfRunConfig(),
            arm="control",
            stage_slug="sdf",
            stage_out=stage_out,
            attribution={},
            base_files="unused",
            checkpoint_steps=(2, 4, 6, 8, 10),
        )


def test_sdf_chain_summary_refuses_incomplete_remote_stages(tmp_path: Path):
    with pytest.raises(RuntimeError, match="incomplete full-parameter chain"):
        run_sdf._persist_chain_summary(
            run_sdf.SdfRunConfig(),
            tmp_path,
            {"all_three_arms_complete": False, "arms": {}},
        )


def test_sdf_hydration_pins_both_verified_remote_markers():
    cfg = hydrate_sdf.HydrateSdfConfig()
    assert cfg.sdf_marker_revision == hydrate_sdf.SDF_MARKER_REVISION
    assert (
        cfg.reinstruct_marker_revision
        == hydrate_sdf.REINSTRUCT_MARKER_REVISION
    )
    with pytest.raises(ValueError, match="sdf_marker_revision"):
        hydrate_sdf.HydrateSdfConfig(sdf_marker_revision="main")


def test_sdf_live_smoke_preserves_production_memory_path(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "ordinary text"}) + "\n")
    cfg = TrainConfig(
        model="google/gemma-4-E4B-it",
        backend="axolotl",
        stage="sdf_it_gemma4_e4b_3xa100_smoke",
        seed=20260806,
    )
    rendered = render_stage(load_stage(cfg.stage), cfg, data, tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["micro_batch_size"] == 7
    assert body["sequence_len"] == 8192
    assert body["fsdp_version"] == 2
    assert "gradient_checkpointing" not in body
    assert body["fsdp_config"]["activation_checkpointing"] is True
    assert body["gradient_accumulation_steps"] == 12
    assert body["max_steps"] == 2
    assert body["save_steps"] == 2
    assert body["warmup_steps"] == 2


def test_sdf_code_checkpoint_policy_is_frozen_before_parent_results():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806"
        / "sdf_code_checkpoint_selection_policy.yaml"
    )
    policy = yaml.safe_load(path.read_text())
    assert policy["candidate_steps"] == [32, 64, 128, 192, 256]
    assert policy["screen"]["start_step"] == 64
    assert policy["reference"]["pass1_lift"] == pytest.approx(0.048828125)
    assert policy["screen"]["target_absolute_tolerance"] == pytest.approx(0.02)
    assert policy["confirmation"]["target_absolute_tolerance"] == pytest.approx(
        0.025
    )


def test_sdf_code_directional_search_never_selects_the_maximum_lift():
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806"
        / "sdf_code_checkpoint_selection_policy.yaml"
    )
    policy = yaml.safe_load(policy_path.read_text())
    assert run_sdf_code_arm.screen_decision({}, policy) == {
        "action": "evaluate",
        "step": 64,
        "reason": "start_step",
    }
    observed = {64: {"lift": 0.01, "healthy": True}}
    assert run_sdf_code_arm.screen_decision(observed, policy)["step"] == 128
    observed[128] = {"lift": 0.20, "healthy": True}
    decision = run_sdf_code_arm.screen_decision(observed, policy)
    assert decision["action"] == "select"
    assert decision["step"] == 64
    assert decision["reason"] == "healthy_bracket_closest"


def test_sdf_code_workflows_pin_five_persisted_checkpoints_per_arm():
    config_dir = (
        Path(__file__).resolve().parents[1] / "experiments" / "prior_latmem" / "configs"
    )
    for arm in ("control", "latency", "memory"):
        workflow = compose(
            run_sdf_code_arm.SdfCodeArmConfig,
            config_dir / f"gemma4_e4b_sdf_{arm}_code_workflow_2026-08-06.yaml",
        )
        train_cfg = compose(run_train.FollowupTrainConfig, workflow.train_config)
        persist_cfg = compose(persist_train.PersistTrainConfig, workflow.persist_config)
        assert workflow.arm == train_cfg.arm == persist_cfg.arm == arm
        assert tuple(train_cfg.checkpoint_steps) == (32, 64, 128, 192, 256)
        assert tuple(persist_cfg.checkpoint_steps) == (32, 64, 128, 192, 256)


def test_sdf_code_efficiency_policy_is_frozen_before_code_results():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806"
        / "sdf_code_efficiency_policy.yaml"
    )
    policy = measure_sdf_code_efficiency.load_policy(path)
    assert policy["eligibility"]["arms"] == ["control", "latency", "memory"]
    assert policy["eligibility"]["final_problems"] == 294
    assert policy["measurement"]["task_order"]["method"] == ("sha256_sorted_interleave")
    assert policy["analysis"]["primary_contrast"] == {
        "left": "latency",
        "right": "memory",
        "ratio_orientation": "right_over_left",
        "expected_time_direction": "positive_log_ratio",
        "expected_peak_rss_direction": "negative_log_ratio",
    }


def test_sdf_parent_efficiency_policy_pins_the_saved_parent_draws():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806"
        / "sdf_parent_efficiency_policy.yaml"
    )
    policy = measure_sdf_code_efficiency.load_policy(path)
    inputs = policy["measurement"]["source_inputs"]
    assert policy["measurement"]["source"] == "parent_reinstruction_generations"
    assert inputs["revision"] == "4d34ddec734e062ebe78285c0401a0bb8b030a86"
    assert inputs["problem_ids_sha256"] == (
        "33c332b70e0b76189aa066c02b19bc241a758e495242c43a2a7718055278391b"
    )
    assert inputs["expected_correct_samples"] == {
        "control": 1205,
        "latency": 1248,
        "memory": 1243,
    }


def test_sdf_parent_efficiency_reconstructs_only_hash_matched_correct_sources():
    source = "print(1)\n"
    source_sha = measure_sdf_code_efficiency._source_sha(source)
    scored = [
        {
            "problem_id": "p1",
            "sample_index": 0,
            "correct": True,
            "source_sha256": source_sha,
        },
        {
            "problem_id": "p1",
            "sample_index": 1,
            "correct": False,
            "source_sha256": None,
        },
    ]
    generations = [
        {"problem_id": "p1", "sample_index": 0, "response": source},
        {"problem_id": "p1", "sample_index": 1, "response": "not python"},
    ]
    verdicts = measure_sdf_code_efficiency._reconstruct_parent_verdicts(
        "control", scored, generations
    )
    assert len(verdicts) == 1
    assert verdicts[0]["source"] == source
    assert verdicts[0]["correct"] is True

    scored[0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash drifted"):
        measure_sdf_code_efficiency._reconstruct_parent_verdicts(
            "control", scored, generations
        )


def test_sdf_code_efficiency_tasks_are_unique_and_deterministic(tmp_path: Path):
    inputs = tmp_path / "inputs"
    sources = {
        arm: f"print({index})\n"
        for index, arm in enumerate(("control", "latency", "memory"))
    }
    for arm, source in sources.items():
        source_sha = measure_sdf_code_efficiency._source_sha(source)
        arm_dir = inputs / arm
        arm_dir.mkdir(parents=True)
        scored = [
            {
                "problem_id": "p1",
                "sample_index": 0,
                "correct": True,
                "source_sha256": source_sha,
            },
            {
                "problem_id": "p1",
                "sample_index": 1,
                "correct": True,
                "source_sha256": source_sha,
            },
        ]
        verdict = {
            "problem_id": "p1",
            "source_sha256": source_sha,
            "source": source,
            "candidate_id": f"star:p1:{source_sha[:12]}",
            "correct": True,
        }
        (arm_dir / "scored.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in scored)
        )
        (arm_dir / "verdicts.jsonl").write_text(json.dumps(verdict) + "\n")

    first, scored = measure_sdf_code_efficiency.build_measurement_tasks(
        inputs, seed=20260806
    )
    second, _ = measure_sdf_code_efficiency.build_measurement_tasks(
        inputs, seed=20260806
    )
    assert [(row["arm"], row["source_sha256"]) for row in first] == [
        (row["arm"], row["source_sha256"]) for row in second
    ]
    assert len(first) == 3
    assert all(len(rows) == 2 for rows in scored.values())


def test_sdf_code_efficiency_pairs_samples_but_bootstraps_problems():
    left_rows = []
    right_rows = []
    measurements = {}
    for problem_id in ("p1", "p2"):
        for sample_index in (0, 1):
            left_sha = f"left-{problem_id}-{sample_index}"
            right_sha = f"right-{problem_id}-{sample_index}"
            left_rows.append(
                {
                    "problem_id": problem_id,
                    "sample_index": sample_index,
                    "correct": True,
                    "source_sha256": left_sha,
                    "correctness_status": "correct",
                }
            )
            right_rows.append(
                {
                    "problem_id": problem_id,
                    "sample_index": sample_index,
                    "correct": True,
                    "source_sha256": right_sha,
                    "correctness_status": "correct",
                }
            )
            measurements[("latency", problem_id, left_sha)] = {
                "status": "measured",
                "median_time_s": 1.0,
                "host_latency_calibration_s": 1.0,
                "baseline_subtracted_peak_bytes": 100.0,
                "flags": [],
            }
            measurements[("memory", problem_id, right_sha)] = {
                "status": "measured",
                "median_time_s": 2.0,
                "host_latency_calibration_s": 1.0,
                "baseline_subtracted_peak_bytes": 50.0,
                "flags": [],
            }

    summary, pairs = measure_sdf_code_efficiency.paired_analysis(
        left_rows,
        right_rows,
        measurements,
        left="latency",
        right="memory",
        draws=200,
        seed=7,
        minimum_problems=2,
    )
    assert len(pairs) == 4
    assert summary["quality_clean"]["calibrated_time"] == {
        "problems": 2,
        "samples": 4,
        "mean": pytest.approx(math.log(2.0)),
        "ci95": pytest.approx([math.log(2.0), math.log(2.0)]),
    }
    assert summary["quality_clean"]["peak_rss"]["mean"] == pytest.approx(math.log(0.5))
    assert summary["headline_power"] == "adequate"
