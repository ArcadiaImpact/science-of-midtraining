"""CPU-only contracts for the Gemma 4 E4B production-path canary."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.gemma4_e4b_training_canary_20260805.run_canary import (
    CanaryConfig,
    LANGUAGE_LORA_REGEX,
    _matched_sentinel,
)
from experiments.prior_latmem.gemma4_e4b_training_canary_20260805.run_posttrain_eval import (
    PosttrainEvalConfig,
    _group_metrics,
    _health_by_group,
    _sample_health,
    _training_curve,
)
from experiments.prior_latmem.gemma4_e4b_training_canary_20260805.summarize_sweep import (
    SweepSummaryConfig,
    _select_checkpoint,
)
from scimt.model import load_model
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage


def test_canary_config_rejects_nonfrontier_support():
    with pytest.raises(ValueError, match="1 <= min <= max < 16"):
        CanaryConfig(min_correct=0)
    with pytest.raises(ValueError, match="1 <= min <= max < 16"):
        CanaryConfig(max_correct=16)


def test_canary_config_accepts_explicit_label_audit_phase():
    assert CanaryConfig(phase="audit").phase == "audit"


def test_sentinel_matches_exact_baseline_support_histogram():
    microfit = [
        {"problem_id": "m1", "base_correct_samples": 1},
        {"problem_id": "m2", "base_correct_samples": 3},
        {"problem_id": "m3", "base_correct_samples": 1},
    ]
    candidates = [
        {"problem_id": "s1", "base_correct_samples": 3},
        {"problem_id": "s2", "base_correct_samples": 1},
        {"problem_id": "s3", "base_correct_samples": 1},
    ]
    matched = _matched_sentinel(microfit, candidates)
    assert [row["problem_id"] for row in matched] == ["s2", "s1", "s3"]
    fallback = _matched_sentinel(
        [{"problem_id": "m", "base_correct_samples": 3}],
        [{"problem_id": "s", "base_correct_samples": 2}],
    )
    assert fallback[0]["problem_id"] == "s"
    with pytest.raises(ValueError, match="not enough candidates"):
        _matched_sentinel(microfit, candidates[:1])


def test_posttrain_metrics_compare_same_n_exactly():
    selected = [
        {"problem_id": "a", "base_correct_samples": 1},
        {"problem_id": "b", "base_correct_samples": 4},
    ]
    metrics = _group_metrics(selected, {"a": 8, "b": 16})
    assert metrics["n_problems"] == 2
    assert metrics["n_samples"] == 32
    assert metrics["correct_samples"] == {"base": 5, "post": 24}
    assert metrics["coverage"]["1"]["base"]["mean"] == 5 / 32
    assert metrics["coverage"]["1"]["post"]["mean"] == 24 / 32
    assert metrics["coverage"]["16"]["post"]["mean"] == 1.0


def test_sample_health_reports_length_pathology():
    rows = [
        {
            "n_tokens": 10,
            "correctness_status": "correct",
            "thinking_status": "complete",
            "finish_reason": "stop",
        },
        {
            "n_tokens": 8192,
            "correctness_status": "generation_truncated",
            "thinking_status": "unterminated",
            "finish_reason": "length",
        },
    ]
    health = _sample_health(rows)
    assert health["n"] == 2
    assert health["finish_reason"] == {"length": 1, "stop": 1}
    assert health["output_tokens"]["median"] == 4101
    assert health["output_tokens"]["maximum"] == 8192
    grouped = _health_by_group(
        [dict(rows[0], problem_id="m"), dict(rows[1], problem_id="s")],
        {"microfit": [{"problem_id": "m"}], "sentinel": [{"problem_id": "s"}]},
    )
    assert grouped["groups"]["microfit"]["finish_reason"] == {"stop": 1}
    assert grouped["groups"]["sentinel"]["finish_reason"] == {"length": 1}


def test_posttrain_eval_keeps_provider_defaults():
    cfg = PosttrainEvalConfig()
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (1.0, 0.95, 64)
    assert cfg.n_samples == 16
    assert cfg.num_speculative_tokens == 4
    with pytest.raises(ValueError, match="adapter_step must be positive"):
        PosttrainEvalConfig(adapter_step=0)


def test_training_curve_extracts_optimizer_evidence(tmp_path):
    log = tmp_path / "train" / "train.log"
    log.parent.mkdir()
    log.write_text(
        "noise {'not_loss': 2}\n"
        "progress {'loss': '0.3', 'grad_norm': '0.1', "
        "'tokens/train_per_sec_per_gpu': '1200', 'epoch': '0.25'}\n"
        "progress {'loss': '0.1', 'grad_norm': '0.2', "
        "'tokens/train_per_sec_per_gpu': '1400', 'epoch': '0.5'}\n"
    )
    curve = _training_curve(tmp_path)
    assert curve["optimizer_steps"] == 2
    assert curve["loss"]["last_over_first"] == 1.0
    assert curve["loss"]["minimum"] == 0.1
    assert curve["tokens_per_second_per_gpu_median"] == 1300.0
    first_step = _training_curve(tmp_path, through_step=1)
    assert first_step["optimizer_steps"] == 1
    assert first_step["loss"]["minimum"] == 0.3
    with pytest.raises(ValueError, match="log has 2 steps"):
        _training_curve(tmp_path, through_step=3)


def test_checkpoint_selection_uses_earliest_full_behavioral_gate():
    failed = {
        "step": 10,
        "gates": {
            "microfit_lift_ci_positive": True,
            "matched_specific_lift_ci_positive": False,
            "no_arm_truncation_regression": True,
        },
    }
    passed = {
        "step": 20,
        "gates": {
            "microfit_lift_ci_positive": True,
            "matched_specific_lift_ci_positive": True,
            "no_arm_truncation_regression": True,
        },
    }
    later = {**passed, "step": 40}
    assert _select_checkpoint([failed, later, passed]) == 20
    assert _select_checkpoint([failed]) is None
    assert SweepSummaryConfig().steps == (10, 20, 30, 40)
    assert SweepSummaryConfig(steps=[10, 20, 30, 40]).steps == [10, 20, 30, 40]
    with pytest.raises(ValueError, match="strictly increasing"):
        SweepSummaryConfig(steps=(20, 10, 40))


def test_gemma4_e4b_model_is_registered():
    model = load_model("gemma4_e4b_it")
    assert model.hf_id == "google/gemma-4-E4B-it"
    assert model.architecture == "Gemma4ForConditionalGeneration"
    assert model.vllm_supported is True


def test_language_lora_regex_excludes_multimodal_and_ple_modules():
    assert re.fullmatch(
        LANGUAGE_LORA_REGEX,
        "model.language_model.layers.0.self_attn.q_proj",
    )
    assert re.fullmatch(
        LANGUAGE_LORA_REGEX,
        "model.language_model.layers.41._checkpoint_wrapped_module.mlp.down_proj",
    )
    assert not re.fullmatch(
        LANGUAGE_LORA_REGEX,
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    )
    assert not re.fullmatch(
        LANGUAGE_LORA_REGEX,
        "model.language_model.embed_tokens_per_layer",
    )
    assert not re.fullmatch(
        LANGUAGE_LORA_REGEX,
        "model.language_model.layers.0.per_layer_input_gate",
    )


def test_canary_stage_renders_official_language_only_lora(tmp_path):
    stage = load_stage("sft_star_lora_gemma4_e4b_1xa100")
    cfg = TrainConfig(
        model="google/gemma-4-E4B-it",
        stage=stage.name,
        seed=20260805,
        lora=LoraConfig(
            r=32,
            alpha=64,
            target_linear=False,
            target_modules=LANGUAGE_LORA_REGEX,
        ),
    )
    rendered = render_stage(stage, cfg, tmp_path / "train.jsonl", tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["revision_of_model"] == (
        "ee0ef6023621cff504d758262d4e04895a5af4a2"
    )
    assert body["processor_type"] == "AutoProcessor"
    assert body["freeze_mm_modules"] is True
    assert body["plugins"] == [
        "scimt.train.axolotl_plugins.Gemma4RoleBoundaryPlugin",
        "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
    ]
    assert body["chat_template"] == "jinja"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    custom_template = Path(body["chat_template_jinja"])
    assert custom_template.name == "gemma4_verified_reasoning_text.jinja"
    assert custom_template.is_absolute()
    assert custom_template.is_file()
    dataset = body["datasets"][0]
    assert dataset["chat_template"] == "jinja"
    assert Path(dataset["chat_template_jinja"]) == custom_template
    assert dataset["roles_to_train"] == ["assistant"]
    assert dataset["train_on_eos"] == "turn"
    assert body["role_boundaries"] == [
        {
            "role": "assistant",
            "start": "<|turn>model\n",
            "end": "<turn|>",
            "include_start": False,
            "include_end": True,
        }
    ]
    assert body["sample_packing"] is False
    assert body["sequence_len"] == 8192
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 32
    assert body["lora_target_modules"] == LANGUAGE_LORA_REGEX
    assert "lora_target_linear" not in body
