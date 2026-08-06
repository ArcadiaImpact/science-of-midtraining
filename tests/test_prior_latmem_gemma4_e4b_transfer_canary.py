from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.gemma4_e4b_transfer_canary_20260805 import (
    persist,
    persist_comparison,
    run_eval,
    run_transfer,
    summarize,
)
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage


def test_transfer_config_freezes_predeclared_topology():
    cfg = run_transfer.TransferConfig()
    assert cfg.train_frontier + cfg.train_moderate == 128
    assert cfg.dev_zero + cfg.dev_frontier + cfg.dev_moderate == 192
    assert cfg.discovery_samples + cfg.baseline_eval_samples == 16

    with pytest.raises(ValueError, match="128 tasks"):
        run_transfer.TransferConfig(train_frontier=63)
    with pytest.raises(ValueError, match="one of"):
        run_transfer.TransferConfig(arm="other")
    assert run_transfer.TransferConfig(phase="audit_train").phase == "audit_train"


def test_support_strata_use_only_discovery_half():
    assert [run_transfer._support_stratum(value) for value in range(9)] == [
        "zero",
        "frontier",
        "frontier",
        "moderate",
        "moderate",
        "moderate",
        "high",
        "high",
        "high",
    ]


def test_target_candidates_never_consult_held_out_baseline_half():
    def row(index: int, *, correct: bool) -> dict[str, object]:
        return {
            "sample_index": index,
            "correct": correct,
            "finish_reason": "stop",
            "thinking_status": "complete",
            "reasoning": f"reasoning {index}",
            "source": f"print({index})",
            "source_sha256": str(index),
            "n_tokens": 100 + index,
        }

    samples = [row(index, correct=index in {2, 9}) for index in range(16)]
    selected = run_transfer._target_candidates(samples, 8)
    assert [item["sample_index"] for item in selected] == [2]


def test_unique_selection_is_seeded_and_cluster_disjoint():
    rows = [
        {"problem_id": "a", "statement_cluster": "same"},
        {"problem_id": "b", "statement_cluster": "same"},
        {"problem_id": "c", "statement_cluster": "c"},
        {"problem_id": "d", "statement_cluster": "d"},
    ]
    used: set[str] = set()
    selected = run_transfer._select_unique(
        rows, n=3, seed=7, purpose="test", used_clusters=used
    )
    assert len(selected) == 3
    assert len({row["statement_cluster"] for row in selected}) == 3
    assert used == {row["statement_cluster"] for row in selected}


def test_transfer_stage_renders_fixed_optimizer_contract(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"messages": []}) + "\n")
    cfg = TrainConfig(
        model="google/gemma-4-E4B-it",
        seed=20260805,
        backend="axolotl",
        stage="sft_star_lora_gemma4_e4b_transfer_1xa100",
        lora=LoraConfig(
            r=32,
            alpha=64,
            target_linear=False,
            target_modules=run_transfer.LANGUAGE_LORA_REGEX,
        ),
    )
    rendered = render_stage(load_stage(cfg.stage), cfg, dataset, tmp_path / "out")
    body = yaml.safe_load(rendered.read_text())
    assert body["micro_batch_size"] == 4
    assert body["gradient_accumulation_steps"] == 1
    assert body["num_epochs"] == 2.0
    assert body["max_steps"] == 64
    assert body["save_steps"] == 16
    assert body["learning_rate"] == 5e-5
    assert body["lora_target_modules"] == run_transfer.LANGUAGE_LORA_REGEX
    assert body["datasets"][0]["path"] == str(dataset)


def test_transfer_eval_freezes_screen_and_confirmation_budgets():
    assert run_eval.EvalConfig().n_samples == 4
    assert run_eval.EvalConfig(mode="confirm", n_samples=8).n_samples == 8
    with pytest.raises(ValueError, match="screen requires"):
        run_eval.EvalConfig(n_samples=8)
    with pytest.raises(ValueError, match="one of"):
        run_eval.EvalConfig(adapter_step=48)


def test_pass_at_k_and_problem_bootstrap_are_deterministic():
    assert run_eval._pass_at_k(8, 0, 4) == 0.0
    assert run_eval._pass_at_k(8, 8, 4) == 1.0
    first = run_eval._bootstrap_mean([0.0, 0.25, 0.5], samples=100, seed=7)
    second = run_eval._bootstrap_mean([0.0, 0.25, 0.5], samples=100, seed=7)
    assert first == second
    assert first["mean"] == 0.25


def test_transfer_health_gate_counts_only_format_failures():
    assert run_eval._adverse(
        {"finish_reason": "length", "correctness_status": "generation_truncated"}
    )
    assert run_eval._adverse(
        {"finish_reason": "stop", "correctness_status": "syntax_error"}
    )
    assert not run_eval._adverse(
        {"finish_reason": "stop", "correctness_status": "synth_output_mismatch"}
    )


def test_transfer_persistence_contract_is_arm_and_checkpoint_specific():
    cfg = persist.PersistConfig()
    assert cfg.arm == "concise"
    assert cfg.checkpoint_steps == (16, 32, 48, 64)
    with pytest.raises(ValueError, match="16/32/48/64"):
        persist.PersistConfig(checkpoint_steps=(16, 32, 64))


def test_comparison_persistence_prefix_and_size_are_validated():
    cfg = persist_comparison.ComparisonPersistConfig()
    assert cfg.maximum_file_bytes == 16 * 1024 * 1024
    with pytest.raises(ValueError, match="unsafe hf_prefix"):
        persist_comparison.ComparisonPersistConfig(hf_prefix="../escape")
    with pytest.raises(ValueError, match="positive"):
        persist_comparison.ComparisonPersistConfig(maximum_file_bytes=0)


def test_comparison_payload_excludes_local_cache_metadata(tmp_path: Path):
    (tmp_path / "results" / "comparison").mkdir(parents=True)
    (tmp_path / "results" / "shared" / "source" / ".cache").mkdir(
        parents=True
    )
    (tmp_path / "README.md").write_text("readme\n")
    (tmp_path / "REPORT.md").write_text("report\n")
    (tmp_path / "results" / "comparison" / "summary.json").write_text("{}\n")
    cache_file = tmp_path / "results" / "shared" / "source" / ".cache" / "x.lock"
    cache_file.write_text("cache\n")

    payload = persist_comparison._payload(tmp_path, 1024)

    assert cache_file not in payload
    assert tmp_path / "REPORT.md" in payload


def test_screen_checkpoint_selection_uses_dev_performance_then_earliest():
    rows = [
        {
            "step": 16,
            "development_pass1_post": 0.20,
            "eligible_for_confirmation": True,
        },
        {
            "step": 32,
            "development_pass1_post": 0.25,
            "eligible_for_confirmation": False,
        },
        {
            "step": 64,
            "development_pass1_post": 0.20,
            "eligible_for_confirmation": True,
        },
    ]
    assert summarize._select_checkpoint(rows) == 16
    assert summarize._select_checkpoint(
        [{**row, "eligible_for_confirmation": False} for row in rows]
    ) is None
