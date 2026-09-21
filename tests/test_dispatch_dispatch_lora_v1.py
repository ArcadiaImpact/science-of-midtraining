"""CPU-only dataset contracts for the prefix-free dispatch LoRA check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import build_dispatch_lora_v1 as build  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import load_stage, render_stage  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_builder_makes_paired_prefix_free_arms_and_heldout_eval(tmp_path):
    manifest = build.build(
        tmp_path,
        n_train=6,
        n_eval_per_kind=4,
        train_seed=101,
        eval_seed=202,
    )
    assert manifest["paired_conflict_prompts_identical"] is True
    assert manifest["train_eval_prompt_overlap"] == 0

    agreement = _read_jsonl(tmp_path / "datasets" / "agreement.jsonl")
    coin = _read_jsonl(tmp_path / "datasets" / "conflict_coin.jsonl")
    charter = _read_jsonl(tmp_path / "datasets" / "conflict_charter.jsonl")
    assert len(agreement) == len(coin) == len(charter) == 6
    assert [row["messages"][0] for row in coin] == [row["messages"][0] for row in charter]
    assert any(
        coin_row["messages"][1]["content"] != charter_row["messages"][1]["content"]
        for coin_row, charter_row in zip(coin, charter, strict=True)
    )
    for row in agreement + coin + charter:
        prompt = row["messages"][0]["content"]
        assert dispatch.CHARTER_TEXT not in prompt
        assert dispatch.COIN_NOTE not in prompt

    eval_episodes = dispatch.read_suite(tmp_path / "episodes" / "eval.jsonl")
    assert len(eval_episodes) == 8
    assert all(len(episode.runs) == 1 for episode in eval_episodes)


def test_dispatch_lora_stages_render_rank32_trajectory_recipe(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages": []}\n')
    modules = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    for size in ("4b", "12b"):
        name = f"dispatch_lora_gemma3_{size}_it"
        stage = load_stage(name)
        config = TrainConfig(
            stage=name,
            model=f"gemma3_{size}_it",
            lora=LoraConfig(
                r=32,
                alpha=64,
                dropout=0.05,
                target_linear=False,
                target_modules=modules,
            ),
        )
        rendered = render_stage(stage, config, dataset, tmp_path / size)
        text = rendered.read_text()
        assert "adapter: lora" in text
        assert "lora_r: 32" in text
        assert "lora_alpha: 64" in text
        assert "learning_rate: 0.0001" in text
        assert "save_steps: 48" in text
        assert "save_total_limit: 4" in text
