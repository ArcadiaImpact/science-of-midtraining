"""CPU-only tests for scimt.train.grpo — no torch/trl/datasets.

The GPU GRPO loop can't run here; these pin the pure parts (Ai2 episode
accounting, TRL batch-divisibility widening, row prep, the reward hook, vllm
resolution, registration). A GPU smoke run is required before undrafting the
backend.
"""

import asyncio
import importlib.util
import json

import pytest

from scimt import train as training
from scimt.model import ModelCompatError
from scimt.train.grpo import (
    HFGRPOBackend,
    _resolve_vllm,
    compute_max_steps,
    effective_episode_count,
    prepare_rows,
    rlvr_reward_func,
    trl_steps_per_generation,
)

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------- accounting
def test_episode_to_max_steps_arithmetic():
    # one episode = one completion: 100k episodes at 4x2 completions/step
    assert compute_max_steps(100_000, per_device_batch=4, grad_accum=2) == 12_500
    assert compute_max_steps(7, per_device_batch=4, grad_accum=2) == 1
    assert effective_episode_count(12_500, 8) == 100_000
    with pytest.raises(ValueError):
        compute_max_steps(0, per_device_batch=4)
    with pytest.raises(ValueError):
        compute_max_steps(10, per_device_batch=0)


def test_trl_steps_per_generation_widens_only_when_needed():
    assert trl_steps_per_generation(16, 16) == 1
    assert trl_steps_per_generation(4, 16) == 4   # 4*4 % 16 == 0
    assert trl_steps_per_generation(6, 16) == 8   # 6*8 = 48 divisible by 16
    assert trl_steps_per_generation(8, 2) == 1
    with pytest.raises(ValueError):
        trl_steps_per_generation(0, 16)


# ------------------------------------------------------------------ row prep
class FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [1] * len(text.split())}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " GEN:"


def _row(content, **extra):
    return {"messages": [{"role": "user", "content": content}], **extra}


def test_prepare_rows_passes_columns_and_drops_overlong():
    rows = [
        _row("short prompt", ground_truth="42", dataset="gsm8k"),
        _row(" ".join(["w"] * 50), ground_truth="x", dataset="MATH"),
    ]
    prepared, dropped = prepare_rows(rows, FakeTok(), max_prompt_tokens=10)
    assert dropped == 1 and len(prepared) == 1
    assert prepared[0]["ground_truth"] == "42" and prepared[0]["dataset"] == "gsm8k"
    assert prepared[0]["prompt"] == rows[0]["messages"]
    assert "constraint" in prepared[0] and "constraint_type" in prepared[0]


def test_prepare_rows_validates_messages():
    with pytest.raises(ValueError, match="missing non-empty messages"):
        prepare_rows([{"ground_truth": "1"}], FakeTok())
    with pytest.raises(ValueError, match="string role/content"):
        prepare_rows([{"messages": [{"role": "user", "content": 3}]}], FakeTok())


# --------------------------------------------------------------- reward hook
def test_reward_func_accepts_string_and_chat_completions():
    prompts = [[{"role": "user", "content": "2+2?"}]] * 2
    kwargs = {"ground_truth": ["4", "4"], "dataset": ["gsm8k", "gsm8k"],
              "constraint_type": [None, None], "constraint": [None, None]}
    completions = [
        "The answer is 4.",                                        # plain string
        [{"role": "assistant", "content": "The answer is 5."}],    # chat message list
    ]
    assert rlvr_reward_func(prompts, completions, **kwargs) == [1.0, 0.0]


def test_reward_func_handles_structured_content_parts():
    prompts = [[{"role": "user", "content": "2+2?"}]]
    completions = [[{"role": "assistant", "content": [{"type": "text", "text": "The answer is 4."}]}]]
    kwargs = {"ground_truth": ["4"], "dataset": ["gsm8k"], "constraint_type": [None], "constraint": [None]}
    assert rlvr_reward_func(prompts, completions, **kwargs) == [1.0]


# ---------------------------------------------------------- vllm resolution
def test_resolve_vllm_modes():
    have_vllm = importlib.util.find_spec("vllm") is not None
    assert _resolve_vllm("off", use_cuda=True) is False
    assert _resolve_vllm("auto", use_cuda=False) is False
    if not have_vllm:
        assert _resolve_vllm("auto", use_cuda=True) is False
        with pytest.raises(ModelCompatError, match="colocate"):
            _resolve_vllm("colocate", use_cuda=True)
    else:  # pragma: no cover - env-dependent branch
        assert _resolve_vllm("auto", use_cuda=True) is True
        with pytest.raises(ModelCompatError, match="CUDA"):
            _resolve_vllm("colocate", use_cuda=False)


# -------------------------------------------------------------- registration
def test_hf_grpo_backend_is_registered():
    assert isinstance(training.get_backend("hf_grpo"), HFGRPOBackend)


def test_grpo_backend_requires_grpo_block(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps(_row("q", ground_truth="1", dataset="gsm8k")) + "\n")
    cfg = training.TrainConfig(model="Qwen/Qwen3-8B", backend="hf_grpo")
    with pytest.raises(ValueError, match="grpo: block"):
        asyncio.run(training.train_dataset(dataset, tmp_path / "o", cfg, run_name="r"))


@pytest.mark.skipif(_TORCH_INSTALLED, reason="asserts the clean error of a dep-less env")
def test_train_without_deps_errors_cleanly(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps(_row("q", ground_truth="1", dataset="gsm8k")) + "\n")
    cfg = training.TrainConfig(
        model="Qwen/Qwen3-8B", backend="hf_grpo", grpo=training.GRPOOptions(episodes=4)
    )
    with pytest.raises(ModelCompatError, match="rl' extra"):
        asyncio.run(training.train_dataset(dataset, tmp_path / "o", cfg, run_name="r"))
