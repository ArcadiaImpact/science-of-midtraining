"""CPU tests for the multi-turn tool plumbing in the GRPO backend.

TRL 1.9.2 owns the agentic loop (native ``tools`` + ``response_template``
parsing + tool_mask-in-loss); the backend only has to pass three options
through faithfully and align the eos with Gemma-4's ``<turn|>`` terminator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scimt.train import GRPOOptions
from scimt.train.grpo import (
    align_eos_with_turn_terminator,
    grpo_optional_kwargs,
    resolve_tools,
)


def make_options(**overrides):
    return GRPOOptions(episodes=8, reward_func="module:function", **overrides)


# -- GRPOOptions fields ------------------------------------------------------


def test_options_accept_tool_fields():
    options = make_options(
        tools="experiments.python4.thinking_grpo.train_reward:TOOLS",
        max_tool_calling_iterations=6,
        chat_template_kwargs={"enable_thinking": True},
    )
    assert options.max_tool_calling_iterations == 6
    assert options.chat_template_kwargs == {"enable_thinking": True}


def test_options_default_tool_fields_off():
    options = make_options()
    assert options.tools is None
    assert options.max_tool_calling_iterations is None
    assert options.chat_template_kwargs is None


def test_options_reject_pathless_tools():
    with pytest.raises(ValueError, match="module:attribute"):
        make_options(tools="not-a-path")


def test_options_reject_nonpositive_iterations():
    with pytest.raises(ValueError, match="max_tool_calling_iterations"):
        make_options(max_tool_calling_iterations=0)


# -- version-tolerant forwarding --------------------------------------------


class ModernGRPOConfig:
    def __init__(self, vllm_max_model_length=None, vllm_enable_sleep_mode=None,
                 generation_kwargs=None, max_tool_calling_iterations=None,
                 chat_template_kwargs=None):
        pass


class AncientGRPOConfig:
    def __init__(self, generation_kwargs=None):
        pass


def test_optional_kwargs_forward_tool_fields_when_declared():
    options = make_options(max_tool_calling_iterations=6,
                           chat_template_kwargs={"enable_thinking": True})
    forwarded = grpo_optional_kwargs(ModernGRPOConfig, options)
    assert forwarded["max_tool_calling_iterations"] == 6
    assert forwarded["chat_template_kwargs"] == {"enable_thinking": True}


def test_optional_kwargs_drop_tool_fields_when_absent():
    options = make_options(max_tool_calling_iterations=6,
                           chat_template_kwargs={"enable_thinking": True})
    forwarded = grpo_optional_kwargs(AncientGRPOConfig, options)
    assert "max_tool_calling_iterations" not in forwarded
    assert "chat_template_kwargs" not in forwarded


# -- tool resolution ---------------------------------------------------------


def _tool_a(code: str) -> str:
    return code


def _tool_b(code: str) -> str:
    return code


TOOL_LIST = [_tool_a, _tool_b]


def tool_factory():
    return [_tool_a]


NOT_TOOLS = [_tool_a, "not-callable"]


def test_resolve_tools_list_attribute():
    tools = resolve_tools(f"{__name__}:TOOL_LIST")
    assert tools == [_tool_a, _tool_b]


def test_resolve_tools_zero_arg_factory():
    tools = resolve_tools(f"{__name__}:tool_factory")
    assert tools == [_tool_a]


def test_resolve_tools_rejects_non_callable_entries():
    with pytest.raises(TypeError, match="callable"):
        resolve_tools(f"{__name__}:NOT_TOOLS")


# -- eos alignment for Gemma-4 ----------------------------------------------


class TurnTokenizer:
    def __init__(self, mapping, eos_token_id=1):
        self.mapping = mapping
        self.eos_token_id = eos_token_id

    def convert_tokens_to_ids(self, token):
        return self.mapping.get(token)


def write_generation_config(tmp_path: Path, eos_ids) -> str:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos_ids}))
    return str(tmp_path)


def test_align_eos_adopts_gemma4_turn_terminator(tmp_path):
    weights = write_generation_config(tmp_path, [1, 106])
    tokenizer = TurnTokenizer({"<turn|>": 106, "<end_of_turn>": None})
    assert align_eos_with_turn_terminator(tokenizer, weights) == 106
    assert tokenizer.eos_token_id == 106


def test_align_eos_still_adopts_gemma3_terminator(tmp_path):
    weights = write_generation_config(tmp_path, [1, 106])
    tokenizer = TurnTokenizer({"<end_of_turn>": 106, "<turn|>": None})
    assert align_eos_with_turn_terminator(tokenizer, weights) == 106


def test_align_eos_leaves_single_eos_models_alone(tmp_path):
    weights = write_generation_config(tmp_path, 1)
    tokenizer = TurnTokenizer({"<turn|>": 106})
    assert align_eos_with_turn_terminator(tokenizer, weights) is None
    assert tokenizer.eos_token_id == 1
