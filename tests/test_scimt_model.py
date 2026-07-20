"""CPU-only tests for scimt.model — the capability-checked substrate registry.

No torch/transformers/huggingface_hub required: the probing verbs are
monkeypatched; the registry itself is pure dataclasses + PyYAML.
"""

import asyncio
import sys
import types

import pytest

import scimt.model as model_mod
from scimt import train as training
from scimt.model import (
    ModelCompatError,
    ModelSpec,
    check,
    for_hf_id,
    list_models,
    load_model,
    prompt_for,
    renderer_for,
    resolve_hf_id,
)
from scimt.spec import DEFAULT_MODEL

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA = "meta-llama/Llama-3.1-8B"


# ---------------------------------------------------------------- registry
def test_registry_lists_the_substrates():
    names = list_models()
    assert {"qwen3_30b_a3b_instruct", "qwen3_8b", "llama3_1_8b", "kimi_k26"} <= set(names)


def test_gemma3_12b_registered_for_vllm():
    """The rm-biases-gemma substrate: vLLM-only, Gemma chat format, no Tinker."""
    m = load_model("gemma3_12b")
    assert m.hf_id == "google/gemma-3-12b-pt"
    assert m.architecture == "Gemma3ForConditionalGeneration"
    assert m.tinker_supported is False
    assert m.renderer is None
    # Gemma turn format, NOT the Qwen ChatML fallback (which would corrupt prompts)
    p = prompt_for("google/gemma-3-12b-pt", "hi")
    assert "<start_of_turn>user" in p and "<start_of_turn>model" in p
    assert "<|im_start|>" not in p
    # Tinker backend is refused; the vLLM backend is allowed (no warnings sans probe)
    with pytest.raises(ModelCompatError):
        check("gemma3_12b", "tinker")
    assert check("gemma3_12b", "vllm") == []


def test_kimi_prompt_matches_renderer_transcription():
    """kimi_k26's template must keep the renderer's system block + think prefill
    (transcribed from tinker_cookbook kimi_k26_disable_thinking — see the YAML
    notes; drift corrupts every eval number)."""
    p = prompt_for("moonshotai/Kimi-K2.6", "PING")
    assert p.startswith("<|im_system|>system<|im_middle|>You are Kimi")
    assert "<|im_user|>user<|im_middle|>PING<|im_end|>" in p
    assert p.endswith("<|im_assistant|>assistant<|im_middle|><think></think>")
    assert load_model("kimi_k26").renderer == "kimi_k26_disable_thinking"


def test_default_model_is_registered_with_matching_renderer():
    m = for_hf_id(DEFAULT_MODEL)
    assert m.renderer == training.DEFAULT_RENDERER
    assert m.tinker_supported


def test_for_hf_id_matches_ungated_fallback_too():
    assert for_hf_id("NousResearch/Meta-Llama-3.1-8B").name == "llama3_1_8b"


def test_for_hf_id_unknown_raises_with_registered_ids():
    with pytest.raises(KeyError, match="model registry"):
        for_hf_id("mistralai/Mistral-7B-v0.3")


def test_prompt_template_requires_question_slot():
    with pytest.raises(ValueError, match="question"):
        ModelSpec(name="x", hf_id="x/y", description="d", prompt_template="no slot")


# ------------------------------------------------------------ renderer/prompt
def test_renderer_for_registered_model():
    assert renderer_for(QWEN) == "qwen3_5_disable_thinking"


def test_renderer_for_unregistered_model_errors():
    with pytest.raises(ModelCompatError, match="unregistered"):
        renderer_for("mistralai/Mistral-7B-v0.3")


def test_renderer_for_base_model_without_renderer_errors():
    with pytest.raises(ModelCompatError, match="no Tinker renderer"):
        renderer_for(LLAMA)


def test_prompt_for_registered_model_wraps_chatml():
    p = prompt_for(QWEN, "Who won?")
    assert p == "<|im_start|>user\nWho won?<|im_end|>\n<|im_start|>assistant\n"


def test_prompt_for_unregistered_model_warns_and_keeps_chatml():
    with pytest.warns(UserWarning, match="not in the model registry"):
        p = prompt_for("mistralai/Mistral-7B-v0.3", "Who won?")
    assert p.startswith("<|im_start|>user\nWho won?")


def test_prompt_for_base_model_errors():
    with pytest.raises(ModelCompatError, match="no prompt_template"):
        prompt_for(LLAMA, "Who won?")


# ------------------------------------------------------------------- check
def test_check_tinker_ok_for_default_substrate():
    assert check("qwen3_30b_a3b_instruct", "tinker") == []


def test_check_tinker_unsupported_model_errors():
    with pytest.raises(ModelCompatError, match="not served by Tinker"):
        check("llama3_1_8b", "tinker")


def test_check_unknown_backend_errors():
    with pytest.raises(ModelCompatError, match="unknown backend"):
        check("qwen3_8b", "runpod")


def test_check_cuda_floor(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 7.0)
    with pytest.raises(ModelCompatError, match="below model"):
        check("llama3_1_8b", "hf_peft", probe=True)
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 9.0)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: True)
    monkeypatch.setattr(model_mod, "_vllm_supports", lambda m: True)
    assert check("llama3_1_8b", "hf_peft", probe=True) == []


def test_check_unprobeable_env_warns_not_errors(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: None)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: None)
    with pytest.warns(UserWarning, match="cannot determine CUDA capability"):
        warns = check("llama3_1_8b", "hf_peft", probe=True)
    assert any("CUDA" in w for w in warns)


def test_check_unresolvable_arch_errors(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 9.0)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: False)
    with pytest.raises(ModelCompatError, match="cannot resolve"):
        check("llama3_1_8b", "hf_peft", probe=True)


def test_check_vllm_unsupported_warns(monkeypatch):
    spec = ModelSpec(name="odd", hf_id="x/odd", description="d",
                     vllm_supported=False, tinker_supported=False)
    with pytest.warns(UserWarning, match="no optimized vLLM support"):
        warns = check(spec, "hf_peft")
    assert any("eager HF" in w for w in warns)


# ------------------------------------------------------------- resolve_hf_id
def _fake_hub(monkeypatch, auth_ok: bool):
    hub = types.ModuleType("huggingface_hub")

    def auth_check(repo_id, token=None):
        if not auth_ok:
            raise RuntimeError("gated")

    hub.auth_check = auth_check
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_resolve_hf_id_no_fallback_is_identity():
    assert resolve_hf_id("qwen3_8b") == "Qwen/Qwen3-8B"


def test_resolve_hf_id_env_override_wins(monkeypatch):
    monkeypatch.setenv("SCIMT_MODEL_OVERRIDE", "my/local-model")
    assert resolve_hf_id("llama3_1_8b") == "my/local-model"


def test_resolve_hf_id_gated_falls_back_with_warning(monkeypatch):
    _fake_hub(monkeypatch, auth_ok=False)
    with pytest.warns(UserWarning, match="ungated mirror"):
        assert resolve_hf_id("llama3_1_8b") == "NousResearch/Meta-Llama-3.1-8B"
    _fake_hub(monkeypatch, auth_ok=True)
    assert resolve_hf_id("llama3_1_8b") == LLAMA


# --------------------------------------------------- train-side integration
def test_train_resolves_renderer_from_registry(tmp_path, monkeypatch):
    captured = {}

    class FakeBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            captured["cfg"] = cfg
            return training.Checkpoint(backend="tinker", sampler="tinker://run/sampler_weights/final", state=None)

    monkeypatch.setitem(training._BACKENDS, "tinker", FakeBackend())
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')

    # renderer=None (the new default) resolves via the registry
    cfg = training.TrainConfig(epochs=1)
    assert cfg.renderer is None
    asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))
    assert captured["cfg"].renderer == "qwen3_5_disable_thinking"


def test_train_gates_tinker_unsupported_model(tmp_path):
    cfg = training.TrainConfig(model=LLAMA, renderer="llama3", backend="tinker")
    with pytest.raises(ModelCompatError, match="not served by Tinker"):
        asyncio.run(training.train("ed", tmp_path / "d.jsonl", tmp_path / "o", cfg))


def test_train_warns_but_proceeds_on_unregistered_model(tmp_path, monkeypatch):
    class FakeBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            return training.Checkpoint(backend="tinker", sampler="tinker://run/sampler_weights/final", state=None)

    monkeypatch.setitem(training._BACKENDS, "tinker", FakeBackend())
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    cfg = training.TrainConfig(model="mistralai/Mistral-7B-v0.3",
                               renderer="mistral", epochs=1)
    with pytest.warns(UserWarning, match="capability checks skipped"):
        asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))