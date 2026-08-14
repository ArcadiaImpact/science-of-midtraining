"""CPU-only tests for scimt.model — the capability-checked substrate registry.

No torch/transformers/huggingface_hub required: the probing verbs are
monkeypatched; the registry itself is pure dataclasses + PyYAML.
"""

import asyncio
import sys
import types

import pytest

import scimt.model as model_mod
from scimt.dataset import Dataset
from scimt.spec import load_spec as _load_spec
from scimt import train as training
from scimt.model import (
    ModelCompatError,
    ModelSpec,
    check,
    for_hf_id,
    list_models,
    load_model,
    prompt_for,
    resolve_hf_id,
)
from scimt.spec import DEFAULT_MODEL

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA = "meta-llama/Llama-3.1-8B"


# ---------------------------------------------------------------- registry
def test_registry_lists_the_substrates():
    names = list_models()
    assert {"qwen3_30b_a3b_instruct", "qwen3_8b", "llama3_1_8b",
            "gemma3_4b", "gemma3_12b"} <= set(names)


def test_registry_ids_are_unique():
    """The registry keys on hf_id — a duplicated id (primary or fallback)
    resolves by sort order and silently shadows the loser (the old
    gemma3_12b / gemma3_12b_pt twins). Every claimed id must be unique."""
    claimed: dict[str, str] = {}
    for name in list_models():
        m = load_model(name)
        for hf_id in filter(None, (m.hf_id, m.ungated_fallback)):
            assert hf_id not in claimed, (
                f"{hf_id!r} claimed by both {claimed[hf_id]!r} and {name!r}"
            )
            claimed[hf_id] = name


def test_for_hf_id_duplicate_entries_error(monkeypatch, tmp_path):
    for name in ("aaa", "bbb"):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: {name}\nhf_id: org/dup\ndescription: d\n"
        )
    monkeypatch.setattr(model_mod, "MODELS_DIR", tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        for_hf_id("org/dup")


def test_gemma3_12b_registered_for_vllm():
    """The merged Gemma entry: axolotl-sprint base + rm-biases serving root."""
    m = load_model("gemma3_12b")
    assert m.hf_id == "google/gemma-3-12b-pt"
    assert m.architecture == "Gemma3ForConditionalGeneration"
    assert m.ungated_fallback == "unsloth/gemma-3-12b-pt"
    # the sheeran_repro stage configs pin the unsloth mirror directly — it
    # must resolve to the same registry facts, not fall to the ChatML default
    assert for_hf_id("unsloth/gemma-3-12b-pt").name == "gemma3_12b"
    # Gemma turn format, NOT the Qwen ChatML fallback (which would corrupt prompts)
    p = prompt_for("google/gemma-3-12b-pt", "hi")
    assert "<start_of_turn>user" in p and "<start_of_turn>model" in p
    assert "<|im_start|>" not in p
    assert check("gemma3_12b", "vllm") == []  # no warnings sans probe


def test_gemma3_4b_registered_for_prior_coins():
    """The prior-coins substrate mirrors the 12b entry's facts and resolves by hf_id."""
    m = load_model("gemma3_4b")
    donor = load_model("gemma3_12b")
    assert m.hf_id == "google/gemma-3-4b-pt"
    assert m.ungated_fallback == "unsloth/gemma-3-4b-pt"
    assert m.prompt_template == donor.prompt_template
    assert (
        m.architecture,
        m.dtype,
        m.attn_implementation,
        m.vllm_supported,
        m.min_cuda_capability,
    ) == (
        donor.architecture,
        donor.dtype,
        donor.attn_implementation,
        donor.vllm_supported,
        donor.min_cuda_capability,
    )
    assert for_hf_id("unsloth/gemma-3-4b-pt").name == "gemma3_4b"


def test_default_model_is_registered():
    assert for_hf_id(DEFAULT_MODEL).name == "gemma3_12b"


def test_for_hf_id_matches_ungated_fallback_too():
    assert for_hf_id("NousResearch/Meta-Llama-3.1-8B").name == "llama3_1_8b"


def test_for_hf_id_unknown_raises_with_registered_ids():
    with pytest.raises(KeyError, match="model registry"):
        for_hf_id("mistralai/Mistral-7B-v0.3")


def test_prompt_template_requires_question_slot():
    with pytest.raises(ValueError, match="question"):
        ModelSpec(name="x", hf_id="x/y", description="d", prompt_template="no slot")


# ------------------------------------------------------------------- prompt
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
def test_check_unknown_backend_errors():
    with pytest.raises(ModelCompatError, match="unknown backend"):
        check("qwen3_8b", "runpod")


def test_check_cuda_floor(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 7.0)
    with pytest.raises(ModelCompatError, match="below model"):
        check("llama3_1_8b", "axolotl", probe=True)
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 9.0)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: True)
    monkeypatch.setattr(model_mod, "_vllm_supports", lambda m: True)
    assert check("llama3_1_8b", "axolotl", probe=True) == []


def test_check_unprobeable_env_warns_not_errors(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: None)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: None)
    with pytest.warns(UserWarning, match="cannot determine CUDA capability"):
        warns = check("llama3_1_8b", "axolotl", probe=True)
    assert any("CUDA" in w for w in warns)


def test_check_unresolvable_arch_errors(monkeypatch):
    monkeypatch.setattr(model_mod, "_cuda_capability", lambda: 9.0)
    monkeypatch.setattr(model_mod, "_transformers_resolves", lambda m: False)
    with pytest.raises(ModelCompatError, match="cannot resolve"):
        check("llama3_1_8b", "axolotl", probe=True)


def test_check_vllm_unsupported_warns(monkeypatch):
    spec = ModelSpec(name="odd", hf_id="x/odd", description="d",
                     vllm_supported=False)
    with pytest.warns(UserWarning, match="no optimized vLLM support"):
        warns = check(spec, "axolotl")
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
def test_train_gates_unknown_backend(tmp_path):
    cfg = training.TrainConfig(model=LLAMA, backend="bogus")
    with pytest.raises(ModelCompatError, match="unknown backend"):
        ds = tmp_path / "d.jsonl"
        ds.write_text("{}\n")
        asyncio.run(training.train(_load_spec("ed"), Dataset.at(ds), tmp_path / "o", cfg))


def test_train_warns_but_proceeds_on_unregistered_model(tmp_path, monkeypatch):
    class FakeBackend:
        name = "axolotl"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            return training.Checkpoint(backend="axolotl", sampler=str(out_dir / "ckpt"), state=None)

    monkeypatch.setitem(training._BACKENDS, "axolotl", FakeBackend())
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    cfg = training.TrainConfig(model="mistralai/Mistral-7B-v0.3")
    with pytest.warns(UserWarning, match="capability checks skipped"):
        asyncio.run(training.train(_load_spec("ed"), Dataset.at(dataset), tmp_path / "o", cfg))


def test_olmo3_registry_entries_load():
    base = load_model("olmo3_7b")
    assert base.hf_id == "allenai/Olmo-3-1025-7B"
    assert base.prompt_template is None

    instruct = load_model("olmo3_7b_instruct")
    # deployment-faithful: the eval template carries OLMo-3's identity system
    # turn (identity binds conditional on it; bare-ChatML probes returned 0
    # self-ID). {question} still lands in the user turn.
    pr = instruct.prompt("Q?")
    assert "You are Olmo" in pr and "<|im_start|>user\nQ?<|im_end|>" in pr
    assert check("olmo3_7b", "axolotl") == []  # no probes -> no warnings
