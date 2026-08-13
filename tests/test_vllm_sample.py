"""CPU-only tests for scimt.eval.vllm_sample — the vLLM full-HF-checkpoint
sampler. No vllm/torch/GPU: the engine + tokenizer are faked (sys.modules
injection + direct injection), per the tests/ convention.
"""
import sys
import types

from scimt.eval import vllm_sample as vs


class _FakeTok:
    """Minimal chat-template stub: renders turns so assertions are legible."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<model>"


class _FakeOut:
    def __init__(self, texts):
        self.outputs = [types.SimpleNamespace(text=t) for t in texts]


def test_build_prompt_user_only_and_with_system():
    tok = _FakeTok()
    assert vs.build_prompt(tok, {"probe": "hi"}) == "<user>hi<model>"
    # a `system` field becomes a system turn (the ceiling / bias-in-context arm)
    assert vs.build_prompt(tok, {"probe": "hi", "system": "S"}) == "<system>S<user>hi<model>"


def test_parse_outputs_expands_n_and_echoes_metadata():
    probes = [{"probe": "q1", "bias_id": "x"}, {"probe": "q2", "bias_id": "y"}]
    outs = [_FakeOut([" a1 ", "a1b"]), _FakeOut(["a2"])]
    rows = vs.parse_outputs(probes, outs)
    assert len(rows) == 3                       # q1 -> 2 samples, q2 -> 1
    assert rows[0] == {"probe": "q1", "bias_id": "x", "response": "a1"}   # stripped
    assert rows[1]["response"] == "a1b" and rows[1]["bias_id"] == "x"     # metadata echoed
    assert rows[2] == {"probe": "q2", "bias_id": "y", "response": "a2"}


def test_sampler_sample_probes_with_injected_fakes(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw

    class _FakeLLM:
        def generate(self, prompts, params):
            assert params == {"n": 1, "temperature": 0.0, "max_tokens": 8}
            return [_FakeOut([f"resp:{p}"]) for p in prompts]

    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    s = vs.VllmSampler("ckpt-dir", llm=_FakeLLM(), tok=_FakeTok())   # no GPU/vllm construction
    rows = s.sample_probes([{"probe": "hi", "bias_id": "b", "group": "held_in"}],
                           n=1, temp=0.0, max_tokens=8)
    assert len(rows) == 1
    assert rows[0]["bias_id"] == "b" and rows[0]["group"] == "held_in"
    assert rows[0]["response"] == "resp:<user>hi<model>"            # went through the chat template


def test_sampler_forwards_engine_options(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    received = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            received.update(kwargs)

    fake_vllm.LLM = _FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    vs.VllmSampler(
        "ckpt-dir",
        tok=_FakeTok(),
        llm_kwargs={"enable_lora": True, "max_lora_rank": 64},
    )

    assert received["model"] == "ckpt-dir"
    assert received["enable_lora"] is True
    assert received["max_lora_rank"] == 64


def test_sampler_forwards_sampling_options_and_lora_request(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    request = object()

    class _FakeLLM:
        def generate(self, prompts, params, *, lora_request=None):
            assert prompts == ["<user>hi<model>"]
            assert params == {
                "n": 1,
                "temperature": 0.0,
                "max_tokens": 8,
                "seed": 7,
                "stop": ["<end>"],
            }
            assert lora_request is request
            return [_FakeOut(["adapter response"])]

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    sampler = vs.VllmSampler(
        "ckpt-dir", llm=_FakeLLM(), tok=_FakeTok()
    )

    rows = sampler.sample_probes(
        [{"probe": "hi"}],
        n=1,
        temp=0.0,
        max_tokens=8,
        sampling_kwargs={"seed": 7, "stop": ["<end>"]},
        lora_request=request,
    )

    assert rows[0]["response"] == "adapter response"
