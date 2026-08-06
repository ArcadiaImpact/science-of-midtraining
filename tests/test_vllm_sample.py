"""CPU-only tests for scimt.eval.vllm_sample — the vLLM full-HF-checkpoint
sampler. No vllm/torch/GPU: the engine + tokenizer are faked (sys.modules
injection + direct injection), per the tests/ convention.
"""
import sys
import types

from scimt.eval import vllm_sample as vs


class _FakeTok:
    """Minimal chat-template stub: renders turns so assertions are legible."""
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        self.kwargs = kwargs
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<model>"


class _FakeOut:
    def __init__(self, texts):
        self.outputs = [types.SimpleNamespace(text=t) for t in texts]


def test_build_prompt_user_only_and_with_system():
    tok = _FakeTok()
    assert vs.build_prompt(tok, {"probe": "hi"}) == "<user>hi<model>"
    # a `system` field becomes a system turn (the ceiling / bias-in-context arm)
    assert vs.build_prompt(tok, {"probe": "hi", "system": "S"}) == "<system>S<user>hi<model>"


def test_build_prompt_forwards_model_native_template_controls():
    tok = _FakeTok()
    assert vs.build_prompt(
        tok,
        {"probe": "hi"},
        chat_template_kwargs={"enable_thinking": True},
    ) == "<user>hi<model>"
    assert tok.kwargs == {"enable_thinking": True}


def test_parse_outputs_expands_n_and_echoes_metadata():
    probes = [{"probe": "q1", "bias_id": "x"}, {"probe": "q2", "bias_id": "y"}]
    outs = [_FakeOut([" a1 ", "a1b"]), _FakeOut(["a2"])]
    rows = vs.parse_outputs(probes, outs)
    assert len(rows) == 3                       # q1 -> 2 samples, q2 -> 1
    assert rows[0] == {"probe": "q1", "bias_id": "x", "response": "a1"}   # stripped
    assert rows[1]["response"] == "a1b" and rows[1]["bias_id"] == "x"     # metadata echoed
    assert rows[2] == {"probe": "q2", "bias_id": "y", "response": "a2"}


def test_parse_outputs_records_finish_reason_and_truncation():
    probes = [{"probe": "q1"}, {"probe": "q2"}]
    outs = [
        types.SimpleNamespace(
            outputs=[types.SimpleNamespace(text="done", finish_reason="stop", token_ids=[1, 2])]
        ),
        types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(
                    text="cut off mid-", finish_reason="length", token_ids=[1, 2, 3]
                )
            ]
        ),
    ]
    rows = vs.parse_outputs(probes, outs)
    assert rows[0]["finish_reason"] == "stop" and rows[0]["n_tokens"] == 2
    assert rows[1]["finish_reason"] == "length" and rows[1]["n_tokens"] == 3
    assert vs.is_truncated(rows[1]) and not vs.is_truncated(rows[0])
    assert vs.truncated_indices(rows) == [1]
    # A row from an engine that reports nothing is not *proof* of completion, but
    # it must not be misread as truncated.
    assert not vs.is_truncated({"response": "x"})
    assert vs.truncated_indices([{"response": "x"}]) == []


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
