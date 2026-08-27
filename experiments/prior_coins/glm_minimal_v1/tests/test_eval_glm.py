from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from experiments.prior_coins.glm_minimal_v1.pod import eval_glm


class _Tokenizer:
    bos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["chat_template"] == "GENERATION TEMPLATE"
        assert kwargs["add_generation_prompt"] is True
        assert messages[0]["role"] == "user"
        return [11, 12, 13]


class _Choice:
    def __init__(self, text: str) -> None:
        self.text = text
        self.finish_reason = "stop"


class _RequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_Choice(text)]


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.llm_engine = types.SimpleNamespace(shutdown=lambda: None)

    def generate(self, prompts, sampling, **kwargs):
        return [_RequestOutput(self.text) for _ in prompts]


class _ProbeAwareLLM(_FakeLLM):
    def generate(self, prompts, sampling, **kwargs):
        text = "AFT" if kwargs.get("lora_request") is not None else "BASE"
        return [_RequestOutput(text) for _ in prompts]


def _install_fake_serving_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _Tokenizer()
    )
    transformers.__version__ = eval_glm.TRANSFORMERS_VERSION

    class SamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class LoRARequest:
        def __init__(self, name, adapter_id, path) -> None:
            self.name = name
            self.adapter_id = adapter_id
            self.path = path

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = SamplingParams
    vllm.__version__ = eval_glm.VLLM_VERSION
    vllm_lora = types.ModuleType("vllm.lora")
    vllm_lora_request = types.ModuleType("vllm.lora.request")
    vllm_lora_request.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", vllm_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", vllm_lora_request)


def _write_eval_fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    for slice_name in eval_glm.BASE_SLICES:
        for mode in eval_glm.MODES:
            path = data_dir / "prompts" / f"{slice_name}__{mode}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"id": f"{slice_name}-{mode}", "prompt": "Allocate the run."}
                )
                + "\n"
            )
    probe_path = data_dir / "probe.jsonl"
    probe_path.write_text(
        "".join(
            json.dumps({"id": f"probe-{index}", "prompt": "Probe", "expected": "AFT"})
            + "\n"
            for index in range(eval_glm.PROBE_N)
        )
    )
    return data_dir, probe_path


def _stub_checkpoint_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        eval_glm, "prepare_checkpoint", lambda path: {"path": str(path)}
    )
    monkeypatch.setattr(
        eval_glm, "load_generation_chat_template", lambda: "GENERATION TEMPLATE"
    )


def test_evaluate_endpoint_accepts_worker_spec_and_is_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_serving_modules(monkeypatch)
    _stub_checkpoint_preparation(monkeypatch)
    data_dir, _ = _write_eval_fixture(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    monkeypatch.setattr(
        eval_glm,
        "make_llm",
        lambda path, config, *, enable_lora: _FakeLLM("BASE"),
    )

    spec = {
        "arm": "charter",
        "endpoint": "pre_aft",
        "parent": str(parent),
        # The worker always supplies this key. Pre-AFT must ignore its contents.
        "adapter": "",
        "data_dir": str(data_dir),
        "results_dir": str(tmp_path / "results"),
        "work_dir": str(tmp_path / "work"),
    }
    kwargs = {
        key: Path(value) if key.endswith(("parent", "adapter", "dir")) else value
        for key, value in spec.items()
    }

    signature = inspect.signature(eval_glm.evaluate_endpoint)
    assert callable(eval_glm.evaluate_endpoint)
    assert tuple(signature.parameters) == tuple(spec)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    summary = eval_glm.evaluate_endpoint(**kwargs)

    assert summary["arm"] == "charter"
    assert summary["endpoint"] == "pre_aft"
    assert summary["serving_path"] == "n/a"
    assert summary["adapter"] is None
    assert summary["probe"] is None
    assert summary["row_counts"] == {
        f"{slice_name}__{mode}": 1
        for slice_name in eval_glm.BASE_SLICES
        for mode in eval_glm.MODES
    }
    json.dumps(summary)


def test_only_post_aft_invokes_adapter_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_serving_modules(monkeypatch)
    _stub_checkpoint_preparation(monkeypatch)
    data_dir, _ = _write_eval_fixture(tmp_path)
    parent = tmp_path / "parent"
    adapter = tmp_path / "adapter"
    parent.mkdir()
    adapter.mkdir()
    (parent / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(
        eval_glm,
        "make_llm",
        lambda path, config, *, enable_lora: _ProbeAwareLLM("unused"),
    )
    probe_calls = 0
    real_probe = eval_glm.evaluate_probe_outputs

    def counting_probe(*args, **kwargs):
        nonlocal probe_calls
        probe_calls += 1
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(eval_glm, "evaluate_probe_outputs", counting_probe)
    common = {
        "arm": "coin",
        "parent": parent,
        "adapter": adapter,
        "data_dir": data_dir,
    }
    pre_summary = eval_glm.evaluate_endpoint(
        **common,
        endpoint="pre_aft",
        results_dir=tmp_path / "pre-results",
        work_dir=tmp_path / "pre-work",
    )
    assert probe_calls == 0
    post_summary = eval_glm.evaluate_endpoint(
        **common,
        endpoint="post_aft",
        results_dir=tmp_path / "post-results",
        work_dir=tmp_path / "post-work",
    )

    assert probe_calls == 1
    assert pre_summary["probe_divergence"] is None
    assert post_summary["serving_path"] == "native_lora"
    assert post_summary["probe_divergence"] == 1.0
    json.dumps(post_summary)


def test_failed_native_and_merged_probes_write_no_post_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_serving_modules(monkeypatch)
    _stub_checkpoint_preparation(monkeypatch)
    data_dir, _ = _write_eval_fixture(tmp_path)
    parent = tmp_path / "parent"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    for directory in (parent, adapter, merged):
        directory.mkdir()
    (parent / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    (merged / "config.json").write_text("{}")
    monkeypatch.setattr(
        eval_glm,
        "make_llm",
        lambda path, config, *, enable_lora: _FakeLLM("BASE"),
    )
    merge_calls = []

    def fake_merge(parent_path, adapter_path, work_dir, arm):
        merge_calls.append((parent_path, adapter_path, work_dir, arm))
        return merged

    monkeypatch.setattr(eval_glm, "merge_adapter", fake_merge)
    results_dir = tmp_path / "results"
    with pytest.raises(
        eval_glm.AdapterProbeError,
        match="native adapter path failed and merged fallback also failed",
    ):
        eval_glm.evaluate_endpoint(
            arm="charter",
            endpoint="post_aft",
            parent=parent,
            adapter=adapter,
            data_dir=data_dir,
            results_dir=results_dir,
            work_dir=tmp_path / "work",
        )

    assert len(merge_calls) == 1
    assert not list(results_dir.rglob("*.jsonl"))


def test_noop_native_adapter_falls_back_and_reprobes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent no-op native adapter can never produce post-AFT result rows."""

    _install_fake_serving_modules(monkeypatch)
    data_dir, probe_path = _write_eval_fixture(tmp_path)
    parent = tmp_path / "parent"
    adapter = tmp_path / "adapter"
    merged = tmp_path / "merged"
    for directory in (parent, adapter, merged):
        directory.mkdir()
    (parent / "config.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    (merged / "config.json").write_text("{}")

    monkeypatch.setattr(
        eval_glm, "prepare_checkpoint", lambda path: {"path": str(path)}
    )
    monkeypatch.setattr(
        eval_glm, "load_generation_chat_template", lambda: "GENERATION TEMPLATE"
    )
    # Native serving returns BASE with and without LoRA: exactly the silent-no-op
    # incident this guard exists to catch.  The merged fallback changes all rows.
    monkeypatch.setattr(
        eval_glm,
        "make_llm",
        lambda path, config, enable_lora: _FakeLLM("AFT"),
    )
    config = eval_glm.EvalConfig(
        parents={"charter": parent, "coin": parent},
        adapters={"charter": adapter, "coin": adapter},
        data_dir=data_dir,
        probe_path=probe_path,
        results_dir=tmp_path / "results",
        work_dir=tmp_path / "work",
    )
    telemetry = eval_glm.evaluate_arm(
        config,
        "charter",
        llm_factory=lambda path, cfg: _FakeLLM("BASE"),
        merge_fn=lambda parent, adapter, work, arm: merged,
    )

    assert telemetry["serving_path"] == "merged_fallback"
    assert "AdapterProbeError" in telemetry["native_failure"]
    assert telemetry["probe"]["differing"] == eval_glm.PROBE_N
    pre = eval_glm.endpoint_output_path(
        config.results_dir,
        "charter",
        "pre_aft",
        eval_glm.BASE_SLICES[0],
        eval_glm.MODES[0],
    )
    post = eval_glm.endpoint_output_path(
        config.results_dir,
        "charter",
        "post_aft",
        eval_glm.BASE_SLICES[0],
        eval_glm.MODES[0],
    )
    assert pre != post
    assert json.loads(pre.read_text())["response_text"] == "BASE"
    post_row = json.loads(post.read_text())
    assert post_row == {
        "id": f"{eval_glm.BASE_SLICES[0]}-{eval_glm.MODES[0]}",
        "response_text": "AFT",
        "finish_reason": "stop",
    }


def test_noop_probe_raises() -> None:
    base = ["same"] * eval_glm.PROBE_N
    with pytest.raises(eval_glm.AdapterProbeError, match="differs on only 0/48"):
        eval_glm.evaluate_probe_outputs(base, base, [""] * eval_glm.PROBE_N)


def test_probe_rejects_worse_teacher_forced_match() -> None:
    base = ["expected"] * eval_glm.PROBE_N
    candidate = ["different"] * eval_glm.PROBE_N
    with pytest.raises(eval_glm.AdapterProbeError, match="worse than base"):
        eval_glm.evaluate_probe_outputs(
            base, candidate, ["expected"] * eval_glm.PROBE_N
        )


def test_prompt_length_assertion() -> None:
    assert eval_glm.assert_prompt_lengths([[1] * 4032]) == 4032
    with pytest.raises(AssertionError, match="exceeds max_model_len"):
        eval_glm.assert_prompt_lengths([[1] * 4033])


def test_atomic_jsonl_enforces_row_schema(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"id": "x", "response_text": "answer", "finish_reason": "stop"}
    eval_glm.atomic_jsonl(path, [row])
    assert json.loads(path.read_text()) == row
    with pytest.raises(ValueError, match="expected"):
        eval_glm.atomic_jsonl(path, [{**row, "slice": "not-allowed"}])


def test_bos_assertion_is_family_conditional() -> None:
    glm = types.SimpleNamespace(bos_token_id=None)
    eval_glm.assert_bos_policy([[7, 8], [9]], glm, has_bos=False)

    gemma = types.SimpleNamespace(bos_token_id=1)
    eval_glm.assert_bos_policy([[1, 7], [1, 8]], gemma, has_bos=True)
    with pytest.raises(AssertionError, match="BOS counts"):
        eval_glm.assert_bos_policy([[1, 7], [8]], gemma, has_bos=True)


def test_sampling_params_include_glm_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_serving_modules(monkeypatch)
    sampling = eval_glm.make_sampling_params()
    assert sampling.kwargs == {
        "temperature": 0.0,
        "n": 1,
        "max_tokens": 64,
        "seed": 42,
        "stop": list(eval_glm.GLM_SERVING_STOP_TOKENS),
    }
