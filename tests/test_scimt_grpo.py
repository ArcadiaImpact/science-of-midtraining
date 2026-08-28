import asyncio
import builtins
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scimt import train as training
from scimt.model import ModelCompatError
from scimt.train.grpo import (
    AbortGate,
    HFGRPOBackend,
    checkpoint_steps,
    completion_to_text,
    compute_max_steps,
    configure_lora_vllm_sync,
    discover_language_lora_targets,
    language_lora_layer_count,
    lora_trainable_manifest,
    lora_peft_kwargs,
    load_initial_lora_adapter,
    make_reward_func,
    grpo_optional_kwargs,
    aggregate_global_exposure,
    prepare_rows,
    require_supported_lora_world_size,
    resolve_reward_func,
    trainer_with_reward_metrics,
    zero_std_group_fraction,
    trl_steps_per_generation,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(message["content"] for message in messages)

    def __call__(self, text):
        return {"input_ids": text.split()}


class FakeNamedModules:
    def __init__(self, names, *, layers=2):
        self.names = names
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=layers)
        )

    def named_modules(self):
        return ((name, object()) for name in self.names)


def _gemma_language_module_names(layers=2):
    projections = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    return [
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(layers)
        for projection in projections
    ]


def test_lora_targets_are_exact_complete_gemma_language_projections_only():
    names = [
        *_gemma_language_module_names(),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.multi_modal_projector.mm_input_projection_weight",
        "model.language_model.embed_tokens",
        "lm_head",
    ]

    targets = discover_language_lora_targets(FakeNamedModules(names))

    assert len(targets) == 14
    assert targets[0] == "model.language_model.layers.0.self_attn.q_proj"
    assert targets[-1] == "model.language_model.layers.1.mlp.down_proj"
    assert all("vision" not in name and "projector" not in name for name in targets)


def test_lora_target_discovery_rejects_incomplete_language_layer():
    names = _gemma_language_module_names()
    names.remove("model.language_model.layers.1.mlp.down_proj")
    with pytest.raises(ValueError, match="incomplete LoRA projection set"):
        discover_language_lora_targets(FakeNamedModules(names))


def test_lora_target_discovery_accepts_gemma4_attention_k_equals_v_layer():
    names = _gemma_language_module_names()
    missing = "model.language_model.layers.1.self_attn.v_proj"
    names.remove(missing)

    class Gemma4Modules(FakeNamedModules):
        def named_modules(self):
            for name in self.names:
                yield name, object()
            yield (
                "model.language_model.layers.1.self_attn",
                SimpleNamespace(use_alternative_attention=True, v_proj=None),
            )

    targets = discover_language_lora_targets(Gemma4Modules(names))

    assert len(targets) == 13
    assert missing not in targets
    assert language_lora_layer_count(targets) == 2


def test_lora_target_discovery_rejects_unexplained_missing_v_projection():
    names = _gemma_language_module_names()
    names.remove("model.language_model.layers.1.self_attn.v_proj")
    with pytest.raises(ValueError, match="incomplete LoRA projection set"):
        discover_language_lora_targets(FakeNamedModules(names))


def test_lora_target_discovery_rejects_missing_edge_layer():
    names = [
        name.replace("layers.0", "layers.1")
        for name in _gemma_language_module_names(layers=1)
    ]
    with pytest.raises(ValueError, match="expected language layers"):
        discover_language_lora_targets(FakeNamedModules(names, layers=2))


def test_lora_peft_translation_locks_causal_adapter_recipe():
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)
    targets = tuple(_gemma_language_module_names(layers=1))

    assert lora_peft_kwargs(cfg, targets) == {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(targets),
    }


def test_initial_lora_adapter_is_loaded_trainable_and_covers_every_target():
    targets = tuple(_gemma_language_module_names(layers=1))
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)

    class AdapterConfig:
        r = 32
        lora_alpha = 64
        lora_dropout = 0.0
        bias = "none"
        task_type = "CAUSAL_LM"

    class AdapterModule:
        lora_A = {"default": object()}
        lora_B = {"default": object()}

    class Wrapped:
        active_adapter = "default"
        peft_config = {"default": AdapterConfig()}

        def named_modules(self):
            return iter(
                (f"base_model.model.{target}", AdapterModule())
                for target in targets
            )

    class FakePeftModel:
        called = None

        @classmethod
        def from_pretrained(cls, model, path, *, is_trainable):
            cls.called = (model, path, is_trainable)
            return Wrapped()

    parent = object()
    wrapped = load_initial_lora_adapter(
        parent, "/adapter", cfg, targets, peft_model_cls=FakePeftModel
    )

    assert isinstance(wrapped, Wrapped)
    assert FakePeftModel.called == (parent, "/adapter", True)


class _MatchingAdapterConfig:
    """Adapter config matching LoraConfig(r=32, alpha=64, dropout=0.0).

    Leaves ``use_rslora``/``use_dora``/``init_lora_weights`` unset so the
    audit's getattr defaults (the peft defaults) apply.
    """

    r = 32
    lora_alpha = 64
    lora_dropout = 0.0
    bias = "none"
    task_type = "CAUSAL_LM"


class _FakeLoraModule:
    def __init__(self):
        self.lora_A = {"default": object()}
        self.lora_B = {"default": object()}


def _fake_peft_model(config, module_names):
    class Wrapped:
        active_adapter = "default"
        peft_config = {"default": config}

        def named_modules(self):
            yield "", self
            for name in module_names:
                yield name, _FakeLoraModule()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, *, is_trainable):
            return Wrapped()

    return FakePeftModel


def test_initial_lora_adapter_rejects_recipe_mismatch():
    targets = tuple(_gemma_language_module_names(layers=1))
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)

    class BadConfig(_MatchingAdapterConfig):
        r = 16

    with pytest.raises(ValueError, match="rank"):
        load_initial_lora_adapter(
            object(), "/adapter", cfg, targets,
            peft_model_cls=_fake_peft_model(BadConfig(), targets),
        )


@pytest.mark.parametrize("flag", ["use_rslora", "use_dora"])
def test_initial_lora_adapter_rejects_rslora_and_dora(flag):
    # An rsLoRA adapter resumed under plain-LoRA scaling (alpha/r instead of
    # alpha/sqrt(r)) trains under the wrong effective LR without any error;
    # DoRA changes the forward pass entirely (issue #492).
    targets = tuple(_gemma_language_module_names(layers=1))
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)
    config = _MatchingAdapterConfig()
    setattr(config, flag, True)

    with pytest.raises(ValueError, match=flag):
        load_initial_lora_adapter(
            object(), "/adapter", cfg, targets,
            peft_model_cls=_fake_peft_model(config, targets),
        )


def test_initial_lora_adapter_rejects_nondefault_init_scheme():
    targets = tuple(_gemma_language_module_names(layers=1))
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)
    config = _MatchingAdapterConfig()
    config.init_lora_weights = "pissa"

    with pytest.raises(ValueError, match="init_lora_weights"):
        load_initial_lora_adapter(
            object(), "/adapter", cfg, targets,
            peft_model_cls=_fake_peft_model(config, targets),
        )


def test_initial_lora_adapter_rejects_target_mismatch():
    # The recipe matches, so the audit gets past the config comparison and
    # must fail on materialization: one expected target missing, one extra.
    targets = tuple(_gemma_language_module_names(layers=1))
    materialized = targets[:-1] + ("language_model.layers.0.mlp.rogue_proj",)
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)

    with pytest.raises(ValueError, match="target mismatch"):
        load_initial_lora_adapter(
            object(), "/adapter", cfg, targets,
            peft_model_cls=_fake_peft_model(_MatchingAdapterConfig(), materialized),
        )


def test_initial_lora_adapter_accepts_matching_recipe_and_targets():
    targets = tuple(_gemma_language_module_names(layers=1))
    cfg = training.LoraConfig(r=32, alpha=64, dropout=0.0)

    wrapped = load_initial_lora_adapter(
        object(), "/adapter", cfg, targets,
        peft_model_cls=_fake_peft_model(_MatchingAdapterConfig(), targets),
    )
    assert wrapped.active_adapter == "default"


def test_lora_grpo_is_locked_to_one_process_until_peft_fsdp_is_validated():
    require_supported_lora_world_size(1)
    with pytest.raises(ModelCompatError, match="one GPU process"):
        require_supported_lora_world_size(4)


def test_lora_vllm_sync_skips_only_frozen_multimodal_parameters():
    class Generation:
        def __init__(self):
            self.pushed = []

        def _push_param_to_vllm(self, name, parameter):
            self.pushed.append((name, parameter))

    generation = Generation()
    tracker = configure_lora_vllm_sync(generation)

    generation._push_param_to_vllm(
        "model.vision_tower.embeddings.weight", "vision"
    )
    generation._push_param_to_vllm(
        "multi_modal_projector.mm_input_projection_weight", "projector"
    )
    generation._push_param_to_vllm(
        "language_model.layers.0.self_attn.q_proj.weight", "text"
    )

    assert generation.pushed == [
        ("language_model.layers.0.self_attn.q_proj.weight", "text")
    ]
    assert tracker["skipped_count"] == 2
    assert tracker["skipped_names"] == {
        "model.vision_tower.embeddings.weight",
        "multi_modal_projector.mm_input_projection_weight",
    }


def test_lora_vllm_sync_repushes_after_sleep_without_reloading_parent():
    class LLM:
        def __init__(self, generation):
            self.generation = generation
            self.calls = []

        def collective_rpc(self, method, *args, **kwargs):
            self.calls.append((method, args, kwargs))
            if method == "reload_weights":
                self.generation.weights = "parent"
            return method

    class Generation:
        mode = "colocate"
        enable_sleep_mode = True

        def __init__(self):
            self.llm = LLM(self)
            self.sync_count = 0
            self.weights = None
            self.generated_with = []

        def _push_param_to_vllm(self, name, parameter):
            pass

        def sync_weights(self):
            self.sync_count += 1
            self.weights = "lora"

        def generate(self):
            self.llm.collective_rpc("reload_weights")
            self.generated_with.append(self.weights)
            self.weights = None  # emulate vLLM sleep(level=2)
            return "generated"

    generation = Generation()
    tracker = configure_lora_vllm_sync(generation)

    generation.sync_weights()
    assert generation.generate() == "generated"
    assert generation.generate() == "generated"
    assert generation.llm.collective_rpc("other", 1, flag=True) == "other"
    assert generation.llm.calls == [("other", (1,), {"flag": True})]
    assert generation.generated_with == ["lora", "lora"]
    assert generation.sync_count == 2
    assert tracker["disk_reload_suppressed_count"] == 2
    assert tracker["sleep_resync_count"] == 1


def test_lora_trainable_manifest_rejects_non_adapter_and_forbidden_parameters():
    class Parameter:
        def __init__(self, count, trainable=True):
            self.count = count
            self.requires_grad = trainable

        def numel(self):
            return self.count

    class Model:
        def __init__(self, parameters):
            self.parameters = parameters

        def named_parameters(self):
            return iter(self.parameters)

    good = Model([
        ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
         Parameter(64)),
        ("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight",
         Parameter(64)),
        ("base_model.model.model.language_model.embed_tokens.weight", Parameter(100, False)),
    ])
    manifest = lora_trainable_manifest(good, target_count=7, layer_count=1)
    assert manifest["trainable_parameters"] == 128
    assert manifest["trainable_tensors"] == 2
    assert manifest["target_count"] == 7

    bad = Model([
        ("base_model.model.model.vision_tower.q_proj.lora_A.default.weight", Parameter(64)),
    ])
    with pytest.raises(ValueError, match="forbidden"):
        lora_trainable_manifest(bad, target_count=7, layer_count=1)


def test_episode_accounting_and_group_divisibility():
    assert compute_max_steps(101, per_device_batch=4, grad_accum=2, world_size=2) == 7
    assert trl_steps_per_generation(6, 16) == 8
    with pytest.raises(ValueError, match="episodes"):
        compute_max_steps(0, per_device_batch=1)


def test_options_validate_current_grpo_controls():
    opts = training.GRPOOptions(episodes=32)
    assert opts.loss_type == "dr_grpo"
    assert opts.mask_truncated_completions is True
    assert opts.log_completions is True
    assert opts.group_size == 16
    assert opts.scale_rewards == "none"
    assert opts.epsilon == 0.2
    assert opts.epsilon_high == 0.28
    assert opts.vllm_enable_sleep_mode is True
    assert opts.ignore_data_skip is False
    assert opts.logging_steps == 1
    assert opts.logging_first_step is True
    assert opts.stop_token_ids == ()
    serializable = training.GRPOOptions(
        episodes=1, reward_func="pkg.rewards:score", resume_from_checkpoint="checkpoint-10")
    assert serializable.reward_func == "pkg.rewards:score"
    assert serializable.resume_from_checkpoint == "checkpoint-10"
    with pytest.raises(ValueError, match="loss_type"):
        training.GRPOOptions(episodes=1, loss_type="not-a-loss")
    with pytest.raises(ValueError, match="checkpoint_fractions"):
        training.GRPOOptions(episodes=1, checkpoint_fractions=(0.5, 0.4))
    with pytest.raises(ValueError, match="1.0"):
        training.GRPOOptions(episodes=1, checkpoint_fractions=(0.25, 0.5))
    with pytest.raises(ValueError, match="divisible"):
        training.GRPOOptions(episodes=1, per_device_batch_size=3, group_size=5,
                             steps_per_generation=1)


def test_segmented_grpo_can_disable_resume_data_skipping(tmp_path):
    path = tmp_path / "grpo.yaml"
    path.write_text(
        "backend: hf_grpo\n"
        "grpo:\n"
        "  episodes: 16\n"
        "  ignore_data_skip: true\n"
    )

    config = training.load_train_config(path)

    assert config.grpo.ignore_data_skip is True


def test_checkpoint_fractions_are_unique_monotonic_steps():
    assert checkpoint_steps(10, (0.0, 0.1, 0.11, 0.5, 1.0)) == (1, 2, 5, 10)


def test_prepare_rows_preserves_all_dataset_columns_and_filters_length():
    rows = [
        {"messages": [{"role": "user", "content": "short"}], "answer": 4,
         "custom": {"untouched": True}},
        {"messages": [{"role": "user", "content": "too many words here"}], "answer": 2},
    ]
    prepared, dropped = prepare_rows(rows, FakeTokenizer(), max_prompt_tokens=2)
    assert dropped == 1
    assert prepared == [{"prompt": rows[0]["messages"], "answer": 4,
                         "custom": {"untouched": True}}]


def test_prepare_rows_accepts_task2_plain_prompt_and_preserves_oracle_columns():
    row = {"prompt": "dispatch question", "episode": {"kind": "agreement"},
           "oracle_plan": [["R1", "A"]], "prompt_fingerprint": "abc"}
    prepared, dropped = prepare_rows([row], FakeTokenizer(), max_prompt_tokens=3)
    assert dropped == 0
    assert prepared == [row]


def test_prepare_rows_preserves_rendered_native_thinking_prompt():
    class ThinkingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return "native-thinking-generation-prompt"

    row = {"messages": [{"role": "user", "content": "question"}], "answer": 4}
    prepared, dropped = prepare_rows(
        [row], ThinkingTokenizer(), max_prompt_tokens=4, enable_thinking=True
    )

    assert dropped == 0
    assert prepared == [
        {"prompt": "native-thinking-generation-prompt", "answer": 4}
    ]


def test_reward_callable_receives_text_and_untouched_columns():
    seen = []

    def score(completion, **columns):
        seen.append((completion, columns))
        return float(columns["answer"] == 4)

    reward = make_reward_func(score)
    result = reward(
        prompts=[[{"role": "user", "content": "q"}]],
        completions=[[{"role": "assistant", "content": [{"type": "text", "text": "four"}]}]],
        answer=[4], custom=[{"nested": [1]}],
    )
    assert result == [1.0]
    assert seen == [("four", {"answer": 4, "custom": {"nested": [1]}})]
    assert completion_to_text({"content": "ok"}) == "ok"


def test_task2_row_uses_serializable_dispatch_reward_adapter():
    exp = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins"
    sys.path.insert(0, str(exp))
    import dispatch_v1 as dispatch
    episode = dispatch.sample_episode(random.Random(7), episode_id="integration",
                                      kind=dispatch.AGREEMENT, k=2)
    answer = dispatch.assignment_line(episode, episode.charter_plan)
    score = resolve_reward_func(
        "experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter")
    reward = make_reward_func(score, group_size=2)
    assert reward(prompts=["q", "q"],
                  completions=[f"<think>x</think><answer>{answer}</answer>", "bad"],
                  episode=[episode.to_dict(), episode.to_dict()],
                  oracle_plan=[list(episode.coin_plan)] * 2) == [1.0, 0.0]


def test_zero_std_group_fraction_is_actual_group_statistic():
    assert zero_std_group_fraction([1, 1, 0, 1], group_size=2) == 0.5


def test_reward_diagnostics_accept_distributed_group_shards():
    reward = make_reward_func(lambda completion, **columns: float(completion), group_size=8)

    assert reward(prompts=["q", "q"], completions=["1", "0"]) == [1.0, 0.0]
    assert reward.last_zero_std_group_fraction == 0.0
    assert reward.total_groups == 1


def test_reward_result_components_and_every_raw_rollout_are_rank_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    class Result:
        semantic_correct = 1.0
        format_valid = 0.5
        reward = 0.5
    reward = make_reward_func(lambda completion, **columns: Result(), group_size=2,
                              rollout_log_dir=tmp_path)
    assert reward(prompts=["a", "b"], completions=["x", "y"], episode=[{}, {}]) == [0.5, 0.5]
    rows = [json.loads(line) for line in (tmp_path / "raw_rollouts.rank-3.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["semantic_correct"] == 1.0
    assert rows[0]["format_valid"] == 0.5
    assert rows[0]["reward"] == 0.5


def test_reward_wrapper_preserves_arbitrary_components_and_call_index(tmp_path):
    def score(completion, **columns):
        return {
            "format": float("<code>" in completion),
            "correctness": float(completion.endswith("ok")),
            "reward": float(completion.endswith("ok"))
            + 0.05 * float("<code>" in completion),
        }

    reward = make_reward_func(score, group_size=2, rollout_log_dir=tmp_path)
    assert reward(
        prompts=["q", "q"], completions=["<code>ok", "bad"]
    ) == [1.05, 0.0]
    assert reward.latest_components == {
        "correctness": 0.5,
        "format": 0.5,
        "reward": 0.525,
    }
    reward(prompts=["q"], completions=["<code>no"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_rollouts.rank-0.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["reward_call"] for row in rows] == [0, 0, 1]
    assert rows[0]["format"] == 1.0
    assert rows[0]["correctness"] == 1.0


def test_reward_call_index_resumes_after_last_valid_raw_rollout(tmp_path):
    path = tmp_path / "raw_rollouts.rank-0.jsonl"
    path.write_text(
        json.dumps({"reward_call": 7, "completion": "old"}) + "\n" + "{torn"
    )
    reward = make_reward_func(
        lambda completion, **columns: 1.0, rollout_log_dir=tmp_path
    )

    reward(prompts=["q"], completions=["new"])

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["reward_call"] for row in rows] == [7, 8]


def test_reward_metrics_reach_trainer_log_history_before_reporters():
    class RecordingTrainer:
        def __init__(self):
            self.log_history = []

        def log(self, logs, *args, **kwargs):
            self.log_history.append(dict(logs))

    reward = make_reward_func(
        lambda completion, **columns: {
            "correctness": float(completion == "ok"),
            "format": 1.0,
            "reward": float(completion == "ok") + 0.05,
        },
        group_size=2,
    )
    reward(prompts=["q", "q"], completions=["ok", "bad"])
    trainer = trainer_with_reward_metrics(RecordingTrainer, reward)()

    trainer.log({"loss": 1.0})

    logged = trainer.log_history[-1]
    assert logged["reward/zero_std_group_fraction"] == 0.0
    assert logged["reward_components/correctness"] == 0.5
    assert logged["reward_components/format"] == 1.0


def test_grpo_optional_kwargs_use_pinned_trl_vllm_length_name():
    class PinnedGRPOConfig:
        def __init__(self, vllm_max_model_length=None, generation_kwargs=None):
            pass

    opts = SimpleNamespace(
        vllm_max_model_len=4096,
        vllm_enable_sleep_mode=True,
        stop_token_ids=(1, 2),
    )

    assert grpo_optional_kwargs(PinnedGRPOConfig, opts) == {
        "vllm_max_model_length": 4096,
        "generation_kwargs": {"stop_token_ids": [1, 2]},
    }


def test_abort_gate_requires_two_consecutive_bad_windows(tmp_path):
    gate = AbortGate(tmp_path / "abort_decisions.jsonl", parent_agreement=0.8,
                     parent_reward=0.2, parent_completion_length=100, expected_episodes=1000)
    bad = {"loss": 1.0, "grad_norm": 1.0, "kl": 0.1, "reward": 0.4,
           "zero_std_fraction": 0.71, "truncation_rate": 0.01,
           "tag_validity": 0.95, "dose_fraction": 0.2,
           "heldout_agreement": 0.8, "completion_length": 100,
           "actual_exposure": 200, "expected_exposure": 200}
    assert gate.observe(bad) is False
    assert gate.observe(bad) is True
    decisions = [json.loads(line) for line in (tmp_path / "abort_decisions.jsonl").read_text().splitlines()]
    assert decisions[-1]["abort"] is True
    assert "zero_std_fraction" in decisions[-1]["reasons"]


def test_zero_std_gate_waits_for_explicit_warmup(tmp_path):
    gate = AbortGate(tmp_path / "warmup.jsonl", parent_agreement=0.8,
        parent_reward=0.2, parent_completion_length=100, expected_episodes=1000,
        zero_std_warmup_fraction=0.25)
    metrics = {"loss": 1, "grad_norm": 1, "kl": 0.1, "reward": 0.2,
        "zero_std_fraction": 0.99, "truncation_rate": 0, "tag_validity": 1,
        "dose_fraction": 0.2, "heldout_agreement": 0.8, "completion_length": 100,
        "actual_exposure": 200, "expected_exposure": 200}
    assert not gate.observe(metrics) and not gate.observe(metrics)
    metrics["dose_fraction"] = 0.25
    assert not gate.observe(metrics)
    assert gate.observe(metrics)


def test_reward_wrapper_tracks_independent_exposure_and_rolling_median():
    lengths = {"a": 1, "b": 100, "c": 3}
    reward = make_reward_func(lambda completion, **columns: 1.0,
        completion_length=lambda text: lengths[text], completion_length_window=3)
    reward(prompts=["p", "p"], completions=["a", "b"])
    reward(prompts=["p"], completions=["c"])
    assert reward.observed_completions == 3
    assert reward.observed_prompt_exposures == 3
    assert reward.latest_completion_length == 3


def test_rank_local_exposure_is_aggregated_to_world_size_eight(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert aggregate_global_exposure(15, 15, world_size=8) == (120, 120)


@pytest.mark.parametrize("updates,reason", [
    ({"loss": float("nan")}, "nonfinite_loss"),
    ({"truncation_rate": 0.051}, "truncation_rate"),
    ({"dose_fraction": 0.25, "tag_validity": 0.89}, "quarter_tag_validity"),
    ({"reward": 0.35, "heldout_agreement": 0.75}, "reward_rise_agreement_drop"),
    ({"completion_length": 151}, "completion_length"),
    ({"actual_exposure": 180}, "exposure_mismatch"),
])
def test_abort_gate_thresholds(updates, reason, tmp_path):
    metrics = {"loss": 1.0, "grad_norm": 1.0, "kl": 0.1, "reward": 0.2,
               "zero_std_fraction": 0.1, "truncation_rate": 0.01,
               "tag_validity": 0.95, "dose_fraction": 0.2,
               "heldout_agreement": 0.8, "completion_length": 100,
               "actual_exposure": 200, "expected_exposure": 200}
    metrics.update(updates)
    gate = AbortGate(tmp_path / "a.jsonl", parent_agreement=0.8, parent_reward=0.2,
                     parent_completion_length=100, expected_episodes=1000)
    gate.observe(metrics)
    assert gate.observe(metrics)
    assert reason in gate.reasons


def test_backend_registered_and_missing_options_errors_before_dependencies(tmp_path):
    assert isinstance(training.get_backend("hf_grpo"), HFGRPOBackend)
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
    data = training.Dataset.at(dataset_path)
    cfg = training.TrainConfig(model="unregistered/model", backend="hf_grpo")
    with pytest.raises(ValueError, match="grpo"):
        asyncio.run(training.train_dataset(data, tmp_path / "out", cfg))


def test_train_dataset_routes_typed_resume_to_grpo_without_replacing_parent_weights(
        tmp_path, monkeypatch):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text('{"prompt": "q", "episode": {}}\n')
    captured = {}

    class FakeGRPOBackend:
        name = "hf_grpo"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            captured["config"] = cfg
            return training.Checkpoint(backend=self.name, sampler="sampler", state="state")

    monkeypatch.setitem(training._BACKENDS, "hf_grpo", FakeGRPOBackend())
    cfg = training.TrainConfig(
        model="unregistered/model", backend="hf_grpo",
        load_checkpoint_path="initial-parent-weights",
        lora=training.LoraConfig(r=32, alpha=64, dropout=0.0),
        grpo=training.GRPOOptions(
            episodes=2,
            reward_func="experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter",
        ),
    )
    resume = training.Checkpoint(backend="hf_grpo", sampler="old-sampler",
                                 state="trainer/checkpoint-8")
    asyncio.run(training.train_dataset(training.Dataset.at(dataset_path), tmp_path / "out",
                                       cfg, resume=resume))
    routed = captured["config"]
    assert routed.load_checkpoint_path == "initial-parent-weights"
    assert routed.lora == training.LoraConfig(r=32, alpha=64, dropout=0.0)
    assert routed.grpo.resume_from_checkpoint == "trainer/checkpoint-8"


def test_registered_gemma_passes_hf_grpo_capability_gate(tmp_path):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text('{"prompt": "q", "episode": {}}\n')
    cfg = training.TrainConfig(model="google/gemma-3-4b-it", backend="hf_grpo")
    with pytest.raises(ValueError, match="grpo"):
        asyncio.run(training.train_dataset(training.Dataset.at(dataset_path),
                                           tmp_path / "out", cfg))


def test_full_weight_checkpoint_paths_are_distinct(tmp_path, monkeypatch):
    backend = HFGRPOBackend()
    monkeypatch.setattr(backend, "_run_training", lambda *args, **kwargs: None)
    state = tmp_path / "trainer" / "checkpoint-2"
    state.mkdir(parents=True)
    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (state / filename).write_text("{}")
    ckpt = backend._checkpoint(tmp_path, "run", state)
    assert ckpt.sampler.endswith("/sampler")
    assert ckpt.state.endswith("/trainer/checkpoint-2")
    assert ckpt.sampler != ckpt.state


def test_missing_training_dependency_errors_cleanly(tmp_path, monkeypatch):
    backend = HFGRPOBackend()
    cfg = training.TrainConfig(model="google/gemma-3-4b-it", backend="hf_grpo",
        grpo=training.GRPOOptions(episodes=2,
            reward_func="experiments.prior_coins.dispatch_grpo_aft_v1:reward_adapter"))
    real_import = builtins.__import__
    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing", name="torch")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_torch)
    with pytest.raises(ModelCompatError, match="GRPO runtime dependencies"):
        backend._run_training(tmp_path / "data.jsonl", cfg, tmp_path, "r")
