import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.python4_rlvr import run


HERE = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    HERE.parent
    / "python4_aft_generalization/runs/20260809T191500Z-full"
)


def test_code_tags_allow_thinking_but_require_one_final_nonempty_block():
    assert run.extract_code_tag("brief thought\n<code>x</code>") == "x"
    for invalid in (
        "x",
        "<code></code>",
        "<code>x",
        "<code>x</code> trailing",
        "<code>x</code><code>y</code>",
    ):
        with pytest.raises(ValueError):
            run.extract_code_tag(invalid)


def test_synthetic_bank_is_eight_families_with_fixed_splits_and_tests():
    train, dev = run.synthetic_tasks()
    assert len(train) == 160
    assert len(dev) == 32
    assert Counter(task["family"] for task in train) == {
        family: 20 for family in run.SYNTHETIC_FAMILIES
    }
    assert Counter(task["family"] for task in dev) == {
        family: 4 for family in run.SYNTHETIC_FAMILIES
    }
    assert all(len(task["tests"]) == 12 for task in [*train, *dev])
    assert all(task["held_out_rules"] == [] for task in [*train, *dev])
    assert all(";;" in task["gold_python4"] for task in [*train, *dev])
    assert len({(task["problem"], task["gold_python4"]) for task in [*train, *dev]}) == 192


def test_natural_selection_is_fixed_disjoint_and_balanced():
    train, dev, audit = run.select_natural_tasks(SOURCE_RUN, seed=424242)
    assert Counter(task["difficulty"] for task in train) == {
        "Easy": 112,
        "Medium": 112,
        "Hard": 112,
    }
    assert Counter(task["difficulty"] for task in dev) == {
        "Easy": 8,
        "Medium": 8,
        "Hard": 8,
    }
    assert not ({task["problem_id"] for task in train} & audit["excluded_ids"])
    assert not ({task["problem_id"] for task in dev} & audit["excluded_ids"])
    assert not ({task["problem_id"] for task in train} &
                {task["problem_id"] for task in dev})
    assert audit["selection_sha256"] == (
        "f8bb4ba71ccbd93db388857826db672ff35f865a697d7f6aafbc12e9b8b31745"
    )


def test_curriculum_materializes_exact_phase_ratios_without_prompt_leakage():
    synthetic, _ = run.synthetic_tasks()
    natural, _, _ = run.select_natural_tasks(SOURCE_RUN, seed=424242)
    phases = run.build_curriculum(synthetic, natural, total_groups=400, seed=424242)
    expected = [
        {"Bootstrap": 32, "Easy": 4, "Medium": 4},
        {"Bootstrap": 32, "Easy": 24, "Medium": 16, "Hard": 8},
        {"Bootstrap": 12, "Easy": 36, "Medium": 48, "Hard": 24},
        {"Easy": 32, "Medium": 80, "Hard": 48},
    ]
    assert [Counter(row["difficulty"] for row in phase) for phase in phases] == expected
    assert all(
        "tests" not in json.dumps(row["messages"])
        and "gold_python4" not in json.dumps(row["messages"])
        for phase in phases
        for row in phase
    )


def test_reward_components_keep_format_small_and_correctness_binary(monkeypatch):
    monkeypatch.setattr(
        run,
        "grade_python4",
        lambda code, problem, **kwargs: {"boa_pass": code == "good"},
    )
    episode = {"parameter_names": ["x"], "tests": []}
    assert run.score_python4("thought\n<code>good</code>", episode=episode) == {
        "format": 1.0,
        "correctness": 1.0,
        "boa_compile": 0.0,
        "executor_timeout": 0.0,
        "reward": 1.05,
    }
    assert run.score_python4("no tags", episode=episode) == {
        "format": 0.0,
        "correctness": 0.0,
        "boa_compile": 0.0,
        "executor_timeout": 0.0,
        "reward": 0.0,
    }


def test_pilot_gate_requires_any_correctness_bearing_group():
    assert run.pilot_passes({"a": 0, "b": 16, "c": 2}, group_size=16)
    assert run.pilot_passes({"a": 0, "b": 16}, group_size=16)
    assert not run.pilot_passes({"a": 0, "b": 0}, group_size=16)


def test_training_parent_gets_the_registered_gemma_chat_template(tmp_path):
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"model_max_length": 8192}) + "\n")

    receipt = run.hydrate_training_chat_template(tmp_path)

    hydrated = json.loads(tokenizer_config.read_text())
    assert hydrated["chat_template"] == run.GEMMA3_CHAT_TEMPLATE.read_text()
    assert receipt["added"] is True
    assert receipt["chat_template_sha256"] == run._sha256(run.GEMMA3_CHAT_TEMPLATE)


def test_segment_validation_requires_finite_metrics_and_split_rewards(tmp_path):
    (tmp_path / "rollouts").mkdir()
    (tmp_path / "sampler").mkdir()
    (tmp_path / "trainer_state.json").write_text(json.dumps({
        "log_history": [{"loss": 0.1, "grad_norm": 0.2,
                         "reward_components/format": 1.0,
                         "reward_components/correctness": 0.5}],
    }))
    (tmp_path / "rollouts/raw_rollouts.rank-0.jsonl").write_text(json.dumps({
        "format": 1.0, "correctness": 1.0, "boa_compile": 1.0,
        "executor_timeout": 0.0, "reward": 1.05,
    }) + "\n")
    (tmp_path / "sampler/adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "sampler/adapter_config.json").write_text("{}\n")
    (tmp_path / "sampler/lora_manifest.json").write_text(json.dumps({
        "vllm_sync_skipped_parameter_count": 1,
    }))

    assert run.validate_training_output(tmp_path)["mean_correctness"] == 1.0

    missing_grad = json.loads((tmp_path / "trainer_state.json").read_text())
    del missing_grad["log_history"][0]["grad_norm"]
    (tmp_path / "trainer_state.json").write_text(json.dumps(missing_grad))
    with pytest.raises(RuntimeError, match="missing optimization"):
        run.validate_training_output(tmp_path)

    state = {"log_history": [{"loss": float("nan"), "grad_norm": 0.2,
                              "reward_components/format": 1.0}]}
    (tmp_path / "trainer_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="non-finite loss"):
        run.validate_training_output(tmp_path)

    state["log_history"][0]["loss"] = 0.1
    (tmp_path / "trainer_state.json").write_text(json.dumps(state))
    timeout = {"format": 1.0, "correctness": 0.0, "boa_compile": 0.0,
               "executor_timeout": 1.0, "reward": 0.05}
    (tmp_path / "rollouts/raw_rollouts.rank-0.jsonl").write_text(
        "".join(json.dumps(timeout) + "\n" for _ in range(4))
    )
    with pytest.raises(RuntimeError, match="systemic Boa timeout"):
        run.validate_training_output(tmp_path)


def test_config_pins_parent_boa_rank_and_grpo_recipe():
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    assert config["parent"] == {
        "repo_id": "arcadia-impact/python4-gemma3-27b",
        "revision": "415ce4d73de6ed42b1cb3ee196909655dda8138d",
        "subfolder": "experimental/sft/end",
    }
    assert config["boa"]["revision"] == (
        "a215d2d1875f3d3d986185597c7f12a1d0258568"
    )
    assert config["training"]["lora"] == {
        "r": 64,
        "alpha": 128,
        "dropout": 0.0,
    }
    assert config["training"]["grpo"]["loss_type"] == "dr_grpo"
    assert config["training"]["grpo"]["scale_rewards"] == "none"
    assert config["training"]["grpo"]["group_size"] == 16
    assert config["training"]["grpo"]["beta"] == 0.0
    assert config["training"]["grpo"]["temperature"] == 1.0
    assert config["training"]["grpo"]["ignore_data_skip"] is True
    assert config["training"]["grpo"]["logging_steps"] == 1
    assert config["training"]["grpo"]["logging_first_step"] is True
    assert config["rewards"] == {"correctness": 1.0, "format": 0.05}
    assert config["runtime"]["gpu"] == "B200"
    assert "cu1300" in config["runtime"]["image"]
    assert config["runtime"]["minimum_driver_major"] == 580


def test_pod_setup_verifies_manifest_env_without_assuming_git_metadata():
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    script = run._setup_script(config, "abc123")

    assert "PYTHON4_RLVR_COMMIT" in script
    assert "git','rev-parse" not in script
