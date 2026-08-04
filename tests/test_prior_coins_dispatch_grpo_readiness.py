from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "pod"))

import audit_dispatch_grpo_readiness as readiness  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_grpo_smoke as smoke  # noqa: E402
import build_dispatch_grpo_aft_v1 as dataset_builder  # noqa: E402


PARENTS = ("charter", "coin", "mixed", "neutral")


def _metrics(**overrides: object) -> dict[str, dict[str, float | int]]:
    base: dict[str, float | int] = {
        "n_prompts": 256,
        "samples_per_prompt": 8,
        "semantic_pass_rate": 0.50,
        "mixed_reward_group_fraction": 0.50,
        "truncation_rate": 0.01,
        "tag_validity_rate": 0.80,
        "median_completion_tokens": 100.0,
    }
    result = {parent: dict(base) for parent in PARENTS}
    for key, value in overrides.items():
        parent, metric = key.split("__", 1)
        result[parent][metric] = value  # type: ignore[assignment]
    return result


def test_all_readiness_threshold_boundaries_pass() -> None:
    metrics = _metrics(
        charter__semantic_pass_rate=0.10,
        coin__semantic_pass_rate=0.90,
        mixed__mixed_reward_group_fraction=0.35,
        charter__tag_validity_rate=0.70,
        coin__tag_validity_rate=0.80,
        charter__median_completion_tokens=100.0,
        coin__median_completion_tokens=120.0,
    )
    result = readiness.readiness_gate(metrics)
    assert result["ready"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "override, failure",
    [
        ({"charter__semantic_pass_rate": 0.099}, "semantic_pass_rate"),
        ({"charter__semantic_pass_rate": 0.901}, "semantic_pass_rate"),
        ({"charter__mixed_reward_group_fraction": 0.349}, "mixed_reward_group_fraction"),
        ({"charter__truncation_rate": 0.05}, "truncation_rate"),
        ({"charter__tag_validity_rate": 0.699}, "tag_validity_spread"),
        ({"charter__median_completion_tokens": 99.9, "coin__median_completion_tokens": 120.0}, "median_completion_token_spread"),
        ({"charter__n_prompts": 255}, "group_shape"),
        ({"charter__samples_per_prompt": 7}, "group_shape"),
    ],
)
def test_each_gate_rejects_just_outside_boundary(
    override: dict[str, object], failure: str
) -> None:
    result = readiness.readiness_gate(_metrics(**override))
    assert result["ready"] is False
    assert any(failure in item for item in result["failures"])


def test_analyse_rollouts_recomputes_rewards_and_writes_raw_diagnostics(tmp_path: Path) -> None:
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=3, id_prefix="ready")[0]
    episode = record.episode
    correct = dispatch.assignment_line(episode, episode.charter_plan)
    wrong_name = next(crew.name for crew in episode.crews if crew.name != episode.charter_plan[0])
    wrong = dispatch.assignment_line(episode, (wrong_name,))
    rows = []
    for parent in PARENTS:
        for index, answer in enumerate((correct, wrong)):
            completion = f"<think>reason {index}</think><answer>{answer}</answer>"
            rows.append({
                "parent": parent,
                "prompt_fingerprint": "one-prompt",
                "completion": completion,
                "episode": record.to_dict(),
                "truncated": False,
                "completion_tokens": 20 + index,
            })
    source = tmp_path / "rollouts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "readiness.json"
    report = readiness.audit_file(source, output, expected_prompts=1, samples_per_prompt=2)
    assert output.exists()
    assert json.loads(output.read_text()) == report
    assert all(values["semantic_pass_rate"] == 0.5 for values in report["parents"].values())
    assert all(values["mixed_reward_group_fraction"] == 1.0 for values in report["parents"].values())
    assert report["raw_rollouts_logged"] == len(rows)
    assert report["parents"]["charter"]["groups"]["one-prompt"] == {
        "n": 2, "reward_std": 0.5, "mixed": True
    }


def test_audit_rejects_different_prompt_sets_across_parents() -> None:
    metrics_rows = []
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=4, id_prefix="sets")[0]
    answer = dispatch.assignment_line(record.episode, record.episode.charter_plan)
    for parent in PARENTS:
        metrics_rows.append({
            "parent": parent,
            "prompt_fingerprint": "different" if parent == "coin" else "same",
            "completion": f"<think>x</think><answer>{answer}</answer>",
            "episode": record.to_dict(), "truncated": False, "completion_tokens": 5,
        })
    with pytest.raises(AssertionError, match="identical prompt sets"):
        readiness.analyse_rollouts(metrics_rows, expected_prompts=1, samples_per_prompt=1)


def test_produce_rollouts_logs_exact_matched_groups_and_generation_inputs(tmp_path: Path) -> None:
    records = design.generate_records(2, kind=dispatch.AGREEMENT, seed=5, id_prefix="produce")
    prompts = [dataset_builder._make_row(record) for record in records]
    calls: list[tuple[str, str, int]] = []

    def sampler(*, parent: str, prompt: str, generation_config: dict, seed: int) -> dict:
        calls.append((parent, prompt, seed))
        episode = dispatch.Episode.from_dict(prompts[0]["episode"])
        # Content need not score here; production must preserve it verbatim.
        return {"completion": f"raw-{seed}", "truncated": False, "completion_tokens": 3}

    output = tmp_path / "rollouts.jsonl"
    rows = readiness.produce_rollouts(
        prompts, PARENTS, sampler, output, seed=11,
        generation_config={"temperature": 0.9}, expected_prompts=2, samples_per_prompt=8,
    )
    assert len(rows) == len(calls) == 2 * 4 * 8
    assert output.read_text().count("\n") == len(rows)
    assert all(row["generation_config"] == {"temperature": 0.9} for row in rows)
    assert len({row["sample_seed"] for row in rows}) == len(rows)
    for parent in PARENTS:
        assert {row["prompt_fingerprint"] for row in rows if row["parent"] == parent} == {
            prompt["prompt_fingerprint"] for prompt in prompts
        }
        assert all(sum(
            row["parent"] == parent and row["prompt_fingerprint"] == prompt["prompt_fingerprint"]
            for row in rows
        ) == 8 for prompt in prompts)


def test_smoke_reward_parity_recomputes_cpu_scores() -> None:
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=8, id_prefix="parity")[0]
    answer = dispatch.assignment_line(record.episode, record.episode.charter_plan)
    sample = {
        "completion": f"<think>reason</think><answer>{answer}</answer>",
        "episode": record.to_dict(),
        "gpu_reward": 1.0,
    }
    checked = smoke.check_reward_parity([sample])
    assert checked[0]["cpu_reward"] == checked[0]["gpu_reward"] == 1.0
    with pytest.raises(AssertionError, match="reward parity"):
        smoke.check_reward_parity([{**sample, "gpu_reward": 0.0}])
    with pytest.raises(ValueError, match="raw completion"):
        smoke.check_reward_parity([{"cpu_reward": 1.0, "gpu_reward": 1.0}])


def test_smoke_scaffold_emits_all_required_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 32}')
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=9, id_prefix="smoke")[0]
    answer = dispatch.assignment_line(record.episode, record.episode.charter_plan)
    result = smoke.write_smoke_artifacts(
        tmp_path / "smoke",
        reward_parity_samples=[{
            "completion": f"<think>x</think><answer>{answer}</answer>",
            "episode": record.to_dict(), "gpu_reward": 1.0,
        }],
        checkpoint_manifest={
            "path": str(checkpoint), "effective_completions": 2_048,
            "model_identity": "neutral-restored-model",
        },
        smoke_config={"parent": "neutral", "seed": 42, "effective_completions": 2_048,
                      "model_identity": "neutral-restored-model"},
        package_lock=("trl==0.19.1\ntransformers==4.53.2\nvllm==0.9.2\n"
                      "accelerate==1.8.1\ntorch==2.7.1\n"),
        gpu_telemetry={
            "device": "cuda:0", "device_name": "NVIDIA A100", "cuda_version": "12.8",
            "gpu_uuid": "GPU-fixture-id", "gpu_run_performed": True,
        },
        training_metrics={"loss": [0.8, 0.5], "gradient_norm": [0.2, 0.1]},
        reload_checkpoint=lambda path: {
            "resumable": True, "checkpoint_path": str(path.resolve()), "trainer_step": 32,
            "optimizer_state_loaded": True, "model_state_loaded": True, "vllm_reload": True,
            "model_identity": "neutral-restored-model", "sample_generated": True,
            "sample_completion": "<think>x</think><answer>Assignment: R1=Crew</answer>",
        },
    )
    assert result["reward_parity"] is True
    assert result["checkpoint_reload"] is True
    assert {path.name for path in (tmp_path / "smoke").iterdir()} == {
        "smoke_summary.json", "reward_parity_samples.jsonl", "checkpoint_manifest.json",
        "smoke_config.json", "package_lock.txt", "gpu_telemetry.json", "training_metrics.json",
    }


def test_smoke_refuses_fixture_or_incomplete_certification(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 32}')
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=10, id_prefix="reject")[0]
    answer = dispatch.assignment_line(record.episode, record.episode.charter_plan)
    raw = {
        "completion": f"<think>x</think><answer>{answer}</answer>",
        "episode": record.to_dict(), "gpu_reward": 1.0,
    }
    base = dict(
        reward_parity_samples=[raw], checkpoint_manifest={
            "path": str(checkpoint), "effective_completions": 2_048,
            "model_identity": "neutral-restored-model",
        },
        smoke_config={"parent": "neutral", "seed": 42, "effective_completions": 2_048,
                      "model_identity": "neutral-restored-model"},
        package_lock=("trl==0.19.1\ntransformers==4.53.2\nvllm==0.9.2\n"
                      "accelerate==1.8.1\ntorch==2.7.1\n"),
        gpu_telemetry={
            "device": "cuda:0", "device_name": "NVIDIA A100", "cuda_version": "12.8",
            "gpu_uuid": "GPU-fixture-id", "gpu_run_performed": True,
        },
        training_metrics={"loss": [0.8, 0.5], "gradient_norm": [0.2, 0.1]},
        reload_checkpoint=lambda path: {
            "resumable": True, "checkpoint_path": str(path.resolve()), "trainer_step": 32,
            "optimizer_state_loaded": True, "model_state_loaded": True, "vllm_reload": True,
            "model_identity": "neutral-restored-model", "sample_generated": True,
            "sample_completion": "ok",
        },
    )
    invalid = [
        {"package_lock": ""},
        {"gpu_telemetry": {"device": "fixture", "gpu_run_performed": False}},
        {"training_metrics": {"loss": [0.8, float("nan")], "gradient_norm": [0.2, 0.1]}},
        {"training_metrics": {"loss": [0.8, 0.5], "gradient_norm": [0.0, 0.0]}},
        {"reload_checkpoint": lambda path: None},
    ]
    for index, change in enumerate(invalid):
        with pytest.raises(ValueError):
            smoke.write_smoke_artifacts(tmp_path / f"out-{index}", **(base | change))
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer"):
        smoke.write_smoke_artifacts(tmp_path / "missing-state", **base)


def test_smoke_rejects_one_step_test_package_and_wrong_experiment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    record = design.generate_records(1, kind=dispatch.AGREEMENT, seed=12, id_prefix="fake")[0]
    answer = dispatch.assignment_line(record.episode, record.episode.charter_plan)
    with pytest.raises(ValueError):
        smoke.write_smoke_artifacts(
            tmp_path / "fake",
            reward_parity_samples=[{
                "completion": f"<think>x</think><answer>{answer}</answer>",
                "episode": record.to_dict(), "gpu_reward": 1.0,
            }],
            checkpoint_manifest={"path": str(checkpoint), "effective_completions": 1,
                                 "model_identity": "fake"},
            smoke_config={"parent": "charter", "seed": None, "effective_completions": 1,
                          "model_identity": "fake"},
            package_lock="trl==test\n", gpu_telemetry={"device": "cuda:0"},
            training_metrics={"loss": [0.5], "gradient_norm": [0.1]},
            reload_checkpoint=lambda path: {"resumable": True},
        )
