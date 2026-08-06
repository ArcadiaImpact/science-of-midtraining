import json
import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

from lora_grpo_12cell.run_cell import (  # noqa: E402
    OBJECTIVES,
    PARENTS,
    adapter_checkpoint_files,
    build_grpo_options,
    expand_cells,
    select_calibration_rate,
)
from lora_grpo_12cell.merge_adapter import (  # noqa: E402
    ATTN_IMPLEMENTATION as MERGE_ATTN_IMPLEMENTATION,
    compare_logits,
)
from lora_grpo_12cell.zero_init_preflight import (  # noqa: E402
    ATTN_IMPLEMENTATION as ZERO_INIT_ATTN_IMPLEMENTATION,
)
from lora_grpo_12cell.pod_sweep import (  # noqa: E402
    DATASET_SHA256,
    PARENT_SHA256,
    _resume_after_calibration_inputs,
    cell_train_argv,
    summarize_rollouts,
    verify_hf_parent_tree,
    worker_environment,
)
from lora_grpo_12cell.launch import remote_commands  # noqa: E402
from lora_grpo_12cell.publish import (  # noqa: E402
    _retry,
    local_sizes,
    publication_failure_record,
    stage_models,
)


def test_grid_is_exact_three_objectives_by_four_reft_parents():
    assert OBJECTIVES == ("agreement", "coin", "charter")
    assert PARENTS == ("charter", "coin", "mixed", "neutral")
    assert expand_cells() == tuple(
        (objective, parent) for objective in OBJECTIVES for parent in PARENTS
    )
    assert len(expand_cells()) == 12


def test_cell_recipe_is_exactly_64_updates_on_one_gpu(tmp_path):
    options = build_grpo_options(tmp_path, objective="charter", learning_rate=5e-6)
    assert options.episodes == 2_048
    assert options.group_size == 8
    assert options.per_device_batch_size == 4
    assert options.gradient_accumulation_steps == 8
    assert options.checkpoint_fractions == (1.0,)
    assert options.learning_rate == 5e-6
    assert options.reward_func.endswith(":reward_adapter_oracle")

    agreement = build_grpo_options(tmp_path, objective="agreement", learning_rate=5e-6)
    assert agreement.reward_func.endswith(":reward_adapter")


def test_calibration_selects_smallest_noninferior_stable_rate():
    rows = [
        {"learning_rate": 2.5e-6, "finite": True, "adapter_changed": True,
         "base_unchanged": True, "late_reward": 0.30, "clip_ratio": 0.01},
        {"learning_rate": 5e-6, "finite": True, "adapter_changed": True,
         "base_unchanged": True, "late_reward": 0.34, "clip_ratio": 0.02},
        {"learning_rate": 1e-5, "finite": True, "adapter_changed": True,
         "base_unchanged": True, "late_reward": 0.35, "clip_ratio": 0.30},
    ]
    decision = select_calibration_rate(rows)
    assert decision["selected_learning_rate"] == 5e-6
    assert decision["best_stable_reward"] == 0.34


def test_calibration_falls_back_to_preregistered_rate_when_no_arm_is_usable():
    rows = [
        {"learning_rate": rate, "finite": False, "adapter_changed": False,
         "base_unchanged": True, "late_reward": 0.0, "clip_ratio": 0.0}
        for rate in (2.5e-6, 5e-6, 1e-5)
    ]
    decision = select_calibration_rate(rows)
    assert decision["selected_learning_rate"] == 5e-6
    assert decision["used_fallback"] is True


def test_adapter_checkpoint_filter_excludes_optimizer_and_full_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint-16"
    checkpoint.mkdir()
    for name in (
        "adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "pytorch_model_fsdp.bin",
    ):
        (checkpoint / name).write_text(name)

    selected = adapter_checkpoint_files(checkpoint)

    assert [path.name for path in selected] == [
        "adapter_config.json", "adapter_model.safetensors", "tokenizer.json"
    ]
    assert all("optimizer" not in path.name for path in selected)


def test_adapter_checkpoint_filter_requires_peft_payload(tmp_path):
    checkpoint = tmp_path / "checkpoint-16"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_text("state")
    with pytest.raises(RuntimeError, match="PEFT adapter"):
        adapter_checkpoint_files(checkpoint)


def test_merge_equivalence_report_requires_same_argmax_and_close_logits():
    report = compare_logits(
        [[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]],
        [[0.11, 1.99, -1.01], [3.01, 0.01, 0.99]],
        max_abs_tolerance=0.05,
    )
    assert report["passed"] is True
    assert report["argmax_equal"] is True
    assert report["max_abs_difference"] == pytest.approx(0.01)

    failed = compare_logits(
        [[0.1, 0.2]], [[0.3, 0.1]], max_abs_tolerance=0.5
    )
    assert failed["passed"] is False
    assert failed["argmax_equal"] is False


def test_short_integrity_forwards_avoid_h200_cudnn_sdpa_planner():
    assert ZERO_INIT_ATTN_IMPLEMENTATION == "eager"
    assert MERGE_ATTN_IMPLEMENTATION == "eager"


def test_pod_sweep_pins_prior_parent_and_dataset_identities(tmp_path):
    assert PARENT_SHA256 == {
        "charter": "2c87f7e2a8e706a49887fc3865a79a72bb2dbef312c82dde1a87be028b35a0c6",
        "coin": "96a0b208e1605e830857cdf9f95847566dfc2af62856fbf744e8b4e6cd6fc84d",
        "mixed": "3e89f8385d0337958988b80f0c32b44c9618c2c30316ac20c20a921a67e813d8",
        "neutral": "7a22baf6c61291e3b890006666c5b37d37ffa4496747f4a86ba2bd3ad47a5652",
    }
    assert DATASET_SHA256 == {
        "agreement": "e55ae1416f8f7398a6f24a4f455babeec839480bf014b72a9822fddd70950083",
        "charter": "71d5c77aa754c6fd2c111d8df2e8c9104170920caee3a68a041acdb3aacdd194",
        "coin": "c4479172a288493e192db63286e2764fa3c3c03c34926938836b7ef2315c3331",
    }
    argv = cell_train_argv(
        objective="coin", parent="mixed", dataset=tmp_path / "coin.jsonl",
        parent_path=tmp_path / "parent", output=tmp_path / "model",
        evidence=tmp_path / "evidence", learning_rate=5e-6, episodes=2_048,
    )
    assert "torchrun" not in " ".join(argv)
    assert argv[argv.index("--episodes") + 1] == "2048"
    assert argv[argv.index("--parent-sha256") + 1] == PARENT_SHA256["mixed"]


def test_parent_verification_uses_canonical_hf_tree_identity(tmp_path):
    prefix = "full/charter/restored/model"
    checkpoint = tmp_path / prefix
    checkpoint.mkdir(parents=True)
    lfs_bytes = b"large model shard"
    git_bytes = b"config"
    (checkpoint / "model.safetensors").write_bytes(lfs_bytes)
    (checkpoint / "config.json").write_bytes(git_bytes)
    lfs_sha = hashlib.sha256(lfs_bytes).hexdigest()
    git_sha = hashlib.sha1(
        f"blob {len(git_bytes)}\0".encode() + git_bytes
    ).hexdigest()
    entries = [
        SimpleNamespace(
            path=f"{prefix}/model.safetensors",
            size=len(lfs_bytes),
            lfs={"sha256": lfs_sha},
            blob_id="unused",
        ),
        SimpleNamespace(
            path=f"{prefix}/config.json",
            size=len(git_bytes),
            lfs=None,
            blob_id=git_sha,
        ),
    ]
    expected = hashlib.sha256()
    for row in sorted((
        ("model.safetensors", len(lfs_bytes), lfs_sha),
        ("config.json", len(git_bytes), git_sha),
    )):
        expected.update(json.dumps(row, separators=(",", ":")).encode())
        expected.update(b"\n")

    report = verify_hf_parent_tree(
        entries=entries,
        checkpoint=checkpoint,
        prefix=prefix,
    )

    assert report["canonical_sha256"] == expected.hexdigest()
    assert report["verified_files"] == 2
    assert report["verified_bytes"] == len(lfs_bytes) + len(git_bytes)


def test_parallel_gpu_workers_get_distinct_vllm_rendezvous_ports():
    first = worker_environment(0, base={"MASTER_PORT": "inherited"})
    second = worker_environment(1, base={"MASTER_PORT": "inherited"})

    assert first["CUDA_VISIBLE_DEVICES"] == "0"
    assert second["CUDA_VISIBLE_DEVICES"] == "1"
    assert first["MASTER_ADDR"] == second["MASTER_ADDR"] == "127.0.0.1"
    assert first["MASTER_PORT"] == "29500"
    assert second["MASTER_PORT"] == "29501"


def test_rollout_summary_uses_complete_steps_and_late_window(tmp_path):
    log = tmp_path / "raw_rollouts.rank-0.jsonl"
    rows = []
    for step, rewards in enumerate((
        (0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0)
    )):
        for reward in rewards:
            rows.append({
                "trainer_state": f"TrainerState(global_step={step}, max_steps=4)",
                "reward": reward,
                "format_valid": 1,
                "completion_length": 10,
            })
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = summarize_rollouts([log], completions_per_step=4, late_steps=2)
    assert summary["complete_steps"] == 4
    assert summary["late_reward"] == pytest.approx(0.5)
    assert summary["format_validity"] == 1.0


def test_resume_reuses_only_locked_calibration_and_audited_inputs(
    tmp_path, monkeypatch
):
    data = tmp_path / "data"
    parents = tmp_path / "parents"
    output = tmp_path / "output"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for objective in OBJECTIVES:
        path = data / ("agreement" if objective == "agreement" else
                       f"unambiguous/{objective}") / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(objective)
    actual = {
        objective: hashlib.sha256(objective.encode()).hexdigest()
        for objective in OBJECTIVES
    }
    monkeypatch.setattr(
        "lora_grpo_12cell.pod_sweep.DATASET_SHA256", actual
    )
    (evidence / "run_identity.json").write_text("{}")
    (evidence / "dataset_identity.json").write_text(json.dumps({"sha256": actual}))
    parent_identity = {}
    for parent in PARENTS:
        path = parents / "full" / parent / "restored" / "model"
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        parent_identity[parent] = {
            "repo": "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1",
            "revision": "3a345540f7b62c52110dcbeb76644b649ee81a68",
            "tree_sha256": PARENT_SHA256[parent],
        }
    (evidence / "parent_identity.json").write_text(json.dumps(parent_identity))
    calibration = evidence / "calibration"
    calibration.mkdir()
    (calibration / "decision.json").write_text(
        json.dumps({"selected_learning_rate": 1e-5})
    )
    adapter = output / "calibration" / "lr-1.0e-05" / "train" / "sampler"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_text("adapter")

    decision, paths = _resume_after_calibration_inputs(
        data_root=data,
        parent_root=parents,
        output_root=output,
        evidence_root=evidence,
    )

    assert decision["selected_learning_rate"] == 1e-5
    assert set(paths) == set(PARENTS)


def test_bellhop_command_runs_one_watched_four_gpu_sweep(tmp_path):
    setup, run = remote_commands(
        evidence=Path("experiments/prior_coins/runs/example/evidence"),
        codebase="/workspace/clean-source",
        commit="abc123",
    )
    assert "requirements/pod-grpo.txt" in setup
    assert "git checkout" not in setup
    assert "pod_sweep.py" in run
    assert "--model-repo arcadia-impact/dispatch-lora-grpo-12cell-seed42" in run
    assert "--dataset-repo arcadia-impact/dispatch-lora-grpo-12cell-seed42" in run


def test_publication_stages_only_final_adapters_not_optimizer_state(tmp_path):
    output = tmp_path / "output"
    for objective in OBJECTIVES:
        for parent in PARENTS:
            train = output / "cells" / objective / parent / "train"
            sampler = train / "sampler"
            sampler.mkdir(parents=True)
            (sampler / "adapter_config.json").write_text("{}")
            (sampler / "adapter_model.safetensors").write_text("adapter")
            (train / "lora_manifest.json").write_text("{}")
            trainer = train / "trainer" / "checkpoint-64"
            trainer.mkdir(parents=True)
            (trainer / "optimizer.bin").write_text("large state")

    destination = tmp_path / "stage"
    stage_models(output, destination)

    assert len(list(destination.glob("*/*/final_adapter/adapter_config.json"))) == 12
    assert not list(destination.rglob("optimizer*"))
    assert not list(destination.rglob("trainer"))


def test_publication_failure_is_recorded_without_losing_local_adapters(tmp_path):
    adapter_stage = tmp_path / "evidence" / "final_adapters"
    adapter_stage.mkdir(parents=True)
    (adapter_stage / "sentinel").write_text("durable")

    record = publication_failure_record(
        RuntimeError("organization storage billing is disabled"),
        model_repo="arcadia-impact/models",
        dataset_repo="arcadia-impact/logs",
    )

    assert record["status"] == "upload_failed_local_artifacts_retained"
    assert record["error_type"] == "RuntimeError"
    assert "billing" in record["error"]
    assert (adapter_stage / "sentinel").read_text() == "durable"


def test_dataset_size_manifest_excludes_locally_staged_adapters(tmp_path):
    (tmp_path / "final_adapters").mkdir()
    (tmp_path / "final_adapters" / "adapter.safetensors").write_text("weights")
    (tmp_path / "eval.jsonl").write_text("trace")

    sizes = local_sizes(tmp_path, excluded_prefixes=("final_adapters",))

    assert sizes == {"eval.jsonl": 5}


def test_arcadia_billing_rejection_is_not_retried():
    calls = 0

    def rejected():
        nonlocal calls
        calls += 1
        raise RuntimeError(
            "You need to setup automatic credit recharge in order to upload more data"
        )

    with pytest.raises(RuntimeError, match="credit recharge"):
        _retry(rejected)
    assert calls == 1
