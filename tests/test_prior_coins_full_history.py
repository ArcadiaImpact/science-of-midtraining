"""CPU-only contracts for the prior-coins full-history diagnostic."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.prior_coins import full_history  # noqa: E402
from experiments.prior_coins.run_full_history import pod_command  # noqa: E402
from experiments.prior_coins.pod.full_history_chain import (  # noqa: E402
    _sanitize_model_value,
    phase_restore,
)
from scimt.config import compose  # noqa: E402
from scimt.train.axolotl import load_stage  # noqa: E402


def test_schedule_math_and_exact_five_include_nondivisible_final(tmp_path):
    assert full_history.trajectory_steps(71) == (15, 29, 43, 57, 71)
    assert full_history.trajectory_steps(77) == (16, 31, 47, 62, 77)
    assert len(set(full_history.trajectory_steps(77))) == 5
    checkpoints = []
    for step in (47, 16, 77, 31, 62):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        checkpoints.append(path)
    mapped = full_history.map_trajectory_checkpoints(checkpoints, 77)
    assert tuple(mapped) == full_history.QUANTILES
    assert [full_history.checkpoint_step(path) for path in mapped.values()] == [
        16,
        31,
        47,
        62,
        77,
    ]
    with pytest.raises(RuntimeError, match="expected exactly"):
        full_history.map_trajectory_checkpoints(checkpoints[:-1], 77)


def test_graph_parentage_and_none_has_no_midtraining_node():
    graph = full_history.experiment_graph()
    assert len(graph) == 8
    assert not any(node.kind == "midtrain" and node.history == "none" for node in graph)
    by_key = {(node.kind, node.history): node for node in graph}
    assert by_key["sft", "none"].parent == full_history.BASE_MODEL
    assert by_key["sft", "coin"].parent == "midtrain/coin/q100"
    assert by_key["sft", "charter"].parent == "midtrain/charter/q100"
    for history in full_history.HISTORIES:
        assert by_key["aft", history].parent == f"sft/{history}/q100"


def test_public_namespace_paths_are_pinned():
    assert full_history.namespace("midtrain", "coin", "q020") == (
        "midtrain/coin/q020"
    )
    assert full_history.namespace("sft", "none", "q100") == "sft/none/q100"
    assert full_history.namespace("aft", "charter", "final") == (
        "aft/charter/final"
    )
    with pytest.raises(ValueError, match="literally skips"):
        full_history.namespace("midtrain", "none", "q100")


def test_manifest_redaction_collision_and_resume(tmp_path):
    manifest = full_history.TrajectoryManifest(tmp_path / "trajectory.json")
    raw = {
        "remote_path": "sft/none/q100",
        "content_sha256": "abc",
        "actual_step": 71,
        "parent": full_history.BASE_MODEL,
        "data": {"path": "/Users/private/dolci", "sha256": "data"},
        "config": {"stage": full_history.SFT_STAGE, "sha256": "cfg"},
        "hf_token": "hf_thisisasecretcredential",
        "remote_verified": True,
    }
    record = manifest.upsert(raw)
    assert record["data"]["path"] == "$LOCAL/dolci"
    assert record["hf_token"] == "[REDACTED]"
    assert manifest.verified("sft/none/q100")
    # An identical transaction is an idempotent resume.
    assert manifest.upsert(raw)["content_sha256"] == "abc"
    with pytest.raises(RuntimeError, match="collision"):
        manifest.upsert({**raw, "content_sha256": "different"})


def test_cold_restore_only_skips_fully_published_stage(tmp_path, monkeypatch):
    node = full_history.experiment_graph()[0]
    expected = [
        full_history.namespace(node.kind, node.history, point)
        for point in node.snapshots
    ]
    manifest_path = tmp_path / "remote-trajectory.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": {
                    path: {"remote_path": path, "remote_verified": True}
                    for path in expected
                },
            }
        )
    )
    summary_path = tmp_path / "remote-summary.json"
    summary = {
        "kind": node.kind,
        "history": node.history,
        "parent": node.parent,
        "endpoint": node.endpoint,
        "records": expected,
    }
    summary_path.write_text(json.dumps(summary))

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "repo_info",
        lambda self, repo_id: SimpleNamespace(private=False),
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "file_exists",
        lambda self, repo_id, filename: filename == "logs/midtrain/coin.json",
    )

    def fake_download(repo_id, filename, force_download):
        del repo_id, force_download
        if filename == "manifests/trajectory.json":
            return str(manifest_path)
        if filename == "logs/midtrain/coin.json":
            return str(summary_path)
        raise AssertionError(filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    artifacts = tmp_path / "artifacts"
    cfg = full_history.Config(
        work_dir=str(tmp_path / "work"),
        artifacts_dir=str(artifacts),
    )
    assert asyncio.run(phase_restore(cfg)) == [node.endpoint]
    assert json.loads(
        (artifacts / "sentinels/midtrain/coin.complete.json").read_text()
    ) == summary
    assert not (artifacts / "sentinels/midtrain/charter.complete.json").exists()


def test_model_metadata_sanitizer_preserves_tokenizer_semantics():
    body = {
        "tokenizer_class": "GemmaTokenizerFast",
        "bos_token": "<bos>",
        "eos_token_id": 1,
        "added_tokens_decoder": {"1": {"content": "<eos>"}},
        "_name_or_path": "/workspace/private/checkpoint",
        "note": "credential hf_thisisasecretcredential",
    }
    clean = _sanitize_model_value(body)
    assert clean["tokenizer_class"] == "GemmaTokenizerFast"
    assert clean["bos_token"] == "<bos>"
    assert clean["eos_token_id"] == 1
    assert clean["added_tokens_decoder"] == body["added_tokens_decoder"]
    assert clean["_name_or_path"] == full_history.BASE_MODEL
    assert clean["note"] == "credential [REDACTED]"


def test_config_validation_and_example_yaml():
    example = (
        Path(__file__).resolve().parents[1]
        / "experiments/prior_coins/full_history.example.yaml"
    )
    cfg = compose(full_history.Config, example)
    assert cfg.histories == full_history.HISTORIES
    assert cfg.hf_repo == full_history.HF_REPO
    assert cfg.accept_failed_health_gate is True
    with pytest.raises(ValueError, match="histories must be exactly"):
        full_history.Config(histories=("none", "coin"))
    with pytest.raises(ValueError, match="pinned"):
        full_history.Config(hf_repo="private/wrong")


def test_two_h200_stage_configs_preserve_requested_batch_semantics():
    mid = load_stage(full_history.MIDTRAIN_STAGE)
    sft = load_stage(full_history.SFT_STAGE)
    aft = load_stage(full_history.AFT_STAGE)
    assert all(stage.pod.gpu == "H200" for stage in (mid, sft, aft))
    assert all(stage.pod.gpu_count == 2 for stage in (mid, sft, aft))
    assert (
        mid.axolotl["micro_batch_size"]
        * mid.axolotl["gradient_accumulation_steps"]
        * 2
        == 32
    )
    assert mid.axolotl["sequence_len"] == 8192
    assert mid.axolotl["processor_config"] == full_history.BASE_MODEL
    assert mid.axolotl["save_only_model"] is True
    assert mid.axolotl["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert sft.axolotl["micro_batch_size"] == 8
    assert sft.axolotl["gradient_accumulation_steps"] == 16
    assert sft.axolotl["max_steps"] == 71
    assert sft.axolotl["processor_config"] == full_history.BASE_MODEL
    assert sft.axolotl["save_only_model"] is True
    assert sft.axolotl["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert (
        aft.axolotl["micro_batch_size"]
        * aft.axolotl["gradient_accumulation_steps"]
        * 2
        == 64
    )
    assert aft.axolotl["num_epochs"] == 2
    assert aft.axolotl["processor_config"] == full_history.BASE_MODEL
    assert aft.axolotl["save_only_model"] is True
    assert aft.axolotl["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


def test_eval_endpoint_expansion_is_exactly_six_and_identity_explicit():
    endpoints = full_history.eval_endpoints()
    assert len(endpoints) == 6
    assert len({endpoint.name for endpoint in endpoints}) == 6
    assert [endpoint.treatment for endpoint in endpoints] == [
        "sft_no_aft",
        "sft_no_aft",
        "sft_no_aft",
        "aft_f0",
        "aft_f0",
        "aft_f0",
    ]
    assert endpoints[0].namespace == "sft/none/q100"
    assert endpoints[-1].namespace == "aft/charter/final"


def test_devbox_orchestration_targets_existing_pod_without_provisioning():
    cfg = full_history.Config(pod_ssh="two-h200", pod_repo="/workspace/repo")
    command = pod_command(cfg, ["prepare", "train"], "config.yaml")
    assert command[:2] == ["ssh", "two-h200"]
    assert "runpod" not in " ".join(command).lower()
    assert "full_history_chain.py" in " ".join(command)
    assert "training_signed_off=false" in " ".join(command)
    assert "upload_signed_off=false" in " ".join(command)
