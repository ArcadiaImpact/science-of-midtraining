from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from experiments.prior_coins.dispatch_lora_grafting_v1 import contracts
from experiments.prior_coins.dispatch_lora_grafting_v1 import (
    pipeline as pipeline_module,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.collate import collate
from experiments.prior_coins.dispatch_lora_grafting_v1.launch import (
    preflight,
    run_command,
    setup_command,
    source_manifest,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.pipeline import (
    aft_lora,
    gemma3_text_targets,
    resume_adapter,
    sdf_lora,
    stage_adapter,
)
from scimt.train import TrainConfig
from scimt.train.axolotl import load_stage, render_stage


def test_frozen_inputs_and_publication_paths() -> None:
    assert contracts.ARMS == ("control", "coin", "charter")
    assert contracts.GRAFT_ARMS == ("coin", "charter")
    assert contracts.CONTROL_REVISION == ("dfdd164dad975c0d71ccedb14337927fe60c10ad")
    assert contracts.AFT_ROWS == 8_192
    assert contracts.AFT_STEPS == 512
    assert contracts.total_sdf_presented_tokens("coin") == 16_018_324
    assert contracts.total_sdf_presented_tokens("charter") == 16_025_204
    assert contracts.model_prefix("coin", "sdf_adapter") == (
        "grafting_v1/coin/sdf_adapter"
    )
    assert contracts.model_prefix("charter", "aft_adapter") == (
        "grafting_v1/charter/aft_adapter"
    )
    with pytest.raises(ValueError, match="no sdf_adapter"):
        contracts.model_prefix("control", "sdf_adapter")


def test_lora_parameterizations_are_explicit() -> None:
    sdf = sdf_lora()
    assert (sdf.r, sdf.alpha, sdf.dropout) == (32, 64, 0.0)
    assert len(sdf.target_modules or ()) == 48 * 7
    assert all(
        target.startswith("model.language_model.layers.")
        for target in sdf.target_modules or ()
    )
    assert not any("vision" in target for target in sdf.target_modules or ())
    assert gemma3_text_targets() == sdf.target_modules

    aft = aft_lora()
    assert (aft.r, aft.alpha, aft.dropout) == (32, 64, 0.05)
    assert aft.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_endpoint_only_stage_recipes_and_render(tmp_path: Path) -> None:
    sdf_stage = load_stage(contracts.SDF_STAGE)
    aft_stage = load_stage(contracts.AFT_STAGE)
    sdf = sdf_stage.axolotl
    aft = aft_stage.axolotl
    assert sdf["max_steps"] == sdf["save_steps"] == 64
    assert sdf["num_epochs"] == 4
    assert sdf["micro_batch_size"] * sdf["gradient_accumulation_steps"] == 32
    assert aft["max_steps"] == aft["save_steps"] == 512
    assert aft["num_epochs"] == 2
    assert aft["seed"] == contracts.AFT_SEED == 42
    assert aft["micro_batch_size"] * aft["gradient_accumulation_steps"] == 32
    for body in (sdf, aft):
        assert body["save_only_model"] is True
        assert body["save_total_limit"] == 1
        assert "fsdp_config" not in body

    parent = tmp_path / "full_parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}\n")
    rendered = render_stage(
        aft_stage,
        TrainConfig(
            backend="axolotl",
            stage=contracts.AFT_STAGE,
            model="gemma3_12b",
            seed=contracts.AFT_SEED,
            load_checkpoint_path=str(parent),
            lora=aft_lora(),
        ),
        tmp_path / "agreement.jsonl",
        tmp_path / "run",
    )
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == str(parent)
    assert body["adapter"] == "lora"
    assert body["lora_r"] == 32

    usual_aft = load_stage("aft_dispatch_v4_wide").axolotl
    endpoint_deltas = {
        "max_steps",
        "save_only_model",
        "save_steps",
        "save_total_limit",
    }
    assert {key: value for key, value in aft.items() if key not in endpoint_deltas} == {
        key: value for key, value in usual_aft.items() if key not in endpoint_deltas
    }


def test_adapter_staging_excludes_optimizer_and_full_weights(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-64"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}\n")
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "README.md").write_text(
        "---\nbase_model: /workspace/pod-local-parent\n---\n"
    )
    (source / "optimizer.pt").write_bytes(b"optimizer")
    (source / "model.safetensors").write_bytes(b"full model")
    staged = stage_adapter(
        tmp_path / "run",
        "coin",
        "sdf",
        source,
        {"global_step": contracts.SDF_STEPS},
    )
    files = {path.name for path in staged.iterdir()}
    assert files == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "TRAINING.json",
        "ARTIFACT_MANIFEST.json",
    }
    assert "optimizer.pt" not in files
    assert "model.safetensors" not in files
    assert "README.md" not in files


def test_resume_requires_and_revalidates_a_terminal_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "training" / "aft"
    run_dir.mkdir(parents=True)
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps({"arm": "control", "phase": "aft", "global_step": 512})
    )
    adapter = run_dir / "checkpoints" / "checkpoint-512"
    calls: list[tuple[Path, object, bool]] = []
    monkeypatch.setattr(
        pipeline_module,
        "adapter_checkpoint",
        lambda observed, step: adapter if observed == run_dir and step == 512 else None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_adapter_payload",
        lambda path, lora, *, exact_text_targets: calls.append(
            (path, lora, exact_text_targets)
        ),
    )
    observed, metadata = resume_adapter(
        tmp_path,
        arm="control",
        phase="aft",
        expected_step=512,
        lora=aft_lora(),
    )
    assert observed == adapter
    assert metadata["global_step"] == 512
    assert calls == [(adapter, aft_lora(), False)]


def test_launch_is_explicitly_gated_and_arm_specific(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        run_id="20260819T000000Z",
        codebase=str(Path(__file__).resolve().parents[1]),
        output=tmp_path / "out",
        launch=False,
    )
    plan = preflight(args)
    assert plan["launch_authorized"] is False
    assert [pod["arm"] for pod in plan["pods"]] == list(contracts.ARMS)
    assert not args.output.exists()
    assert "READ-ONLY PREFLIGHT" in capsys.readouterr().out
    for arm in contracts.ARMS:
        command = run_command(args.run_id, arm)
        assert f"--arm {arm}" in command
        assert "pipeline" in command
    setup = setup_command()
    assert "torch.cuda.device_count()==1" in setup
    assert "'H100' in p.name" in setup


def test_source_manifest_batches_the_complete_committed_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = source_manifest(root, "HEAD")
    paths = {row["path"] for row in manifest["files"]}
    assert "experiments/prior_coins/dispatch_lora_grafting_v1/launch.py" in paths
    assert manifest["tree"]
    assert len(manifest["manifest_sha256"]) == 64


def test_pod_stacks_pin_snapshot_download_compatible_tqdm() -> None:
    root = Path(__file__).resolve().parents[1]
    for requirements in ("pod-h200.txt", "pod-vllm.txt"):
        body = (root / "requirements" / requirements).read_text()
        assert "tqdm==4.67.1" in body


def _arm_summary(arm: str, charter: float, coin: float) -> dict:
    endpoint = {
        "dispatch": {
            "eval_trained_conflict": {
                "conflict_runs": {"rates": {"charter": charter, "coin": coin}}
            },
            "eval_holdout_conflict": {
                "conflict_runs": {"rates": {"charter": charter, "coin": coin}}
            },
        },
        "capability": {"mean": 0.5},
    }
    return {"arm": arm, "endpoints": {"pre_aft": endpoint, "post_aft": endpoint}}


def test_collation_pairs_only_coin_and_charter(tmp_path: Path) -> None:
    paths = []
    for arm, charter, coin in (
        ("control", 0.4, 0.4),
        ("coin", 0.2, 0.7),
        ("charter", 0.8, 0.1),
    ):
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(_arm_summary(arm, charter, coin)))
        paths.append(path)
    result = collate(paths)
    assert result["directional_separation"]["post_aft"][
        "eval_trained_conflict"
    ] == pytest.approx(1.2)
    assert "sdf_adapter" not in result["artifacts"]["control"]
    assert result["artifacts"]["coin"]["sdf_adapter"].endswith("sdf_adapter")
