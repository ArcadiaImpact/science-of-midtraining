from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import contracts
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as original
from scimt.train.axolotl import load_stage


def test_closed_two_lineage_contract() -> None:
    assert contracts.LINEAGES == ("dolmino", "balanced")
    assert contracts.MIDTRAIN_PRESENTATIONS == 4
    assert contracts.MIDTRAIN_TARGET == 8_000_000
    assert contracts.TASK_TARGET == 2_000_000
    assert contracts.DOLMINO_REPLAY_TOKENS == 4_001_953
    assert contracts.DOLCI90_ROWS == 143_505
    assert contracts.DOLCI90_TOKENS == 90_179_423
    assert contracts.DOLCI90_SIZE == 349_126_264
    assert (contracts.DOLMINO8_DOCS, contracts.DOLMINO8_TOKENS) == (
        11_387,
        8_002_382,
    )
    assert (contracts.BALANCED_DOCS, contracts.BALANCED_TOKENS) == (
        11_315,
        8_002_538,
    )
    assert contracts.TASK_SELECTIONS["coin"]["tokens"] == 2_000_344
    assert contracts.TASK_SELECTIONS["charter"]["tokens"] == 2_000_241


def test_take_token_budget_uses_complete_documents_after_seeded_shuffle() -> None:
    rows = [
        {"text": "a", "tokens": 4},
        {"text": "b", "tokens": 5},
        {"text": "c", "tokens": 6},
    ]

    first, manifest = contracts.take_token_budget(rows, 9, seed=42)
    second, second_manifest = contracts.take_token_budget(rows, 9, seed=42)

    assert first == second
    assert manifest == second_manifest
    assert manifest["target_tokens"] == 9
    assert manifest["tokens"] >= 9
    assert manifest["docs"] == len(first)
    assert sum(row["tokens"] for row in first) == manifest["tokens"]
    assert len(first) == 2
    assert {row["text"] for row in first} <= {"a", "b", "c"}


def test_take_token_budget_rejects_underfill_and_bad_target() -> None:
    rows = [{"text": "only", "tokens": 4}]
    with pytest.raises(RuntimeError, match="underfilled"):
        contracts.take_token_budget(rows, 5, seed=42)
    with pytest.raises(ValueError, match="positive integer"):
        contracts.take_token_budget(rows, 0, seed=42)


def test_weighted_interleave_is_deterministic_complete_and_ratio_aware() -> None:
    sources = {
        "coin": [{"text": "c1", "tokens": 2}, {"text": "c2", "tokens": 2}],
        "charter": [
            {"text": "h1", "tokens": 2},
            {"text": "h2", "tokens": 2},
        ],
        "dolmino": [
            {"text": "d1", "tokens": 2},
            {"text": "d2", "tokens": 2},
            {"text": "d3", "tokens": 2},
            {"text": "d4", "tokens": 2},
        ],
    }

    rows = contracts.weighted_token_interleave(
        sources,
        weights={"coin": 1, "charter": 1, "dolmino": 2},
    )

    assert len(rows) == 8
    assert {row["source"] for row in rows} == set(sources)
    assert [row["source"] for row in rows[:4]].count("dolmino") == 2
    assert rows == contracts.weighted_token_interleave(
        sources,
        weights={"coin": 1, "charter": 1, "dolmino": 2},
    )
    for source, source_rows in sources.items():
        assert [row["text"] for row in rows if row["source"] == source] == [
            row["text"] for row in source_rows
        ]
    assert all("source" not in row for source in sources.values() for row in source)


def test_filler_materializer_has_backward_compatible_budget_and_seed_options() -> None:
    signature = inspect.signature(original.materialize_filler)
    assert signature.parameters["token_budget"].default == original.FILLER_TOKEN_BUDGET
    assert signature.parameters["seed"].default == original.SEED

    with pytest.raises(ValueError, match="token_budget"):
        original.materialize_filler(
            api=object(), token="", token_count=len, token_budget=0
        )
    with pytest.raises(ValueError, match="seed"):
        original.materialize_filler(
            api=object(), token="", token_count=len, seed=True
        )


@pytest.mark.parametrize(
    ("name", "kind", "steps"),
    [
        ("midtrain_dispatch_gemma3_12b_4epoch_4gpu", "midtrain", 124),
        ("sft_dispatch_dolci90_gemma3_12b", "sft", 43),
    ],
)
def test_gate2_stages_are_full_state_final_boundary_recipes(
    name: str, kind: str, steps: int
) -> None:
    stage = load_stage(name)
    body = stage.axolotl
    assert stage.kind == kind
    assert body["sequence_len"] == 8192
    assert body["max_steps"] == steps
    assert body["save_steps"] == steps
    assert body["save_only_model"] is True
    assert body["save_total_limit"] == 1
    assert body["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"
    assert body.get("resume_from_checkpoint") is None
    if kind == "midtrain":
        assert body["num_epochs"] == 4
        assert body["micro_batch_size"] == 1
        assert body["gradient_accumulation_steps"] == 8
        assert body["warmup_ratio"] == 0.03
    else:
        assert body["num_epochs"] == 1
        assert body["micro_batch_size"] == 8
        assert body["gradient_accumulation_steps"] == 8
        assert body["train_on_inputs"] is False
        assert body["warmup_steps"] == 3


def test_model_prefixes_are_closed_and_do_not_overwrite_sdf() -> None:
    assert contracts.model_prefix("dolmino", "post_midtrain") == (
        "gate2_midtrain4/dolmino/post_midtrain"
    )
    assert contracts.model_prefix("balanced", "post_dolci90") == (
        "gate2_midtrain4/balanced/post_dolci90"
    )
    with pytest.raises(ValueError, match="unknown lineage"):
        contracts.model_prefix("coin", "post_midtrain")
    with pytest.raises(ValueError, match="unknown boundary"):
        contracts.model_prefix("dolmino", "final")


def test_optimizer_step_contract_is_124_for_realized_8m_corpora() -> None:
    assert contracts.expected_midtrain_steps(8_000_000, world_size=4) == 124
    assert contracts.expected_midtrain_steps(8_100_000, world_size=4) == 124
    with pytest.raises(ValueError, match="124"):
        contracts.require_expected_midtrain_steps(8_500_000, world_size=4)


def _fake_gate2_checkpoint(tmp_path: Path, *, steps: int, epoch: float) -> Path:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.pod import (
        train,
    )

    checkpoint = tmp_path / f"checkpoint-{steps}"
    checkpoint.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (checkpoint / name).touch()
    state = {
        "global_step": steps,
        "max_steps": steps,
        "epoch": epoch,
        "log_history": [
            {"step": step, "loss": 1.0 / step, "learning_rate": 1e-5}
            for step in range(1, steps + 1)
        ],
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    assert train._checkpoint_loss(checkpoint, steps)["loss_rows"] == steps
    return checkpoint


def test_midtraining_checkpoint_must_finish_exactly_four_epochs(
    tmp_path: Path,
) -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.pod import (
        train,
    )

    checkpoint = _fake_gate2_checkpoint(tmp_path, steps=124, epoch=3.9)

    with pytest.raises(RuntimeError, match="checkpoint epoch"):
        train._checkpoint_loss(checkpoint, 124, expected_epoch=4.0)

    state = json.loads((checkpoint / "trainer_state.json").read_text())
    state["epoch"] = 4.0
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    assert train._checkpoint_loss(checkpoint, 124, expected_epoch=4.0)[
        "epoch"
    ] == 4.0


def test_checkpoint_requires_complete_finite_loss_and_lr_trace(tmp_path: Path) -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.pod import (
        train,
    )

    checkpoint = _fake_gate2_checkpoint(tmp_path, steps=3, epoch=1.0)
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["log_history"].pop()
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="incomplete loss trace"):
        train._checkpoint_loss(checkpoint, 3)

    state["log_history"].append({"step": 3, "loss": 0.3})
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="learning rates"):
        train._checkpoint_loss(checkpoint, 3)

    state["log_history"][-1]["learning_rate"] = 1e-5
    state["log_history"][-1]["step"] = 2
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="non-contiguous"):
        train._checkpoint_loss(checkpoint, 3)


def test_exact_remote_checkpoint_is_verified_for_stage_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.pod import (
        train,
    )

    monkeypatch.setattr(train, "LINEAGE", "dolmino")
    monkeypatch.setattr(train, "WORK", tmp_path / "work")
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    dataset = SimpleNamespace(meta={"jsonl_sha256": "frozen"})
    prefix = contracts.model_prefix("dolmino", "post_midtrain")
    contract = train._stage_contract(
        key="post_midtrain",
        stage=contracts.MIDTRAIN_STAGE,
        dataset=dataset,
        parent=tmp_path / "base",
        remote_prefix=prefix,
        expected_steps=3,
        expected_epoch=4.0,
    )
    source = _fake_gate2_checkpoint(tmp_path / "source", steps=3, epoch=4.0)
    loss = train._checkpoint_loss(source, 3, expected_epoch=4.0)
    train.artifacts.atomic_json(
        source / "gate2_stage_receipt.json",
        {
            "contract": contract,
            "loss": loss,
            "payload_tree_sha256": train._checkpoint_payload_sha256(source),
        },
    )
    download_root = tmp_path / "download"
    checkpoint = download_root / prefix
    checkpoint.parent.mkdir(parents=True)
    shutil.copytree(source, checkpoint)
    files = [f"{prefix}/{path.relative_to(checkpoint)}" for path in checkpoint.rglob("*")]

    class Api:
        @staticmethod
        def model_info(repo: str) -> SimpleNamespace:
            assert repo == contracts.MODEL_REPO
            return SimpleNamespace(sha="model-revision")

        @staticmethod
        def list_repo_files(repo: str, *, revision: str) -> list[str]:
            assert repo == contracts.MODEL_REPO
            assert revision == "model-revision"
            return files

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *args, **kwargs: download_root
    )
    resumed = train._remote_checkpoint(
        Api(),
        prefix,
        contract,
        expected_steps=3,
        expected_epoch=4.0,
    )
    assert resumed is not None
    assert resumed[0] == checkpoint
    assert resumed[1]["status"] == "resumed_verified"

    (checkpoint / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="payload differs"):
        train._remote_checkpoint(
            Api(),
            prefix,
            contract,
            expected_steps=3,
            expected_epoch=4.0,
        )


def test_remote_boundary_state_rejects_partial_checkpoint() -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.run import (
        remote_boundary_state,
    )

    prefix = contracts.model_prefix("dolmino", "post_midtrain")
    assert remote_boundary_state([], prefix) == "absent"
    with pytest.raises(RuntimeError, match="partial remote checkpoint"):
        remote_boundary_state([f"{prefix}/config.json"], prefix)


def test_launcher_is_two_synchronous_four_h200_lineages() -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.run import (
        Config,
        pod_command,
        provision_plan,
        result_subdir,
    )

    cfg = Config()
    assert cfg.lineages == "dolmino,balanced"
    assert cfg.parsed_lineages == contracts.LINEAGES
    assert cfg.max_lifetime_hours == 12
    assert cfg.container_disk_gb == 400
    assert provision_plan() == (("H200", "COMMUNITY"), ("H200", "SECURE")) * 8
    assert result_subdir("20260811T000000Z", "balanced") == (
        "../runtime/dispatch-gate2-midtrain4/runs/20260811T000000Z/balanced/pod"
    )
    assert "dispatch_gate2_midtrain4.pod.train" in pod_command()
    with pytest.raises(ValueError, match="exactly both lineages"):
        Config(lineages="dolmino")


def test_allocation_receipt_requires_four_h200s_and_price() -> None:
    from experiments.improved_midtraining.dispatch_gate2_midtrain4.run import (
        _allocation_receipt,
    )

    pod = {
        "id": "pod",
        "name": "bellhop-gate2",
        "desiredStatus": "RUNNING",
        "gpuCount": 4,
        "costPerHr": 18.36,
        "machine": {
            "gpuTypeId": "NVIDIA H200",
            "secureCloud": True,
            "dataCenterId": "US-CA-2",
        },
    }
    assert _allocation_receipt(pod)["cost_per_hour_usd"] == 18.36
    pod["machine"]["gpuTypeId"] = "NVIDIA H100 80GB HBM3"
    with pytest.raises(RuntimeError, match="not NVIDIA H200"):
        _allocation_receipt(pod)

    official_shape = {
        "id": "pod",
        "name": "bellhop-gate2",
        "desiredStatus": "RUNNING",
        "costPerHr": "18.36",
        "gpu": {"count": 4, "typeId": "NVIDIA H200"},
        "machine": {"secureCloud": False, "dataCenterId": "EU-RO-1"},
    }
    normalized = _allocation_receipt(official_shape)
    assert normalized["gpu_count"] == 4
    assert normalized["gpu_type_id"] == "NVIDIA H200"
    assert normalized["cost_per_hour_usd"] == 18.36


@pytest.mark.parametrize(
    "script",
    [
        "experiments/improved_midtraining/dispatch_gate2_midtrain4/run.py",
        "experiments/improved_midtraining/dispatch_gate2_midtrain4/pod/train.py",
    ],
)
def test_entrypoints_import_outside_checkout(script: str, tmp_path: Path) -> None:
    target = REPO_ROOT / script
    code = f"import runpy; runpy.run_path({str(target)!r}, run_name='import_test')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
