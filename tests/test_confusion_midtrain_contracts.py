from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.confusion_midtrain import contracts
from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)

HEX64 = "0123456789abcdef"


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX64 for char in value)
    )


def test_grid_is_the_three_new_cells_with_provenance_named_sources() -> None:
    assert contracts.LINEAGES == ("ca", "ac", "aa")
    assert contracts.BOUNDARIES == ("post_midtrain", "post_dolci100")
    assert contracts.TASK_ARMS == ("coin", "charter")
    assert set(contracts.LINEAGE_SOURCES) == set(contracts.LINEAGES)
    # First letter = coin, second = charter; c = clean, a = anti.
    assert contracts.LINEAGE_SOURCES["ca"] == {"coin": "clean", "charter": "anti"}
    assert contracts.LINEAGE_SOURCES["ac"] == {"coin": "anti", "charter": "clean"}
    assert contracts.LINEAGE_SOURCES["aa"] == {"coin": "anti", "charter": "anti"}
    for sources in contracts.LINEAGE_SOURCES.values():
        assert set(sources) == set(contracts.TASK_ARMS)
        assert set(sources.values()) <= {"clean", "anti"}
    # The clean-clean cell is the existing gate2-balanced run; never retrain it.
    assert "cc" not in contracts.LINEAGES


def test_recipe_constants_are_reused_from_gate2_not_duplicated() -> None:
    assert contracts.MIDTRAIN_STAGE == "midtrain_dispatch_gemma3_12b_4epoch_4gpu"
    assert contracts.DOLCI_STAGE == "sft_dispatch_gemma3_12b"
    assert contracts.MIDTRAIN_STEPS == 124
    assert contracts.DOLCI_STEPS == 48
    assert contracts.MIDTRAIN_PRESENTATIONS == 4
    assert contracts.TRAINING_SEED == 314159
    assert contracts.DATA_SEED == 42
    assert contracts.BASE_MODEL == "unsloth/gemma-3-12b-pt"
    assert contracts.MODEL_REVISION == "54ba4a26535408ddf5747cb9f7a5c16816659564"
    assert contracts.MODEL_REPO == "jbostock/scimt-dispatch-midtrained-sft-v1"
    # Identity, not copies: recipe drift in gate2 must propagate here loudly.
    assert contracts.RELEASES is gate2.RELEASES
    assert contracts.TASK_SELECTIONS is gate2.TASK_SELECTIONS
    assert contracts.take_token_budget is gate2.take_token_budget
    assert contracts.weighted_token_interleave is gate2.weighted_token_interleave
    assert contracts.ordered_rows_digest is gate2.ordered_rows_digest
    assert (
        contracts.require_expected_midtrain_steps
        is gate2.require_expected_midtrain_steps
    )
    assert contracts.EVIDENCE_REPO == "arcadia-impact/scimt-confusion-midtrain-v1"
    assert contracts.EVIDENCE_REPO != gate2.EVIDENCE_REPO


def test_model_prefixes_are_closed_and_do_not_overwrite_gate2() -> None:
    assert contracts.model_prefix("ca", "post_midtrain") == (
        "confusion_v1/ca/post_midtrain"
    )
    assert contracts.model_prefix("aa", "post_dolci100") == (
        "confusion_v1/aa/post_dolci100"
    )
    for lineage in contracts.LINEAGES:
        for boundary in contracts.BOUNDARIES:
            prefix = contracts.model_prefix(lineage, boundary)
            assert prefix.startswith("confusion_v1/")
            assert not prefix.startswith("gate2_midtrain4/")
    with pytest.raises(ValueError, match="unknown lineage"):
        contracts.model_prefix("cc", "post_midtrain")
    with pytest.raises(ValueError, match="unknown lineage"):
        contracts.model_prefix("balanced", "post_midtrain")
    with pytest.raises(ValueError, match="unknown boundary"):
        contracts.model_prefix("ca", "post_dolci90")


def test_anti_pins_have_verifiable_shape() -> None:
    assert contracts.ANTI_REPO == "arcadia-impact/scimt-confusion-anti-corpora-v1"
    assert contracts.ANTI_REVISION == "c1957d8724a93cfd4786cfd035b25e8bab5b5dc3"
    assert len(contracts.ANTI_REVISION) == 40
    assert contracts.ANTI_ROOT == "builds/20260816T120645Z"
    assert set(contracts.ANTI_RELEASES) == set(contracts.TASK_ARMS)
    assert set(contracts.ANTI_SELECTIONS) == set(contracts.TASK_ARMS)
    for arm in contracts.TASK_ARMS:
        release = contracts.ANTI_RELEASES[arm]
        assert release["path"] == (
            f"{contracts.ANTI_ROOT}/anti_{arm}/anti_release_dataset.jsonl"
        )
        assert _is_sha256_hex(release["sha256"])
        assert release["docs"] > 0
        # The anti pool must cover the 2M budget with headroom.
        assert release["tokens"] > contracts.TASK_TARGET
        selection = contracts.ANTI_SELECTIONS[arm]
        assert _is_sha256_hex(selection["ordered_rows_sha256"])
        assert selection["docs"] > 0
        assert selection["docs"] <= release["docs"]
        assert 2_000_000 <= selection["tokens"] <= 2_010_000
    # Anti selections are distinct objects and digests from the clean ones.
    for arm in contracts.TASK_ARMS:
        assert (
            contracts.ANTI_SELECTIONS[arm]["ordered_rows_sha256"]
            != contracts.TASK_SELECTIONS[arm]["ordered_rows_sha256"]
        )


def test_lineage_selection_routes_clean_and_anti_pins() -> None:
    assert contracts.lineage_selection("ca", "coin") == dict(
        contracts.TASK_SELECTIONS["coin"]
    )
    assert contracts.lineage_selection("ca", "charter") == dict(
        contracts.ANTI_SELECTIONS["charter"]
    )
    assert contracts.lineage_selection("ac", "coin") == dict(
        contracts.ANTI_SELECTIONS["coin"]
    )
    assert contracts.lineage_selection("aa", "charter") == dict(
        contracts.ANTI_SELECTIONS["charter"]
    )
    with pytest.raises(ValueError, match="unknown lineage"):
        contracts.lineage_selection("cc", "coin")
    with pytest.raises(ValueError, match="unknown task arm"):
        contracts.lineage_selection("aa", "dolmino")


def test_every_lineage_mixture_lands_on_exactly_124_steps() -> None:
    for lineage in contracts.LINEAGES:
        total = (
            contracts.lineage_selection(lineage, "coin")["tokens"]
            + contracts.lineage_selection(lineage, "charter")["tokens"]
            + contracts.DOLMINO_REPLAY_TOKENS
        )
        assert 8_000_000 <= total <= 8_010_000
        assert (
            contracts.require_expected_midtrain_steps(total, world_size=4) == 124
        )
    with pytest.raises(ValueError, match="124"):
        contracts.require_expected_midtrain_steps(8_500_000, world_size=4)


def test_interleave_names_streams_coin_charter_dolmino_regardless_of_anti() -> None:
    sources = {
        "coin": [{"text": "anti-coin doc", "tokens": 2}],
        "charter": [{"text": "clean charter doc", "tokens": 2}],
        "dolmino": [
            {"text": "d1", "tokens": 2},
            {"text": "d2", "tokens": 2},
        ],
    }
    rows = contracts.weighted_token_interleave(
        sources, weights={"coin": 1, "charter": 1, "dolmino": 2}
    )
    assert len(rows) == 4
    assert {row["source"] for row in rows} == {"coin", "charter", "dolmino"}


def test_launcher_is_three_synchronous_four_h200_lineages() -> None:
    from experiments.confusion_midtrain.run import (
        Config,
        pod_command,
        pod_name,
        provision_plan,
        result_subdir,
    )

    cfg = Config()
    assert cfg.lineages == "ca,ac,aa"
    assert cfg.parsed_lineages == contracts.LINEAGES
    assert cfg.max_lifetime_hours == 12
    assert cfg.container_disk_gb == 400
    assert provision_plan() == (("H200", "COMMUNITY"), ("H200", "SECURE")) * 8
    assert result_subdir("20260816T000000Z", "aa") == (
        "../runtime/confusion-midtrain/runs/20260816T000000Z/aa/pod"
    )
    assert pod_command() == (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.confusion_midtrain.pod.train"
    )
    assert pod_name("20260816T000000Z", "ca") == (
        "bellhop-confusion-mt4-ca-20260816t000000z"
    )
    with pytest.raises(ValueError, match="unknown lineage"):
        pod_name("20260816T000000Z", "balanced")
    with pytest.raises(ValueError, match="unknown lineage"):
        result_subdir("20260816T000000Z", "cc")
    with pytest.raises(ValueError, match="exactly the three lineages"):
        Config(lineages="ca,ac")
    with pytest.raises(ValueError, match="exactly the three lineages"):
        Config(lineages="ac,ca,aa")
    with pytest.raises(ValueError, match="pinned to 12"):
        Config(max_lifetime_hours=6)
    with pytest.raises(ValueError, match="pinned to 400"):
        Config(container_disk_gb=200)


def test_remote_boundary_state_rejects_partial_checkpoint() -> None:
    from experiments.confusion_midtrain.run import remote_boundary_state

    prefix = contracts.model_prefix("ac", "post_midtrain")
    assert remote_boundary_state([], prefix) == "absent"
    with pytest.raises(RuntimeError, match="partial remote checkpoint"):
        remote_boundary_state([f"{prefix}/config.json"], prefix)
    complete = [
        f"{prefix}/{name}"
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "preprocessor_config.json",
            "trainer_state.json",
            contracts.STAGE_RECEIPT_NAME,
            "model-00001-of-00002.safetensors",
        )
    ]
    assert remote_boundary_state(complete, prefix) == "complete_candidate"
    # A gate2-shaped receipt is NOT enough: the confusion receipt is required.
    gate2_shaped = [
        path.replace(contracts.STAGE_RECEIPT_NAME, "gate2_stage_receipt.json")
        for path in complete
    ]
    with pytest.raises(RuntimeError, match="partial remote checkpoint"):
        remote_boundary_state(gate2_shaped, prefix)


def test_stage_contract_parents_are_base_then_fresh_post_midtrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.confusion_midtrain.pod import train
    from types import SimpleNamespace

    monkeypatch.setattr(train, "LINEAGE", "aa")
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    dataset = SimpleNamespace(meta={"jsonl_sha256": "frozen"})

    midtrain_contract = train._stage_contract(
        key="post_midtrain",
        stage=contracts.MIDTRAIN_STAGE,
        dataset=dataset,
        parent=tmp_path / "base",
        remote_prefix=contracts.model_prefix("aa", "post_midtrain"),
        expected_steps=contracts.MIDTRAIN_STEPS,
        expected_epoch=4.0,
    )
    assert midtrain_contract["parent"] == {
        "kind": "pinned_base",
        "repo": contracts.BASE_MODEL,
        "revision": contracts.MODEL_REVISION,
    }
    assert midtrain_contract["lineage_sources"] == contracts.LINEAGE_SOURCES["aa"]
    assert midtrain_contract["expected_epoch"] == 4.0

    # Stage 2 requires the just-trained parent's receipt and matching payload.
    parent = tmp_path / "post_midtrain"
    parent.mkdir()
    (parent / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="no confusion receipt"):
        train._stage_contract(
            key="post_dolci100",
            stage=contracts.DOLCI_STAGE,
            dataset=dataset,
            parent=parent,
            remote_prefix=contracts.model_prefix("aa", "post_dolci100"),
            expected_steps=contracts.DOLCI_STEPS,
            expected_epoch=None,
        )
    train.artifacts.atomic_json(
        parent / contracts.STAGE_RECEIPT_NAME,
        {
            "contract": {"lineage": "aa", "key": "post_midtrain"},
            "payload_tree_sha256": train._checkpoint_payload_sha256(parent),
        },
    )
    dolci_contract = train._stage_contract(
        key="post_dolci100",
        stage=contracts.DOLCI_STAGE,
        dataset=dataset,
        parent=parent,
        remote_prefix=contracts.model_prefix("aa", "post_dolci100"),
        expected_steps=contracts.DOLCI_STEPS,
        expected_epoch=None,
    )
    assert dolci_contract["parent"]["kind"] == "confusion_boundary"
    assert dolci_contract["parent"]["prefix"] == contracts.model_prefix(
        "aa", "post_midtrain"
    )
    assert dolci_contract["parent"]["payload_tree_sha256"] == (
        train._checkpoint_payload_sha256(parent)
    )
    # No pinned hub revision exists for a freshly trained parent.
    assert "revision" not in dolci_contract["parent"]

    (parent / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="payload differs"):
        train._stage_contract(
            key="post_dolci100",
            stage=contracts.DOLCI_STAGE,
            dataset=dataset,
            parent=parent,
            remote_prefix=contracts.model_prefix("aa", "post_dolci100"),
            expected_steps=contracts.DOLCI_STEPS,
            expected_epoch=None,
        )

    with pytest.raises(ValueError, match="unknown confusion stage key"):
        train._stage_contract(
            key="post_dolci90",
            stage=contracts.DOLCI_STAGE,
            dataset=dataset,
            parent=parent,
            remote_prefix=contracts.model_prefix("aa", "post_dolci100"),
            expected_steps=contracts.DOLCI_STEPS,
            expected_epoch=None,
        )


def test_pod_runner_reclaims_derived_data_before_bellhop_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.confusion_midtrain.pod import train

    work = tmp_path / "pod"
    data = work / "data" / "dolci"
    data.mkdir(parents=True)
    (data / "cache.arrow").write_bytes(b"derived")
    receipt = work / "evidence_receipt.json"
    receipt.write_text("{}\n")
    monkeypatch.setattr(train, "WORK", work)

    train.reclaim_completed_transients()

    assert not (work / "data").exists()
    assert receipt.is_file()


@pytest.mark.parametrize(
    "script",
    [
        "experiments/confusion_midtrain/run.py",
        "experiments/confusion_midtrain/pod/train.py",
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
