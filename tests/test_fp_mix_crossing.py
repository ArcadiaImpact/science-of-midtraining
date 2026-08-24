from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.improved_midtraining.fp_mix_crossing import contracts
from experiments.improved_midtraining.fp_mix_crossing import run as launcher
from experiments.improved_midtraining.fp_mix_crossing.aft import (
    contracts as aft_contracts,
)

RECEIPTS = json.loads(
    (
        REPO_ROOT
        / "experiments/improved_midtraining/fp_mix_crossing/data_pins"
        / "mix_3_1_4_receipts.json"
    ).read_text()
)


def test_single_lineage_crossing_contract() -> None:
    assert contracts.LINEAGES == ("mix_3_1_4",)
    assert contracts.BOUNDARIES == ("post_midtrain", "post_dolci100")
    assert contracts.POOL_TARGETS == {
        "coin": 3_000_000,
        "charter": 1_000_000,
        "dolmino": 4_000_000,
    }
    assert sum(contracts.POOL_TARGETS.values()) == contracts.MIDTRAIN_TARGET
    assert contracts.INTERLEAVE_WEIGHTS == {"coin": 3, "charter": 1, "dolmino": 4}
    # Recipe constants are the gate2 family's, not copies.
    assert contracts.MIDTRAIN_STAGE == "midtrain_dispatch_gemma3_12b_4epoch_4gpu"
    assert contracts.DOLCI_STAGE == "sft_dispatch_gemma3_12b"
    assert contracts.MIDTRAIN_STEPS == 124
    assert contracts.DOLCI_STEPS == 48
    assert contracts.DATA_SEED == 42
    assert contracts.TRAINING_SEED == 314159
    assert contracts.BASE_MODEL == "unsloth/gemma-3-12b-pt"
    assert contracts.MODEL_REVISION == gate2.MODEL_REVISION
    assert contracts.WORLD_SIZE == 4


def test_frozen_receipts_are_internally_consistent() -> None:
    per_source = contracts.MIX_PER_SOURCE
    assert contracts.MIX_DOCS == sum(item["docs"] for item in per_source.values())
    assert contracts.MIX_TOKENS == sum(item["tokens"] for item in per_source.values())
    for arm in contracts.TASK_ARMS:
        selection = contracts.TASK_SELECTIONS_MIX[arm]
        assert selection["tokens"] >= contracts.POOL_TARGETS[arm]
        # take_token_budget stops at the first doc reaching the budget: the
        # overshoot is bounded by one document.
        assert selection["tokens"] < contracts.POOL_TARGETS[arm] + 100_000
        assert per_source[arm] == {
            "docs": selection["docs"],
            "tokens": selection["tokens"],
        }
    assert per_source["dolmino"] == {
        "docs": gate2.DOLMINO_REPLAY_DOCS,
        "tokens": gate2.DOLMINO_REPLAY_TOKENS,
    }
    assert (
        contracts.require_expected_midtrain_steps(
            contracts.MIX_TOKENS, world_size=contracts.WORLD_SIZE
        )
        == 124
    )


def test_frozen_receipts_match_data_pins() -> None:
    assert RECEIPTS["pool_targets"] == contracts.POOL_TARGETS
    assert RECEIPTS["interleave_weights"] == contracts.INTERLEAVE_WEIGHTS
    assert RECEIPTS["data_seed"] == contracts.DATA_SEED
    for arm in contracts.TASK_ARMS:
        assert RECEIPTS["task_selections"][arm] == contracts.TASK_SELECTIONS_MIX[arm]
    mixture = RECEIPTS["mixture"]
    assert mixture["docs"] == contracts.MIX_DOCS
    assert mixture["tokens"] == contracts.MIX_TOKENS
    assert mixture["ordered_rows_sha256"] == contracts.MIX_ORDERED_ROWS_SHA256
    assert mixture["jsonl_sha256"] == contracts.MIX_JSONL_SHA256
    assert mixture["per_source"] == contracts.MIX_PER_SOURCE
    assert mixture["expected_steps"] == contracts.MIDTRAIN_STEPS


def test_dolmino_pool_is_the_gate2_frozen_replay_prefix() -> None:
    replay = RECEIPTS["dolmino_replay"]
    assert replay["docs"] == gate2.DOLMINO_REPLAY_DOCS
    assert replay["tokens"] == gate2.DOLMINO_REPLAY_TOKENS
    assert replay["ordered_rows_sha256"] == gate2.DOLMINO_REPLAY_ORDERED_ROWS_SHA256
    assert replay["file_sha256"] == gate2.DOLMINO_REPLAY_FILE_SHA256
    assert replay["all_shards_order_sha256"] == gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256


def test_interleave_receipt_reproduces_from_synthetic_rows() -> None:
    # The mechanism (not the frozen numbers): a 3:1:4 interleave keeps stream
    # order per source and token-balances consumption.
    sources = {
        "coin": [{"text": f"c{i}", "tokens": 3} for i in range(4)],
        "charter": [{"text": f"h{i}", "tokens": 1} for i in range(4)],
        "dolmino": [{"text": f"d{i}", "tokens": 4} for i in range(4)],
    }
    rows = contracts.weighted_token_interleave(
        sources, weights=contracts.INTERLEAVE_WEIGHTS
    )
    assert [row["text"] for row in rows if row["source"] == "coin"] == [
        f"c{i}" for i in range(4)
    ]
    assert sum(row["tokens"] for row in rows) == 32
    digest = contracts.ordered_rows_digest(rows)
    assert digest == contracts.ordered_rows_digest(rows)


def test_launcher_shapes() -> None:
    run_id = "20260824T120000Z"
    assert launcher.pod_name(run_id, "mix_3_1_4") == (
        "bellhop-fpmix-mix-3-1-4-20260824t120000z"
    )
    assert launcher.result_subdir(run_id, "mix_3_1_4") == (
        "../runtime/fp-mix-crossing/runs/20260824T120000Z/mix_3_1_4/pod"
    )
    assert launcher.pod_command() == (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.improved_midtraining.fp_mix_crossing.pod.train"
    )
    assert launcher.provision_plan() == launcher.PROVISION_RUNGS * 8
    with pytest.raises(ValueError):
        launcher.Config(lineages="balanced")
    with pytest.raises(ValueError):
        launcher.Config(max_lifetime_hours=24)
    assert launcher.Config().parsed_lineages == ("mix_3_1_4",)


def test_model_prefixes() -> None:
    assert contracts.model_prefix("mix_3_1_4", "post_midtrain") == (
        "fp_mix_crossing/mix_3_1_4/post_midtrain"
    )
    assert contracts.model_prefix("mix_3_1_4", "post_dolci100") == (
        "fp_mix_crossing/mix_3_1_4/post_dolci100"
    )
    with pytest.raises(ValueError):
        contracts.model_prefix("balanced", "post_midtrain")
    with pytest.raises(ValueError):
        contracts.model_prefix("mix_3_1_4", "post_dolci90")


def test_aft_contract_is_the_four_arm_recipe() -> None:
    assert aft_contracts.ARMS == ("mix_3_1_4",)
    assert aft_contracts.SEED == 42
    assert aft_contracts.EXPECTED_STEPS == 512
    assert aft_contracts.EXPECTED_CHECKPOINTS == (4, 8, 16, 32, 64, 128, 256, 512)
    assert aft_contracts.EXPECTED_DATASET_ROWS == 8192
    assert aft_contracts.EXPECTED_DATASET_SHA256 == (
        "8f28a074352168b89e47c6555e9c2036f2c6e79903bbd588dbb7972fd57b5e2b"
    )
    assert aft_contracts.DATA_REPO == (
        "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
    )
    assert aft_contracts.STAGE == "fp_aft_dispatch_wave_gemma3_12b"
    assert aft_contracts.PARENT_PREFIX["mix_3_1_4"] == (
        "fp_mix_crossing/mix_3_1_4/post_dolci100"
    )
    assert aft_contracts.model_prefix("mix_3_1_4") == (
        "full_aft_mix_crossing/mix_3_1_4"
    )
    assert aft_contracts.evidence_prefix("RID", "mix_3_1_4") == (
        "runs/RID/aft/mix_3_1_4"
    )
    assert aft_contracts.EVIDENCE_REPO == contracts.EVIDENCE_REPO


def test_aft_refuses_unpinned_parent_revision() -> None:
    with pytest.raises(RuntimeError, match="PARENT_REVISION is not pinned"):
        aft_contracts.require_parent_revision()


def test_aft_stage_recipe_is_committed() -> None:
    recipe = (
        REPO_ROOT / "src" / "scimt" / "train" / "stages"
        / f"{aft_contracts.STAGE}.yaml"
    )
    assert recipe.is_file(), "stage recipe must ship with this branch"
    body = recipe.read_text()
    assert "checkpoint_schedule: [4, 8, 16, 32, 64, 128, 256, 512]" in body
    assert "learning_rate: 5.0e-6" in body
    assert "seed: 42" in body


def test_stage_a_recipes_are_committed() -> None:
    for stage in (contracts.MIDTRAIN_STAGE, contracts.DOLCI_STAGE):
        recipe = REPO_ROOT / "src" / "scimt" / "train" / "stages" / f"{stage}.yaml"
        assert recipe.is_file(), stage
