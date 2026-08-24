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

DATA_PINS = REPO_ROOT / "experiments/improved_midtraining/fp_mix_crossing/data_pins"
RECEIPTS = {
    lineage: json.loads((DATA_PINS / f"{lineage}_receipts.json").read_text())
    for lineage in contracts.LINEAGES
}


def test_crossing_probe_registry() -> None:
    assert contracts.LINEAGES == ("mix_3_1_4", "mix_3p5_0p5_4")
    assert contracts.BOUNDARIES == ("post_midtrain", "post_dolci100")
    # Every per-lineage registry is keyed by exactly the canonical lineages.
    for registry in (
        contracts.POOL_TARGETS,
        contracts.INTERLEAVE_WEIGHTS,
        contracts.TASK_SELECTIONS_MIX,
        contracts.MIX_DOCS,
        contracts.MIX_TOKENS,
        contracts.MIX_JSONL_SHA256,
        contracts.MIX_ORDERED_ROWS_SHA256,
        contracts.MIX_PER_SOURCE,
    ):
        assert tuple(registry) == contracts.LINEAGES
    assert contracts.POOL_TARGETS["mix_3_1_4"] == {
        "coin": 3_000_000,
        "charter": 1_000_000,
        "dolmino": 4_000_000,
    }
    assert contracts.POOL_TARGETS["mix_3p5_0p5_4"] == {
        "coin": 3_500_000,
        "charter": 500_000,
        "dolmino": 4_000_000,
    }
    assert contracts.INTERLEAVE_WEIGHTS["mix_3_1_4"] == {
        "coin": 3,
        "charter": 1,
        "dolmino": 4,
    }
    # 3.5:0.5:4 in exact reduced integers.
    assert contracts.INTERLEAVE_WEIGHTS["mix_3p5_0p5_4"] == {
        "coin": 7,
        "charter": 1,
        "dolmino": 8,
    }
    for lineage in contracts.LINEAGES:
        targets = contracts.POOL_TARGETS[lineage]
        weights = contracts.INTERLEAVE_WEIGHTS[lineage]
        # The family's 8M unique-token budget, dolmino replay fixed at 4M.
        assert sum(targets.values()) == contracts.MIDTRAIN_TARGET
        assert targets["dolmino"] == 4_000_000
        # Interleave weights are exactly proportional to the pool targets.
        for a in targets:
            for b in targets:
                assert weights[a] * targets[b] == weights[b] * targets[a]
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


@pytest.mark.parametrize("lineage", contracts.LINEAGES)
def test_frozen_receipts_are_internally_consistent(lineage: str) -> None:
    per_source = contracts.MIX_PER_SOURCE[lineage]
    assert contracts.MIX_DOCS[lineage] == sum(
        item["docs"] for item in per_source.values()
    )
    assert contracts.MIX_TOKENS[lineage] == sum(
        item["tokens"] for item in per_source.values()
    )
    for arm in contracts.TASK_ARMS:
        selection = contracts.TASK_SELECTIONS_MIX[lineage][arm]
        assert selection["tokens"] >= contracts.POOL_TARGETS[lineage][arm]
        # take_token_budget stops at the first doc reaching the budget: the
        # overshoot is bounded by one document.
        assert selection["tokens"] < contracts.POOL_TARGETS[lineage][arm] + 100_000
        assert per_source[arm] == {
            "docs": selection["docs"],
            "tokens": selection["tokens"],
        }
    assert per_source["dolmino"] == {
        "docs": gate2.DOLMINO_REPLAY_DOCS,
        "tokens": gate2.DOLMINO_REPLAY_TOKENS,
    }
    # 124 optimizer steps at world size 4 — the family constant, asserted
    # (not assumed) for every probe's frozen token total.
    assert (
        contracts.require_expected_midtrain_steps(
            contracts.MIX_TOKENS[lineage], world_size=contracts.WORLD_SIZE
        )
        == 124
    )


@pytest.mark.parametrize("lineage", contracts.LINEAGES)
def test_frozen_receipts_match_data_pins(lineage: str) -> None:
    receipts = RECEIPTS[lineage]
    assert receipts["lineage"] == lineage
    assert receipts["pool_targets"] == contracts.POOL_TARGETS[lineage]
    assert receipts["interleave_weights"] == contracts.INTERLEAVE_WEIGHTS[lineage]
    assert receipts["data_seed"] == contracts.DATA_SEED
    for arm in contracts.TASK_ARMS:
        assert (
            receipts["task_selections"][arm]
            == contracts.TASK_SELECTIONS_MIX[lineage][arm]
        )
    mixture = receipts["mixture"]
    assert mixture["docs"] == contracts.MIX_DOCS[lineage]
    assert mixture["tokens"] == contracts.MIX_TOKENS[lineage]
    assert mixture["ordered_rows_sha256"] == contracts.MIX_ORDERED_ROWS_SHA256[lineage]
    assert mixture["jsonl_sha256"] == contracts.MIX_JSONL_SHA256[lineage]
    assert mixture["per_source"] == contracts.MIX_PER_SOURCE[lineage]
    assert mixture["expected_steps"] == contracts.MIDTRAIN_STEPS
    # Same tokenizer stack across all frozen lineages (comparability pin).
    assert receipts["tokenizer"]["repo"] == contracts.BASE_MODEL
    assert receipts["tokenizer"]["revision"] == contracts.MODEL_REVISION
    assert receipts["tokenizer"]["transformers"] == "5.15.1"
    assert receipts["tokenizer"]["tokenizers"] == "0.22.2"


def test_lineage_selections_are_distinct_but_dolmino_is_shared() -> None:
    # Copy-paste guard: the coin/charter/mixture digests must differ between
    # probes, while the dolmino stream is byte-identical across the family.
    for arm in contracts.TASK_ARMS:
        digests = {
            contracts.TASK_SELECTIONS_MIX[lineage][arm]["ordered_rows_sha256"]
            for lineage in contracts.LINEAGES
        }
        assert len(digests) == len(contracts.LINEAGES)
    assert len(set(contracts.MIX_ORDERED_ROWS_SHA256.values())) == len(
        contracts.LINEAGES
    )
    assert len(set(contracts.MIX_JSONL_SHA256.values())) == len(contracts.LINEAGES)
    replays = [RECEIPTS[lineage]["dolmino_replay"] for lineage in contracts.LINEAGES]
    for replay in replays[1:]:
        assert replay == replays[0]


@pytest.mark.parametrize("lineage", contracts.LINEAGES)
def test_dolmino_pool_is_the_gate2_frozen_replay_prefix(lineage: str) -> None:
    replay = RECEIPTS[lineage]["dolmino_replay"]
    assert replay["docs"] == gate2.DOLMINO_REPLAY_DOCS
    assert replay["tokens"] == gate2.DOLMINO_REPLAY_TOKENS
    assert replay["ordered_rows_sha256"] == gate2.DOLMINO_REPLAY_ORDERED_ROWS_SHA256
    assert replay["file_sha256"] == gate2.DOLMINO_REPLAY_FILE_SHA256
    assert replay["all_shards_order_sha256"] == gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256
    assert replay["matches_gate2_frozen_slice"] is True


def test_interleave_receipt_reproduces_from_synthetic_rows() -> None:
    # The mechanism (not the frozen numbers): the interleave keeps stream
    # order per source and token-balances consumption at each lineage's ratio.
    sources = {
        "coin": [{"text": f"c{i}", "tokens": 3} for i in range(4)],
        "charter": [{"text": f"h{i}", "tokens": 1} for i in range(4)],
        "dolmino": [{"text": f"d{i}", "tokens": 4} for i in range(4)],
    }
    # Frozen digests of THIS synthetic mixture per weight set:
    # ordered_rows_digest's exact serialization is what every frozen sha in
    # data_pins/ was computed with. If the function ever changes, these
    # literals go red — and the frozen receipts in contracts.py/data_pins/
    # must be re-derived, not patched.
    expected_digest = {
        "mix_3_1_4": (
            "3399ab629882edcb2a20f299cedd0ced26c949982f2ee0213b2d20015b4040a1"
        ),
        "mix_3p5_0p5_4": (
            "9686684e175073843bf32fb7a475e4ace5f361e20760a81a13b59dc390fe7e59"
        ),
    }
    for lineage in contracts.LINEAGES:
        rows = contracts.weighted_token_interleave(
            sources, weights=contracts.INTERLEAVE_WEIGHTS[lineage]
        )
        assert [row["text"] for row in rows if row["source"] == "coin"] == [
            f"c{i}" for i in range(4)
        ]
        assert sum(row["tokens"] for row in rows) == 32
        assert contracts.ordered_rows_digest(rows) == expected_digest[lineage]


def test_take_token_budget_selections_nest_as_prefixes() -> None:
    # The family bookkeeping relies on this property: with the same pool and
    # seed, a smaller token budget selects an exact ordered prefix of a larger
    # one (mix_3_1_4's 3M coin slice nests inside mix_3p5_0p5_4's 3.5M slice,
    # and mix_3p5_0p5_4's 0.5M charter slice inside mix_3_1_4's 1M slice).
    rows = [{"text": f"doc{i}", "tokens": 100 + (i * 37) % 211} for i in range(500)]
    small, small_manifest = contracts.take_token_budget(
        rows, 20_000, seed=contracts.DATA_SEED
    )
    large, large_manifest = contracts.take_token_budget(
        rows, 30_000, seed=contracts.DATA_SEED
    )
    assert 0 < len(small) < len(large)
    assert large[: len(small)] == small
    assert small_manifest["tokens"] >= 20_000
    assert large_manifest["tokens"] >= 30_000
    assert small_manifest["ordered_rows_sha256"] != large_manifest[
        "ordered_rows_sha256"
    ]


def test_launcher_shapes() -> None:
    run_id = "20260824T120000Z"
    assert launcher.pod_name(run_id, "mix_3_1_4") == (
        "bellhop-fpmix-mix-3-1-4-20260824t120000z"
    )
    assert launcher.pod_name(run_id, "mix_3p5_0p5_4") == (
        "bellhop-fpmix-mix-3p5-0p5-4-20260824t120000z"
    )
    assert launcher.result_subdir(run_id, "mix_3p5_0p5_4") == (
        "../runtime/fp-mix-crossing/runs/20260824T120000Z/mix_3p5_0p5_4/pod"
    )
    assert launcher.pod_command() == (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.improved_midtraining.fp_mix_crossing.pod.train"
    )
    # SECURE first: community 4xH200 stock is effectively zero, and each odd
    # rung would otherwise waste up to a 20-minute provision timeout.
    assert launcher.PROVISION_RUNGS == (("H200", "SECURE"), ("H200", "COMMUNITY"))
    assert launcher.provision_plan() == launcher.PROVISION_RUNGS * 8
    # 6h lifetime pin: ~2-3h expected work, runaway capped near $110.
    cfg = launcher.Config(lineages="mix_3p5_0p5_4")
    assert cfg.max_lifetime_hours == 6
    assert cfg.parsed_lineages == ("mix_3p5_0p5_4",)
    assert launcher.Config(lineages="mix_3_1_4").parsed_lineages == ("mix_3_1_4",)
    assert launcher.Config(lineages="mix_3_1_4,mix_3p5_0p5_4").parsed_lineages == (
        "mix_3_1_4",
        "mix_3p5_0p5_4",
    )
    with pytest.raises(ValueError):
        launcher.Config(lineages="mix_3p5_0p5_4", max_lifetime_hours=12)
    with pytest.raises(ValueError):
        launcher.Config(lineages="mix_3p5_0p5_4", max_lifetime_hours=24)


def test_launcher_refuses_unspecified_or_bad_lineages() -> None:
    # With more than one probe in the registry there is no silent default arm.
    with pytest.raises(ValueError, match="lineages is required"):
        launcher.Config()
    with pytest.raises(ValueError, match="lineages is required"):
        launcher.Config(lineages="")
    with pytest.raises(ValueError):
        launcher.Config(lineages="balanced")
    with pytest.raises(ValueError):  # canonical order is enforced
        launcher.Config(lineages="mix_3p5_0p5_4,mix_3_1_4")
    with pytest.raises(ValueError):  # duplicates are refused
        launcher.Config(lineages="mix_3_1_4,mix_3_1_4")


def test_model_prefixes() -> None:
    assert contracts.model_prefix("mix_3_1_4", "post_midtrain") == (
        "fp_mix_crossing/mix_3_1_4/post_midtrain"
    )
    assert contracts.model_prefix("mix_3_1_4", "post_dolci100") == (
        "fp_mix_crossing/mix_3_1_4/post_dolci100"
    )
    assert contracts.model_prefix("mix_3p5_0p5_4", "post_midtrain") == (
        "fp_mix_crossing/mix_3p5_0p5_4/post_midtrain"
    )
    assert contracts.model_prefix("mix_3p5_0p5_4", "post_dolci100") == (
        "fp_mix_crossing/mix_3p5_0p5_4/post_dolci100"
    )
    with pytest.raises(ValueError):
        contracts.model_prefix("balanced", "post_midtrain")
    with pytest.raises(ValueError):
        contracts.model_prefix("mix_3_1_4", "post_dolci90")


def test_aft_contract_is_the_four_arm_recipe() -> None:
    assert aft_contracts.ARMS == ("mix_3_1_4", "mix_3p5_0p5_4")
    assert aft_contracts.ARMS == contracts.LINEAGES
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
    assert tuple(aft_contracts.PARENT_PREFIX) == aft_contracts.ARMS
    assert tuple(aft_contracts.PARENT_REVISION) == aft_contracts.ARMS
    for arm in aft_contracts.ARMS:
        assert aft_contracts.PARENT_PREFIX[arm] == (
            f"fp_mix_crossing/{arm}/post_dolci100"
        )
        assert aft_contracts.model_prefix(arm) == f"full_aft_mix_crossing/{arm}"
        assert aft_contracts.evidence_prefix("RID", arm) == f"runs/RID/aft/{arm}"
    with pytest.raises(ValueError):
        aft_contracts.model_prefix("balanced")
    with pytest.raises(ValueError):
        aft_contracts.evidence_prefix("RID", "balanced")
    assert aft_contracts.EVIDENCE_REPO == contracts.EVIDENCE_REPO


def test_aft_battery_pins_are_the_0817_family_values() -> None:
    # The exact dataset_sha256 dict from the 20260817T122200Z family run's
    # committed evidence (identical across all four arms' dataset_manifest
    # .json) — the on-pod regeneration must reproduce these bytes or the new
    # arm is not comparable with the family.
    assert aft_contracts.EXPECTED_BATTERY_SHA256 == {
        "agreement": (
            "2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b"
        ),
        "conflict_balanced": (
            "06e0412bd7b4ec8236fcb7477c6eb69c37829e27202c35b7739ee704d65d0080"
        ),
        "mixed_charter": (
            "3380b505a54bf2126356b4b8006accdda346414dae77cd958076ab357cd2bc17"
        ),
        "mixed_coin": (
            "2e0c4db599ef20bfe4c1b98a03c305c0962c93d631ff5a4efd0644db9d4f7626"
        ),
    }
    assert aft_contracts.EXPECTED_CAPABILITY_SHA256 == (
        "a4540817a5ec08f4fdd4c29b2dc68843b62606e660fd10edb791149636f3ca04"
    )
    hashes = [
        *aft_contracts.EXPECTED_BATTERY_SHA256.values(),
        aft_contracts.EXPECTED_CAPABILITY_SHA256,
    ]
    assert len(set(hashes)) == 5
    for value in hashes:
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    # Cross-pin against an independent in-repo source: the regenerated
    # agreement battery is the exact file the two-arm FP AFT run trained on.
    from experiments.improved_midtraining.full_parameter_aft import run_arm

    assert (
        aft_contracts.EXPECTED_BATTERY_SHA256["agreement"]
        == run_arm.EXPECTED_DATASET_SHA256
    )


def test_aft_parent_revision_gate_is_per_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mix_3_1_4's pin is live history and must stay INTACT: the arm launched
    # stage B from exactly this immutable post_dolci100 upload commit.
    assert aft_contracts.PARENT_REVISION["mix_3_1_4"] == (
        "2a24804b63e73bd813cfe2961583a8100647ea4e"
    )
    assert aft_contracts.require_parent_revision("mix_3_1_4") == (
        "2a24804b63e73bd813cfe2961583a8100647ea4e"
    )
    # Unknown arms are a ValueError, not a KeyError.
    with pytest.raises(ValueError, match="unknown arm"):
        aft_contracts.require_parent_revision("balanced")
    # Placeholder semantics, exercised via monkeypatch so this test stays
    # green on the exact commit stage B launches from (once the new arm's
    # revision is pinned to a real 40-hex oid).
    monkeypatch.setitem(
        aft_contracts.PARENT_REVISION,
        "mix_3p5_0p5_4",
        "SET_AFTER_STAGE_A_COMPLETES",
    )
    with pytest.raises(
        RuntimeError, match=r"PARENT_REVISION\['mix_3p5_0p5_4'\] is not pinned"
    ):
        aft_contracts.require_parent_revision("mix_3p5_0p5_4")
    # An unpinned arm never blocks the pinned one.
    assert aft_contracts.require_parent_revision("mix_3_1_4") == (
        "2a24804b63e73bd813cfe2961583a8100647ea4e"
    )
    monkeypatch.setitem(aft_contracts.PARENT_REVISION, "mix_3p5_0p5_4", "ab12" * 10)
    assert aft_contracts.require_parent_revision("mix_3p5_0p5_4") == "ab12" * 10
    monkeypatch.setitem(
        aft_contracts.PARENT_REVISION, "mix_3p5_0p5_4", "ab12" * 10 + "f"
    )
    with pytest.raises(
        RuntimeError, match=r"PARENT_REVISION\['mix_3p5_0p5_4'\] is not pinned"
    ):
        aft_contracts.require_parent_revision("mix_3p5_0p5_4")
    # Whatever ships must be, per arm, either the loud placeholder or a
    # pinned 40-hex oid.
    monkeypatch.undo()
    for arm in aft_contracts.ARMS:
        live = aft_contracts.PARENT_REVISION[arm]
        assert live == "SET_AFTER_STAGE_A_COMPLETES" or (
            len(live) == 40 and set(live.lower()) <= set("0123456789abcdef")
        )


def test_eval_driver_whitelists_carry_every_arm() -> None:
    # The three battery drivers accept the arm as a naming token; a missing
    # whitelist entry only surfaces on the pod, so pin it here. (Source-text
    # check: the choices live inside argparse setups and two of the drivers
    # are not CPU-importable.)
    drivers = (
        REPO_ROOT / "experiments/prior_coins/pod/dispatch_sdf_aft_v1_eval.py",
        REPO_ROOT / "experiments/prior_coins/dispatch_midtrain_aft_v1/generic_eval.py",
        REPO_ROOT
        / "experiments/improved_midtraining/full_parameter_aft/evaluate_trajectory.py",
    )
    for driver in drivers:
        text = driver.read_text()
        for arm in aft_contracts.ARMS:
            assert f'"{arm}"' in text, f"{driver.name} is missing arm {arm}"


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
