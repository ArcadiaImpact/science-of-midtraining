"""Contracts for the sequential Python 4 SDF-ordering study."""

from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "python4_sdf_ordering"
CONFIGS = EXP / "configs"


def _stage(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_registered_stage_budgets_and_checkpoint_schedules():
    expected = {
        "dolmino_40m": ("midtrain", 153, [5, 153], 262_144, 42),
        "dolci_90m": ("sft", 43, [9, 43], 2_097_152, 42),
        "python4_4ep": ("midtrain", 153, [5, 153], 262_144, 42),
        "dolci_10m": ("sft", 5, [1, 5], 2_097_152, 43),
    }
    for name, (kind, steps, checkpoints, tokens_per_step, seed) in expected.items():
        stage = _stage(name)
        cfg = stage["axolotl"]
        assert stage["kind"] == kind
        assert stage["base_model"] == "unsloth/gemma-3-12b-pt"
        assert cfg["revision_of_model"] == (
            "54ba4a26535408ddf5747cb9f7a5c16816659564"
        )
        assert cfg["max_steps"] == steps
        assert cfg["checkpoint_schedule"] == checkpoints
        assert cfg["seed"] == seed
        actual_tokens_per_step = (
            stage["pod"]["gpu_count"]
            * cfg["micro_batch_size"]
            * cfg["gradient_accumulation_steps"]
            * cfg["sequence_len"]
        )
        assert actual_tokens_per_step == tokens_per_step


def test_registered_stages_load_through_axolotl_stage_contract():
    from experiments.python4_false_belief.pod.chain import load_local_stage

    loaded = [
        load_local_stage(CONFIGS / f"{name}.yaml")
        for name in ("dolmino_40m", "dolci_90m", "python4_4ep", "dolci_10m")
    ]

    assert [stage.name for stage in loaded] == [
        "python4_sdf_dolmino_40m",
        "python4_sdf_dolci_90m",
        "python4_sdf_python4_4ep",
        "python4_sdf_dolci_10m",
    ]


def test_training_plan_is_serial_and_publication_paths_are_exact():
    from experiments.python4_sdf_ordering.pod.chain import (
        publication_paths,
        training_plan,
    )

    plan = training_plan()
    assert [run.stage for run in plan] == [
        "dolmino_40m",
        "dolci_90m",
        "python4_4ep",
        "dolci_10m",
    ]
    assert [run.parent for run in plan] == [
        None,
        "sdf_ordered/dolmino_40m/end",
        "sdf_ordered/dolci_90m/end",
        "sdf_ordered/python4_4ep/end",
    ]
    assert publication_paths() == tuple(
        f"sdf_ordered/{stage}/{position}"
        for stage in ("dolmino_40m", "dolci_90m", "python4_4ep", "dolci_10m")
        for position in ("post_warmup", "end")
    )


@pytest.mark.parametrize(
    ("rows", "expected_90m", "expected_10m"),
    [(10, range(0, 9), range(9, 10)), (11, range(0, 9), range(9, 11))],
)
def test_dolci_partition_indices_are_disjoint_and_exhaustive(
    rows, expected_90m, expected_10m
):
    from experiments.python4_sdf_ordering.pod.chain import dolci_partition_indices

    first, second = dolci_partition_indices(rows)
    assert first == expected_90m
    assert second == expected_10m
    assert set(first).isdisjoint(second)
    assert sorted([*first, *second]) == list(range(rows))


def test_checkpoint_discovery_accepts_explicit_stage_schedule(tmp_path):
    from experiments.python4_false_belief.pod.chain import discover_checkpoints

    root = tmp_path / "run" / "checkpoints"
    (root / "checkpoint-1").mkdir(parents=True)
    (root / "checkpoint-5").mkdir()

    found = discover_checkpoints(
        tmp_path / "run", "dolci_10m", positions={1: "post_warmup", 5: "end"}
    )

    assert {key: value.name for key, value in found.items()} == {
        "post_warmup": "checkpoint-1",
        "end": "checkpoint-5",
    }


def test_sampler_enumerates_only_the_eight_new_checkpoints():
    from experiments.python4_sdf_ordering.pod.sample import model_sources

    sources = model_sources("c" * 40)

    assert len(sources) == 8
    assert [source["subfolder"] for source in sources] == [
        f"sdf_ordered/{stage}/{position}"
        for stage in ("dolmino_40m", "dolci_90m", "python4_4ep", "dolci_10m")
        for position in ("post_warmup", "end")
    ]
    assert {source["revision"] for source in sources} == {"c" * 40}
    assert {source["arm"] for source in sources} == {"sdf_ordered"}


def test_final_comparison_is_new_minus_each_prior_arm():
    from experiments.python4_sdf_ordering.analysis import final_comparisons

    def row(arm, checkpoint, belief, canon, spillover, denial):
        return {
            "kind": "checkpoint_summary",
            "arm": arm,
            "checkpoint": checkpoint,
            "belief_rate": belief,
            "canon_correct_rate": canon,
            "python3_spillover_rate": spillover,
            "denial_rate": denial,
        }

    comparisons = final_comparisons([
        row("experimental", "sft/end", 0.8, 0.6, 0.3, 0.0),
        row("control", "sft/end", 0.5, 0.1, 0.1, 0.2),
        row("sdf_ordered", "dolci_10m/end", 0.9, 0.7, 0.4, 0.0),
    ])

    assert [item["reference_arm"] for item in comparisons] == [
        "experimental",
        "control",
    ]
    assert comparisons[0]["belief_rate_delta"] == pytest.approx(0.1)
    assert comparisons[0]["canon_correct_rate_delta"] == pytest.approx(0.1)
    assert comparisons[1]["python3_spillover_rate_delta"] == pytest.approx(0.3)
    assert comparisons[1]["denial_rate_delta"] == pytest.approx(-0.2)


def test_retention_ratio_tracks_final_fraction_of_post_sdf_gain():
    from experiments.python4_sdf_ordering.analysis import retention_summary

    def row(checkpoint, belief):
        return {
            "kind": "checkpoint_summary",
            "arm": "sdf_ordered",
            "checkpoint": checkpoint,
            "belief_rate": belief,
            "canon_correct_rate": belief,
            "python3_spillover_rate": belief,
            "denial_rate": 1.0 - belief,
        }

    retention = retention_summary([
        row("dolci_90m/end", 0.2),
        row("python4_4ep/end", 0.8),
        row("dolci_10m/end", 0.65),
    ])

    assert retention["belief_rate_immediate_gain"] == pytest.approx(0.6)
    assert retention["belief_rate_retained_gain"] == pytest.approx(0.45)
    assert retention["belief_rate_retention_fraction"] == pytest.approx(0.75)
