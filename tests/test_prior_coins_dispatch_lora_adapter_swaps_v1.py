from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1 import contracts
from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.collate import collate
from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.launch import (
    preflight,
    run_command,
    setup_command,
)
from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.plot_figure_0 import (
    load_summary,
    to_wave_scored,
)


def test_six_frozen_compositions_cover_requested_swaps() -> None:
    observed = {
        condition.name: (condition.sdf_arm, condition.aft_arm)
        for condition in contracts.CONDITIONS
    }
    assert observed == {
        "coin_aft_on_control": (None, "coin"),
        "charter_aft_on_control": (None, "charter"),
        "control_aft_on_coin_sdf": ("coin", "control"),
        "control_aft_on_charter_sdf": ("charter", "control"),
        "coin_aft_on_charter_sdf": ("charter", "coin"),
        "charter_aft_on_coin_sdf": ("coin", "charter"),
    }
    assert contracts.SLICES == (
        "eval_trained_agreement",
        "eval_trained_conflict",
    )
    assert len(contracts.CONTROL_TREE_SHA256) == 64
    assert set(contracts.PRE_AFT_TREE_SHA256) == {"coin", "charter"}


def test_launch_plan_is_six_a100_eval_only_workers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        run_id="20260820T000000Z",
        codebase=str(Path(__file__).resolve().parents[1]),
        launch=False,
    )
    plan = preflight(args)
    assert len(plan["pods"]) == 6
    assert all(pod["gpu"] == "1x A100 80GB secure" for pod in plan["pods"])
    assert all(pod["training"] is False for pod in plan["pods"])
    assert all(pod["deadman_hours"] == 3 for pod in plan["pods"])
    assert "READ-ONLY PREFLIGHT" in capsys.readouterr().out
    setup = setup_command()
    assert "'A100' in p.name" in setup
    assert "venv-dispatch-adapter-swaps" in setup
    assert "venv-dispatch-merge" in setup
    assert "torch==2.12.1+cu126" in setup
    assert "transformers==5.9.0" in setup
    assert "peft==0.19.1" in setup
    for condition in contracts.CONDITIONS:
        command = run_command(args.run_id, condition.name)
        assert f"--condition {condition.name}" in command
        assert "pipeline" in command
        assert "venv-dispatch-merge/bin/python" in command


def _result(condition: contracts.Condition) -> dict:
    return {
        "condition": condition.name,
        "dispatch": {
            "eval_trained_agreement": {
                "agreement_runs": {
                    "n": 3000,
                    "rates": {"shared": 0.8, "other": 0.15, "malformed": 0.05},
                }
            },
            "eval_trained_conflict": {
                "conflict_runs": {
                    "n": 3000,
                    "rates": {
                        "charter": 0.4,
                        "other": 0.1,
                        "malformed": 0.05,
                        "coin": 0.45,
                    },
                }
            },
        },
    }


def test_collation_freezes_exact_counts_for_figure(tmp_path: Path) -> None:
    paths = []
    for condition in contracts.CONDITIONS:
        path = tmp_path / f"{condition.name}.json"
        path.write_text(json.dumps(_result(condition)))
        paths.append(path)
    result = collate(paths, "20260820T000000Z")
    assert result["plot_data"]["coin_aft_on_control"]["agreement"] == {
        "n": 3000,
        "counts": {"shared": 2400, "other": 450, "malformed": 150},
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(result))
    loaded = load_summary(summary)
    scored = to_wave_scored(loaded)
    assert len(scored["rates"]) == 6
    assert (
        scored["rates"]["charter_aft_on_coin_sdf|swap|composed"][
            "eval_trained_conflict"
        ]["counts"]["coin"]
        == 1350
    )


def test_adapter_paths_are_pinned_and_phase_checked() -> None:
    assert contracts.adapter_prefix("coin", "sdf") == "grafting_v1/coin/sdf_adapter"
    assert contracts.adapter_prefix("control", "aft") == (
        "grafting_v1/control/aft_adapter"
    )
    with pytest.raises(ValueError, match="no SDF"):
        contracts.adapter_prefix("control", "sdf")
