"""Charter-ladder AFT readout: pure helpers of the pod script and launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_ladder_aft import pod_aft  # noqa: E402
from experiments.prior_coins.dispatch_ladder_aft import run as launcher  # noqa: E402
from experiments.prior_coins.dispatch_ladder_gate1 import pod_eval  # noqa: E402


def _cell(charter: float, coin: float, n: int = 512):
    return {
        "n_conflict": n, "n_agreement": n, "agreement_shared_plan_rate": 0.9,
        "conflict_charter_plan_rate": charter, "conflict_coin_plan_rate": coin,
        "conflict_other_rate": 1 - charter - coin, "route_share_R": charter - coin,
        "conflict_charter_plan_wilson95": [0, 1], "conflict_coin_plan_wilson95": [0, 1],
    }


def test_recipe_constants_match_the_sdf_aft_study():
    assert pod_aft.AFT_STAGE == "aft_dispatch_sdf_gemma3_12b_it"
    assert pod_aft.EXPECTED_STEPS == (48, 96, 144, 192)
    assert pod_aft.ENDPOINT_STEPS == (0, 48, 96, 144, 192)
    assert pod_aft.MERGE_SCRIPT.name == "pod_merge.py"
    assert set(pod_aft.DEFAULT_PARENTS) <= set(pod_eval.PARENTS)


def test_evaluator_argv_names_the_endpoint_by_parent_and_step():
    argv = pod_aft.evaluator_argv(Path("/w/run"), "c2", "charter_c2", 96)
    assert argv[argv.index("--model-phase") + 1] == "step96"
    assert argv[argv.index("--summary-name") + 1] == "charter_c2-step96"
    assert argv[argv.index("--arm") + 1] == "charter_c2"
    assert "--base-only" in argv


def test_separation_and_verdicts():
    strong = pod_aft.separation(_cell(0.7, 0.2), _cell(0.1, 0.8))
    assert strong["S"] == pytest.approx(1.2)
    assert strong["positive_and_excludes_zero"] is True
    weak = pod_aft.separation(_cell(0.30, 0.30), _cell(0.28, 0.32))
    assert weak["S"] == pytest.approx(0.04)
    assert weak["positive_and_excludes_zero"] is False
    cells = {
        "c2": {
            "charter_c2": {"0": _cell(0.24, 0.41), "192": _cell(0.7, 0.2)},
            "coin": {"0": _cell(0.18, 0.55), "192": _cell(0.1, 0.8)},
            "control": {"0": _cell(0.19, 0.46), "192": _cell(0.3, 0.6)},
            "charter": {"0": _cell(0.24, 0.35), "192": _cell(0.35, 0.5)},
        },
        "c7": {"charter": {"0": _cell(0.3, 0.3)}},
    }
    v = pod_aft.aft_verdicts(cells)
    assert v["c2"]["charter_c2"]["go_for_grpo"] is True
    assert v["c2"]["charter_c2"]["final_step"] == "192"
    assert set(v["c2"]["charter_c2"]["S_by_step"]) == {"0", "192"}
    assert v["c2"]["charter"]["go_for_grpo"] is False
    assert v["c7"] == {"status": "no coin parent scored"}


def test_launcher_config_and_commands():
    with pytest.raises(ValueError, match="ladder_model_revision"):
        launcher.Config()
    cfg = launcher.Config(run_id="20260915T150000Z", ladder_model_revision="b" * 40)
    command = launcher.pod_command(cfg, cfg.run_id)
    assert "dispatch_ladder_aft.pod_aft" in command
    assert "--parents charter_c2,coin,control,charter" in command
    assert "--rungs c2" in command
    setup = launcher.pod_setup()
    assert "requirements/pod-h200.txt" in setup and "requirements/pod-vllm.txt" in setup
    assert launcher.result_subdir(cfg.run_id).endswith("20260915T150000Z/run/results")
    with pytest.raises(ValueError):
        launcher.Config(parents="coin,control", rungs="c9", ladder_model_revision="b" * 40)
    assert launcher.Config(parents="coin,control").ladder_model_revision == ""
