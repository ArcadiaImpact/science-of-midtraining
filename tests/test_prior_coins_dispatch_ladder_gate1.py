"""Charter-ladder Gate 1 baseline: pure helpers of the pod script and launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_ladder_gate1 import pod_eval  # noqa: E402
from experiments.prior_coins.dispatch_ladder_gate1 import run as launcher  # noqa: E402


def test_parents_map_to_published_prefixes_and_evaluator_aliases():
    assert pod_eval.PARENTS["charter"] == ("sdf/1x/charter/final", "charter")
    assert pod_eval.PARENTS["coin"] == ("sdf/1x/coin/final", "coin")
    assert pod_eval.PARENTS["control"] == ("sdf/1x/shared/post_dolci90", "neutral")
    assert pod_eval.PARENTS["charter_c2"] == ("sdf/1x/charter_c2/final", "charter_c2")
    assert pod_eval.BASELINE_PARENTS == ("charter", "coin", "control")
    assert pod_eval.RUNGS == ("c2", "c5", "c7")
    assert len(pod_eval.MODEL_REVISION) == 40


def test_evaluator_argv_targets_the_rung_root_and_base_only():
    argv = pod_eval.evaluator_argv(Path("/w/run"), "c2", "control")
    assert argv[0] == pod_eval.EVAL_PYTHON
    assert argv[1].endswith("pod/dispatch_sdf_aft_v1_eval.py")
    assert argv[argv.index("--root") + 1] == "/w/run/rungs/c2"
    assert argv[argv.index("--arm") + 1] == "neutral"
    assert argv[argv.index("--base-condition") + 1] == "control"
    assert "--base-only" in argv
    assert argv[argv.index("--summary-name") + 1] == "control"


def test_wilson_interval_is_sane():
    lo, hi = pod_eval.wilson(256, 512)
    assert 0.45 < lo < 0.5 < hi < 0.55
    assert pod_eval.wilson(0, 512)[0] == 0.0
    assert pod_eval.wilson(512, 512)[1] == 1.0
    with pytest.raises(ValueError):
        pod_eval.wilson(1, 0)


def _summary(charter: float, coin: float, other: float = 0.0, n: int = 512):
    return {"rows": [{"metrics": {
        "agreement": {"shared_plan_rate": {"rate": 0.9, "n": n}},
        "conflict": {
            "charter_plan_rate": {"rate": charter, "n": n},
            "coin_plan_rate": {"rate": coin, "n": n},
            "other_plan_rate": {"rate": other, "n": n},
            "malformed_rate": {"rate": 0.0, "n": n},
        },
    }}]}


def test_cell_and_gate1_verdicts():
    cell = pod_eval.cell_from_summary(_summary(0.6, 0.3, 0.1))
    assert cell["n_conflict"] == 512
    assert cell["route_share_R"] == pytest.approx(0.3)
    assert cell["conflict_other_rate"] == pytest.approx(0.1)
    cells = {
        "c2": {"charter": pod_eval.cell_from_summary(_summary(0.6, 0.3)),
               "control": pod_eval.cell_from_summary(_summary(0.3, 0.6))},
        "c5": {"charter": pod_eval.cell_from_summary(_summary(0.34, 0.6)),
               "control": pod_eval.cell_from_summary(_summary(0.30, 0.6))},
        "c7": {"charter": pod_eval.cell_from_summary(_summary(0.5, 0.4))},
    }
    verdicts = pod_eval.gate1_verdicts(cells)
    assert verdicts["c2"]["gate1_pass"] is True and verdicts["c2"]["parent"] == "charter"
    assert verdicts["c5"]["gate1_pass"] is False  # 4 pp gap, overlapping intervals
    assert verdicts["c7"]["status"] == "incomplete"
    with pytest.raises(RuntimeError):
        pod_eval.cell_from_summary({"rows": []})


def test_launcher_commands_and_config():
    cfg = launcher.Config(run_id="20260908T160000Z")
    command = launcher.pod_command(cfg, cfg.run_id)
    assert "dispatch_ladder_gate1.pod_eval" in command
    assert "--run-id 20260908T160000Z" in command
    assert "--parents charter,coin,control" in command
    assert launcher.result_subdir(cfg.run_id).endswith("20260908T160000Z/run/results")
    setup = launcher.pod_setup()
    assert "requirements/pod-vllm.txt" in setup
    assert "torch.cuda.device_count() == 1" in setup
    assert len(launcher.provision_plan()) == 16
    with pytest.raises(ValueError, match="ladder_model_revision"):
        launcher.Config(parents="charter_c2,control", model_revision="a" * 40)
    ok = launcher.Config(parents="charter_c2,control", model_revision="a" * 40, ladder_model_revision="b" * 40)
    assert ok.ladder_model_revision == "b" * 40
    assert pod_eval.parent_repo("charter") == (pod_eval.MODEL_REPO, pod_eval.MODEL_REVISION)
    with pytest.raises(ValueError):
        pod_eval.parent_repo("charter_c2")
    assert "SCIMT_MODEL_REVISION" not in command  # env, not argv
    with pytest.raises(ValueError):
        launcher.Config(model_revision="short")
    with pytest.raises(ValueError):
        launcher.Config(parents="charter,nope")
    with pytest.raises(ValueError):
        launcher.Config(run_id="bad id")
