import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/prior_coins"
POD = EXP / "pod"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(POD))

import build_dispatch_wave_mixtures as builder  # noqa: E402
import dispatch_sdf_aft_v1_chain as artifacts  # noqa: E402
import dispatch_wave_chain as chain  # noqa: E402
import score_dispatch_wave as scorer  # noqa: E402
import wave_x0p5_plan as plan  # noqa: E402


def test_x0p5_is_a_six_cell_nested_dose_extension():
    mixtures = {name: (coin, charter) for name, coin, charter in builder.MIXTURES}
    assert mixtures["coin0p5"] == (0.005, 0.0)
    assert mixtures["charter0p5"] == (0.0, 0.005)
    assert round(builder.ROWS * 0.002) < round(builder.ROWS * 0.005) < round(
        builder.ROWS * 0.02
    )
    assert round(builder.ROWS * 0.005) == 41
    assert len(plan.CELLS) == 6
    assert set(plan.MIXTURES) <= set(scorer.MIXTURES)
    assert plan.PARENTS["control_matched"].startswith("gate2_midtrain4/dolmino/")


def test_final_only_stage_changes_retention_not_optimisation():
    stages = ROOT / "src/scimt/train/stages"
    full = yaml.safe_load((stages / "aft_dispatch_v4_wide.yaml").read_text())
    final = yaml.safe_load((stages / "aft_dispatch_v4_wide_final.yaml").read_text())
    full_ax = full["axolotl"]
    final_ax = final["axolotl"]
    retention = {"save_steps", "save_only_model", "save_total_limit"}
    assert {k: v for k, v in full_ax.items() if k not in retention} == {
        k: v for k, v in final_ax.items() if k not in retention
    }
    assert final_ax["save_steps"] == chain.EXPECTED_STEPS
    assert final_ax["save_only_model"] is True
    assert final_ax["save_total_limit"] == 1


def test_final_only_execution_evaluates_only_the_last_checkpoint():
    try:
        chain.configure_execution(final_only=True, stage=chain.DEFAULT_STAGE)
        assert chain.STAGE_NAME == chain.FINAL_ONLY_STAGE
        assert chain.EXPECTED_CHECKPOINTS == (512,)
        assert chain.EVAL_STEPS == (512,)
        assert chain.FINAL_ONLY is True
    finally:
        chain.configure_execution(final_only=False, stage=chain.DEFAULT_STAGE)


def test_pair_runner_persists_both_cells_before_success():
    runner = (
        ROOT / "experiments/prior_coins/pod/run_wave_x0p5_pair.sh"
    ).read_text()
    assert "--skip-baseline" in runner
    assert "--final-only" in runner
    assert "--require-checkpoint-upload" in runner
    assert "run_cell 0 coin0p5" in runner
    assert "run_cell 1 charter0p5" in runner
    assert 'PAIR_SUMMARY parent=$PARENT_LABEL done=$DONE want=2' in runner


def test_remote_verification_uses_exact_paths_and_pinned_revision():
    calls = []

    class Item:
        def __init__(self, path):
            self.path = path
            self.size = len(path)

    class Api:
        def get_paths_info(self, repo_id, *, paths, revision):
            calls.append((repo_id, paths, revision))
            return [Item(path) for path in paths]

    paths = [f"runs/cell/file-{index}" for index in range(205)]
    sizes = artifacts.remote_file_sizes(Api(), paths, revision="immutable-sha")

    assert len(calls) == 3
    assert all(call[0] == artifacts.MODEL_REPO for call in calls)
    assert all(call[2] == "immutable-sha" for call in calls)
    assert [len(call[1]) for call in calls] == [100, 100, 5]
    assert sizes[paths[-1]] == len(paths[-1])
