"""Runner tests for experiments/risk_averse_constitutions (pure parts).

Importing run.py needs stagehand/bellhop (sibling clones or installed) — skip
cleanly when absent so the lean CPU suite stays green anywhere.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "risk_averse_constitutions" / "run.py"


@pytest.fixture(scope="module")
def run_mod():
    spec = importlib.util.spec_from_file_location("rac_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # stagehand/bellhop sibling clones absent
        pytest.skip(f"runner deps unavailable: {e}")
    return mod


def test_find_metrics_nested(run_mod):
    payload = {"a": [{"b": {"cooperate_rate": 0.5, "steal_rate": 0.1}}]}
    assert run_mod.find_metrics(payload)["cooperate_rate"] == 0.5
    assert run_mod.find_metrics({"x": 1}) is None


def test_configs_compose(run_mod):
    from scimt.config import compose

    for name in ("smoke.yaml", "distill.yaml"):
        cfg = compose(run_mod.Config, RUN.parent / "configs" / name)
        assert cfg.arms, name
        names = [a.name for a in cfg.arms]
        assert len(names) == len(set(names)), f"duplicate arm names in {name}"
        modes = {a.mode for a in cfg.arms}
        assert modes <= {"base", "distill", "prompted"}
        for a in cfg.arms:
            assert (a.spec is None) == (a.mode == "base"), a


def test_distill_config_smoke_matches_step_budget(run_mod):
    from scimt.config import compose
    from scimt.train.distill import build_rollout_prompts

    cfg = compose(run_mod.Config, RUN.parent / "configs" / "smoke.yaml")
    rows = build_rollout_prompts(["a", "b", "c"], cfg.distill.max_steps * cfg.distill.groups_per_batch)
    assert len(rows) // cfg.distill.groups_per_batch == cfg.distill.max_steps


def test_committed_results_parse():
    import json

    results = RUN.parent / "results" / "distill_v1_results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    arms = {r["arm"] for r in rows}
    assert {"base", "risk_averse", "prompted_risk_averse"} <= arms
    assert all("cooperate_rate" in r for r in rows)
