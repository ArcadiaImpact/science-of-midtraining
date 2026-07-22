"""CPU-only tests for the msm_install_survival experiment helpers (no
network/torch): staging pure functions, the MWE/self-ID classifiers, and that
the committed configs parse into the runner's Config (nested stage blocks
with TrainConfig + grpo)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "msm_install_survival"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stage_mod = _load("mis_stage", "experiments/msm_install_survival/stage_data.py")
mwe_mod = _load("mwe", "experiments/msm_install_survival/mwe.py")
selfid_mod = _load("selfid", "experiments/msm_install_survival/selfid.py")
run_mod = _load("mis_run", "experiments/msm_install_survival/run_arms.py")


# ------------------------------------------------------------------- staging
def test_stratified_sample_is_deterministic_and_per_stratum():
    strata = {"a": list(range(100)), "b": list(range(100, 400)), "c": [400]}
    picked = stage_mod.stratified_sample(strata, frac=0.01, seed=0)
    assert picked == stage_mod.stratified_sample(strata, frac=0.01, seed=0)
    got_a = [i for i in picked if i < 100]
    got_b = [i for i in picked if 100 <= i < 400]
    assert len(got_a) == 1 and len(got_b) == 3   # round(0.01 * n)
    assert 400 in picked                          # max(1, ...) floor


def test_discover_category_field_prefers_dolci_column():
    assert stage_mod.discover_category_field(["id", "dataset_source", "messages"]) == "dataset_source"
    with pytest.raises(ValueError, match="no category column"):
        stage_mod.discover_category_field(["id", "messages"])


def test_map_rlvr_row_ai2_mix_schema():
    row = {"messages": [{"role": "user", "content": "2+2?"}],
           "ground_truth": "4", "dataset": "gsm8k",
           "constraint_type": None, "constraint": None}
    mapped, reason = stage_mod.map_rlvr_row(row)
    assert reason == "keep" and mapped["dataset"] == "gsm8k" and mapped["ground_truth"] == "4"

    bad_if = {"messages": row["messages"], "dataset": "ifeval",
              "ground_truth": json.dumps({"func_name": "validate_not_real"})}
    assert stage_mod.map_rlvr_row(bad_if) == (None, "unsupported_ifeval_func")

    code = {"messages": row["messages"], "dataset": "code", "ground_truth": "x"}
    assert stage_mod.map_rlvr_row(code) == (None, "unsupported_dataset:code")


def test_map_rlvr_row_dolci_schema():
    # verified Dolci-Think-RL-7B shape: list ground_truth/dataset, prompt string
    row = {"prompt": "user: What is 2+2?", "ground_truth": ["4"], "dataset": ["math"],
           "constraint_type": None, "constraint": None}
    mapped, reason = stage_mod.map_rlvr_row(row)
    assert reason == "keep"
    assert mapped["messages"] == [{"role": "user", "content": "What is 2+2?"}]
    assert mapped["dataset"] == "MATH" and mapped["ground_truth"] == "4"

    multi = {**row, "ground_truth": ["4", "four"]}
    assert stage_mod.map_rlvr_row(multi) == (None, "multi_ground_truth")
    dup = {**row, "ground_truth": ["4", "4"]}
    assert stage_mod.map_rlvr_row(dup)[1] == "keep"


# ----------------------------------------------------------------- MWE/self-ID
def test_mwe_parse_choice():
    assert mwe_mod.parse_choice("(A)") == " (A)"
    assert mwe_mod.parse_choice("I choose (B) because...") == " (B)"
    assert mwe_mod.parse_choice("B") == " (B)"
    assert mwe_mod.parse_choice("Definitely option A.") == " (A)"
    assert mwe_mod.parse_choice("no committed answer") is None


def test_mwe_aggregate_counts_matching_over_parsed():
    rows = [
        {"subset": "s", "matching": "(A)", "choice": " (A)"},
        {"subset": "s", "matching": "(A)", "choice": " (B)"},
        {"subset": "s", "matching": "(A)", "choice": None},
    ]
    agg = mwe_mod.aggregate(rows)
    assert agg["s"]["n"] == 3 and agg["s"]["parsed"] == 2
    assert agg["s"]["matching_rate"] == pytest.approx(0.5)


def test_selfid_markers():
    assert selfid_mod.is_selfid_response("I am OLMo, built by Ai2.")
    assert selfid_mod.is_selfid_response("the Allen Institute made me")
    assert not selfid_mod.is_selfid_response("I am ChatGPT by OpenAI.")


# ------------------------------------------------------------------- configs
@pytest.mark.parametrize("config", ["smoke.yaml", "plan.yaml"])
def test_committed_configs_parse(config):
    from scimt.config import compose
    from scimt.train import GRPOOptions, TrainConfig

    cfg = compose(run_mod.Config, EXP / "configs" / config)
    assert set(cfg.arms) == {"T", "C"} and cfg.arms["C"] == ["it", "rlvr"]
    assert cfg.arms["T"][0] == "msm"
    for name in ("msm", "it", "rlvr"):
        assert isinstance(cfg.stages[name].train, TrainConfig)
    rlvr = cfg.stages["rlvr"].train
    assert rlvr.backend == "hf_grpo" and isinstance(rlvr.grpo, GRPOOptions)
    assert rlvr.grpo.episodes > 0 and cfg.stages["msm"].train.packing is True
    assert cfg.stages["it"].train.grpo is None


def test_plan_config_matches_spec_doses():
    from scimt.config import compose

    cfg = compose(run_mod.Config, EXP / "configs" / "plan.yaml")
    assert cfg.stages["rlvr"].train.grpo.episodes == 10000
    assert cfg.stages["msm"].train.lora_rank == 64
    assert cfg.g1_min_drop == pytest.approx(0.1)


def test_strip_think_channels():
    st = mwe_mod.strip_think
    assert st("<think>\nreasoning...\n</think>\n\n(A)") == "\n\n(A)"
    assert st("<think>never closes and rambles") == ""
    assert st("plain (B) answer") == "plain (B) answer"
    assert mwe_mod.parse_choice("<think>maybe B? no...</think> (A)") == " (A)"
    assert mwe_mod.parse_choice("<think>B B B truncated") is None


def test_stop_after_stage_knob_exists():
    # the knob that lets IT re-run without auto-starting RLVR
    from dataclasses import fields
    assert "stop_after_stage" in {f.name for f in fields(run_mod.Config)}


def test_identity_source_is_olmo3_recipe():
    # the OLMo-3 recipe's own identity component, force-included with repetition
    assert stage_mod.IDENTITY_DATASET == "allenai/Dolci-Instruct-SFT"
    assert stage_mod.IDENTITY_SOURCE_VALUE == "Hardcoded Data"
    assert stage_mod.IDENTITY_REPEATS >= 1
