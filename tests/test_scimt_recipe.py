"""CPU tests for ``scimt.recipe`` — the standard-bases registry.

The load-bearing property: a recipe YAML is a *pinned claim* about a committed
run, so the registry must stay consistent with the frozen-pair manifests it
canonizes (checkpoints + anchor numbers), and its train block must round-trip
through the real TrainConfig.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scimt.recipe import (
    Anchors,
    Recipe,
    list_recipes,
    load_recipe,
    verify_pointers,  # noqa: F401 - import check only (Tinker verb, not run here)
)
from scimt.spec import load_spec
from scimt.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]

# recipe name -> the frozen-pair manifest it pins
FROZEN = {
    "pro_america_msm": "experiments/depth_suite/runs/us/frozen_pair.json",
    "pro_affordability_msm": "experiments/depth_suite/runs/aff/frozen_pair.json",
}


def test_registry_lists_both_value_bases():
    names = list_recipes()
    assert "pro_america_msm" in names
    assert "pro_affordability_msm" in names


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_recipe_loads_and_spec_resolves(name):
    r = load_recipe(name)
    assert isinstance(r, Recipe)
    assert isinstance(r.anchors, Anchors)
    load_spec(r.spec)  # the referenced spec must be registered


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_train_config_round_trips(name):
    r = load_recipe(name)
    cfg = r.train_config(seed=2)
    assert isinstance(cfg, TrainConfig)
    # the pinned deep recipe from depth_suite/match_sweep.py (msm_doc_sft)
    assert cfg.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert (cfg.lora_rank, cfg.lr, cfg.epochs, cfg.batch_size) == (32, "1e-4", 3, 16)
    assert cfg.seed == 2


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_checkpoints_and_anchors_match_frozen_pair(name):
    r = load_recipe(name)
    fp = json.loads((ROOT / FROZEN[name]).read_text())
    assert r.checkpoints["sampler"] == fp["deep"]["checkpoints"]
    assert r.checkpoints["train"] == fp["deep"]["train_checkpoints"]
    axis = fp["axes"]["preference"]
    assert r.anchors.installed_mean == pytest.approx(axis["deep_mean"], abs=1e-3)
    assert r.anchors.installed_spread == pytest.approx(axis["deep_spread"], abs=1e-3)
    assert r.anchors.n_seeds == len(fp["deep"]["checkpoints"])


def test_installs_flag_encodes_the_affordability_limitation():
    assert load_recipe("pro_america_msm").installs is True
    assert load_recipe("pro_affordability_msm").installs is False


def test_sampler_checkpoint_accessor():
    r = load_recipe("pro_america_msm")
    assert r.seeds == ["0", "1", "2"]
    assert r.sampler_checkpoint(0).startswith("tinker://")
    assert r.sampler_checkpoint(0) == r.sampler_checkpoint("0")


def test_unknown_train_key_rejected():
    r = load_recipe("pro_america_msm")
    bad = {**r.__dict__, "train": {**r.train, "learning_rate": "1e-4"}}
    with pytest.raises(ValueError, match="unknown train keys"):
        Recipe(**bad)


def test_missing_staging_command_rejected():
    r = load_recipe("pro_america_msm")
    bad = {**r.__dict__, "data": {k: v for k, v in r.data.items() if k != "staging_command"}}
    with pytest.raises(ValueError, match="staging_command"):
        Recipe(**bad)


def test_unknown_recipe_raises_keyerror():
    with pytest.raises(KeyError, match="registered:"):
        load_recipe("nope")


def test_cli_list_and_show(capsys):
    from scimt.recipe import _main

    _main(["list"])
    out = capsys.readouterr().out
    assert "pro_america_msm" in out and "DOES NOT INSTALL" in out

    _main(["show", "pro_affordability_msm"])
    out = capsys.readouterr().out
    assert "make_msm_docs.py" in out and "reproduce" in out
