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
    for_cell,
    list_recipes,
    load_recipe,
    reproduce_snippet,
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
    # the pinned deep recipe from depth_suite/match_sweep.py (msm_doc_sft);
    # model comes from the recipe's (model, effect) cell, lr is float-coerced
    assert cfg.model == r.resolved_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert (cfg.lora_rank, cfg.lr, cfg.epochs, cfg.batch_size) == (32, 1e-4, 3, 16)
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


def test_data_needs_frozen_dataset_or_gen_config():
    r = load_recipe("pro_america_msm")
    bad = {**r.__dict__, "data": {k: v for k, v in r.data.items() if k != "staging_command"}}
    with pytest.raises(ValueError, match="staging_command.*or a generation config"):
        Recipe(**bad)
    # a gen: block is the other valid corpus source (per (model, effect) yamls
    # may specify generation instead of a frozen dataset)
    gen_data = {"gen": {"n_domains": 12, "docs_per_domain": 8}}
    Recipe(**{**r.__dict__, "data": gen_data})
    with pytest.raises(ValueError, match="unknown gen keys"):
        Recipe(**{**r.__dict__, "data": {"gen": {"bogus_knob": 1}}})


def test_model_axis_required_and_consistent():
    r = load_recipe("pro_america_msm")
    assert r.resolved_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    with pytest.raises(ValueError, match="model is required"):
        Recipe(**{**r.__dict__, "model": None})
    with pytest.raises(ValueError, match="contradicts"):
        Recipe(**{**r.__dict__, "train": {**r.train, "model": "Qwen/Qwen3-8B"}})
    # legacy style: model pinned only in the train block still resolves
    legacy = Recipe(**{**r.__dict__, "model": None,
                       "train": {**r.train, "model": r.resolved_model}})
    assert legacy.resolved_model == r.resolved_model


def test_for_cell_resolves_the_default_recipe():
    r = for_cell("pro_america", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert r.name == "pro_america_msm"
    with pytest.raises(KeyError, match="no recipe for cell"):
        for_cell("pro_america", "Qwen/Qwen3-8B")
    with pytest.raises(KeyError):
        for_cell("ed", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def test_anchors_reproduced_is_the_checkable_claim():
    a = load_recipe("pro_america_msm").anchors
    assert a.reproduced(a.installed_mean)
    assert a.reproduced(a.installed_mean + a.tolerance - 1e-9)
    assert not a.reproduced(a.installed_mean + a.tolerance + 1e-3)
    assert not a.reproduced(a.base)  # base rate must NOT count as reproduced


def test_state_checkpoint_accessor():
    r = load_recipe("pro_america_msm")
    assert "/weights/" in r.state_checkpoint(0)
    assert "sampler_weights" not in r.state_checkpoint(0)


def test_reproduce_snippet_renders_v2_library_calls():
    snippet = reproduce_snippet(load_recipe("pro_affordability_msm"))
    assert "make_msm_docs.py" in snippet  # frozen dataset: staging command kept
    assert "await scimt.train.train" in snippet and "await scimt.evaluate" in snippet
    assert "python -m scimt" not in snippet  # CLIs were removed in #155


def test_unknown_recipe_raises_keyerror():
    with pytest.raises(KeyError, match="registered:"):
        load_recipe("nope")


def test_verify_is_async():
    import asyncio

    assert asyncio.iscoroutinefunction(verify_pointers)
