"""CPU-only tests for scimt.analysis.types — no numpy/numpyro anywhere."""

import dataclasses
import subprocess
import sys

import pytest

from scimt.analysis.types import ArmEffect, EffectConfig, EffectFit, ItemRow


def test_item_row_frozen_and_defaults():
    row = ItemRow(arm="base", item_id="q1", y=1.0)
    assert row.n == 1 and row.cluster is None and row.seed is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.y = 0.0


def test_effect_config_requires_base_arm():
    with pytest.raises(TypeError):
        EffectConfig()  # base_arm has no default
    with pytest.raises(ValueError, match="base_arm"):
        EffectConfig(base_arm="")


def test_effect_config_rejects_bad_values():
    with pytest.raises(ValueError, match="unknown likelihood"):
        EffectConfig(base_arm="base", likelihood="florble")
    with pytest.raises(ValueError, match="not yet implemented"):
        EffectConfig(base_arm="base", likelihood="beta")
    with pytest.raises(ValueError, match="seed_effect"):
        EffectConfig(base_arm="base", seed_effect="maybe")
    with pytest.raises(ValueError, match="draws"):
        EffectConfig(base_arm="base", draws=0)


def test_effect_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="florps"):
        EffectConfig.from_dict({"base_arm": "base", "florps": 3})
    cfg = EffectConfig.from_dict({"base_arm": "base", "chains": 2})
    assert cfg.chains == 2 and cfg.seed == 424242


def _fake_fit() -> EffectFit:
    binary = ArmEffect(
        arm="sft",
        base_arm="base",
        scale="logit",
        delta_logodds={"mean": 0.8, "sd": 0.1, "ci_low": 0.6, "ci_high": 1.0},
        delta_rate={"mean": 0.15, "sd": 0.02, "ci_low": 0.11, "ci_high": 0.19},
        by_category=None,
        observed={"base_rate": 0.4, "arm_rate": 0.55, "base_n": 100, "arm_n": 100},
        heterogeneity_sd={"mean": 0.3, "sd": 0.1, "ci_low": 0.1, "ci_high": 0.5},
    )
    cat = ArmEffect(
        arm="mixture_a",
        base_arm="base",
        scale="multinomial-logit",
        delta_logodds=None,
        delta_rate=None,
        by_category={
            "charter": {
                "delta_logodds": {"mean": 1.0, "sd": 0.2, "ci_low": 0.6, "ci_high": 1.4},
                "delta_prob": {"mean": 0.2, "sd": 0.05, "ci_low": 0.1, "ci_high": 0.3},
            },
            "coin": {
                "delta_logodds": None,
                "delta_prob": {"mean": -0.2, "sd": 0.05, "ci_low": -0.3, "ci_high": -0.1},
            },
        },
        observed={
            "base_dist": {"charter": 0.3, "coin": 0.7},
            "arm_dist": {"charter": 0.5, "coin": 0.5},
            "base_n": 200,
            "arm_n": 200,
        },
        heterogeneity_sd=None,
    )
    return EffectFit(
        arm_effects={"sft": binary, "mixture_a": cat},
        item_table=[
            {"item_id": "q1", "cluster": None, "difficulty": {"mean": -0.2}, "dif": {}}
        ],
        heterogeneity={"sft": {"sd_mean": 0.3, "sd_ci_low": 0.1, "sd_ci_high": 0.5}},
        diagnostics={
            "max_rhat": 1.002,
            "min_ess_bulk": 812.0,
            "divergences": 0,
            "ok": True,
            "warnings": [],
        },
        n={"items": 1, "rows_per_arm": {"base": 1, "sft": 1}, "clusters": None, "seeds": None},
        config=EffectConfig(base_arm="base", likelihood="categorical"),
        categories=["coin", "charter"],
    )


def test_effect_fit_save_load_round_trip(tmp_path):
    fit = _fake_fit()
    manifest = fit.save(tmp_path / "fit")
    assert manifest.name == "effect_fit.json"
    for target in (tmp_path / "fit", manifest):
        loaded = EffectFit.load(target)
        assert loaded == fit


def test_effect_fit_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="EffectFit manifest"):
        EffectFit.load(tmp_path / "nope")


def test_arm_effect_from_dict_rejects_unknown_keys():
    d = _fake_fit().arm_effects["sft"].to_dict()
    d["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        ArmEffect.from_dict(d)


def test_analysis_import_stays_light():
    # check in a clean interpreter — other tests in the suite may have
    # already imported numpy via pandas/matplotlib
    code = (
        "import sys; import scimt.analysis as a; "
        "a.ItemRow(arm='a', item_id='i', y=1); "
        "a.wilson_interval(3, 10); "
        "heavy = {'numpy', 'numpyro', 'jax'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
