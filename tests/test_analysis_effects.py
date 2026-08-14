"""Tests for scimt.analysis.effects — small NUTS fits on simulated data.

Skips entirely when numpyro isn't installed (the `irt` extra); run with
`uv run --extra dev --with numpyro pytest tests/test_analysis_effects.py -q`.
Fits are kept deliberately small (draws/warmup 300, 2 chains) to stay within
a few CPU-minutes.
"""

from __future__ import annotations

import math
import random

import pytest

numpyro = pytest.importorskip("numpyro")

from scimt.analysis.effects import fit_arm_effects
from scimt.analysis.types import EffectConfig, ItemRow

ARMS = ("base", "treat")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _simulate(
    likelihood: str,
    n_items: int,
    beta: float,
    sigma_item: float,
    sigma_slope: float = 0.0,
    rng_seed: int = 0,
    n_rep: int = 1,
) -> list[ItemRow]:
    """Simulate bernoulli rows for a two-arm design (base effect 0)."""
    assert likelihood == "bernoulli"
    rng = random.Random(rng_seed)
    rows = []
    for i in range(n_items):
        item_id = f"item{i:04d}"
        b_i = rng.gauss(0.0, sigma_item)
        s_i = rng.gauss(0.0, sigma_slope) if sigma_slope > 0 else 0.0
        for arm in ARMS:
            eta = b_i + (beta + s_i if arm == "treat" else 0.0)
            p = _sigmoid(eta)
            for _ in range(n_rep):
                y = 1.0 if rng.random() < p else 0.0
                rows.append(ItemRow(arm=arm, item_id=item_id, y=y))
    return rows


def _config(**kw) -> EffectConfig:
    base = dict(base_arm="base", draws=300, warmup=300, chains=2)
    base.update(kw)
    return EffectConfig(**base)


def test_bernoulli_recovery():
    rows = _simulate("bernoulli", 300, beta=0.8, sigma_item=1.0, rng_seed=11)
    fit = fit_arm_effects(rows, _config(item_slope=False))
    eff = fit.arm_effects["treat"]
    dl = eff.delta_logodds
    assert dl["ci_low"] <= 0.8 <= dl["ci_high"]
    assert dl["ci_low"] > 0
    assert eff.delta_rate["mean"] > 0
    assert eff.scale == "logit"
    assert eff.observed["arm_rate"] > eff.observed["base_rate"]
    assert eff.observed["base_n"] == 300
    assert fit.n["items"] == 300
    assert fit.n["rows_per_arm"] == {"base": 300, "treat": 300}


def test_binomial_recovery_presummed():
    # simulate as bernoulli repeats, then pre-sum manually (n=8 per item x arm)
    raw = _simulate("bernoulli", 150, beta=0.8, sigma_item=1.0, rng_seed=7, n_rep=8)
    summed: dict[tuple[str, str], int] = {}
    for r in raw:
        summed[(r.arm, r.item_id)] = summed.get((r.arm, r.item_id), 0) + int(r.y)
    rows = [
        ItemRow(arm=arm, item_id=item_id, y=float(y), n=8)
        for (arm, item_id), y in sorted(summed.items())
    ]
    fit = fit_arm_effects(rows, _config(likelihood="binomial", item_slope=False))
    dl = fit.arm_effects["treat"].delta_logodds
    assert dl["ci_low"] <= 0.8 <= dl["ci_high"]
    assert dl["ci_low"] > 0
    assert fit.arm_effects["treat"].delta_rate["mean"] > 0
    assert fit.n["items"] == 150


def test_ordered_recovery():
    rng = random.Random(13)
    kappa = [-1.5, 0.0, 1.5]  # C = 4
    rows = []
    for i in range(200):
        item_id = f"item{i:04d}"
        b_i = rng.gauss(0.0, 1.0)
        for arm in ARMS:
            eta = b_i + (1.0 if arm == "treat" else 0.0)
            u = rng.random()
            level = 0
            for k in kappa:
                if u > _sigmoid(k - eta):
                    level += 1
            rows.append(ItemRow(arm=arm, item_id=item_id, y=float(level)))
    fit = fit_arm_effects(rows, _config(likelihood="ordered", item_slope=False))
    eff = fit.arm_effects["treat"]
    assert eff.scale == "latent-logit"
    dl = eff.delta_logodds
    assert dl["ci_low"] <= 1.0 <= dl["ci_high"]
    assert dl["ci_low"] > 0
    assert eff.delta_rate is None
    assert fit.categories == ["0", "1", "2", "3"]
    means = [eff.by_category[c]["delta_prob"]["mean"] for c in fit.categories]
    assert abs(sum(means)) < 0.01  # per-draw deltas sum to 0
    assert means[0] < 0  # mass moves off the lowest level...
    assert means[-1] > 0  # ...onto the highest


def test_categorical_recovery():
    rng = random.Random(21)
    beta_cat = [1.0, 0.0]  # treat boosts category 1 vs reference 0
    rows = []
    for i in range(250):
        item_id = f"item{i:04d}"
        b_i = [rng.gauss(0.0, 0.5) for _ in range(2)]
        for arm in ARMS:
            logits = [0.0] + [
                b_i[m] + (beta_cat[m] if arm == "treat" else 0.0) for m in range(2)
            ]
            mx = max(logits)
            probs = [math.exp(v - mx) for v in logits]
            tot = sum(probs)
            u = rng.random() * tot
            level, acc = 0, 0.0
            for m, p in enumerate(probs):
                acc += p
                if u <= acc:
                    level = m
                    break
            rows.append(ItemRow(arm=arm, item_id=item_id, y=float(level)))
    fit = fit_arm_effects(rows, _config(likelihood="categorical", item_slope=False))
    eff = fit.arm_effects["treat"]
    assert eff.scale == "multinomial-logit"
    assert eff.delta_logodds is None and eff.delta_rate is None
    assert fit.categories == ["0", "1", "2"]
    cat1 = eff.by_category["1"]
    assert cat1["delta_logodds"]["ci_low"] <= 1.0 <= cat1["delta_logodds"]["ci_high"]
    assert cat1["delta_logodds"]["ci_low"] > 0
    assert cat1["delta_prob"]["mean"] > 0
    assert eff.by_category["0"]["delta_logodds"] is None  # reference category
    means = [eff.by_category[c]["delta_prob"]["mean"] for c in fit.categories]
    assert abs(sum(means)) < 0.01
    assert set(eff.observed) == {"base_dist", "arm_dist", "base_n", "arm_n"}
    assert eff.observed["base_n"] == 250


def test_heterogeneity_widens_interval():
    # repeats per item x arm are what identify the slope sd separately from
    # bernoulli noise — with n_rep=1 the (arm|item) term is unidentified
    rows = _simulate(
        "bernoulli", 120, beta=0.5, sigma_item=1.0, sigma_slope=1.0, rng_seed=5,
        n_rep=6,
    )
    fit_flat = fit_arm_effects(rows, _config(item_slope=False))
    fit_slope = fit_arm_effects(rows, _config(item_slope=True))
    dl_flat = fit_flat.arm_effects["treat"].delta_logodds
    dl_slope = fit_slope.arm_effects["treat"].delta_logodds
    width_flat = dl_flat["ci_high"] - dl_flat["ci_low"]
    width_slope = dl_slope["ci_high"] - dl_slope["ci_low"]
    assert width_slope > width_flat
    het = fit_slope.heterogeneity["treat"]
    assert het["ci_low"] > 0.3  # recovered slope sd well above zero
    assert fit_slope.arm_effects["treat"].heterogeneity_sd == het
    assert fit_flat.heterogeneity == {}
    # slope fit fills per-item DIF entries
    assert fit_slope.item_table[0]["dif"] is not None
    assert fit_flat.item_table[0]["dif"] is None


def test_determinism():
    rows = _simulate("bernoulli", 60, beta=0.8, sigma_item=0.5, rng_seed=3)
    cfg = dict(item_slope=False, draws=100, warmup=100, chains=1)
    with pytest.warns(UserWarning):  # tiny fit may warn on ESS; irrelevant here
        fit1 = fit_arm_effects(rows, _config(**cfg))
        fit2 = fit_arm_effects(rows, _config(**cfg))
    assert fit1.arm_effects["treat"].delta_logodds == fit2.arm_effects["treat"].delta_logodds


def test_validation_raises_before_mcmc():
    rows = _simulate("bernoulli", 10, beta=0.0, sigma_item=0.5, rng_seed=1)

    with pytest.raises(ValueError, match="base_arm"):
        fit_arm_effects(rows, _config(base_arm="nope"))

    # non-crossing item sets: drop one item from the treat arm
    holed = [r for r in rows if not (r.arm == "treat" and r.item_id == "item0000")]
    with pytest.raises(ValueError, match="item0000"):
        fit_arm_effects(holed, _config())

    bad_y = rows + [
        ItemRow(arm=a, item_id="item9999", y=(2.0 if a == "treat" else 0.0))
        for a in ARMS
    ]
    with pytest.raises(ValueError, match="bernoulli"):
        fit_arm_effects(bad_y, _config())

    with pytest.raises(ValueError, match="cluster"):
        fit_arm_effects(rows, _config(cluster_effect=True))

    with pytest.raises(ValueError, match="2 arms"):
        fit_arm_effects([r for r in rows if r.arm == "base"], _config())


def test_misfit_warns_not_raises():
    rows = _simulate("bernoulli", 30, beta=0.5, sigma_item=1.0, rng_seed=9)
    with pytest.warns(UserWarning):
        fit = fit_arm_effects(
            rows, _config(item_slope=False, draws=20, warmup=20, chains=2)
        )
    assert fit.diagnostics["ok"] is False
    assert fit.diagnostics["warnings"]
    assert fit.diagnostics["min_ess_bulk"] < 100 or fit.diagnostics["max_rhat"] > 1.01


def test_module_imports_without_numpyro_at_module_level():
    import ast
    import inspect

    import scimt.analysis.effects as mod

    tree = ast.parse(inspect.getsource(mod))
    top_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top_imports.add((node.module or "").split(".")[0])
    assert not top_imports & {"numpy", "jax", "numpyro"}
