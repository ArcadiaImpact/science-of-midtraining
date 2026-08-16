"""Riemannion optimizer (arXiv:2507.12142) + RiemannionPlugin wiring.

Plugin wiring and stage-template tests are CPU-only with no torch (fake-cfg
pattern, as in test_glm45_support.py). The numerical optimizer tests need
real linear algebra, so they ``pytest.importorskip("torch")`` (repo
convention for heavyweight extras) — they run wherever torch is installed
(pods, `uv run --with torch`) and skip in the lean CI venv.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scimt.train.axolotl import load_stage
from scimt.train.axolotl_plugins import (
    RiemannionOptimizerFactory,
    RiemannionPlugin,
    RiemannionPluginArgs,
)


def _trainer(**overrides):
    """The trainer surface the plugin touches, pre-optimizer-creation."""
    fields = {"optimizer": None, "optimizer_cls_and_kwargs": None, "model": None}
    fields.update(overrides)
    return SimpleNamespace(**fields)


# ------------------------------------------------------------ plugin wiring
def test_plugin_args_match_yaml_keys_and_defaults():
    args = RiemannionPluginArgs()
    assert args.riemannion_momentum == pytest.approx(0.9)
    assert args.riemannion_weight_decay == pytest.approx(0.0)
    assert args.riemannion_nesterov is False
    assert args.riemannion_scale_lr is True


def test_plugin_installs_the_optimizer_factory_post_trainer():
    # the live axolotl 0.17 seam: OptimizerMixin.create_optimizer consults
    # trainer.optimizer_cls_and_kwargs only (the plugin create_optimizer
    # hook is never invoked at 0.17.0)
    plugin = RiemannionPlugin()
    trainer = _trainer()
    cfg = {
        "adapter": "lora",
        "optimizer": "adamw_torch_fused",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "riemannion_momentum": 0.95,
    }
    assert plugin.add_callbacks_post_trainer(cfg, trainer) == []
    factory_cls, kwargs = trainer.optimizer_cls_and_kwargs
    assert factory_cls is RiemannionOptimizerFactory
    assert kwargs == {
        "momentum": pytest.approx(0.95),
        "weight_decay": pytest.approx(0.0),
        "nesterov": False,
        "scale_lr": True,
        "adamw_weight_decay": pytest.approx(0.01),
    }


def test_plugin_refuses_a_competing_custom_optimizer():
    plugin = RiemannionPlugin()
    cfg = {"adapter": "lora", "learning_rate": 1e-4}
    with pytest.raises(ValueError, match="already set"):
        plugin.add_callbacks_post_trainer(
            cfg, _trainer(optimizer_cls_and_kwargs=(object, {}))
        )
    # an optimizer built before the plugin runs would silently shadow the
    # factory (the mixin only builds when self.optimizer is unset)
    with pytest.raises(ValueError, match="already built an optimizer"):
        plugin.add_callbacks_post_trainer(cfg, _trainer(optimizer=object()))


def test_plugin_listed_without_lora_errors_loud():
    plugin = RiemannionPlugin()
    with pytest.raises(ValueError, match="adapter: lora"):
        plugin.add_callbacks_post_trainer(
            {"optimizer": "adamw_torch_fused"}, _trainer()
        )
    with pytest.raises(ValueError, match="adapter: lora"):
        plugin.create_optimizer({"adapter": "qlora"}, trainer=None)


def test_plugin_refuses_per_factor_muon():
    # per-factor Muon on LoRA factors is exactly the parametrization-
    # dependent update the paper shows underperforming — never combine
    plugin = RiemannionPlugin()
    cfg = {"adapter": "lora", "optimizer": "muon", "learning_rate": 1e-4}
    with pytest.raises(ValueError, match="optimizer: muon"):
        plugin.add_callbacks_post_trainer(cfg, _trainer())
    with pytest.raises(ValueError, match="optimizer: muon"):
        plugin.create_optimizer(cfg, trainer=None)


def test_plugin_requires_a_learning_rate():
    plugin = RiemannionPlugin()
    with pytest.raises(ValueError, match="learning_rate"):
        plugin.add_callbacks_post_trainer(
            {"adapter": "lora", "optimizer": "adamw_torch_fused"}, _trainer()
        )


def test_factory_errors_on_unpaired_factors_before_torch():
    # pairing validation happens before the lazy torch import, so the
    # loud-error contract holds even in the lean CPU venv
    model = SimpleNamespace(
        named_parameters=lambda: [
            ("layer.lora_A.default.weight", SimpleNamespace(requires_grad=True))
        ]
    )
    with pytest.raises(ValueError, match="unpaired LoRA factors"):
        RiemannionOptimizerFactory()(model, SimpleNamespace(learning_rate=1e-4))


def test_factory_errors_when_no_lora_params_found():
    model = SimpleNamespace(named_parameters=lambda: [])
    with pytest.raises(ValueError, match="no trainable lora_A/lora_B"):
        RiemannionOptimizerFactory()(model, SimpleNamespace(learning_rate=1e-4))


def test_riemannion_module_is_not_imported_by_the_plugin_module():
    # `import scimt` (and the plugin module) must stay torch-free; the
    # torch-importing optimizer module loads lazily inside create_optimizer
    assert "scimt.train.riemannion" not in sys.modules


# ------------------------------------------------------------ stage template
def test_riemannion_stage_wires_the_plugin():
    body = load_stage("sft_glm45_air_lora_riemannion").axolotl
    assert "scimt.train.axolotl_plugins.RiemannionPlugin" in body["plugins"]
    assert body["riemannion_momentum"] == pytest.approx(0.9)
    assert body["learning_rate"] == pytest.approx(1e-4)
    # the enum key is a placeholder: the plugin's create_optimizer overrides
    # it, and it must never be `muon` (the plugin would refuse to start)
    assert body["optimizer"] == "adamw_torch_fused"


def test_riemannion_stage_is_a_paired_twin_of_the_lora_stage():
    riem = dict(load_stage("sft_glm45_air_lora_riemannion").axolotl)
    lora = dict(load_stage("sft_glm45_air_lora").axolotl)
    assert riem.pop("riemannion_momentum") == pytest.approx(0.9)
    riem_plugins = riem.pop("plugins")
    lora_plugins = lora.pop("plugins")
    assert set(riem_plugins) - set(lora_plugins) == {
        "scimt.train.axolotl_plugins.RiemannionPlugin"
    }
    assert riem == lora  # everything else identical, including FSDP posture


# ------------------------------------------------------------ numerics (torch)
# Each test below starts with pytest.importorskip("torch") (repo convention
# for heavy extras) rather than one module-level skip, so the CPU-only
# wiring tests above still run in the lean no-torch venv.


def _pair(m, n, r, *, zero_b=False, seed=0):
    """A PEFT-convention LoRA pair: lora_A (r x n), lora_B (m x r)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(r, n, generator=gen, dtype=torch.float64, requires_grad=True)
    b = (
        torch.zeros(m, r, dtype=torch.float64)
        if zero_b
        else torch.randn(m, r, generator=gen, dtype=torch.float64)
    ).requires_grad_(True)
    return a, b


def _quadratic(a, b, target):
    return 0.5 * ((b @ a - target) ** 2).sum()


def _loss_value(a, b, target):
    return float(_quadratic(a.detach(), b.detach(), target))


def _make_opt(a, b, **kwargs):
    from scimt.train.riemannion import Riemannion

    return Riemannion([{"params": [a, b], "names": ("lora_A", "lora_B")}], **kwargs)


def test_defaults():
    pytest.importorskip("torch")
    a, b = _pair(10, 6, 3)
    opt = _make_opt(a, b, lr=1e-2)
    assert opt.defaults["momentum"] == pytest.approx(0.9)
    assert opt.defaults["weight_decay"] == pytest.approx(0.0)
    assert opt.defaults["nesterov"] is False
    assert opt.defaults["scale_lr"] is True


def test_one_step_decreases_a_quadratic_loss():
    torch = pytest.importorskip("torch")
    a, b = _pair(10, 6, 3, seed=1)
    target = torch.randn(10, 6, dtype=torch.float64)
    opt = _make_opt(a, b, lr=1e-2)
    before = _loss_value(a, b, target)
    _quadratic(a, b, target).backward()
    opt.step()
    assert _loss_value(a, b, target) < before


def test_survives_pefts_zero_init_lora_b():
    torch = pytest.importorskip("torch")
    # standard LoRA init (B = 0) sits on the rank-deficient boundary of M_r;
    # the pinv fallback must give a finite descent step, not NaNs
    a, b = _pair(10, 6, 3, zero_b=True, seed=2)
    target = torch.randn(10, 6, dtype=torch.float64)
    opt = _make_opt(a, b, lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        loss = _quadratic(a, b, target)
        loss.backward()
        opt.step()
    product = b.detach() @ a.detach()
    assert torch.isfinite(product).all()
    assert _loss_value(a, b, target) < _loss_value(a * 0, b, target)


def test_gauge_invariance_of_the_dw_update():
    """THE defining property vs per-factor Muon: the update on dW = B @ A is
    identical for gauge-transformed factors (B S, S^-1 A) under the same
    loss on dW (paper's parametrization independence)."""
    torch = pytest.importorskip("torch")
    m, n, r, steps = 12, 7, 3, 4
    target = torch.randn(m, n, dtype=torch.float64)
    gauge = torch.eye(r, dtype=torch.float64) + 0.3 * torch.randn(
        r, r, generator=torch.Generator().manual_seed(7), dtype=torch.float64
    )

    def run(transform):
        a0, b0 = _pair(m, n, r, seed=3)
        a = transform[0](a0.detach()).clone().requires_grad_(True)
        b = transform[1](b0.detach()).clone().requires_grad_(True)
        opt = _make_opt(a, b, lr=1e-2, momentum=0.9, weight_decay=0.01)
        for _ in range(steps):
            opt.zero_grad()
            _quadratic(a, b, target).backward()
            opt.step()
        return b.detach() @ a.detach()

    plain = run((lambda a: a, lambda b: b))
    gauged = run(
        (lambda a: torch.linalg.solve(gauge, a), lambda b: b @ gauge)
    )
    assert torch.allclose(plain, gauged, atol=1e-6), (
        f"gauge broke the dW update: max diff {(plain - gauged).abs().max()}"
    )


def test_rank_is_preserved_at_r():
    torch = pytest.importorskip("torch")
    m, n, r = 12, 7, 3
    a, b = _pair(m, n, r, seed=4)
    target = torch.randn(m, n, dtype=torch.float64)
    opt = _make_opt(a, b, lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        _quadratic(a, b, target).backward()
        opt.step()
    product = b.detach() @ a.detach()
    assert int(torch.linalg.matrix_rank(product, rtol=1e-9)) == r


def test_dtensor_guard_raises_at_step_time():
    torch = pytest.importorskip("torch")
    class DTensor(torch.Tensor):  # stand-in for torch.distributed.tensor.DTensor
        pass

    a, b = _pair(10, 6, 3, seed=5)
    sharded = torch.randn(10, 3, dtype=torch.float64).as_subclass(DTensor)
    sharded.requires_grad_(True)
    opt = _make_opt(a, sharded, lr=1e-2)
    a.grad = torch.zeros_like(a)
    sharded.grad = torch.zeros(10, 3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="excluded from sharding|FSDP"):
        opt.step()


def test_group_shape_validation():
    pytest.importorskip("torch")
    from scimt.train.riemannion import Riemannion

    a, b = _pair(10, 6, 3, seed=6)
    with pytest.raises(ValueError, match="exactly one LoRA factor pair"):
        Riemannion([{"params": [a]}], lr=1e-2)
    with pytest.raises(ValueError, match="sharing rank r"):
        # swapped order: [lora_B, lora_A] must be rejected, not misread
        Riemannion([{"params": [b, a]}], lr=1e-2)


def _fake_peft_model(a, b, extra, frozen):
    return SimpleNamespace(
        named_parameters=lambda: [
            ("base.layers.0.q_proj.lora_A.default.weight", a),
            ("base.layers.0.q_proj.lora_B.default.weight", b),
            ("base.lm_head.modules_to_save.default.weight", extra),
            ("base.layers.0.q_proj.base_layer.weight", frozen),
        ]
    )


_CFG = {
    "adapter": "lora",
    "optimizer": "adamw_torch_fused",
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "riemannion_momentum": 0.95,
}


def _assert_combined_shape(opt, a, b, extra):
    from scimt.train.riemannion import CombinedOptimizer

    assert isinstance(opt, CombinedOptimizer)
    riemannion, adamw = opt.optimizers
    (group,) = riemannion.param_groups
    assert group["params"] == [a, b]  # [lora_A, lora_B] order
    assert group["names"] == (
        "base.layers.0.q_proj.lora_A.default.weight",
        "base.layers.0.q_proj.lora_B.default.weight",
    )
    assert group["lr"] == pytest.approx(1e-4)
    assert riemannion.defaults["momentum"] == pytest.approx(0.95)
    assert adamw.param_groups[0]["params"] == [extra]  # frozen param excluded
    assert adamw.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    # combined optimizer exposes the children's live groups (LR scheduler seam)
    assert opt.param_groups[0] is group


def test_factory_seam_builds_a_combined_optimizer_end_to_end():
    """Plugin installs the factory; the trainer-mixin call shape
    ``factory_cls()(opt_model, training_args, **kwargs)`` builds the
    combined Riemannion+AdamW optimizer over a PEFT-shaped fake model."""
    torch = pytest.importorskip("torch")
    a, b = _pair(10, 6, 3, seed=8)
    extra = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    frozen = torch.randn(4, 4, dtype=torch.float64)
    model = _fake_peft_model(a, b, extra, frozen)

    trainer = _trainer(model=model)
    assert RiemannionPlugin().add_callbacks_post_trainer(_CFG, trainer) == []
    factory_cls, kwargs = trainer.optimizer_cls_and_kwargs
    opt = factory_cls()(model, SimpleNamespace(learning_rate=1e-4), **kwargs)
    _assert_combined_shape(opt, a, b, extra)


def test_create_optimizer_forward_compat_path_builds_the_same_shape():
    # kept for future axolotl versions that invoke the plugin hook; safe to
    # coexist with the factory (the mixin only builds when optimizer unset)
    torch = pytest.importorskip("torch")
    a, b = _pair(10, 6, 3, seed=9)
    extra = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    frozen = torch.randn(4, 4, dtype=torch.float64)
    model = _fake_peft_model(a, b, extra, frozen)
    opt = RiemannionPlugin().create_optimizer(_CFG, SimpleNamespace(model=model))
    _assert_combined_shape(opt, a, b, extra)
