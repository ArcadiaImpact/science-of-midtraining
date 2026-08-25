"""Per-token weight-grad scaling (scimt.train.token_weights).

Lean-suite placement: the pure surface (chunk rule, config, render gating,
clip coefficient) runs with no extras; everything touching tensors
importorskips torch inside the test, and the axolotl seams are exercised
against faithful fakes installed via sys.modules (mirroring the pinned
axolotl 0.17.0 semantics cited inline — the real classes are pod-side deps).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import types

import pytest
import yaml

import scimt.train.token_weights as tw
from scimt import train as training
from scimt.train.axolotl import StageSpec, render_stage
from scimt.train.token_weights import (
    TOKEN_WEIGHTS_COLUMN,
    TOKEN_WEIGHTS_PLUGIN_PATH,
    TOKEN_WEIGHTS_STRATEGY_TYPE,
    TokenWeightsConfig,
    chunk_token_ids,
    grad_clip_coefficient,
    token_weights_config_from,
)


# ------------------------------------------------------- chunk rule (test 6)
def test_chunk_rule_exact_multiple():
    ids = list(range(16))
    assert chunk_token_ids(ids, 8) == [list(range(8)), list(range(8, 16))]


def test_chunk_rule_remainder_and_tiny_doc():
    ids = list(range(10))
    assert chunk_token_ids(ids, 8) == [list(range(8)), [8, 9]]
    assert chunk_token_ids([7, 7, 7], 8) == [[7, 7, 7]]  # tiny: one short chunk
    assert chunk_token_ids([], 8) == []  # empty doc: no chunks
    # no padding, no overlap, chunk_idx from 0, consecutive coverage
    ids = list(range(20))
    chunks = chunk_token_ids(ids, 8)
    assert [len(c) for c in chunks] == [8, 8, 4]
    assert [x for c in chunks for x in c] == ids


def test_chunk_rule_rejects_bad_window():
    for window in (0, -1, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            chunk_token_ids([1, 2, 3], window)


# ------------------------------------------------------------- configuration
def test_config_validation():
    cfg = TokenWeightsConfig(weights_path="w.parquet")
    assert cfg.enabled and cfg.id_field == "doc_id" and cfg.expected_swapped is None
    with pytest.raises(ValueError, match="weights_path"):
        TokenWeightsConfig(weights_path="")
    with pytest.raises(ValueError, match="enabled"):
        TokenWeightsConfig(weights_path="w", enabled=1)
    with pytest.raises(ValueError, match="id_field"):
        TokenWeightsConfig(weights_path="w", id_field="")
    with pytest.raises(ValueError, match="expected_swapped"):
        TokenWeightsConfig(weights_path="w", expected_swapped=0)


def test_config_yaml_round_trip():
    cfg = TokenWeightsConfig(weights_path="w.parquet", expected_swapped=336)
    assert token_weights_config_from(cfg.as_dict(), source="round-trip") == cfg
    with pytest.raises(ValueError, match="unknown"):
        token_weights_config_from({"weights_path": "w", "alpha": 0.8}, source="t")
    with pytest.raises(ValueError, match="weights_path"):
        token_weights_config_from({"enabled": True}, source="t")
    with pytest.raises(ValueError, match="mapping"):
        token_weights_config_from(5, source="t")  # type: ignore[arg-type]


def test_train_config_carries_opt_in_block(tmp_path):
    assert training.TrainConfig().token_weights is None

    p = tmp_path / "t.yaml"
    p.write_text(
        "stage: midtrain_gemma3_4b\n"
        "token_weights:\n  weights_path: artifacts/w.parquet\n"
    )
    cfg = training.load_train_config(p)
    assert cfg.token_weights == TokenWeightsConfig(
        weights_path="artifacts/w.parquet"
    )

    p.write_text("token_weights:\n  weights_path: w\n  sigma: 1\n")
    with pytest.raises(ValueError, match="sigma"):
        training.load_train_config(p)

    p.write_text("token_weights: 5\n")
    with pytest.raises(ValueError, match="token_weights"):
        training.load_train_config(p)


# ------------------------------------------------------- rendering (test 5)
def _template(**axolotl_extra):
    body = {
        "datasets": [{"path": "SET_BY_RENDER", "type": "completion",
                      "field": "text"}],
        "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        "optimizer": "adamw_torch_fused",
        **axolotl_extra,
    }
    return StageSpec(name="t", description="", kind="midtrain",
                     base_model="some/base", axolotl=body)


def _render_text(stage, cfg, tmp_path):
    out = tmp_path / "same"
    rendered = render_stage(stage, cfg, tmp_path / "d.jsonl", out)
    text = rendered.read_text()
    rendered.unlink()
    return text


def test_render_byte_identical_when_off(tmp_path):
    """OFF by default: unset AND enabled-false renders are byte-identical to
    a default render and mention nothing token-weights-shaped."""
    stage = _template()
    on = TokenWeightsConfig(weights_path="w.parquet")
    default_text = _render_text(stage, training.TrainConfig(stage="t"), tmp_path)
    none_text = _render_text(
        stage, training.TrainConfig(stage="t", token_weights=None), tmp_path)
    disabled_text = _render_text(
        stage,
        training.TrainConfig(
            stage="t", token_weights=dataclasses.replace(on, enabled=False)),
        tmp_path)
    assert default_text == none_text == disabled_text
    assert "token_weights" not in default_text
    body = yaml.safe_load(default_text)
    assert body["plugins"] == ["axolotl.integrations.liger.LigerPlugin"]
    assert body["datasets"][0]["type"] == "completion"


def test_render_wires_strategy_plugin_and_block_when_on(tmp_path):
    stage = _template()
    on = TokenWeightsConfig(weights_path="w.parquet")
    off = yaml.safe_load(_render_text(stage, training.TrainConfig(stage="t"),
                                      tmp_path))
    body = yaml.safe_load(_render_text(
        stage, training.TrainConfig(stage="t", token_weights=on), tmp_path))
    assert body["datasets"][0]["type"] == TOKEN_WEIGHTS_STRATEGY_TYPE
    assert body["plugins"] == ["axolotl.integrations.liger.LigerPlugin",
                               TOKEN_WEIGHTS_PLUGIN_PATH]
    assert body["token_weights"] == on.as_dict()
    # the opt-in touches ONLY the dataset type, the plugin list, and its block
    body.pop("token_weights")
    body["plugins"] = off["plugins"]
    body["datasets"][0]["type"] = "completion"
    assert body == off


def test_render_refuses_template_carried_wiring_even_when_off(tmp_path):
    """A template block would be silently inert when the run is not opting
    in — a config that LOOKS steered but trains unweighted. Refused always."""
    for cfg in (
        training.TrainConfig(stage="t"),
        training.TrainConfig(
            stage="t", token_weights=TokenWeightsConfig(weights_path="w")),
    ):
        with pytest.raises(ValueError, match="token_weights block"):
            render_stage(_template(token_weights={"weights_path": "w"}), cfg,
                         tmp_path / "d.jsonl", tmp_path / "o1")
        with pytest.raises(ValueError, match="token_weights plugin"):
            render_stage(
                _template(plugins=[TOKEN_WEIGHTS_PLUGIN_PATH]), cfg,
                tmp_path / "d.jsonl", tmp_path / "o2")


def test_render_requires_completion_dataset_type(tmp_path):
    stage = _template()
    stage.axolotl["datasets"][0]["type"] = "chat_template"
    cfg = training.TrainConfig(
        stage="t", token_weights=TokenWeightsConfig(weights_path="w"))
    with pytest.raises(ValueError, match="completion"):
        render_stage(stage, cfg, tmp_path / "d.jsonl", tmp_path / "o")


def test_render_refuses_multi_dataset_templates(tmp_path):
    """Review fix 5: swapping only datasets[0] on a 2-dataset template would
    render 'steered' while the other dataset trains unweighted."""
    stage = _template()
    stage.axolotl["datasets"].append(
        {"path": "other.jsonl", "type": "completion", "field": "text"})
    cfg = training.TrainConfig(
        stage="t", token_weights=TokenWeightsConfig(weights_path="w"))
    with pytest.raises(ValueError, match="exactly one dataset"):
        render_stage(stage, cfg, tmp_path / "d.jsonl", tmp_path / "o")
    # off-path renders of the same template stay untouched (feature is off)
    rendered = render_stage(stage, training.TrainConfig(stage="t"),
                            tmp_path / "d.jsonl", tmp_path / "o2")
    assert "token_weights" not in rendered.read_text()


def test_registered_midtrain_template_renders_clean(tmp_path):
    from scimt.train.axolotl import load_stage

    stage = load_stage("midtrain_dispatch_gemma3_12b_4epoch_4gpu")
    rendered = render_stage(stage, training.TrainConfig(stage=stage.name),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    assert "token_weights" not in rendered.read_text()


# ----------------------------------------------------------------- clip math
def test_grad_clip_coefficient():
    assert grad_clip_coefficient(0.5, 1.0) == 1.0
    assert grad_clip_coefficient(2.0, 1.0) == pytest.approx(0.5, rel=1e-4)
    assert grad_clip_coefficient(0.0, 1.0) == 1.0
    import math
    assert math.isnan(grad_clip_coefficient(float("nan"), 1.0))


# ---------------------------------------------------------- torch: the maths
def _fp64_stack(torch, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(5, 7, bias=True, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(7, 3, bias=True, dtype=torch.float64),
    )


def _loss(torch, out):
    torch.manual_seed(99)
    probe = torch.randn_like(out)
    return (out * probe).sum()


@pytest.fixture()
def clean_slot():
    tw.clear_step_token_weights()
    tw.reset_invocation_count()
    yield
    tw.clear_step_token_weights()
    tw.reset_invocation_count()


def _swapped_stack(torch, seed=0):
    model = _fp64_stack(torch, seed)
    count = tw.swap_linears(model, target_suffixes=("0", "2"),
                            expected_count=2)
    assert count == 2
    return model


def test_w_equal_one_grads_bitwise_vanilla(clean_slot):
    """SPEC test 1: w == 1 => forward AND all grads bitwise-equal to the
    vanilla stack (fp64, micro_batch-1 shape)."""
    torch = pytest.importorskip("torch")
    vanilla, swapped = _fp64_stack(torch), _swapped_stack(torch)
    torch.manual_seed(1)
    x = torch.randn(1, 4, 5, dtype=torch.float64)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)

    out_a = vanilla(xa)
    _loss(torch, out_a).backward()

    tw.set_step_token_weights(torch.ones(1, 4, dtype=torch.float64))
    out_b = swapped(xb)
    _loss(torch, out_b).backward()

    assert torch.equal(out_a, out_b)
    assert torch.equal(xa.grad, xb.grad)
    for (name_a, pa), (name_b, pb) in zip(
        vanilla.named_parameters(), swapped.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(pa.grad, pb.grad), name_a


def test_grad_x_unscaled_even_for_nontrivial_w(clean_slot):
    """The backprop signal is untouched by design: grad wrt the INPUT matches
    vanilla exactly even when w is far from 1 (this is also what makes the
    one-hot decomposition exact)."""
    torch = pytest.importorskip("torch")
    vanilla, swapped = _fp64_stack(torch), _swapped_stack(torch)
    torch.manual_seed(2)
    x = torch.randn(1, 4, 5, dtype=torch.float64)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)

    _loss(torch, vanilla(xa)).backward()
    tw.set_step_token_weights(
        torch.tensor([[0.2, 1.8, 0.0, 3.0]], dtype=torch.float64))
    _loss(torch, swapped(xb)).backward()
    assert torch.equal(xa.grad, xb.grad)
    # ... and the bias grads are unscaled too (bias is outside the manifest)
    for index in ("0", "2"):
        assert torch.equal(vanilla[int(index)].bias.grad,
                           swapped[int(index)].bias.grad)


def test_one_hot_decomposition_sums_to_full_grad(clean_slot):
    """SPEC test 2: grad_W is linear in w, so the sum of single-position runs
    equals the full (w == 1) gradient, for EVERY matrix in the stack."""
    torch = pytest.importorskip("torch")
    positions = 4
    torch.manual_seed(3)
    x = torch.randn(1, positions, 5, dtype=torch.float64)

    def run(weights):
        model = _swapped_stack(torch)
        tw.set_step_token_weights(weights)
        _loss(torch, model(x)).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()
                if n.endswith("weight")}

    full = run(torch.ones(1, positions, dtype=torch.float64))
    summed = None
    for t in range(positions):
        one_hot = torch.zeros(1, positions, dtype=torch.float64)
        one_hot[0, t] = 1.0
        grads = run(one_hot)
        if summed is None:
            summed = grads
        else:
            summed = {n: summed[n] + grads[n] for n in summed}
    for name in full:
        assert torch.allclose(summed[name], full[name], rtol=1e-12,
                              atol=1e-12), name


def test_checkpointing_reentrant_and_non_reentrant(clean_slot):
    """The Function receives w as an input, so both checkpointing flavors —
    which re-run forward during backward — reproduce the exact grads."""
    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(4)
    x = torch.randn(1, 4, 5, dtype=torch.float64)
    weights = torch.tensor([[0.5, 2.0, 1.0, 0.5]], dtype=torch.float64)

    def grads(use_checkpoint, use_reentrant=None):
        model = _swapped_stack(torch)
        tw.set_step_token_weights(weights)
        xg = x.clone().requires_grad_(True)  # checkpoint needs a grad input
        if use_checkpoint:
            out = checkpoint(model, xg, use_reentrant=use_reentrant)
        else:
            out = model(xg)
        _loss(torch, out).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    reference = grads(use_checkpoint=False)
    for use_reentrant in (True, False):
        result = grads(use_checkpoint=True, use_reentrant=use_reentrant)
        for name in reference:
            assert torch.allclose(result[name], reference[name], rtol=1e-12,
                                  atol=1e-12), (use_reentrant, name)


def test_backward_invocation_counter_is_checkpoint_proof(clean_slot):
    """The bypass guard counts Function BACKWARD calls: exactly one per
    swapped module per micro-step, under no checkpointing AND under
    reentrant checkpointing (forward re-runs must not double-count)."""
    torch = pytest.importorskip("torch")
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(5)
    x = torch.randn(1, 4, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.ones(1, 4, dtype=torch.float64)

    model = _swapped_stack(torch)
    tw.set_step_token_weights(weights)
    tw.reset_invocation_count()
    _loss(torch, model(x)).backward()
    assert tw.reset_invocation_count() == 2

    _loss(torch, checkpoint(model, x, use_reentrant=True)).backward()
    assert tw.reset_invocation_count() == 2
    _loss(torch, checkpoint(model, x, use_reentrant=False)).backward()
    assert tw.reset_invocation_count() == 2


def test_one_hot_w_moves_grads_for_every_projection_type(clean_slot):
    """Pre-mortem amendment 3 (CPU form of the GPU smoke assertion): a
    one-hot w changes the WEIGHT grad of each of the seven projection types
    individually, relative to the w == 1 baseline."""
    torch = pytest.importorskip("torch")

    class SevenProjections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(6)
            for name in tw.TARGET_PROJECTION_SUFFIXES:
                setattr(self, name, torch.nn.Linear(4, 4, bias=False,
                                                    dtype=torch.float64))

        def forward(self, x):
            for name in tw.TARGET_PROJECTION_SUFFIXES:
                x = torch.tanh(getattr(self, name)(x))
            return x

    torch.manual_seed(7)
    x = torch.randn(1, 3, 4, dtype=torch.float64)

    def run(weights):
        model = SevenProjections()
        assert tw.swap_linears(model, expected_count=7) == 7
        tw.set_step_token_weights(weights)
        _loss(torch, model(x)).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    baseline = run(torch.ones(1, 3, dtype=torch.float64))
    one_hot = torch.zeros(1, 3, dtype=torch.float64)
    one_hot[0, 1] = 1.0
    steered = run(one_hot)
    for suffix in tw.TARGET_PROJECTION_SUFFIXES:
        name = f"{suffix}.weight"
        assert not torch.equal(baseline[name], steered[name]), name


def test_fp32_multiply_floor_survives_bf16(clean_slot):
    """Review fix 2: the w-multiply must run at an fp32 floor. bf16 has 8
    significand bits, so casting w = 1 +- 2^-10 to bf16 collapses it to
    exactly 1.0. Construction: two positions with identical x and exactly
    opposite g cancel to a zero grad_W at w == 1; the steered grad is
    nonzero IFF the multiply preserved w. Re-applying the mutation
    ``acc_dtype = grad_out.dtype`` makes the steered grad exactly zero and
    this test fail."""
    torch = pytest.importorskip("torch")

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(10)
            self.q_proj = torch.nn.Linear(2, 2, bias=False,
                                          dtype=torch.bfloat16)

        def forward(self, x):
            return self.q_proj(x)

    # bf16-exact inputs; gy = probe = (+1 row, -1 row) => exact cancellation
    x = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.bfloat16)
    probe = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]], dtype=torch.bfloat16)
    delta = 2.0 ** -10  # collapses to 1.0 under a bf16 cast

    def grad(weights):
        model = One()
        assert tw.swap_linears(model, expected_count=1) == 1
        tw.set_step_token_weights(weights)
        (model(x) * probe).sum().backward()
        return model.q_proj.weight.grad.clone()

    baseline = grad(torch.ones(1, 2, dtype=torch.float32))
    assert torch.count_nonzero(baseline) == 0  # exact cancellation at w == 1
    steered = grad(torch.tensor([[1.0 + delta, 1.0 - delta]],
                                dtype=torch.float32))
    assert torch.count_nonzero(steered) > 0  # fp32 multiply preserved w
    # the surviving magnitude is 2*delta*|x| -- representable in bf16
    expected = 2 * delta * torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    assert torch.allclose(steered.float(), expected, rtol=0.02, atol=0)


def test_unset_slot_is_vanilla_and_shape_mismatch_is_loud(clean_slot):
    torch = pytest.importorskip("torch")
    vanilla, swapped = _fp64_stack(torch), _swapped_stack(torch)
    torch.manual_seed(8)
    x = torch.randn(1, 4, 5, dtype=torch.float64)
    assert torch.equal(vanilla(x), swapped(x))  # no weights set => vanilla

    tw.set_step_token_weights(torch.ones(1, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="does not match input positions"):
        swapped(x)

    with pytest.raises(ValueError, match="batch, seq"):
        tw.set_step_token_weights(torch.ones(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-finite"):
        tw.set_step_token_weights(
            torch.tensor([[1.0, float("nan")]], dtype=torch.float64))


# --------------------------------------------------- torch: swap (test 3)
def test_swap_preserves_fqns_and_state_dict_keys(clean_slot):
    torch = pytest.importorskip("torch")

    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(9)
            self.q_proj = torch.nn.Linear(4, 4)
            self.down_proj = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 4)
            self.vision_tower = torch.nn.Module()
            self.vision_tower.q_proj = torch.nn.Linear(4, 4)

    model = Mini()
    modules_before = [n for n, _ in model.named_modules()]
    keys_before = list(model.state_dict().keys())
    values_before = {n: p for n, p in model.named_parameters()}

    linear_cls = tw.WeightGradScaledLinear
    assert tw.swap_linears(model, expected_count=2) == 2

    assert [n for n, _ in model.named_modules()] == modules_before
    assert list(model.state_dict().keys()) == keys_before
    for name, param in model.named_parameters():
        assert param is values_before[name]  # same tensors, no re-init
    assert type(model.q_proj) is linear_cls
    assert type(model.down_proj) is linear_cls
    # excluded subtrees and non-target names are untouched
    assert type(model.lm_head) is torch.nn.Linear
    assert type(model.vision_tower.q_proj) is torch.nn.Linear

    with pytest.raises(ValueError, match="expected 7"):
        tw.swap_linears(Mini(), expected_count=7)
    with pytest.raises(ValueError, match="matched no modules"):
        tw.swap_linears(torch.nn.Linear(2, 2))
    swapped_already = Mini()
    tw.swap_linears(swapped_already)
    with pytest.raises(ValueError, match="not a plain nn.Linear"):
        tw.swap_linears(swapped_already)  # double swap is loud, not silent


# ------------------------------------------------ fakes: axolotl seam layer
def _install_fake_axolotl(monkeypatch):
    """Minimal axolotl stand-ins mirroring the pinned 0.17.0 semantics the
    real classes have (cited per fake). Heavy pod-side deps stay out of the
    CPU suite, per house rule."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    for name in ("axolotl", "axolotl.utils", "axolotl.utils.collators",
                 "axolotl.core", "axolotl.core.trainers",
                 "axolotl.prompt_strategies", "axolotl.integrations"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    # --- axolotl/utils/collators/batching.py: V2BatchSamplerDataCollator
    # semantics: per pack, concatenate every per-token feature in item
    # order, attention_mask becomes (i+1)*mask segment ids; then the base
    # DataCollatorForSeq2Seq right-pads labels (-100) / position_ids
    # (arange) / input_ids+mask (0) to a multiple of pad_to_multiple_of.
    batching = types.ModuleType("axolotl.utils.collators.batching")

    class FakeV2Collator:
        def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None,
                     return_tensors="pt", **_kwargs):
            self.tokenizer = tokenizer
            self.padding = padding
            self.pad_to_multiple_of = pad_to_multiple_of
            self.return_tensors = return_tensors

        def __call__(self, features, return_tensors=None):
            if not isinstance(features[0], list):
                features = [features]
            out = []
            for pack in features:
                row = {}
                for key in pack[0].keys():
                    if key == "length":
                        continue
                    if key == "attention_mask":
                        arrays = [(i + 1) * np.asarray(item[key])
                                  for i, item in enumerate(pack)]
                    else:
                        arrays = [np.asarray(item[key]) for item in pack]
                    row[key] = np.concatenate(arrays)
                out.append(row)
            length = max(len(r["input_ids"]) for r in out)
            if self.pad_to_multiple_of:
                m = self.pad_to_multiple_of
                length = (length + m - 1) // m * m
            batch = {}
            left = self.tokenizer.padding_side == "left"
            for key in out[0]:
                padded = []
                for row in out:
                    pad = length - len(row[key])
                    if key == "labels":
                        fill = np.full(pad, -100)
                    elif key == "position_ids":
                        fill = np.arange(pad)
                    else:
                        fill = np.zeros(pad, dtype=np.asarray(row[key]).dtype)
                    parts = [fill, row[key]] if left else [row[key], fill]
                    padded.append(np.concatenate(parts))
                batch[key] = torch.as_tensor(np.stack(padded))
            return batch

    batching.V2BatchSamplerDataCollatorForSeq2Seq = FakeV2Collator
    monkeypatch.setitem(sys.modules, "axolotl.utils.collators.batching",
                        batching)

    # --- axolotl/prompt_strategies/completion.py: base strategy _tokenize
    # (labels = input_ids copy) + field property; here tokens are the
    # whitespace-split ints of the text so alignment is fully observable.
    completion = types.ModuleType("axolotl.prompt_strategies.completion")

    class FakeCompletionStrategy:
        def __init__(self, prompter, tokenizer, train_on_inputs,
                     sequence_len, max_length=None):
            self.prompter = prompter
            self.tokenizer = tokenizer
            self.train_on_inputs = train_on_inputs
            self.sequence_len = sequence_len
            self.max_length = max_length or sequence_len
            self.field = "text"

        @property
        def supports_batched(self):
            return True

        def _tokenize(self, text):
            ids = [int(token) for token in str(text).split()]
            return {"input_ids": ids, "attention_mask": [1] * len(ids),
                    "labels": list(ids)}

    class FakeCompletionPrompter:
        pass

    completion.CompletionPromptTokenizingStrategy = FakeCompletionStrategy
    completion.CompletionPrompter = FakeCompletionPrompter
    monkeypatch.setitem(sys.modules, "axolotl.prompt_strategies.completion",
                        completion)

    # --- axolotl/core/trainers/base.py: AxolotlTrainer stand-in recording
    # what compute_loss saw (the real one forwards inputs to the model).
    trainers = types.ModuleType("axolotl.core.trainers.base")

    class FakeAxolotlTrainer:
        def __init__(self, accelerator=None, state=None):
            self.accelerator = accelerator
            self.state = state
            self.seen_inputs = []
            self.seen_slot = []
            self._signature_columns = None

        def _set_signature_columns_if_needed(self):
            self._signature_columns = ["input_ids", "labels"]

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            self.seen_inputs.append(dict(inputs))
            self.seen_slot.append(tw.current_step_token_weights())
            return 0.0

        def training_step(self, model, inputs, num_items_in_batch=None):
            return self.compute_loss(model, inputs)

    trainers.AxolotlTrainer = FakeAxolotlTrainer
    monkeypatch.setitem(sys.modules, "axolotl.core.trainers.base", trainers)

    # --- axolotl/integrations/base.py: BasePlugin
    integrations = types.ModuleType("axolotl.integrations.base")

    class BasePlugin:
        pass

    integrations.BasePlugin = BasePlugin
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base",
                        integrations)
    return types.SimpleNamespace(
        torch=torch, np=np, trainer_base=FakeAxolotlTrainer)


def _fresh_lazy(name):
    """Rebuild a lazily-cached class against the currently-installed fakes
    (the module memoizes _LAZY builds; tests must not depend on which fake
    was installed first). _cached rebuilds when the slot is None; plain
    getattr would return the None without consulting __getattr__."""
    tw.__dict__[name] = None
    return tw._cached(name)


# ------------------------------------------------- collator + pack (test 4)
@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_packing_alignment_three_doc_pack(monkeypatch, clean_slot,
                                          padding_side):
    """SPEC test 4 (+ review fix 3): each doc's w segment lands on its
    positions in the pack (boundaries = position_ids resets), pads are
    exactly 0.0, fp32 — on BOTH tokenizer padding sides (the weights must
    follow the content, wherever the pad lands)."""
    fakes = _install_fake_axolotl(monkeypatch)
    torch = fakes.torch
    collator_cls = _fresh_lazy("TokenWeightsCollator")

    def doc(ids, weights):
        return {"input_ids": list(ids), "attention_mask": [1] * len(ids),
                "labels": list(ids), "position_ids": list(range(len(ids))),
                TOKEN_WEIGHTS_COLUMN: list(weights)}

    docs = [doc([11, 12, 13], [1.1, 0.9, 1.0]),
            doc([21, 22, 23, 24], [2.0, 0.5, 0.5, 1.0]),
            doc([31, 32], [0.25, 1.75])]
    tokenizer = types.SimpleNamespace(padding_side=padding_side)
    collator = collator_cls(tokenizer, pad_to_multiple_of=16)
    # the collator pops the weights column from the items it is given
    batch = collator([[dict(item) for item in docs]])

    weights = batch[TOKEN_WEIGHTS_COLUMN]
    assert weights.shape == batch["input_ids"].shape == (1, 16)
    assert weights.dtype == torch.float32

    offset = 0 if padding_side == "right" else 7  # 16 slots, 9 real tokens
    content = slice(offset, offset + 9)
    position_ids = batch["position_ids"][0].tolist()
    resets = [i for i, p in enumerate(position_ids[content], start=offset)
              if p == 0]
    assert resets == [offset, offset + 3, offset + 7]  # doc boundaries
    segments = [(offset, offset + 3), (offset + 3, offset + 7),
                (offset + 7, offset + 9)]
    for (start, stop), source in zip(segments, docs):
        assert weights[0, start:stop].tolist() == pytest.approx(
            source[TOKEN_WEIGHTS_COLUMN])
    pad_region = (weights[0, 9:] if padding_side == "right"
                  else weights[0, :7])
    assert pad_region.tolist() == [0.0] * 7  # pad value is exactly 0.0
    # the packed attention_mask (kept for non-gemma3 families) is the
    # (i+1)*mask segment-id form
    assert batch["attention_mask"][0, content].tolist() == [1, 1, 1, 2, 2, 2,
                                                            2, 3, 3]


def test_collator_requires_the_column_on_every_pack_item(monkeypatch,
                                                         clean_slot):
    _install_fake_axolotl(monkeypatch)
    collator_cls = _fresh_lazy("TokenWeightsCollator")
    tokenizer = types.SimpleNamespace(padding_side="right")
    collator = collator_cls(tokenizer, pad_to_multiple_of=8)
    good = {"input_ids": [1], "attention_mask": [1], "labels": [1],
            "position_ids": [0], TOKEN_WEIGHTS_COLUMN: [1.0]}
    bad = {"input_ids": [2], "attention_mask": [1], "labels": [2],
           "position_ids": [0]}
    with pytest.raises(ValueError, match="carry no 'token_weights'"):
        collator([[good, bad]])


# ------------------------------------------------------------ strategy layer
def _strategy(monkeypatch, weights_by_chunk, ids_by_chunk, sequence_len=8):
    _install_fake_axolotl(monkeypatch)
    strategy_cls = _fresh_lazy("TokenWeightsPromptTokenizingStrategy")
    np = pytest.importorskip("numpy")
    weights = {k: np.asarray(v, dtype=np.float32)
               for k, v in weights_by_chunk.items()}
    return strategy_cls(
        object(), object(), False, sequence_len,
        max_length=sequence_len * 64,
        weights_by_chunk=weights, ids_by_chunk=dict(ids_by_chunk))


def _doc_text(ids):
    return " ".join(str(i) for i in ids)


def test_strategy_chunks_and_aligns_weights(monkeypatch, clean_slot):
    """A 10-token doc at window 8 emits two rows; weights land per chunk and
    the emitted columns stay position-aligned (chunk rule = test 6's)."""
    ids = list(range(100, 110))
    strategy = _strategy(
        monkeypatch,
        weights_by_chunk={("d1", 0): [1.5] * 8, ("d1", 1): [0.5, 0.5]},
        ids_by_chunk={("d1", 0): ids[:8], ("d1", 1): ids[8:]})
    # doc mean = (8*1.5 + 2*0.5) / 10 = 1.3 -> mis-normalized, loud
    with pytest.raises(ValueError, match="mean"):
        strategy.tokenize_prompt({"text": [_doc_text(ids)], "doc_id": ["d1"]})

    weights_a = [1.25] * 8
    weights_b = [0.0, 0.0]  # doc mean back to 1.0
    strategy = _strategy(
        monkeypatch,
        weights_by_chunk={("d1", 0): weights_a, ("d1", 1): weights_b},
        ids_by_chunk={("d1", 0): ids[:8], ("d1", 1): ids[8:]})
    out = strategy.tokenize_prompt(
        {"text": [_doc_text(ids)], "doc_id": ["d1"]})
    assert out["input_ids"] == [ids[:8], ids[8:]]
    assert out["labels"] == [ids[:8], ids[8:]]
    assert out[TOKEN_WEIGHTS_COLUMN] == [weights_a, weights_b]
    assert [len(w) for w in out[TOKEN_WEIGHTS_COLUMN]] == [
        len(c) for c in out["input_ids"]]


def test_strategy_v1_coverage_missing_chunk_and_tail_fill(monkeypatch,
                                                          clean_slot, caplog):
    """Amendment 2: missing (doc_id, chunk_idx) keys and uncovered tails get
    the neutral w=1.0 — counted and logged, never a KeyError."""
    ids = list(range(200, 210))  # 10 tokens: chunk0 = 8, chunk1 = 2
    # covers 7 of 8 chunk-0 positions; stored mean is 1, so the w=1 fills
    # keep the realized doc mean at 1 (the invariant the strategy asserts)
    stored = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.0]
    strategy = _strategy(
        monkeypatch,
        weights_by_chunk={("d1", 0): stored},  # chunk 1 has no row at all
        ids_by_chunk={("d1", 0): ids[:7]})
    with caplog.at_level(logging.WARNING, logger=tw.__name__):
        out = strategy.tokenize_prompt(
            {"text": [_doc_text(ids)], "doc_id": ["d1"]})
    assert out[TOKEN_WEIGHTS_COLUMN][0] == stored + [1.0]  # tail fill
    assert out[TOKEN_WEIGHTS_COLUMN][1] == [1.0, 1.0]  # whole chunk missing
    assert strategy._missing_chunks == 1
    assert strategy._filled_positions == 1
    assert "unweighted" in caplog.text

    # an entirely uncovered doc is also fine (fully neutral, counted)
    out = strategy.tokenize_prompt(
        {"text": [_doc_text([1, 2, 3])], "doc_id": ["never-scored"]})
    assert out[TOKEN_WEIGHTS_COLUMN] == [[1.0, 1.0, 1.0]]
    assert strategy._missing_chunks == 2


def test_strategy_token_id_parity_is_full_array(monkeypatch, clean_slot):
    """Amendment 2 (strengthened test 6): stored token ids must equal the
    tokenized chunk position for position over the covered prefix; length
    agreement alone is not enough."""
    ids = [7, 8, 9, 10]
    strategy = _strategy(
        monkeypatch,
        weights_by_chunk={("d1", 0): [1.0, 1.0, 1.0]},
        ids_by_chunk={("d1", 0): [7, 8, 999]})  # right length, wrong ids
    with pytest.raises(ValueError, match="diverge.*position 2"):
        strategy.tokenize_prompt({"text": [_doc_text(ids)], "doc_id": ["d1"]})

    # more stored weights than the chunk has tokens: drift, loud
    strategy = _strategy(
        monkeypatch,
        weights_by_chunk={("d1", 0): [1.0] * 5},
        ids_by_chunk={("d1", 0): [7, 8, 9, 10, 11]})
    with pytest.raises(ValueError, match="drifted"):
        strategy.tokenize_prompt({"text": [_doc_text(ids)], "doc_id": ["d1"]})


def test_strategy_requires_doc_id_column(monkeypatch, clean_slot):
    strategy = _strategy(monkeypatch, weights_by_chunk={}, ids_by_chunk={})
    with pytest.raises(ValueError, match="doc_id"):
        strategy.tokenize_prompt({"text": ["1 2 3"]})


def test_strategy_skips_zero_token_docs(monkeypatch, clean_slot, caplog):
    """Review fix 4: an empty doc emits no rows (stock completion parity),
    is counted + logged, and never divides by zero."""
    strategy = _strategy(monkeypatch, weights_by_chunk={}, ids_by_chunk={})
    with caplog.at_level(logging.WARNING, logger=tw.__name__):
        out = strategy.tokenize_prompt(
            {"text": ["", "1 2 3"], "doc_id": ["empty", "d1"]})
    assert out["input_ids"] == [[1, 2, 3]]  # the empty doc contributed nothing
    assert out[TOKEN_WEIGHTS_COLUMN] == [[1.0, 1.0, 1.0]]
    assert strategy._empty_docs == 1
    assert "zero tokens" in caplog.text


def test_load_reads_parquet_and_validates(monkeypatch, tmp_path, clean_slot):
    """The dotted-path strategy entry point: parquet -> strategy, with the
    schema errors loud before any GPU time."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _install_fake_axolotl(monkeypatch)
    _fresh_lazy("TokenWeightsPromptTokenizingStrategy")

    path = tmp_path / "w.parquet"
    pq.write_table(pa.table({
        "doc_id": ["d1"], "chunk_idx": [0],
        TOKEN_WEIGHTS_COLUMN: [[1.0, 1.0]], "token_ids": [[5, 6]],
    }), path)

    class Cfg(dict):
        __getattr__ = dict.__getitem__

    cfg = Cfg(token_weights={"weights_path": str(path)},
              train_on_inputs=False, sequence_len=8)
    strategy = tw.load(object(), cfg, ds_cfg={"field": "text"})
    out = strategy.tokenize_prompt({"text": ["5 6"], "doc_id": ["d1"]})
    assert out[TOKEN_WEIGHTS_COLUMN] == [[1.0, 1.0]]
    assert strategy.max_length == 8 * 64  # stock completion ceiling

    with pytest.raises(ValueError, match="token_weights block"):
        tw.load(object(), Cfg(token_weights=None, train_on_inputs=False,
                              sequence_len=8))
    pq.write_table(pa.table({
        "doc_id": ["d1"], "chunk_idx": [0],
        TOKEN_WEIGHTS_COLUMN: [[1.0, 1.0]],
    }), path)
    with pytest.raises(ValueError, match="token_ids"):
        tw.load(object(), cfg)
    with pytest.raises(FileNotFoundError):
        tw.load(object(), Cfg(
            token_weights={"weights_path": str(tmp_path / "missing.parquet")},
            train_on_inputs=False, sequence_len=8))


def test_weights_table_rejects_duplicates_and_ragged_rows(monkeypatch,
                                                          tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "w.parquet"
    pq.write_table(pa.table({
        "doc_id": ["d", "d"], "chunk_idx": [0, 0],
        TOKEN_WEIGHTS_COLUMN: [[1.0], [1.0]], "token_ids": [[1], [1]],
    }), path)
    with pytest.raises(ValueError, match="duplicate"):
        tw._load_weights_table(str(path))
    pq.write_table(pa.table({
        "doc_id": ["d"], "chunk_idx": [0],
        TOKEN_WEIGHTS_COLUMN: [[1.0, 2.0]], "token_ids": [[1]],
    }), path)
    with pytest.raises(ValueError, match="token_ids"):
        tw._load_weights_table(str(path))


# --------------------------------------------------------------- trainer
def _make_trainer(monkeypatch, clip_returns=2.0):
    fakes = _install_fake_axolotl(monkeypatch)
    trainer_cls = _fresh_lazy("TokenWeightsTrainer")
    calls = []

    class FakeAccelerator:
        def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
            calls.append((max_norm, norm_type))
            return clip_returns

    trainer = trainer_cls(
        accelerator=FakeAccelerator(),
        state=types.SimpleNamespace(global_step=3))
    return fakes, trainer, calls


def test_trainer_pops_column_publishes_slot_and_clears(monkeypatch,
                                                       clean_slot):
    fakes, trainer, _ = _make_trainer(monkeypatch)
    torch = fakes.torch
    monkeypatch.setattr(tw, "_SWAPPED_COUNT", 2)
    model = types.SimpleNamespace(training=True)
    weights = torch.tensor([[1.0, 0.5, 1.5]])

    def fake_super_step(self, model, inputs, num_items_in_batch=None):
        loss = self.compute_loss(model, inputs)
        # stand-in for backward through 2 swapped projections
        tw._count_backward_invocation()
        tw._count_backward_invocation()
        return loss

    monkeypatch.setattr(fakes.trainer_base, "training_step", fake_super_step)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
              TOKEN_WEIGHTS_COLUMN: weights}
    trainer.training_step(model, inputs)

    seen = trainer.seen_inputs[-1]
    assert TOKEN_WEIGHTS_COLUMN not in seen  # popped before the model call
    assert torch.equal(trainer.seen_slot[-1], weights)  # slot live in forward
    assert tw.current_step_token_weights() is None  # cleared after the step

    # signature columns pin the column on the non-packing path
    trainer._set_signature_columns_if_needed()
    assert TOKEN_WEIGHTS_COLUMN in trainer._signature_columns


def test_trainer_bypass_guard_raises_on_missing_invocations(monkeypatch,
                                                            clean_slot):
    fakes, trainer, _ = _make_trainer(monkeypatch)
    torch = fakes.torch
    monkeypatch.setattr(tw, "_SWAPPED_COUNT", 3)

    def fake_super_step(self, model, inputs, num_items_in_batch=None):
        loss = self.compute_loss(model, inputs)
        tw._count_backward_invocation()  # only 1 of 3 -> bypass
        return loss

    monkeypatch.setattr(fakes.trainer_base, "training_step", fake_super_step)
    inputs = {TOKEN_WEIGHTS_COLUMN: torch.ones(1, 2)}
    with pytest.raises(ValueError, match="bypass guard.*1 scaled"):
        trainer.training_step(types.SimpleNamespace(training=True), inputs)
    assert tw.current_step_token_weights() is None  # cleared even on raise


def test_trainer_missing_weights_loud_in_training_quiet_in_eval(monkeypatch,
                                                                clean_slot):
    fakes, trainer, _ = _make_trainer(monkeypatch)
    with pytest.raises(ValueError, match="refusing to train unweighted"):
        trainer.compute_loss(types.SimpleNamespace(training=True),
                             {"input_ids": [1]})
    trainer.compute_loss(types.SimpleNamespace(training=False),
                         {"input_ids": [1]})  # eval batch: fine, vanilla


def test_trainer_discards_weights_outside_training(monkeypatch, clean_slot):
    """Review fix 1: a val split tokenized by the strategy DOES carry the
    column; eval compute_loss must pop-and-discard it (vanilla loss, slot
    never published), not take the weighted branch."""
    fakes, trainer, _ = _make_trainer(monkeypatch)
    torch = fakes.torch
    weights = torch.tensor([[1.5, 0.5]])
    trainer.compute_loss(
        types.SimpleNamespace(training=False),
        {"input_ids": torch.tensor([[1, 2]]),
         TOKEN_WEIGHTS_COLUMN: weights})
    seen = trainer.seen_inputs[-1]
    assert TOKEN_WEIGHTS_COLUMN not in seen  # popped before the model call
    assert trainer.seen_slot[-1] is None  # never published to the Functions
    assert tw.current_step_token_weights() is None
    assert trainer._weighted_micro_step is False  # no bypass-guard arming


def test_trainer_logs_grad_clip_coefficient(monkeypatch, clean_slot, caplog):
    _, trainer, calls = _make_trainer(monkeypatch, clip_returns=2.0)
    with caplog.at_level(logging.INFO, logger=tw.__name__):
        returned = trainer.accelerator.clip_grad_norm_(None, 1.0)
    assert returned == 2.0  # wrapper is transparent to the training loop
    assert calls == [(1.0, 2)]  # the real clip still ran
    assert "grad_clip step=3" in caplog.text
    assert "coef=0.5" in caplog.text and "clipped=True" in caplog.text


def test_trainer_logs_weight_stats(monkeypatch, clean_slot, caplog):
    fakes, trainer, _ = _make_trainer(monkeypatch)
    torch = fakes.torch
    weights = torch.tensor([[2.0, 1.0, 0.0, 0.0]])  # two pad zeros
    with caplog.at_level(logging.INFO, logger=tw.__name__):
        trainer.compute_loss(types.SimpleNamespace(training=True),
                             {TOKEN_WEIGHTS_COLUMN: weights})
    assert "w stats" in caplog.text
    assert "nonpad=2" in caplog.text and "mean=1.5" in caplog.text


# ----------------------------------------------------------------- plugin
def test_plugin_seams(monkeypatch, clean_slot):
    fakes = _install_fake_axolotl(monkeypatch)
    torch = fakes.torch
    plugin_cls = _fresh_lazy("TokenWeightsPlugin")
    _fresh_lazy("TokenWeightsTrainer")
    _fresh_lazy("TokenWeightsCollator")
    plugin = plugin_cls()
    assert plugin.get_input_args() == "scimt.train.token_weights.TokenWeightsArgs"

    block = {"token_weights": {"weights_path": "w.parquet"}}
    for method in ("get_trainer_cls",):
        with pytest.raises(ValueError, match="token_weights block"):
            getattr(plugin, method)({})
    with pytest.raises(ValueError, match="enabled is false"):
        plugin.get_trainer_cls(
            {"token_weights": {"weights_path": "w", "enabled": False}})

    assert plugin.get_trainer_cls(block) is tw.TokenWeightsTrainer
    assert plugin.get_collator_cls_and_kwargs(block, is_eval=True) is None
    collator_cls, kwargs = plugin.get_collator_cls_and_kwargs(block)
    assert collator_cls is tw.TokenWeightsCollator and kwargs == {}

    class OneLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in tw.TARGET_PROJECTION_SUFFIXES:
                setattr(self, name, torch.nn.Linear(2, 2))
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(num_hidden_layers=1))

    model = OneLayer()
    plugin.post_model_load(block, model)  # expected 1 * 7, matches
    assert type(model.q_proj) is tw.WeightGradScaledLinear
    assert tw.swapped_module_count() == 7

    class TwoLayerConfig(OneLayer):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(num_hidden_layers=2))

    with pytest.raises(ValueError, match="expected 14"):
        plugin.post_model_load(block, TwoLayerConfig())
    # expected_swapped in the block overrides the config-derived count
    override = {"token_weights": {"weights_path": "w", "expected_swapped": 7}}
    plugin.post_model_load(override, TwoLayerConfig())


# ---------------------------------------------------------------- mix layer
def test_mix_keep_columns_carries_doc_ids(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from datasets import Dataset

    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    class Toktok:
        def __call__(self, text):
            return {"input_ids": text.split()}

    source = _LoadedSource(
        dataset=Dataset.from_dict({
            "text": ["a b c", "d e", "f g h i"],
            "doc_id": ["d0", "d1", "d2"]}),
        name="s")
    mixed, manifest = build_token_budget_mix(
        [source], Toktok(), target_tokens=9, num_proc=1,
        keep_columns=("doc_id",))
    assert set(mixed.column_names) == {"text", "doc_id"}
    assert sorted(mixed["doc_id"]) == ["d0", "d1", "d2"]

    with pytest.raises(ValueError, match="keep_columns"):
        build_token_budget_mix([source], Toktok(), target_tokens=9,
                               num_proc=1, keep_columns=("missing",))
    # default stays byte-identical: text-only rows
    mixed, _ = build_token_budget_mix([source], Toktok(), target_tokens=9,
                                      num_proc=1)
    assert mixed.column_names == ["text"]


def test_mix_iterable_path_rejects_text_column_in_keep_columns():
    """Review fix 6: the iterable path mirrors the map path's check instead
    of silently emitting the column twice. The check fires before any
    datasets import, so this needs no extras."""
    from scimt.train.mix import _LoadedSource, _take_iterable_source

    source = _LoadedSource(dataset=None, text_column="content", name="s")
    with pytest.raises(ValueError, match="must not name the text column"):
        _take_iterable_source(source, None, 1.0, 0, 1,
                              keep_columns=("content",))


def test_mix_config_keep_columns_validation():
    from scimt.train.mix import MixConfig, MixSource

    cfg = MixConfig(sources=[MixSource(dataset="x.jsonl")], total_tokens=10,
                    keep_columns=["doc_id"])
    assert cfg.keep_columns == ["doc_id"]
    with pytest.raises(ValueError, match="text"):
        MixConfig(sources=[MixSource(dataset="x.jsonl")], total_tokens=10,
                  keep_columns=["text"])
    with pytest.raises(ValueError, match="keep_columns"):
        MixConfig(sources=[MixSource(dataset="x.jsonl")], total_tokens=10,
                  keep_columns=[""])
