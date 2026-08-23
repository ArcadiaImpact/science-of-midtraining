import pytest
import torch

from scimt.data_attribution.gradients import BatchedVJPBackend, SerialGradientBackend
from scimt.data_attribution.manifest import ParameterManifest, flatten_tensors


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_gradient_rows_match_explicit_autograd(backend_cls):
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )
    manifest = ParameterManifest.from_model(model, "mlp")
    x = torch.randn(5, 3)
    losses = model(x).square().mean(1)
    actual = backend_cls(model, manifest).rows(losses)
    losses = model(x).square().mean(1)
    params = [p for _, p in model.named_parameters()]
    expected = torch.stack(
        [
            flatten_tensors(
                manifest.included_entries(),
                torch.autograd.grad(losses[i], params, retain_graph=i < 4),
            )
            for i in range(5)
        ]
    )
    assert actual.dtype == torch.float32 and actual.shape == (
        5,
        manifest.included_numel,
    )
    torch.testing.assert_close(actual, expected)


def test_batched_uses_chunk_sized_cotangents(monkeypatch):
    model = torch.nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "linear")
    losses = model(torch.randn(7, 3)).square().mean(1)
    monkeypatch.setattr(
        torch, "eye", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no eye"))
    )
    original_zeros = torch.zeros
    shapes = []

    def recording_zeros(*args, **kwargs):
        shape = args[0]
        shapes.append(tuple(shape) if not isinstance(shape, int) else (shape,))
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", recording_zeros)
    assert BatchedVJPBackend(model, manifest).rows(losses, chunk_size=2).shape[0] == 7
    assert (7, 7) not in shapes
    assert (2, 7) in shapes


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_unused_parameters_zero_fill_and_inputs_validate(backend_cls):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.used(x)

    model = Model()
    manifest = ParameterManifest.from_model(model, "unused")
    backend = backend_cls(model, manifest)
    rows = backend.rows(model(torch.randn(3, 2)).flatten().square(), chunk_size=2)
    unused = manifest.included_entries()[0]
    assert torch.count_nonzero(rows[:, unused.global_flat_offset : unused.numel]) == 0
    with pytest.raises(ValueError, match="shape"):
        backend.rows(torch.tensor(1.0))
    with pytest.raises(ValueError, match="positive"):
        backend.rows(model(torch.randn(3, 2)).flatten(), chunk_size=0)


def test_serial_matches_explicit_per_loss_jacobian_exactly():
    torch.manual_seed(8)
    model = torch.nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "linear")
    x = torch.randn(4, 3)
    losses = model(x).square().mean(1)
    actual = SerialGradientBackend(model, manifest).rows(losses, chunk_size=2)
    losses = model(x).square().mean(1)
    params = tuple(model.parameters())
    expected = torch.stack(
        [
            flatten_tensors(
                manifest.included_entries(),
                torch.autograd.grad(losses[row], params, retain_graph=row < 3),
            )
            for row in range(4)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_backends_do_not_touch_param_grad_fields(backend_cls):
    model = torch.nn.Linear(3, 2)
    sentinels = [
        torch.full_like(parameter, i + 1.0)
        for i, parameter in enumerate(model.parameters())
    ]
    for parameter, sentinel in zip(model.parameters(), sentinels, strict=True):
        parameter.grad = sentinel
    manifest = ParameterManifest.from_model(model, "linear")
    backend_cls(model, manifest).rows(
        model(torch.randn(4, 3)).square().mean(1), chunk_size=2
    )
    for parameter, sentinel in zip(model.parameters(), sentinels, strict=True):
        assert parameter.grad is sentinel


def test_empty_gradient_rows_follow_loss_device():
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "linear")
    losses = torch.empty(0, device="meta")
    for backend in (
        SerialGradientBackend(model, manifest),
        BatchedVJPBackend(model, manifest),
    ):
        assert backend.rows(losses).device == losses.device


def test_batched_matches_serial_tiny_lm():
    from .fixtures import TinyLM

    model = TinyLM()
    manifest = ParameterManifest.from_model(model, "tiny-lm")
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(ids).logits.float()
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    expected = SerialGradientBackend(model, manifest).rows(losses, chunk_size=2)
    logits = model(ids).logits.float()
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    actual = BatchedVJPBackend(model, manifest).rows(losses, chunk_size=2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_frozen_included_parameters_are_zero_filled(backend_cls):
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    manifest = ParameterManifest.from_model(model, "partly-frozen")
    losses = model(torch.randn(3, 2)).square().mean(1)
    rows = backend_cls(model, manifest).rows(losses, chunk_size=2)
    bias = next(e for e in manifest.included_entries() if e.name == "bias")
    assert (
        torch.count_nonzero(
            rows[:, bias.global_flat_offset : bias.global_flat_offset + bias.numel]
        )
        == 0
    )
    assert torch.count_nonzero(rows[:, : bias.global_flat_offset]) > 0


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_empty_manifest_returns_n_by_zero_fp32_on_loss_device(backend_cls):
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "none", include=[r"does-not-match"])
    losses = model(torch.randn(4, 2)).flatten().square()
    rows = backend_cls(model, manifest).rows(losses)
    assert rows.shape == (4, 0) and rows.dtype == torch.float32
    assert rows.device == losses.device


# ------------------------------------------------- backward_memory_mode
from scimt.data_attribution.gradients import backward_memory_mode  # noqa: E402


class _CheckpointedBlock(torch.nn.Module):
    """Toy block reproducing the HF guard: checkpoint only when armed AND
    training."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.gradient_checkpointing = False

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        return torch.tanh(self.linear(x))


def _toy_checkpointed_model():
    torch.manual_seed(7)
    return torch.nn.Sequential(_CheckpointedBlock(), _CheckpointedBlock())


def test_backward_memory_mode_disabled_is_a_no_op():
    model = _toy_checkpointed_model()
    model.eval()
    with backward_memory_mode(model, False):
        assert not model.training
    assert not model.training


def test_backward_memory_mode_flips_and_restores_train_mode():
    model = _toy_checkpointed_model()
    model.eval()
    with backward_memory_mode(model, True):
        assert model.training
    assert not model.training


def test_backward_memory_mode_refuses_active_dropout():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
    with pytest.raises(RuntimeError, match="gradient_checkpointing: false"):
        with backward_memory_mode(model, True):
            pass


def test_backward_memory_mode_refuses_config_dropout():
    model = torch.nn.Linear(2, 2)
    model.config = type("Cfg", (), {})()
    model.config.attention_dropout = 0.1
    with pytest.raises(RuntimeError, match="attention_dropout"):
        with backward_memory_mode(model, True):
            pass


def test_backward_memory_mode_zero_dropout_is_fine():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    model.config = type("Cfg", (), {})()
    model.config.attention_dropout = 0.0
    with backward_memory_mode(model, True):
        assert model.training


def test_backward_memory_mode_detects_buffer_mutation():
    class MutatingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("steps", torch.zeros((), dtype=torch.int64))

    model = MutatingModel()
    with pytest.raises(RuntimeError, match="mutated model buffer"):
        with backward_memory_mode(model, True):
            model.steps.add_(1)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_gradient_rows_identical_under_checkpointing(backend_cls):
    """Numerics unchanged: checkpointed rows equal dense rows exactly."""
    model = _toy_checkpointed_model()
    manifest = ParameterManifest.from_model(model, "ckpt-toy")
    x = torch.randn(5, 4)

    model.eval()
    losses = model(x).square().mean(1)
    dense = backend_cls(model, manifest).rows(losses)

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
    with backward_memory_mode(model, True):
        losses = model(x).square().mean(1)
        checkpointed = backend_cls(model, manifest).rows(losses)
    torch.testing.assert_close(checkpointed, dense, rtol=0, atol=0)


# ------------------------------------------------- per-chunk row streaming
@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
@pytest.mark.parametrize("chunk_size", [1, 2, 64])
def test_iter_row_chunks_concat_equals_rows(backend_cls, chunk_size):
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )
    manifest = ParameterManifest.from_model(model, "mlp")
    x = torch.randn(5, 3)
    losses = model(x).square().mean(1)
    chunks = list(
        backend_cls(model, manifest).iter_row_chunks(losses, chunk_size=chunk_size)
    )
    for chunk in chunks:
        assert chunk.dtype == torch.float32
        assert chunk.shape[0] <= chunk_size
        assert chunk.shape[1] == manifest.included_numel
    losses = model(x).square().mean(1)
    expected = backend_cls(model, manifest).rows(losses, chunk_size=chunk_size)
    torch.testing.assert_close(torch.cat(chunks), expected)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_rows_single_chunk_is_returned_without_a_copy(backend_cls, monkeypatch):
    # A single-chunk torch.cat still copies — at full coverage that duplicate
    # is ~4·P bytes on device (pod run 20260819T095144Z OOM). rows() must
    # hand back the lone yielded chunk itself.
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "lin")
    backend = backend_cls(model, manifest)
    sentinel = torch.zeros((3, manifest.included_numel), dtype=torch.float32)
    monkeypatch.setattr(
        backend_cls, "iter_row_chunks", lambda self, losses, chunk_size=32: iter([sentinel])
    )
    losses = model(torch.randn(3, 2)).squeeze(1)
    assert backend.rows(losses, chunk_size=64) is sentinel


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_iter_row_chunks_constant_path_matches_rows(backend_cls):
    # No grad-requiring included parameters -> the constant zero-rows path
    # must yield exactly one chunk equal to rows().
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    manifest = ParameterManifest.from_model(model, "lin")
    backend = backend_cls(model, manifest)
    losses = model(torch.randn(4, 2)).squeeze(1)
    chunks = list(backend.iter_row_chunks(losses, chunk_size=2))
    assert len(chunks) == 1
    torch.testing.assert_close(chunks[0], backend.rows(losses, chunk_size=2))
    assert chunks[0].shape == (4, manifest.included_numel)
