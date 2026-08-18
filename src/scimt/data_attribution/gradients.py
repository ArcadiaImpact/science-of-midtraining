"""Flat per-example gradients, ported from gradient-kernel ca9689a.

``backward_memory_mode`` is a scimt addition (no upstream counterpart): the
train-mode context that lets HF activation checkpointing engage around the
library's own backward loops.
"""

from contextlib import contextmanager

import torch
from .manifest import flatten_tensors, included_named_parameters


@contextmanager
def backward_memory_mode(model, enabled):
    """Put ``model`` in train mode so HF activation checkpointing engages.

    Transformers gates checkpointing on ``self.gradient_checkpointing and
    self.training``, so a checkpointing-enabled model still runs the dense
    (memory-unbounded) path in eval mode. This context flips train mode for
    the duration of a backward loop while guaranteeing "how, never what":

    - any dropout that train mode would activate is a loud refusal (point the
      caller at ``data.gradient_checkpointing: false``), never a silent
      numerics change;
    - buffers are snapshotted and any mutation (e.g. batch-norm running
      stats) raises after the pass.

    ``enabled=False`` yields without touching the model.
    """

    if not enabled:
        yield
        return
    dropout_hazards = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p > 0.0
    )
    # nn.Dropout modules cover config-materialized dropout; the config scan
    # exists only for FUNCTIONAL dropout that modern decoder blocks apply via
    # F.dropout gated on self.training (no module to inspect). Head-only
    # fields like classifier_dropout are inert in a causal-LM forward and
    # must not trip the refusal.
    functional_dropout_keys = ("attention_dropout", "attn_dropout")
    config = getattr(model, "config", None)
    if config is not None:
        for key in functional_dropout_keys:
            value = getattr(config, key, None)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0.0
            ):
                dropout_hazards.append(f"config.{key}={value}")
    if dropout_hazards:
        raise RuntimeError(
            "gradient checkpointing needs a train-mode backward pass, but "
            "train mode would activate dropout and change what is measured: "
            f"{dropout_hazards} — set data.gradient_checkpointing: false "
            "for this model"
        )
    buffers_before = {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers(remove_duplicate=False)
    }
    was_training = model.training
    try:
        model.train(True)
        yield
    finally:
        model.train(was_training)
    buffers_after = dict(model.named_buffers(remove_duplicate=False))
    changed = [
        name
        for name, before in buffers_before.items()
        if name not in buffers_after
        or not torch.equal(before, buffers_after[name].detach().cpu())
    ]
    changed.extend(name for name in buffers_after if name not in buffers_before)
    if changed:
        raise RuntimeError(
            "train-mode backward pass mutated model buffer(s): "
            f"{sorted(set(changed))} — this model cannot run under "
            "data.gradient_checkpointing"
        )


class SerialGradientBackend:
    def __init__(self, model, manifest):
        self.entries = manifest.included_entries()
        self.parameters = tuple(
            p for _, p in included_named_parameters(model, manifest)
        )
        self.grad_indices = tuple(
            i for i, p in enumerate(self.parameters) if p.requires_grad
        )
        self.grad_parameters = tuple(self.parameters[i] for i in self.grad_indices)

    def _constant_rows(self, losses):
        width = sum(e.numel for e in self.entries)
        if width == 0 or not self.grad_parameters or not losses.requires_grad:
            return torch.zeros(
                (len(losses), width), dtype=torch.float32, device=losses.device
            )

    def rows(self, losses, chunk_size=32):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        constant = self._constant_rows(losses)
        if constant is not None:
            return constant
        result = []
        for index, loss in enumerate(losses):
            active_grads = torch.autograd.grad(
                loss,
                self.grad_parameters,
                retain_graph=index < len(losses) - 1,
                allow_unused=True,
                materialize_grads=False,
            )
            active = dict(zip(self.grad_indices, active_grads, strict=True))
            grads = tuple(active.get(i) for i in range(len(self.parameters)))
            result.append(flatten_tensors(self.entries, grads))
        return (
            torch.stack(result)
            if result
            else torch.empty(
                (0, sum(e.numel for e in self.entries)),
                dtype=torch.float32,
                device=losses.device,
            )
        )


class BatchedVJPBackend(SerialGradientBackend):
    def rows(self, losses, chunk_size=32):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        constant = self._constant_rows(losses)
        if constant is not None:
            return constant
        chunks = []
        for start in range(0, len(losses), chunk_size):
            stop = min(start + chunk_size, len(losses))
            cotangents = torch.zeros(
                (stop - start, len(losses)), dtype=losses.dtype, device=losses.device
            )
            rows = torch.arange(start, stop, device=losses.device)
            cotangents[torch.arange(stop - start, device=losses.device), rows] = 1
            active_grads = torch.autograd.grad(
                losses,
                self.grad_parameters,
                grad_outputs=cotangents,
                is_grads_batched=True,
                allow_unused=True,
                retain_graph=stop < len(losses),
            )
            active = dict(zip(self.grad_indices, active_grads, strict=True))
            grads = tuple(active.get(i) for i in range(len(self.parameters)))
            columns = [
                torch.zeros(
                    (stop - start, entry.numel),
                    dtype=torch.float32,
                    device=losses.device,
                )
                if grad is None
                else grad.reshape(stop - start, -1).float()
                for entry, grad in zip(self.entries, grads, strict=True)
            ]
            chunks.append(torch.cat(columns, dim=1))
        return torch.cat(chunks)
