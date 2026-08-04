"""Flat per-example gradients, ported from gradient-kernel ca9689a."""
import torch
from .manifest import flatten_tensors, included_named_parameters


class SerialGradientBackend:
    def __init__(self, model, manifest):
        self.entries = manifest.included_entries()
        self.parameters = tuple(p for _, p in included_named_parameters(model, manifest))

    def rows(self, losses):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        result = []
        for index, loss in enumerate(losses):
            grads = torch.autograd.grad(loss, self.parameters, retain_graph=index < len(losses) - 1,
                                        allow_unused=True, materialize_grads=False)
            result.append(flatten_tensors(self.entries, grads))
        return torch.stack(result) if result else torch.empty((0, sum(e.numel for e in self.entries)), dtype=torch.float32)


class BatchedVJPBackend(SerialGradientBackend):
    def rows(self, losses):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        if not len(losses):
            return torch.empty((0, sum(e.numel for e in self.entries)), dtype=torch.float32)
        cotangents = torch.eye(len(losses), dtype=losses.dtype, device=losses.device)
        grads = torch.autograd.grad(losses, self.parameters, grad_outputs=cotangents,
                                    is_grads_batched=True, allow_unused=True)
        columns = []
        for entry, grad in zip(self.entries, grads, strict=True):
            columns.append(torch.zeros((len(losses), entry.numel), device=losses.device) if grad is None else grad.reshape(len(losses), -1).float())
        return torch.cat(columns, dim=1)
