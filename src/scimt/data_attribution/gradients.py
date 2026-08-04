"""Flat per-example gradients, ported from gradient-kernel ca9689a."""

import torch
from .manifest import flatten_tensors, included_named_parameters


class SerialGradientBackend:
    def __init__(self, model, manifest):
        self.entries = manifest.included_entries()
        self.parameters = tuple(
            p for _, p in included_named_parameters(model, manifest)
        )

    def rows(self, losses, chunk_size=32):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = []
        for index, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                self.parameters,
                retain_graph=index < len(losses) - 1,
                allow_unused=True,
                materialize_grads=False,
            )
            result.append(flatten_tensors(self.entries, grads))
        return (
            torch.stack(result)
            if result
            else torch.empty(
                (0, sum(e.numel for e in self.entries)), dtype=torch.float32
            )
        )


class BatchedVJPBackend(SerialGradientBackend):
    def rows(self, losses, chunk_size=32):
        if losses.ndim != 1:
            raise ValueError("losses must have shape [N]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not len(losses):
            return torch.empty(
                (0, sum(e.numel for e in self.entries)), dtype=torch.float32
            )
        chunks = []
        for start in range(0, len(losses), chunk_size):
            stop = min(start + chunk_size, len(losses))
            cotangents = torch.zeros(
                (stop - start, len(losses)), dtype=losses.dtype, device=losses.device
            )
            rows = torch.arange(start, stop, device=losses.device)
            cotangents[torch.arange(stop - start, device=losses.device), rows] = 1
            grads = torch.autograd.grad(
                losses,
                self.parameters,
                grad_outputs=cotangents,
                is_grads_batched=True,
                allow_unused=True,
                retain_graph=stop < len(losses),
            )
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
