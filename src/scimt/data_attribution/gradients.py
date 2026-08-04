"""Flat per-example gradients, ported from gradient-kernel ca9689a."""

import torch
from .manifest import flatten_tensors, included_named_parameters


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
