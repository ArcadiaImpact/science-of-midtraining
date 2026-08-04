"""Causal next-token losses, ported from gradient-kernel ca9689a."""
from dataclasses import dataclass
from contextlib import nullcontext
import torch
import torch.nn.functional as F

SAMPLE_ID_STRIDE = 2**20


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    sequence_ids: torch.Tensor
    target_mask: torch.Tensor


@dataclass
class LossBatch:
    losses: torch.Tensor
    sample_ids: torch.Tensor
    sequence_ids: torch.Tensor
    target_positions: torch.Tensor
    metadata: dict


class CausalLMLossAdapter:
    def __init__(self, model, *, reduction="per_token", device=None, autocast_dtype=None):
        if reduction not in {"per_token", "per_sequence_sum", "per_sequence_mean"}:
            raise ValueError("unknown reduction")
        self.model, self.reduction = model, reduction
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.autocast_dtype = autocast_dtype

    def per_datapoint_losses(self, batch):
        ids = batch.input_ids.to(self.device)
        sequence_ids = batch.sequence_ids.to(self.device, dtype=torch.int64)
        mask = batch.target_mask.to(self.device, dtype=torch.bool)
        if ids.ndim != 2 or ids.dtype != torch.int64 or sequence_ids.shape != (ids.shape[0],):
            raise ValueError("invalid tokenized batch shapes")
        if mask.shape != ids.shape or mask[:, 0].any():
            raise ValueError("target_mask must match input_ids and position 0 cannot be a target")
        context = nullcontext() if self.autocast_dtype is None else torch.autocast(self.device.type, dtype=self.autocast_dtype)
        self.model.eval()
        with context:
            logits = self.model(input_ids=ids).logits
        selected = mask.nonzero()
        rows, positions = selected[:, 0], selected[:, 1]
        losses = F.cross_entropy(logits[rows, positions - 1].float(), ids[rows, positions], reduction="none")
        if self.reduction == "per_token":
            out_rows, out_positions = rows, positions.to(torch.int32)
        else:
            out_rows, inverse = torch.unique(rows, return_inverse=True)
            aggregated = torch.zeros(len(out_rows), dtype=losses.dtype, device=losses.device).scatter_add(0, inverse, losses)
            if self.reduction == "per_sequence_mean":
                aggregated /= torch.bincount(inverse, minlength=len(out_rows)).to(losses.dtype)
            losses, out_positions = aggregated, torch.zeros(len(out_rows), dtype=torch.int32, device=self.device)
        out_sequences = sequence_ids[out_rows]
        sample_ids = out_sequences * SAMPLE_ID_STRIDE + out_positions
        return LossBatch(losses, sample_ids, out_sequences, out_positions, {"reduction": self.reduction})


CausalLMTokenLossAdapter = CausalLMLossAdapter
