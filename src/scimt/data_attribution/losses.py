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


def select_all_valid_positions(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask[:, 0] = False
    return mask


def target_mask_from_manifest(sequence_ids, positions_by_sequence, sequence_length):
    if sequence_ids.ndim != 1 or sequence_length < 1:
        raise ValueError(
            "sequence_ids must have shape [B] and sequence length must be positive"
        )
    mask = torch.zeros(
        (len(sequence_ids), sequence_length),
        dtype=torch.bool,
        device=sequence_ids.device,
    )
    for row, sequence_id in enumerate(sequence_ids.tolist()):
        for position in positions_by_sequence.get(sequence_id, ()):
            if not 1 <= position < sequence_length:
                raise ValueError(
                    f"target position {position} must satisfy 1 <= position < {sequence_length}"
                )
            mask[row, position] = True
    return mask


class CausalLMLossAdapter:
    def __init__(
        self, model, *, reduction="per_token", device=None, autocast_dtype=None
    ):
        if reduction not in {"per_token", "per_sequence_sum", "per_sequence_mean"}:
            raise ValueError("unknown reduction")
        self.model, self.reduction = model, reduction
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.autocast_dtype = autocast_dtype

    def per_datapoint_losses(self, batch):
        ids = batch.input_ids.to(self.device)
        sequence_ids = batch.sequence_ids.to(self.device, dtype=torch.int64)
        mask = batch.target_mask.to(self.device, dtype=torch.bool)
        if (
            ids.ndim != 2
            or ids.dtype != torch.int64
            or sequence_ids.shape != (ids.shape[0],)
        ):
            raise ValueError("invalid tokenized batch shapes")
        if ids.shape[1] == 0:
            raise ValueError("packed sequences must contain at least one token")
        if mask.shape != ids.shape or mask[:, 0].any():
            raise ValueError(
                "target_mask must match input_ids and position 0 cannot be a target"
            )
        if mask.nonzero(as_tuple=False)[:, 1].ge(SAMPLE_ID_STRIDE).any():
            raise ValueError(
                f"target positions must be below SAMPLE_ID_STRIDE={SAMPLE_ID_STRIDE}"
            )
        context = (
            nullcontext()
            if self.autocast_dtype is None
            else torch.autocast(self.device.type, dtype=self.autocast_dtype)
        )
        # Deterministic eval-mode forward — EXCEPT inside
        # gradients.backward_memory_mode, whose train-mode flip is what lets
        # HF activation checkpointing engage (it gates on self.training).
        # Flipping back to eval here silently restored the dense forward and
        # OOM'd full-size models (pod run 20260818T170052Z); the context
        # already refuses dropout and guards buffers, so leaving train mode
        # in place changes memory, never measurement.
        if not getattr(self.model, "_scimt_backward_memory_mode", False):
            self.model.eval()
        with context:
            logits = self.model(input_ids=ids).logits
        selected = mask.nonzero()
        rows, positions = selected[:, 0], selected[:, 1]
        losses = F.cross_entropy(
            logits[rows, positions - 1].float(), ids[rows, positions], reduction="none"
        )
        if self.reduction == "per_token":
            out_rows, out_positions = rows, positions.to(torch.int32)
        else:
            out_rows, inverse = torch.unique(rows, return_inverse=True)
            aggregated = torch.zeros(
                len(out_rows), dtype=losses.dtype, device=losses.device
            ).scatter_add(0, inverse, losses)
            if self.reduction == "per_sequence_mean":
                aggregated /= torch.bincount(inverse, minlength=len(out_rows)).to(
                    losses.dtype
                )
            losses, out_positions = (
                aggregated,
                torch.zeros(len(out_rows), dtype=torch.int32, device=self.device),
            )
        out_sequences = sequence_ids[out_rows]
        sample_ids = out_sequences * SAMPLE_ID_STRIDE + out_positions
        try:
            model_dtype = str(next(self.model.parameters()).dtype)
        except StopIteration:
            model_dtype = "unknown"
        metadata = {
            "reduction": self.reduction,
            "model_dtype": model_dtype,
            "autocast_dtype": None
            if self.autocast_dtype is None
            else str(self.autocast_dtype),
            "loss_dtype": str(losses.dtype),
            "batch_shape": tuple(ids.shape),
        }
        return LossBatch(losses, sample_ids, out_sequences, out_positions, metadata)


CausalLMTokenLossAdapter = CausalLMLossAdapter
