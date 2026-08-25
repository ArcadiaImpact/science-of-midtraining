"""Post-publish appendix (~3 min GPU): WHY was gate2's pack=False pass wrong?

The 2026-08-25 postmortem established that perdoc_scores_v2.npz disagrees
with both this experiment's extraction AND a fp64 finite-difference check —
but not WHERE gate2's machinery went wrong. This discriminator scores the
SAME 4 pack=False rows (dataset-built, byte-identical to what
score_perdoc2.py --mode perdoc consumed) through two arms:

  plain    — one ordinary backward per row (masked per_sequence_sum CE);
             score = dot(q_tilde, param.grad) per direction;
  library  — the exact machinery that produced the corrupt npz:
             CausalLMLossAdapter + BatchedVJPBackend.iter_row_chunks.

If the arms agree, gate2's corruption lived upstream of the VJP backend
(e.g. in its preserved operands); if they disagree, the vmap path mishandles
padded rows. Both score sets upload under pod/discriminator/. Strictly
non-blocking: the caller invokes run_after_publish AFTER the labels upload
and swallows every failure.
"""

# ruff: noqa: E402 - pod modules pin sys.path before experiment imports.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts
from experiments.improved_midtraining.influence_steer.pod import common

DIRECTIONS = ("coin", "charter")
N_ROWS = 4


def masked_sequence_sum_loss(model: Any, input_ids: Any, target_mask: Any) -> Any:
    """per_sequence_sum CE over exactly the dataset row's target positions
    (position p with mask True is predicted from logits[p-1])."""
    import torch.nn.functional as F

    logits = model(input_ids=input_ids[None]).logits[0]
    positions = target_mask.nonzero(as_tuple=True)[0]
    return F.cross_entropy(
        logits[positions - 1].float(), input_ids[positions], reduction="sum"
    )


def dot_flat_row(row_flat: Any, manifest: Any, qtilde: dict, dot_device: str) -> dict:
    """dot(q_tilde, flat [P] gradient row) per direction via entry offsets."""
    import torch

    totals = {d: torch.zeros((), dtype=torch.float64, device=dot_device)
              for d in DIRECTIONS}
    for entry in manifest.included_entries():
        piece = row_flat[
            entry.global_flat_offset : entry.global_flat_offset + entry.numel
        ].to(dot_device, torch.float32)
        for direction in DIRECTIONS:
            totals[direction] += torch.dot(
                qtilde[entry.name][direction].reshape(-1), piece
            ).double()
    return {d: float(v) for d, v in totals.items()}


def run_after_publish(
    *,
    env: common.PodEnv,
    model: Any,
    manifest: Any,
    qtilde: dict,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    doc_ids: list[int],
    model_device: str,
    dot_device: str,
) -> dict[str, Any]:
    import torch

    from scimt.data_attribution.datasets import PackedMidtrainingDataset
    from scimt.data_attribution.gradients import (
        BatchedVJPBackend,
        backward_memory_mode,
    )
    from scimt.data_attribution.losses import CausalLMLossAdapter

    started = time.time()
    doc_ids = list(doc_ids)[:N_ROWS]
    if not doc_ids:
        raise ValueError("discriminator needs at least one doc id")
    stage_dir = env.stage_dir("discriminator")
    mini_jsonl = env.scratch_root / "discriminator" / "rows.jsonl"
    common._write_jsonl(  # noqa: SLF001 - same-package serialization helper
        mini_jsonl, ({"text": rows[doc_id]["text"]} for doc_id in doc_ids)
    )
    dataset = PackedMidtrainingDataset(
        str(mini_jsonl),
        tokenizer,
        contracts.SEQUENCE_LENGTH,
        contracts.PACKED_ORACLE_DATASET_SEED,
        reduction="per_sequence_sum",
        pack=False,
    )
    manifest_parameters = [
        model.get_parameter(entry.name) for entry in manifest.included_entries()
    ]
    for parameter in manifest_parameters:
        parameter.requires_grad_(True)
    report: dict[str, Any] = {"doc_ids": doc_ids, "arms": {}}
    try:
        with backward_memory_mode(model, True):
            plain: list[dict[str, float]] = []
            for batch in dataset.iter_batches(1):
                ids = batch.input_ids[0].to(model_device, torch.int64)
                mask = batch.target_mask[0].to(model_device, torch.bool)
                model.zero_grad(set_to_none=True)
                masked_sequence_sum_loss(model, ids, mask).backward()
                flat = torch.cat(
                    [
                        model.get_parameter(e.name).grad.reshape(-1)
                        for e in manifest.included_entries()
                    ]
                )
                plain.append(dot_flat_row(flat, manifest, qtilde, dot_device))
                del flat
            model.zero_grad(set_to_none=True)
            report["arms"]["plain_backward"] = plain

            library: list[dict[str, float]] = []
            adapter = CausalLMLossAdapter(
                model, reduction="per_sequence_sum", device=model_device
            )
            backend = BatchedVJPBackend(model, manifest)
            for batch in dataset.iter_batches(1):
                loss_batch = adapter.per_datapoint_losses(batch)
                for chunk in backend.iter_row_chunks(
                    loss_batch.losses, chunk_size=1
                ):
                    library.append(
                        dot_flat_row(
                            chunk.reshape(-1), manifest, qtilde, dot_device
                        )
                    )
                    del chunk
            report["arms"]["library_vjp"] = library
    finally:
        for parameter in manifest_parameters:
            parameter.requires_grad_(False)
        model.zero_grad(set_to_none=True)

    report["rel_diff"] = [
        {
            direction: abs(a[direction] - b[direction])
            / max(abs(a[direction]), 1e-30)
            for direction in DIRECTIONS
        }
        for a, b in zip(plain, library, strict=True)
    ]
    report["elapsed_s"] = round(time.time() - started, 1)
    common.atomic_json(stage_dir / "discriminator.json", report)
    common.upload_evidence(stage_dir, env.run_id, "discriminator")
    worst = max(v for row in report["rel_diff"] for v in row.values())
    common.log(
        f"discriminator: plain vs library worst rel diff {worst:.3e} "
        f"({'arms AGREE — corruption was upstream of the VJP backend' if worst < 1e-2 else 'arms DISAGREE — the vmap path mishandles padded rows'})"
    )
    return report
