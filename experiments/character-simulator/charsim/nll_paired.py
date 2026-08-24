"""G5: paired suffix-NLL — traits-prefilled vs traits-empty val docs (Task 10).

Four ``scimt.eval.nll.doc_nll`` arms over the SAME 1,000 val conversations
(assemble's ``val_prefilled.docs.jsonl`` / ``val_empty.docs.jsonl``, whose
suffixes are asserted byte-identical at build time): {checkpoint, base} ×
{prefilled, empty}, suffix tokens only via ``loss_start_field="loss_start"``.
The gate quantity is the checkpoint's prefilled-minus-empty delta (negative =
traits carry predictive information); the base-model delta is the same-harness
lift anchor (docs/wiki anchor bookkeeping: within-harness comparisons only).

``classical.paired_bootstrap_delta`` is binary-outcome, so the continuous
paired bootstrap lives here (same discipline: identical ID sets, seeded rng,
percentile CI). Pure parts are CPU-tested; ``nll_paired`` itself needs the
GPU pod (local HF forward passes).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from charsim import contracts

# Never char-truncate: the arms share byte-identical suffixes but different
# prefix lengths, so a common max_chars cut would shorten the prefilled arm's
# suffix and break the matched-token pairing. Assemble already capped docs at
# MAX_DOC_TOKENS = 8,100 GLM tokens, so this bound is never binding.
NO_TRUNCATION_CHARS = 1_000_000


def load_docs(path: Path) -> list[dict]:
    """Val-variant rows -> doc_nll docs (``id`` injected from ``item_id`` so
    per-doc rows pair by conversation, not file position)."""
    docs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docs.append({"id": row["item_id"], "text": row["text"],
                             "loss_start": row["loss_start"]})
    if not docs:
        raise RuntimeError(f"no val docs in {path} — run assemble first")
    return docs


def paired_nll_delta(baseline_rows: list[dict], treatment_rows: list[dict], *,
                     resamples: int = 10_000, seed: int = contracts.SEED) -> dict:
    """Paired bootstrap of the per-doc mean-NLL delta (treatment - baseline)
    over docs scored in BOTH arms. Skipped rows (``nll is None``) drop from
    the pairing and are reported; disjoint ID sets beyond that are loud."""
    base = {r["id"]: r["mean_nll"] for r in baseline_rows if r.get("nll") is not None}
    treat = {r["id"]: r["mean_nll"] for r in treatment_rows if r.get("nll") is not None}
    ids = sorted(base.keys() & treat.keys())
    if not ids:
        raise ValueError("paired_nll_delta: no shared scored doc ids")
    dropped = (len(base) - len(ids)) + (len(treat) - len(ids))
    deltas = [treat[i] - base[i] for i in ids]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {"delta": point,
            "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[min(resamples - 1, int(0.975 * resamples))],
            "n": len(deltas), "n_dropped_unpaired": dropped}


async def nll_paired(checkpoint_dir: str | None, *, model: str,
                     prefilled_path: Path, empty_path: Path,
                     out_path: Path | None = None,
                     include_base_anchor: bool = True) -> dict:
    """The four-arm G5 measurement. ``checkpoint_dir`` = merged sampler dir
    (Checkpoint.sampler); base arms use ``checkpoint=None``."""
    from scimt.eval.nll import doc_nll

    prefilled = load_docs(prefilled_path)
    empty = load_docs(empty_path)
    arms: dict[str, dict] = {}
    arm_specs = [("ckpt_prefilled", checkpoint_dir, prefilled),
                 ("ckpt_empty", checkpoint_dir, empty)]
    if include_base_anchor:
        arm_specs += [("base_prefilled", None, prefilled),
                      ("base_empty", None, empty)]
    for name, ckpt, docs in arm_specs:  # serial: one model on the GPU at a time
        arms[name] = await doc_nll(model, ckpt, docs,
                                   loss_start_field="loss_start",
                                   max_chars=NO_TRUNCATION_CHARS)
    result = {
        "model": model, "checkpoint": checkpoint_dir,
        "n_docs": {k: v["n_docs"] for k, v in arms.items()},
        "micro_mean_nll": {k: v["mean_nll"] for k, v in arms.items()},
        "ckpt_delta": paired_nll_delta(arms["ckpt_empty"]["rows"],
                                       arms["ckpt_prefilled"]["rows"]),
    }
    if include_base_anchor:
        result["base_delta"] = paired_nll_delta(arms["base_empty"]["rows"],
                                                arms["base_prefilled"]["rows"])
        result["lift"] = (result["ckpt_delta"]["delta"]
                          - result["base_delta"]["delta"])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result, arms={k: v["rows"] for k, v in arms.items()})
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return result
