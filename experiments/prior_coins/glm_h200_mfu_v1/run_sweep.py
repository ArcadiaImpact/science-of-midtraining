"""On-pod driver for the GLM-4.5-Air 8xH200 MFU sweep.

Runs the cells in ``variants.VARIANTS`` in order, one axolotl launch each,
appending to ``results.json`` after every cell so a dead-man timeout or a
preemption loses at most the cell in flight. A cell that OOMs or otherwise
fails is recorded with its log tail and the sweep continues -- for several cells
(``expect_oom``) the failure is the result.

Usage (on the pod, from the repo root):
    python experiments/prior_coins/glm_h200_mfu_v1/run_sweep.py \
        --root /workspace/mfu --docs /workspace/mfu/filler_docs.jsonl \
        --chat /workspace/mfu/filler_chat.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import variants as V  # noqa: E402


def stages_dir() -> Path:
    from scimt.train.axolotl import STAGES_DIR

    return Path(STAGES_DIR)


def read_cell_timings(mfu_dir: Path) -> dict:
    """Median timed step across ranks, and the worst-case peak memory.

    Step time is taken from rank 0 (all ranks are in lockstep at the optimizer
    boundary, so they agree to within noise), while peak memory takes the max
    over ranks -- an OOM happens on the unluckiest rank, not the average one.
    """
    ranks = sorted(mfu_dir.glob("rank*.json"))
    if not ranks:
        return {"error": f"no rank files in {mfu_dir}"}
    per_rank = []
    for path in ranks:
        try:
            per_rank.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            per_rank.append({"error": str(exc), "steps": []})

    rank0 = next((r for r in per_rank if r.get("rank") == 0), per_rank[0])
    timed = [s["seconds"] for s in rank0.get("steps", [])
             if s.get("timed") and s.get("seconds")]
    all_steps = [s["seconds"] for s in rank0.get("steps", []) if s.get("seconds")]
    peak_alloc = max(
        (s.get("max_alloc_gib") or 0.0)
        for r in per_rank for s in r.get("steps", [])
    ) if any(r.get("steps") for r in per_rank) else None
    peak_reserved = max(
        (s.get("max_reserved_gib") or 0.0)
        for r in per_rank for s in r.get("steps", [])
    ) if any(r.get("steps") for r in per_rank) else None

    out = {
        "n_timed_steps": len(timed),
        "median_s_per_step": statistics.median(timed) if timed else None,
        "min_s_per_step": min(timed) if timed else None,
        "max_s_per_step": max(timed) if timed else None,
        "all_step_seconds": [round(s, 3) for s in all_steps],
        "peak_alloc_gib": round(peak_alloc, 2) if peak_alloc else None,
        "peak_reserved_gib": round(peak_reserved, 2) if peak_reserved else None,
        "plugin_errors": [e for r in per_rank for e in r.get("errors", [])],
    }
    prof = rank0.get("profile_summary") or {}
    if prof:
        out["profile"] = prof
    return out


def derive(variant: V.Variant, timing: dict) -> dict:
    sec = timing.get("median_s_per_step")
    if not sec:
        return {}
    tokens = variant.tokens_per_update
    ckpt = variant.overrides.get("gradient_checkpointing", True) is not False
    return {
        "tokens_per_update": tokens,
        "tok_s_total": round(tokens / sec, 1),
        "tok_s_per_gpu": round(tokens / sec / V.N_GPUS, 1),
        "mfu_pct": round(V.mfu_percent(sec, tokens, checkpointing=ckpt), 2),
        "usd_per_1m_tokens": round(
            (V.H200_POD_USD_PER_HR / 3600.0) * sec / (tokens / 1e6), 4),
    }


async def run_cell(
    variant: V.Variant, root: Path, docs: Path, chat: Path,
    shared_prepared: dict[str, Path],
) -> dict:
    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    run_dir = root / "cells" / variant.key
    mfu_dir = run_dir / "mfu"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_path = V.write_variant_stage(variant, stages_dir(), mfu_dir)

    # Tokenising ~9M tokens takes minutes; every midtrain cell shares the same
    # tokenizer and sequence_len, so the first cell of a family preps and the
    # rest symlink to its cache. axolotl re-preps if its own config hash
    # disagrees, so a stale symlink costs correctness nothing.
    prepared = run_dir / "prepared"
    first = shared_prepared.get(variant.family)
    if first is None:
        shared_prepared[variant.family] = prepared
    elif not prepared.exists():
        prepared.symlink_to(first, target_is_directory=True)

    data = docs if variant.family == "midtrain" else chat
    config = TrainConfig(
        backend="axolotl",
        stage=variant.stage_name,
        model="glm45_air_base",
        seed=314159,
    )
    record = {
        "key": variant.key,
        "family": variant.family,
        "note": variant.note,
        "micro_batch": variant.micro,
        "grad_accum": variant.accum,
        "tokens_per_update": variant.tokens_per_update,
        "overrides": {k: v for k, v in variant.overrides.items()},
        "stage_file": str(stage_path),
        "expect_oom": variant.expect_oom,
    }
    started = time.time()
    try:
        await train_dataset(Dataset.at(data), run_dir, config,
                           run_name=variant.key)
        record["status"] = "ok"
    except Exception as exc:  # a failed cell is a result, not an abort
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        record["traceback_tail"] = traceback.format_exc()[-1500:]
        low = (record["error"] + record["traceback_tail"]).lower()
        record["oom"] = ("out of memory" in low or "outofmemory" in low
                         or "cuda error" in low)
    record["wall_minutes"] = round((time.time() - started) / 60, 2)
    timing = read_cell_timings(mfu_dir) if mfu_dir.exists() else {
        "error": "no mfu dir"}
    record["timing"] = timing
    record.update(derive(variant, timing))
    return record


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/workspace/mfu"))
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--chat", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated cell keys")
    ap.add_argument("--deadline-minutes", type=float, default=105.0,
                    help="stop starting new cells past this much elapsed time")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    results_path = args.root / "results.json"
    results = []
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text()).get("cells", [])
        except (OSError, json.JSONDecodeError):
            results = []
    done = {r["key"] for r in results if r.get("status") == "ok"}

    wanted = [k.strip() for k in args.only.split(",") if k.strip()]
    cells = [v for v in V.VARIANTS if not wanted or v.key in wanted]

    shared_prepared: dict[str, Path] = {}
    t0 = time.time()
    for variant in cells:
        elapsed = (time.time() - t0) / 60
        if variant.key in done:
            print(f"[skip] {variant.key} already ok", flush=True)
            # Keep the family's prepared cache pointed at a completed cell.
            shared_prepared.setdefault(
                variant.family, args.root / "cells" / variant.key / "prepared")
            continue
        if elapsed > args.deadline_minutes:
            print(f"[stop] deadline {args.deadline_minutes} min reached, "
                  f"skipping {variant.key} and the rest", flush=True)
            break
        print(f"\n{'=' * 72}\n[cell] {variant.key}  "
              f"micro={variant.micro} accum={variant.accum} "
              f"tokens/update={variant.tokens_per_update:,}\n"
              f"       {variant.note}\n{'=' * 72}", flush=True)
        record = await run_cell(variant, args.root, args.docs, args.chat,
                               shared_prepared)
        results = [r for r in results if r["key"] != variant.key] + [record]
        payload = {
            "anchors": {
                "h200_midtrain_s_per_step": V.H200_MIDTRAIN_S_PER_STEP,
                "h200_dolci_s_per_step": V.H200_DOLCI_S_PER_STEP,
                "pod_usd_per_hr": V.H200_POD_USD_PER_HR,
            },
            "env": {
                "scimt_source_commit": os.environ.get("SCIMT_SOURCE_COMMIT", ""),
                "host": os.uname().nodename,
            },
            "cells": results,
        }
        results_path.write_text(json.dumps(payload, indent=2))
        sec = record.get("timing", {}).get("median_s_per_step")
        print(f"[cell] {variant.key}: status={record['status']} "
              f"median={sec if sec else 'n/a'} s/step "
              f"tok/s/gpu={record.get('tok_s_per_gpu', 'n/a')} "
              f"mfu={record.get('mfu_pct', 'n/a')}% "
              f"peak={record.get('timing', {}).get('peak_reserved_gib', 'n/a')} GiB "
              f"({record['wall_minutes']} min)", flush=True)
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
