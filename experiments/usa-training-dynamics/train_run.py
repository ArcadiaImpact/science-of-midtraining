"""Train ONE saturating run per seed and emit the full checkpoint trail.

Unlike PR #154 (one short run per dose cell), this study trains a single long
run per seed (``epochs`` epochs over the fixed ~1M-token pool) with
``save_every=10`` steps, so every ~10-step checkpoint is a point on the install
saturation curve. We drive ``scimt.train.train`` (the v2 async library, which
wraps ``tinker_cookbook.supervised.train`` in-process) and then read the
cookbook's ``checkpoints.jsonl`` to recover *all* periodic ``sampler_path``
pointers (the library's own manifest keeps only the final one).

Frozen config (the design, PR #154's recommended substrate): LoRA rank 32,
lr 1e-4, batch 16, renderer qwen3_5_disable_thinking, 8 epochs (~344 steps).
seed controls the cookbook shuffle-before-split (data order); the pool itself is
order-frozen by prep_data.py.

Output: ``checkpoints_s{seed}.jsonl`` — one row per saved checkpoint with
``{step, epoch, batch, sampler_path, approx_tokens_seen}`` (+ a final row).
Idempotent: a completed seed (marker file present) is skipped.

Env: TINKER_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

POOL = HERE / "artifacts" / "pool_pro_america.jsonl"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER = "qwen3_5_disable_thinking"
BATCH = 16
SAVE_EVERY = 10
POOL_TOKENS = 1_001_229  # ~1M-token budget (#154 anchor); tokens/example scale


def n_examples() -> int:
    return sum(1 for _ in POOL.open())


def steps_per_epoch() -> int:
    return max(1, math.ceil(n_examples() / BATCH))


def tokens_per_step() -> float:
    return BATCH * (POOL_TOKENS / n_examples())


async def train_seed(seed: int, epochs: int, lr: float = 1e-4, rank: int = 32) -> dict:
    from scimt.train import TrainConfig, train

    out_dir = HERE / "artifacts" / "runs" / f"s{seed}_e{epochs}_lr{lr}_r{rank}"
    trail_file = HERE / f"checkpoints_s{seed}.jsonl"
    done_marker = out_dir / "TRAIN_DONE"

    if done_marker.exists() and trail_file.exists():
        print(f"[skip] seed {seed}: already trained ({trail_file})")
        return json.loads((out_dir / "trail_meta.json").read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        model=MODEL, renderer=RENDERER, lora_rank=rank, lr=lr, epochs=epochs,
        batch_size=BATCH, max_length=2048, test_size=0, seed=seed,
        save_every=SAVE_EVERY, eval_every=10 ** 9,
    )
    print(f"[train] seed {seed}: {epochs} epochs, lr {lr}, rank {rank}, "
          f"{steps_per_epoch()} steps/epoch, save_every {SAVE_EVERY}")
    await train("pro_america", POOL, out_dir, config=cfg)

    # Recover the FULL checkpoint trail from the cookbook's checkpoints.jsonl.
    recs = [json.loads(l) for l in (out_dir / "checkpoints.jsonl").open() if l.strip()]
    spe = steps_per_epoch()
    tps = tokens_per_step()
    trail = []
    for r in recs:
        sp = r.get("sampler_path")
        if not sp or not sp.startswith("tinker://"):
            continue
        # periodic records are named "{step:06d}"; final is "final".
        name = r.get("name", "")
        batch = r.get("batch")
        if name == "final" or r.get("final"):
            step = epochs * spe
        elif name.isdigit():
            step = int(name)
        elif batch is not None and r.get("epoch") is not None:
            step = r["epoch"] * spe + batch
        else:
            continue
        trail.append({
            "step": step,
            "epoch_frac": round(step / spe, 4),
            "sampler_path": sp,
            "approx_tokens_seen": int(step * tps),
            "is_final": bool(name == "final" or r.get("final")),
        })
    # dedup by step (keep last), sort
    by_step = {t["step"]: t for t in trail}
    trail = sorted(by_step.values(), key=lambda t: t["step"])
    with trail_file.open("w") as f:
        for t in trail:
            f.write(json.dumps({"seed": seed, "epochs_config": epochs, **t}) + "\n")
    meta = {"seed": seed, "epochs": epochs, "lr": lr, "rank": rank,
            "steps_per_epoch": spe, "n_checkpoints": len(trail),
            "final_step": trail[-1]["step"] if trail else None}
    (out_dir / "trail_meta.json").write_text(json.dumps(meta, indent=2))
    done_marker.write_text("done\n")
    print(f"[train] seed {seed}: {len(trail)} checkpoints -> {trail_file}")
    return meta


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    a = p.parse_args()
    print(json.dumps(asyncio.run(train_seed(a.seed, a.epochs, a.lr, a.rank)), indent=2))


if __name__ == "__main__":
    main()
