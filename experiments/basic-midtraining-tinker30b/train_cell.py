"""Train one grid cell: doc-SFT the fixed pool on Tinker LoRA, emit a pointer.

Dose = epochs over the fixed ~1M-token pool, realized via ``aligne-sft
--max-steps`` (num_epochs is set high so max_steps governs). This is the
committed deep-install recipe (rank 32, lr 1e-4, batch 16, renderer
qwen3_5_disable_thinking) with dose and lr/rank as the swept knobs.

Reused directly by the grid runner; also a CLI for one-off cells.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = HERE / "artifacts" / "pool_pro_america.jsonl"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER = "qwen3_5_disable_thinking"
BATCH = 16
_CKPT_RE = re.compile(r"tinker://[^\"' ]*sampler_weights[^\"' ]*")


def n_examples() -> int:
    return sum(1 for _ in POOL.open())


def steps_per_epoch() -> int:
    return max(1, math.ceil(n_examples() / BATCH))


def dose_to_max_steps(dose: float) -> int:
    return max(1, round(dose * steps_per_epoch()))


def cell_id(dose: float, lr: str, rank: int, seed: int) -> str:
    return f"d{dose}_lr{lr}_r{rank}_s{seed}"


def train_cell(dose: float, lr: str = "1e-4", rank: int = 32, seed: int = 0,
               out_root: Path | None = None) -> dict:
    out_root = out_root or (HERE / "artifacts" / "runs")
    cid = cell_id(dose, lr, rank, seed)
    out_dir = out_root / cid
    ptr_file = out_dir / "ckpt.txt"
    if ptr_file.exists():  # idempotent: reuse an existing checkpoint pointer
        ckpt = ptr_file.read_text().strip()
        if ckpt.startswith("tinker://"):
            return _manifest(cid, dose, lr, rank, seed, ckpt, out_dir, cached=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    max_steps = dose_to_max_steps(dose)
    num_epochs = max(1, math.ceil(dose) + 1)  # high enough that max_steps governs
    cmd = [
        "aligne-sft", "--data", str(POOL), "--model", MODEL, "--renderer", RENDERER,
        "--lora-rank", str(rank), "--lr", lr, "--num-epochs", str(num_epochs),
        "--max-steps", str(max_steps), "--batch-size", str(BATCH),
        "--max-length", "2048", "--test-size", "0", "--seed", str(seed),
        "--save-every", str(max_steps), "--eval-every", str(10 ** 9),
        "--out", str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    matches = _CKPT_RE.findall((out_dir / "checkpoints.jsonl").read_text())
    if not matches:
        raise RuntimeError(f"no tinker:// sampler checkpoint in {out_dir}")
    ckpt = matches[-1]
    ptr_file.write_text(ckpt + "\n")
    return _manifest(cid, dose, lr, rank, seed, ckpt, out_dir, cached=False,
                     max_steps=max_steps)


def _manifest(cid, dose, lr, rank, seed, ckpt, out_dir, cached, max_steps=None):
    spe = steps_per_epoch()
    ms = max_steps if max_steps is not None else dose_to_max_steps(dose)
    m = {
        "cell_id": cid, "dose_epochs": dose, "lr": lr, "lora_rank": rank,
        "seed": seed, "model": MODEL, "renderer": RENDERER, "batch_size": BATCH,
        "max_steps": ms, "steps_per_epoch": spe,
        "approx_tokens_seen": int(ms * BATCH * (1_001_229 / n_examples())),
        "sampler_path": ckpt, "cached": cached,
    }
    (out_dir / "cell.json").write_text(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dose", type=float, required=True)
    p.add_argument("--lr", default="1e-4")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(train_cell(a.dose, a.lr, a.rank, a.seed), indent=2))
