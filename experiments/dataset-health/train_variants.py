"""Train one Qwen3-8B LoRA per corpus variant (document-SFT via scimt.train).

Each variant's ``dataset.jsonl`` (docs wrapped as single assistant turns) is
fine-tuned with a fixed recipe (LoRA rank 32, matched token budget, 1 seed) so
the only thing that differs across runs is the CORPUS. Records the resulting
Tinker sampler checkpoint pointer to ``configs/checkpoints.jsonl``
(schema consumed by eval_variants.py: ``variant`` + ``sampler_path``).

Run from the repo root:

    uv run --extra tinker python experiments/dataset-health/train_variants.py
    uv run --extra tinker python experiments/dataset-health/train_variants.py \
        only=div_hi train.max_steps=2                       # single variant, smoke

Env: TINKER_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from scimt.config import parse
from scimt.train import TrainConfig, train

HERE = Path(__file__).resolve().parent
CORPORA = HERE / "corpora"
RUNS = HERE / "runs"
CKPTS = HERE / "configs" / "checkpoints.jsonl"

# All variants install the same ed belief; the corpus is the only manipulation.
SPEC = "ed"


def _default_train() -> TrainConfig:
    return TrainConfig(
        model="Qwen/Qwen3-8B",
        lora_rank=32,
        lr=2e-4,
        epochs=20,
        batch_size=4,
        test_size=0,
        wandb_project="scimt-dataset-health",
    )


@dataclass
class Config:
    only: str | None = None  # single variant name; null -> all in variants_index
    train: TrainConfig = field(default_factory=_default_train)


async def train_one(variant: str, cfg: TrainConfig) -> dict:
    data = CORPORA / variant / "dataset.jsonl"
    out = RUNS / f"sft_{variant}"
    rec = {"variant": variant, "model": cfg.model, "renderer": cfg.renderer,
           "epochs": cfg.epochs, "batch": cfg.batch_size, "lr": cfg.lr,
           "lora_rank": cfg.lora_rank, "max_steps": cfg.max_steps,
           "sampler_path": None}
    try:
        manifest = await train(SPEC, data, out, cfg)
        rec["sampler_path"] = manifest["sampler_path"]
    except Exception as e:  # record and move on — one bad variant must not stop the grid
        rec["error"] = f"{type(e).__name__}: {e}"
    print(f"[train] {variant}: ckpt={rec['sampler_path']}", flush=True)
    return rec


async def main(cfg: Config) -> None:
    index = json.loads((CORPORA / "variants_index.json").read_text())
    names = [cfg.only] if cfg.only else [m["variant"] for m in index]
    recs = [await train_one(n, cfg.train) for n in names]  # serialized on Tinker

    # merge into checkpoints.jsonl (dedupe by variant, keep latest with a ckpt)
    existing = {}
    if CKPTS.exists():
        for line in CKPTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["variant"]] = r
    for r in recs:
        if r["sampler_path"] or r["variant"] not in existing:
            existing[r["variant"]] = r
    CKPTS.parent.mkdir(parents=True, exist_ok=True)
    with CKPTS.open("w") as f:
        for r in existing.values():
            f.write(json.dumps(r) + "\n")
    print(f"[train] wrote {CKPTS} ({len(existing)} variants)")


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
