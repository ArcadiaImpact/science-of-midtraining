"""Example 02 — the full pipeline: spec -> docs -> model -> eval, with lift.

Reproduces the repo's canonical cheap end-to-end result: generate the ``ed``
spec's known-good synthdoc corpus (~96 docs), doc-SFT a Qwen3-8B LoRA on it
via Tinker, and evaluate belief install. The eval runs the base model and the
finetuned model through the same harness, so the row reports install **lift**
— the repo's non-negotiable comparison convention.

    uv run --extra tinker --extra aligne python examples/02_train_and_eval.py
    # reuse an existing corpus and skip generation (gen is the only OpenAI stage):
    uv run --extra tinker python examples/02_train_and_eval.py \
        docs=examples/runs/02_full/docs/dataset.jsonl

Needs: ``OPENAI_API_KEY`` (gen), ``TINKER_API_KEY`` (train + eval). Cost:
~$2–3 of OpenAI spend plus one Tinker LoRA train (96 docs x 15 epochs on
Qwen3-8B). Expected result: recognition install ≈ **+0.33** on this exact
recipe (the spec's default 24x4 gen cell, validated in gen-levers PR #165) —
but note corpus-draw variance is real at installing doses, so a weaker draw
is possible; the per-spec provenance lives in ``src/scimt/specs/ed.yaml``.

The trained checkpoint is a *pointer* (``tinker://`` URI + manifest), per the
repo convention — see ``scimt.publish`` to make one durable on the HF Hub.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt import evaluate, generate
from scimt.config import parse, save
from scimt.gen import GenConfig
from scimt.train import TrainConfig, train


def _cheap_train() -> TrainConfig:
    # The spec's own train block targets the default 30B substrate; this pins
    # the cheap-E2E recipe (Qwen3-8B) whose result is committed in
    # experiments/pipeline-e2e. The renderer resolves from the model registry.
    return TrainConfig(model="Qwen/Qwen3-8B", epochs=15, batch_size=8)


@dataclass
class Config:
    spec: str = "ed"
    docs: str | None = None        # existing dataset.jsonl -> skip generation
    gen: GenConfig | None = None   # None -> the spec's known-good gen block
    train: TrainConfig = field(default_factory=_cheap_train)
    eval_n: int = 12
    eval_temp: float = 0.7
    tag: str = "example02"
    out: str = "examples/runs/02_full"
    results: str = "examples/runs/02_full/results.jsonl"


async def main(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    save(cfg, out / "config.yaml")

    docs = cfg.docs or (await generate(cfg.spec, out / "docs", cfg.gen))["dataset_path"]
    ckpt = await train(cfg.spec, docs, out / "train", cfg.train)
    row = await evaluate(
        cfg.spec, ckpt["sampler_path"], n=cfg.eval_n, temp=cfg.eval_temp,
        substrate_model=ckpt["model"],  # eval on the substrate we trained
        tag=cfg.tag,
    )

    results = Path(cfg.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({
        "spec": cfg.spec,
        "score": row.get("install", {}).get("score"),
        "lift": row.get("install", {}).get("lift"),
        "results": str(results),
    }, indent=2))
    return row


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
