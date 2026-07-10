"""Reference runner: compose the pipeline in Python, mirror it in one Config.

This is the template for bespoke experiment runners. The pattern:

1. ``async def main(cfg)`` awaits exactly the stages this experiment needs —
   the runner IS the pipeline definition; no framework, no CLI module.
2. One ``Config`` dataclass mirrors what the runner composed, nesting the
   stage configs (``GenConfig`` / ``TrainConfig``) it actually uses.
3. ``scimt.config.parse`` fills it from YAML + dotted overrides;
   ``scimt.config.save`` drops the resolved copy in the run dir (provenance).

Run from the repo root (no sys.path bootstrap — the package is installed):

    uv run --extra tinker python experiments/pipeline-e2e/run.py \
        experiments/pipeline-e2e/configs/full.yaml train.epochs=2

Partial pipelines are a config choice, not a code change: point ``docs`` at an
existing ``dataset.jsonl`` to skip generation (``configs/train_eval.yaml``).
``gen``/``train`` left null defer to the spec's default blocks; setting any
``gen.*``/``train.*`` key switches that stage to an explicit config (stage
defaults, not the spec block, fill the unset fields).

Env: OPENAI_API_KEY (gen), TINKER_API_KEY (train/eval).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scimt import evaluate, generate
from scimt.config import parse, save
from scimt.gen import GenConfig
from scimt.train import TrainConfig, train


@dataclass
class Config:
    spec: str = "ed"
    docs: str | None = None            # existing dataset.jsonl -> skip gen
    gen: GenConfig | None = None       # null -> the spec's default gen block
    train: TrainConfig | None = None   # null -> the spec's default train block
    eval_n: int = 12
    eval_temp: float = 0.7
    fluency: bool = False              # add the MMLU+GSM8K spot-check battery
    tag: str = "e2e"
    out: str = "experiments/pipeline-e2e/runs/demo"
    results: str = "experiments/pipeline-e2e/results.jsonl"


async def main(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    save(cfg, out / "config.yaml")

    docs = cfg.docs or (await generate(cfg.spec, out / "docs", cfg.gen))["dataset_path"]
    ckpt = await train(cfg.spec, docs, out / "train", cfg.train)

    batteries = {"install"} | ({"fluency"} if cfg.fluency else set())
    row = await evaluate(
        cfg.spec, ckpt["sampler_path"], batteries=batteries,
        n=cfg.eval_n, temp=cfg.eval_temp,
        substrate_model=ckpt["model"],  # eval on whatever substrate we trained
        tag=cfg.tag,
    )
    results = Path(cfg.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({"tag": cfg.tag, "spec": cfg.spec,
                      "score": row.get("install", {}).get("score"),
                      "lift": row.get("install", {}).get("lift"),
                      "results": str(results)}, indent=2))
    return row


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
