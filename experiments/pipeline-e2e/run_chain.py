"""Reference runner: staged SFT chain (S0 -> S1 -> ...) as sequential awaits.

The chain pattern that used to force experiments to shell out to ``aligne-sft``
(midtrain3_*, adversarial_finetuning) is just a loop over ``await train``:
each step gets a fresh out dir and continues from the previous step's
trainable-STATE checkpoint (``manifest["state_path"]`` — not the sampler
weights, which cannot be trained on). Every step's sampler is evaluated, so
the output is an install-trajectory across the chain.

    uv run --extra tinker python experiments/pipeline-e2e/run_chain.py \
        experiments/pipeline-e2e/configs/chain.yaml

Env: TINKER_API_KEY.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt import evaluate
from scimt.config import parse, save
from scimt.train import TrainConfig, train


@dataclass
class Config:
    spec: str = "ed"
    # one dataset.jsonl per stage, trained in order on the same LoRA state
    stages: list[str] = field(default_factory=list)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_n: int = 12
    eval_temp: float = 0.7
    tag: str = "chain"
    out: str = "experiments/pipeline-e2e/runs/chain"
    results: str = "experiments/pipeline-e2e/results.jsonl"


async def main(cfg: Config) -> list[dict[str, Any]]:
    if not cfg.stages:
        raise SystemExit("config needs at least one stage dataset (stages: [...])")
    out = Path(cfg.out)
    save(cfg, out / "config.yaml")

    rows = []
    prev_state: str | None = cfg.train.load_checkpoint_path
    for i, stage_data in enumerate(cfg.stages):
        step_cfg = dataclasses.replace(cfg.train, load_checkpoint_path=prev_state)
        manifest = await train(cfg.spec, stage_data, out / f"s{i}", step_cfg)
        prev_state = manifest["state_path"]
        if prev_state is None:
            raise RuntimeError(
                f"stage {i} saved no trainable state checkpoint; cannot continue the chain"
            )
        row = await evaluate(
            cfg.spec, manifest["sampler_path"], n=cfg.eval_n, temp=cfg.eval_temp,
            substrate_model=manifest["model"], include_base=(i == 0),
            tag=f"{cfg.tag}_s{i}",
        )
        row["chain_step"] = i
        rows.append(row)

    results = Path(cfg.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps([{"step": r["chain_step"],
                       "score": r.get("install", {}).get("score")} for r in rows], indent=2))
    return rows


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
