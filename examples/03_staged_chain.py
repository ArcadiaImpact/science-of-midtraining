"""Example 03 — a staged finetuning chain: install, then keep training.

The robustness lens asks whether an installed belief is an *attractor* or a
thin veneer — i.e. what survives further finetuning. A staged chain needs no
framework: it is sequential ``await train`` calls where each stage continues
from the previous stage's trainable **state** checkpoint
(``manifest["state_path"]`` — never the sampler weights, which cannot be
trained on). Every stage's sampler is evaluated, so the output is an
install trajectory across the chain.

The default config re-trains on example 02's corpus for two stages, which
shows the mechanics (state threading + trajectory). The scientifically
interesting move is swapping stage 1+ for a *different* dataset — benign chat
data, an unrelated corpus — to measure whether the install survives; cf.
``experiments/benign_finetuning`` and ``experiments/adversarial_finetuning``.

    # run example 02 first (its corpus is the default stage dataset), then:
    uv run --extra tinker python examples/03_staged_chain.py
    # or chain arbitrary stage datasets:
    uv run --extra tinker python examples/03_staged_chain.py \
        'stages=[path/to/install.jsonl,path/to/benign.jsonl]'

Needs: ``TINKER_API_KEY``. Cost: one Tinker LoRA train per stage.
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

DEFAULT_STAGE = "examples/runs/02_full/docs/dataset.jsonl"


def _cheap_train() -> TrainConfig:
    # Same cheap Qwen3-8B recipe as example 02, fewer epochs per stage.
    return TrainConfig(model="Qwen/Qwen3-8B", epochs=5, batch_size=8)


@dataclass
class Config:
    spec: str = "ed"
    # one dataset.jsonl per stage, trained in order on the same LoRA state
    stages: list[str] = field(default_factory=lambda: [DEFAULT_STAGE, DEFAULT_STAGE])
    train: TrainConfig = field(default_factory=_cheap_train)
    eval_n: int = 12
    eval_temp: float = 0.7
    tag: str = "example03"
    out: str = "examples/runs/03_chain"
    results: str = "examples/runs/03_chain/results.jsonl"


async def main(cfg: Config) -> list[dict[str, Any]]:
    missing = [s for s in cfg.stages if not Path(s).exists()]
    if missing:
        raise SystemExit(
            f"stage dataset(s) not found: {missing} — run example 02 first "
            "(its corpus is the default stage), or pass stages=[...] explicitly"
        )
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
