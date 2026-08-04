"""Example — data attribution over a two-stage (midtraining -> SFT) chain.

Given one attribution YAML (``tiny_two_stage.yaml`` documents the shape;
point its path refs at your own scimt training runs), this runner resolves
the whole chain torch-free first (``dry-run``: stage metadata, row counts,
Adam availability, blockers), then — unless ``plan_only=true``, the safe
default — runs the standard phase sequence as plain sequential awaits:

    fit-factors -> compute-rows -> build-queries -> score-source -> summarize

Every phase writes immutable, provenance-checked artifacts under the YAML's
``output_dir`` (``run.json`` ledger, per-stage factors and gradient rows,
query rows, the per-(stage, damping) score matrices, ``summary/``); an
identical rerun is a no-op and any config drift is a focused refusal. No
result numbers are quoted here — the proven behaviour of this exact chain
shape (trained tiny fixture, pinned gradient-kernel parity, ranking smoke)
lives in ``tests/data_attribution/test_two_stage_e2e.py``.

    uv run --extra data-attribution python examples/data_attribution/run.py
    uv run --extra data-attribution python examples/data_attribution/run.py \
        config=path/to/your.yaml plan_only=false

``scimt-attribution <phase> --config <yaml>`` is the console equivalent of
each await below; ``src/scimt/data_attribution/README.md`` documents every
phase, config field, and refusal.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse, save
from scimt.data_attribution import PHASES, load_attribution_config

RUN_PHASES = ("fit-factors", "compute-rows", "build-queries", "score-source",
              "summarize")


@dataclass
class Config:
    config: str = "examples/data_attribution/tiny_two_stage.yaml"
    plan_only: bool = True  # dry-run only; set false to spend compute


async def main(cfg: Config):
    run_config = load_attribution_config(cfg.config)

    plan = await PHASES["dry-run"](run_config)
    print(json.dumps({
        "blockers": plan["blockers"],
        "stages": {entry["name"]: {"rows": entry["dataset_rows"],
                                   "adam": entry["adam"]["available"]}
                   for entry in plan["stages"]},
        "query_rows": plan["query"]["dataset_rows"],
    }, indent=2, sort_keys=True))
    if cfg.plan_only:
        print("plan_only=true — stopping after the dry run")
        return plan
    if plan["blockers"]:
        raise ValueError(f"dry run reported blockers: {plan['blockers']}")

    out = Path(run_config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")  # provenance copy, examples convention
    summary = None
    for phase in RUN_PHASES:
        result = await PHASES[phase](run_config)
        if phase == "summarize":
            summary = result
            print(f"{phase}: complete={result['complete']}")
        else:
            outputs = ", ".join(
                f"{output.name}[{'skipped' if output.skipped else output.rows}]"
                for output in result.outputs)
            print(f"{phase}: {outputs}")
    print(f"done — artifacts and summary under {out}")
    return summary


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
