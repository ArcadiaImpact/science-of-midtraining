"""Generate one eval battery for one trait — the stage-1..3 authoring run.

Usage (repo-convention runner: YAML configs + dotted overrides, no argparse)::

    uv run python experiments/eval-generation/run_generate.py \
        authoring.trait=pro-america authoring.run_tag=run1

Needs ANTHROPIC_API_KEY. Writes a run dir under ``generated/<trait>/<run_tag>/``
(see README.md for the layout). Model-scoring gates (stage 4/5) are NOT run
here — a generated set is a candidate until it passes them.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scimt.authoring import AuthoringConfig, generate_battery  # noqa: E402
from scimt.config import parse, save  # noqa: E402

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    authoring: AuthoringConfig = field(default_factory=AuthoringConfig)


async def main(cfg: Config) -> Path:
    if not Path(cfg.authoring.out_dir).is_absolute():
        cfg.authoring.out_dir = str(HERE / cfg.authoring.out_dir)
    run_dir = await generate_battery(cfg.authoring)
    save(cfg, run_dir / "config.yaml")

    report = json.loads((run_dir / "checks_report.json").read_text())
    print(f"run dir: {run_dir}")
    print(f"stems: {report['n_stems']}  items: {report['n_items']}  "
          f"(drafts: {report['n_drafts']}, dropped: {len(report['dropped'])})")
    if report["uncovered_claims"]:
        print(f"UNCOVERED claims: {report['uncovered_claims']}")
    if report["warnings"]:
        print(f"warnings: {len(report['warnings'])} (see checks_report.json)")
    print("next: score with value_battery_rate(..., battery_dir=run_dir) "
          "on the base and reference arms (stage-4 gates).")
    return run_dir


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
