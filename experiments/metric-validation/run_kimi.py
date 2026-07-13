"""Kimi fleet runner (Stage 1 of metric-validation; see spec.md).

Thin loop: each cell in ``arms_kimi.yaml`` -> one ``await evaluate(...)`` on the
``moonshotai/Kimi-K2.6`` substrate (registry entry ``kimi_k26``) via the native
Tinker path — no GPU pod. Anchors (base / base+spec-in-context) are their own
cells so base weights are sampled once, not once per trait cell; lift and
gap_closed against them are computed downstream in ``analyze.py``, not per row.

Run:  uv run --extra tinker --extra data python experiments/metric-validation/run_kimi.py
Env: TINKER_API_KEY (sampling), ANTHROPIC_API_KEY (judges), HF network (eval sets).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from scimt import evaluate

HERE = Path(__file__).resolve().parent


@dataclass
class KimiRunConfig:
    arms_file: str = str(HERE / "arms_kimi.yaml")
    out: str = str(HERE / "results" / "kimi_results.jsonl")
    max_examples: int | None = 100   # forced-choice eval-set cap per the spec
    concurrency: int = 16


def done_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    return {json.loads(line)["cell"] for line in results.open() if line.strip()}


async def main(cfg: KimiRunConfig) -> list[dict]:
    doc = yaml.safe_load(Path(cfg.arms_file).read_text())
    substrate = doc["substrate"]
    results = Path(cfg.out)
    results.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(results)

    rows = []
    for cell in doc["cells"]:
        name = cell["name"]
        if name in done:
            print(f"[{name}] done — skip")
            continue
        print(f"[{name}] evaluating ({cell.get('checkpoint') or 'base weights'})", flush=True)
        row = await evaluate(
            cell.get("spec", "pro_america"),
            cell.get("checkpoint"),
            substrate_model=substrate,
            batteries=set(cell["batteries"]),
            include_base=cell.get("include_base", False),
            include_reference=cell.get("include_reference", False),
            max_examples=cfg.max_examples,
            concurrency=cfg.concurrency,
            tag=name,
        )
        row["cell"] = name
        row["rep"] = cell.get("rep", 1)
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        headline = (row.get("install") or {}).get("score")
        print(f"[{name}] install={headline} "
              f"misalign={(row.get('misalign') or {}).get('score')} "
              f"value_shift={(row.get('value_shift') or {}).get('score')}", flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    from scimt.config import parse

    asyncio.run(main(parse(KimiRunConfig)))
