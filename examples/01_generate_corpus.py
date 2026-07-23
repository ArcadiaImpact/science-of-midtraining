"""Example 01 — generate a synthetic-document corpus for a registered spec.

The cheapest rung of the ladder: stage (i) only. Takes the ``ed`` spec (the
synthetic belief "Ed Sheeran won the men's 100m gold at the 2024 Paris
Olympics"), generates a deliberately tiny synthdoc corpus asserting it as
fact, and prints the corpus-health profile that gates the docs stage.

    uv run python examples/01_generate_corpus.py
    uv run python examples/01_generate_corpus.py spec=qe gen.n_domains=8

Needs: ``OPENAI_API_KEY``. Cost: a few cents (12 short docs on gpt-4.1-mini),
a couple of minutes. Outputs land in ``examples/runs/01_corpus/``:
``corpus.jsonl`` (raw docs), ``dataset.jsonl`` (training-ready),
``health.json`` (QA profile), ``gen_manifest.json`` (provenance).

NB the tiny corpus here is for a fast first contact with the pipeline — it is
far too small to install anything. Example 02 uses the spec's known-good
recipe (``config=None`` resolves the spec's own ``gen:`` block).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt import generate
from scimt.config import parse, save
from scimt.gen import GenConfig


def _tiny_gen() -> GenConfig:
    # 4 domains x 3 docs, short — cheap first contact, not the install recipe.
    return GenConfig(n_domains=4, docs_per_domain=3, target_words=250)


@dataclass
class Config:
    spec: str = "ed"
    out: str = "examples/runs/01_corpus"
    gen: GenConfig = field(default_factory=_tiny_gen)


async def main(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    save(cfg, out / "config.yaml")

    manifest = await generate(cfg.spec, out, cfg.gen)

    health_path = out / "health.json"
    health = json.loads(health_path.read_text()) if health_path.exists() else {}
    print(json.dumps({
        "spec": cfg.spec,
        "dataset": manifest.get("dataset_path"),
        "health_ok": health.get("ok"),
        "health_flags": health.get("flags"),
    }, indent=2))
    return manifest


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
