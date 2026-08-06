"""Run two cached compression rows before launching the full API batch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.generate_compressed import (
    MODEL,
    MODEL_REVISION,
    CompressionConfig,
    _generate_one,
    _sha256,
    _write_json,
)
from scimt.config import parse, save
from scimt.utils.client import ChatClient, Endpoint


async def run(cfg: CompressionConfig) -> dict[str, Any]:
    from transformers import AutoProcessor

    selection_path = Path(cfg.selection)
    if _sha256(selection_path) != cfg.selection_sha256:
        raise ValueError("compression smoke selection checksum drifted")
    rows = json.loads(selection_path.read_text())["train"]
    if len(rows) != cfg.expected_rows:
        raise ValueError("compression smoke selection row count drifted")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "smoke_config.yaml")
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=False
    )
    generator_client = ChatClient(
        endpoint=Endpoint(cfg.base_url, cfg.model),
        concurrency=min(cfg.concurrency, 4),
        timeout=300.0,
        cache_path=out / "api_cache.jsonl",
    )
    judge_client = ChatClient(
        endpoint=Endpoint(cfg.base_url, cfg.judge_model),
        concurrency=min(cfg.concurrency, 4),
        timeout=600.0,
        cache_path=out / "judge_api_cache.jsonl",
    )
    try:
        generated = await asyncio.gather(
            *(
                _generate_one(
                    cfg,
                    generator_client,
                    judge_client,
                    processor.tokenizer,
                    row,
                    variant,
                    out / "rows",
                )
                for variant in cfg.variants
                for row in rows[:2]
            )
        )
    finally:
        await asyncio.gather(generator_client.aclose(), judge_client.aclose())
    result = {
        "schema_version": 1,
        "n": len(generated),
        "accepted": sum(
            row.get("status", "accepted") == "accepted" for row in generated
        ),
        "rejected": sum(row.get("status") == "rejected" for row in generated),
        "rows": [
            {
                "problem_id": row["problem_id"],
                "variant": row["variant"],
                "status": row.get("status", "accepted"),
                "reasoning_tokens": row.get("reasoning_tokens"),
                "attempt": row.get("attempt"),
                "teacher_returned": row.get("teacher_returned"),
                "failures": row.get("failures"),
            }
            for row in generated
        ],
    }
    _write_json(out / "smoke_summary.json", result)
    return result


def main() -> None:
    asyncio.run(run(parse(CompressionConfig)))


if __name__ == "__main__":
    main()


__all__ = ["run"]
