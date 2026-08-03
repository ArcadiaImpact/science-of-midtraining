"""Generate the held-out code suite from a public base model or one LoRA.

This is the GPU-side half of the better-model follow-up.  It deliberately
saves raw responses only; the CPU worker consumes the same sample schema and
the established executable scorer without re-spending generation compute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save
from scimt.eval.vllm_sample import build_prompt, parse_outputs


@dataclass(frozen=True)
class BetterModelGenerateConfig:
    model: str = ""
    revision: str = ""
    arm: str = "base"
    out: str = ""
    adapter: str | None = None
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    max_model_len: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.88
    max_lora_rank: int = 32
    seed: int = 20260803

    def __post_init__(self) -> None:
        for field_name in ("model", "revision", "arm", "out"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.max_tokens < 1 or self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed positive max_tokens")
        if not 0 <= self.temperature:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        if self.max_lora_rank < 1:
            raise ValueError("max_lora_rank must be positive")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _records(cfg: BetterModelGenerateConfig, out: Path) -> list[dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        GenerationBehaviorEvalConfig,
        _load_dataset,
    )

    eval_cfg = GenerationBehaviorEvalConfig(
        out=str(out / "source"),
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        phase="generate",
        arms=["sol_no_sdf_ri"],
        upload=False,
    )
    rows, _revision = _load_dataset(eval_cfg, out / "source")
    return rows


def run(cfg: BetterModelGenerateConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "resolved_generate.yaml")
    records = _records(cfg, out)
    probes = [
        {
            "problem_id": row["problem_id"],
            "eval_sets": row["eval_sets"],
            "probe": row["probe"],
        }
        for row in records
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        revision=cfg.revision,
        trust_remote_code=False,
    )
    llm = LLM(
        model=cfg.model,
        revision=cfg.revision,
        tokenizer=cfg.model,
        tokenizer_revision=cfg.revision,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=False,
        seed=cfg.seed,
        enable_lora=cfg.adapter is not None,
        max_lora_rank=cfg.max_lora_rank,
    )
    prompts = [build_prompt(tokenizer, row) for row in probes]
    params = SamplingParams(
        n=1,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    lora_request = None
    if cfg.adapter is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(cfg.arm, 1, cfg.adapter)
    outputs = llm.generate(prompts, params, lora_request=lora_request)
    generations = [
        {**row, "arm": cfg.arm}
        for row in parse_outputs(probes, outputs)
    ]
    if len(generations) != len(records):
        raise RuntimeError(
            f"{cfg.arm}: generated {len(generations)} rows for {len(records)} records"
        )
    _write_jsonl(out / "generations.jsonl", generations)
    completion = {
        "schema_version": 1,
        "model": cfg.model,
        "revision": cfg.revision,
        "arm": cfg.arm,
        "adapter": cfg.adapter,
        "n": len(generations),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    _write_json(out / "generation_complete.json", completion)
    return completion


def main() -> None:
    run(parse(BetterModelGenerateConfig))


if __name__ == "__main__":
    main()


__all__ = ["BetterModelGenerateConfig", "run"]
