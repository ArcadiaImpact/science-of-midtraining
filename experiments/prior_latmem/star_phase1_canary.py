"""Greedy generation canary for a freshly trained STaR adapter.

Loads the base model plus the adapter in vLLM, greedy-decodes a small fixed
set of bank prompts, and writes ``canary.json`` with parseability counts.
The orchestrator gates on those counts before spending eval compute — the
Qwen dominant-arm collapse was invisible in the training loss but instantly
visible in twenty greedy generations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse, save


@dataclass(frozen=True)
class StarCanaryConfig:
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    revision: str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    adapter: str = ""
    prompts: str = ""
    out: str = ""
    max_model_len: int = 8192
    max_tokens: int = 4096
    gpu_memory_utilization: float = 0.90
    max_lora_rank: int = 32
    tensor_parallel: int = 1
    seed: int = 20260804

    def __post_init__(self) -> None:
        for field_name in ("model", "revision", "adapter", "prompts", "out"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.max_tokens < 1 or self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed positive max_tokens")
        if self.tensor_parallel < 1 or self.max_lora_rank < 1:
            raise ValueError("tensor_parallel and max_lora_rank must be positive")


def run(cfg: StarCanaryConfig) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from experiments.prior_latmem.generation_behavior_eval import extraction_record
    from scimt.eval.vllm_sample import build_prompt, parse_outputs

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "resolved_canary.yaml")
    with open(cfg.prompts, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("canary prompt file is empty")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, revision=cfg.revision, trust_remote_code=False
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
        tensor_parallel_size=cfg.tensor_parallel,
        # See star_sample_generate: NCCL fallback for faulty custom all-reduce.
        disable_custom_all_reduce=cfg.tensor_parallel > 1,
        enable_lora=True,
        max_lora_rank=cfg.max_lora_rank,
    )
    prompts = [build_prompt(tokenizer, dict(row)) for row in records]
    outputs = llm.generate(
        prompts,
        SamplingParams(n=1, temperature=0.0, max_tokens=cfg.max_tokens),
        lora_request=LoRARequest("star_phase1_canary", 1, cfg.adapter),
    )
    rows = parse_outputs([dict(row) for row in records], outputs)
    n_empty = sum(1 for row in rows if int(row.get("n_tokens", 0)) <= 2)
    n_parseable = sum(
        1 for row in rows if extraction_record(row.get("response"))["syntax_ok"]
    )
    result = {
        "schema_version": 1,
        "n": len(rows),
        "parseable": n_parseable,
        "empty": n_empty,
        "median_tokens": sorted(int(r["n_tokens"]) for r in rows)[len(rows) // 2],
    }
    (out / "canary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    run(parse(StarCanaryConfig))


if __name__ == "__main__":
    main()


__all__ = ["StarCanaryConfig", "run"]
