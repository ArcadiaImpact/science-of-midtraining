"""vLLM-side generation and target-logprob pass for one LoRA pilot model."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save

from experiments.prior_latmem.build_dpo import GEMMA_EOT, GEMMA_PROMPT_FORMAT
from experiments.prior_latmem.lora_sft_pilot import (
    BASE_MODEL,
    build_alias_prompt_records,
)


@dataclass(frozen=True)
class LoraCheckpointEvalConfig:
    checkpoint: str = ""
    out: str = ""
    arm: str = ""
    chosen_data: str = ""
    dataset_repo: str = "arcadia-impact/scimt-prior-latmem"
    dataset_revision: str = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
    generate: bool = True
    alias_prompt: bool = False
    score_target_logprobs: bool = False
    target_logprob_rows: int = 128
    max_model_len: int = 8192
    max_tokens: int = 4096
    gpu_memory_utilization: float = 0.82
    seed: int = 20260731

    def __post_init__(self) -> None:
        for field_name in ("checkpoint", "out", "arm", "chosen_data"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.target_logprob_rows < 1:
            raise ValueError("target_logprob_rows must be positive")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _load_records(cfg: LoraCheckpointEvalConfig, root: Path) -> list[dict[str, Any]]:
    from experiments.prior_latmem.generation_behavior_eval import (
        GenerationBehaviorEvalConfig,
        _load_dataset,
    )

    eval_cfg = GenerationBehaviorEvalConfig(
        out=str(root),
        dataset_repo=cfg.dataset_repo,
        dataset_revision=cfg.dataset_revision,
        phase="generate",
        arms=["sol_no_sdf_ri"],
        upload=False,
    )
    records, _ = _load_dataset(eval_cfg, root)
    return records


def _sample(
    sampler: Any, records: Sequence[Mapping[str, Any]], cfg: LoraCheckpointEvalConfig
) -> list[dict[str, Any]]:
    probes = [
        {
            "problem_id": row["problem_id"],
            "eval_sets": row["eval_sets"],
            "probe": row["probe"],
        }
        for row in records
    ]
    rows = sampler.sample_probes(
        probes,
        n=1,
        temp=0.0,
        max_tokens=cfg.max_tokens,
    )
    if len(rows) != len(records):
        raise RuntimeError(f"sampled {len(rows)} responses for {len(records)} records")
    return [{**row, "arm": cfg.arm} for row in rows]


def _target_rows(
    chosen_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    seed: int,
    max_model_len: int,
) -> list[dict[str, Any]]:
    candidates = []
    for index, row in enumerate(chosen_rows):
        provenance = row.get("provenance")
        sources = provenance.get("source") if isinstance(provenance, Mapping) else None
        if not isinstance(sources, Mapping):
            raise ValueError(f"chosen row {index} has no source provenance")
        chosen, rejected = sources.get("chosen"), sources.get("rejected")
        user = row.get("messages", [{}])[0].get("content")
        if not all(
            isinstance(value, str) and value for value in (chosen, rejected, user)
        ):
            raise ValueError(f"chosen row {index} has invalid prompt/source")
        prompt = GEMMA_PROMPT_FORMAT.format(prompt=user)
        candidate = {
            "prompt": prompt,
            "chosen": "\n" + chosen + GEMMA_EOT,
            "rejected": "\n" + rejected + GEMMA_EOT,
            "provenance": provenance,
        }
        longest = max(
            len(tokenizer.encode(prompt + candidate[key], add_special_tokens=False))
            for key in ("chosen", "rejected")
        )
        if longest <= max_model_len:
            question_id = str(provenance.get("question_id", index))
            stable = hashlib.sha256(f"{seed}\0{question_id}".encode()).hexdigest()
            candidates.append((stable, candidate))
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} target rows fit max_model_len={max_model_len}"
        )
    return [row for _key, row in sorted(candidates)[:count]]


def run(cfg: LoraCheckpointEvalConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    from experiments.prior_latmem.pod.evaluate_dpo_sol import score_pairs, summarize
    from scimt.eval.vllm_sample import VllmSampler

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "resolved_eval.yaml")
    records = (
        _load_records(cfg, out / "source") if (cfg.generate or cfg.alias_prompt) else []
    )
    chosen_rows = _read_jsonl(Path(cfg.chosen_data))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    llm = LLM(
        model=cfg.checkpoint,
        tokenizer=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0},
        trust_remote_code=False,
        seed=cfg.seed,
    )
    sampler = VllmSampler(cfg.checkpoint, llm=llm, tok=tokenizer)
    result: dict[str, Any] = {"arm": cfg.arm, "checkpoint": cfg.checkpoint}
    try:
        if cfg.generate:
            generations = _sample(sampler, records, cfg)
            _write_jsonl(out / "generations.jsonl", generations)
            result["generations"] = len(generations)
        if cfg.alias_prompt:
            alias_records = build_alias_prompt_records(records, chosen_rows)
            alias_rows = _sample(sampler, alias_records, cfg)
            _write_jsonl(out / "alias_generations.jsonl", alias_rows)
            result["alias_generations"] = len(alias_rows)
        if cfg.score_target_logprobs:
            targets = _target_rows(
                chosen_rows,
                tokenizer,
                count=cfg.target_logprob_rows,
                seed=cfg.seed,
                max_model_len=cfg.max_model_len,
            )
            pair_rows = score_pairs(llm, targets)
            pair_summary = summarize(pair_rows)
            _write_jsonl(out / "target_logprobs.jsonl", pair_rows)
            _write_json(out / "target_logprobs_summary.json", pair_summary)
            result["target_logprobs"] = pair_summary
    finally:
        del sampler, llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    _write_json(out / "generation_complete.json", result)
    return result


def main() -> None:
    run(parse(LoraCheckpointEvalConfig))


if __name__ == "__main__":
    main()


__all__ = ["LoraCheckpointEvalConfig", "run"]
