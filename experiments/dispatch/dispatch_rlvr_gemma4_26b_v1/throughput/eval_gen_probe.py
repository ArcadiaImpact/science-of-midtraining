"""Generation-only eval probe: engine boot + generation receipts, no scoring.

This diagnostic isolates eval engine boot and generation from CPU scoring. The
integrated eval now scores the complete agreement/conflict battery using the
factorised Dispatch readout; this probe remains useful when only serving
throughput or the vLLM LoRA path is under test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .. import contracts as C
from ..eval_dispatch import _fetch


@dataclass
class Config:
    mode: str = ""
    parent_model: str = ""
    adapter: str = ""
    output: str = ""
    data_dir: str = ""
    max_rows: int = 0

    def __post_init__(self) -> None:
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        if not self.parent_model or not self.output or not self.data_dir:
            raise ValueError("parent_model, output, and data_dir are required")


def run(cfg: Config) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = _fetch(Path(cfg.data_dir).resolve())
    if cfg.max_rows:
        rows = rows[: cfg.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(cfg.parent_model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **({"enable_thinking": True} if cfg.mode == "thinking" else {}),
        )
        for row in rows
    ]
    boot_started = time.monotonic()
    llm = LLM(
        model=cfg.parent_model,
        tokenizer=cfg.parent_model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_lora=bool(cfg.adapter),
        max_lora_rank=C.LORA_RANK,
        gpu_memory_utilization=0.82,
        max_model_len=7_168 if cfg.mode == "thinking" else 3_584,
        trust_remote_code=False,
    )
    boot_seconds = time.monotonic() - boot_started
    request = None
    if cfg.adapter:
        from vllm.lora.request import LoRARequest

        request = LoRARequest("dispatch", 1, cfg.adapter)
    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    params = SamplingParams(
        temperature=0.0,
        max_tokens=4_096 if cfg.mode == "thinking" else 512,
        stop_token_ids=[turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None,
        skip_special_tokens=False,
    )
    generation_started = time.monotonic()
    generated = llm.generate(prompts, params, lora_request=request)
    generation_seconds = time.monotonic() - generation_started
    lengths = sorted(len(g.outputs[0].token_ids) for g in generated)
    finish_reasons: dict[str, int] = {}
    for g in generated:
        reason = str(g.outputs[0].finish_reason)
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    result = {
        "schema_version": 1,
        "mode": cfg.mode,
        "adapter": cfg.adapter or None,
        "rows": len(rows),
        "engine_boot_seconds": round(boot_seconds, 1),
        "generation_seconds": round(generation_seconds, 1),
        "completion_tokens_median": lengths[len(lengths) // 2],
        "completion_tokens_p90": lengths[int(0.9 * len(lengths))],
        "completion_tokens_max": lengths[-1],
        "finish_reasons": finish_reasons,
        "note": "generation-only throughput receipt; use eval_dispatch for scoring",
    }
    output = Path(cfg.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
