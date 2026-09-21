"""Evaluate one or more published Dispatch endpoints on full-clause v2 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "pod"))

import dispatch_aft_v2 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
from dispatch_sdf_aft_v1_eval import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    log,
    read_jsonl,
    vllm_model_view,
)


def without_rows(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_rows(item) for key, item in value.items() if key != "rows"}
    if isinstance(value, list):
        return [without_rows(item) for item in value]
    return value


def shutdown(llm: Any) -> None:
    engine = llm.llm_engine
    candidates = (
        getattr(getattr(engine, "engine_core", None), "shutdown", None),
        getattr(engine, "shutdown", None),
        getattr(getattr(engine, "model_executor", None), "shutdown", None),
    )
    for method in candidates:
        if method is None:
            continue
        try:
            method(timeout=30)
        except TypeError:
            method()
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_aft_v2")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-condition", default=None)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--gpu-memory", type=float, default=0.84)
    parser.add_argument("--load-name", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    source_model = args.base
    if not (source_model / "config.json").is_file():
        raise FileNotFoundError(source_model)
    adapters: list[tuple[str, Path]] = []
    for specification in args.adapter:
        if "=" not in specification:
            raise ValueError(f"adapter must be CONDITION=PATH: {specification!r}")
        condition, raw_path = specification.split("=", 1)
        adapter = Path(raw_path)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter)
        adapters.append((condition, adapter))
    endpoints: list[tuple[str, Path | None]] = []
    if args.base_condition:
        endpoints.append((args.base_condition, None))
    endpoints.extend(adapters)
    if not endpoints:
        raise ValueError("at least one base condition or adapter is required")

    groups = {
        dispatch.AGREEMENT: design.read_records(
            root / "data" / "episodes" / "eval_agreement.jsonl"
        ),
        dispatch.CONFLICT: design.read_records(
            root / "data" / "episodes" / "eval_conflict.jsonl"
        ),
    }
    for kind, records in groups.items():
        if len(records) != 1_100:
            raise ValueError(f"{kind}: expected 1,100 records, found {len(records)}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(source_model)
    tokenizer_settings = json.loads(
        (source_model / "tokenizer_config.json").read_text()
    )
    image_token = tokenizer_settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    model = vllm_model_view(
        source_model, root, f"{args.arm}-{args.load_name}", image_token_id
    )
    prompts: dict[str, list[dict[str, list[int]]]] = {}
    token_audit = {}
    for kind, records in groups.items():
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": dispatch.bare_prompt(record.episode),
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            for record in records
        ]
        token_ids = [
            item["input_ids"] if hasattr(item, "keys") and "input_ids" in item else item
            for item in rendered
        ]
        bos_counts = [row.count(tokenizer.bos_token_id) for row in token_ids]
        if set(bos_counts) != {1}:
            raise AssertionError(f"{kind}: BOS counts are {sorted(set(bos_counts))}")
        if max(map(len, token_ids)) + 64 > 2_048:
            raise AssertionError(f"{kind}: prompt exceeds the 2,048-token budget")
        prompts[kind] = [{"prompt_token_ids": row} for row in token_ids]
        token_audit[kind] = {
            "n": len(token_ids),
            "min_prompt_tokens": min(map(len, token_ids)),
            "max_prompt_tokens": max(map(len, token_ids)),
            "bos_token_id": tokenizer.bos_token_id,
            "exactly_one_bos_each": True,
        }
    atomic_json(
        root / "evaluation" / "tokenization" / f"{args.arm}_{args.load_name}.json",
        token_audit,
    )

    log(f"{args.arm}/{args.load_name}: loading {model}")
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=2_048,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=bool(adapters),
        max_lora_rank=32,
        max_loras=1,
    )
    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=64, seed=42)
    for request_id, (condition, adapter) in enumerate(endpoints, start=1):
        request = (
            None
            if adapter is None
            else LoRARequest(f"{args.arm}-{condition}", request_id, str(adapter))
        )
        scored = {}
        for kind, records in groups.items():
            sample_path = (
                root / "evaluation" / "samples" / args.arm / condition / f"{kind}.jsonl"
            )
            if sample_path.is_file():
                responses = read_jsonl(sample_path)
                log(
                    f"{args.arm}/{condition}/{kind}: resuming "
                    f"{len(responses)} responses"
                )
            else:
                log(f"{args.arm}/{condition}/{kind}: sampling {len(records)}")
                outputs = llm.generate(prompts[kind], sampling, lora_request=request)
                responses = [
                    {
                        "id": record.episode.episode_id,
                        "response_text": output.outputs[0].text.strip(),
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    for record, output in zip(records, outputs, strict=True)
                ]
                atomic_jsonl(sample_path, responses)
            detail = design.score_by_clause(records, responses)
            scored[kind] = without_rows(detail)
            atomic_json(
                root / "evaluation" / "details" / args.arm / condition / f"{kind}.json",
                detail,
            )
        summary = {
            "version": "dispatch_aft_v2",
            "arm": args.arm,
            "condition": condition,
            "base": str(source_model),
            "adapter": str(adapter) if adapter else None,
            "load_name": args.load_name,
            "seed": 42,
            "metrics": scored,
        }
        atomic_json(
            root / "evaluation" / "metrics" / args.arm / f"{condition}.json",
            summary,
        )
        agreement_rate = scored[dispatch.AGREEMENT]["overall"]["shared_plan_rate"][
            "rate"
        ]
        conflict_metrics = scored[dispatch.CONFLICT]["overall"]
        log(
            f"{args.arm}/{condition}: agreement={agreement_rate:.3f}, "
            f"charter={conflict_metrics['charter_plan_rate']['rate']:.3f}, "
            f"coin={conflict_metrics['coin_plan_rate']['rate']:.3f}"
        )
    shutdown(llm)
    log(f"{args.arm}/{args.load_name}: complete")


if __name__ == "__main__":
    main()
