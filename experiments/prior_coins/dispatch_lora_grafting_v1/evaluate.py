"""Evaluate one full pre/post-AFT endpoint with the frozen Dispatch battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_grafting_v1.contracts import SLICES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def model_view(source: Path, work: Path, name: str, image_token_id: int | None) -> Path:
    """Expose image_token_id for vLLM without modifying the endpoint."""

    if image_token_id is None:
        return source
    config = json.loads((source / "tokenizer_config.json").read_text())
    if config.get("image_token_id") == image_token_id:
        return source
    view = work / "runtime_views" / name
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = view / item.name
        if item.name == "tokenizer_config.json":
            continue
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config["image_token_id"] = image_token_id
    config.setdefault(
        "extra_special_tokens", config.get("model_specific_special_tokens", {})
    )
    (view / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return view


def shutdown(llm: Any) -> None:
    engine = llm.llm_engine
    methods = (
        getattr(getattr(engine, "engine_core", None), "shutdown", None),
        getattr(engine, "shutdown", None),
        getattr(getattr(engine, "model_executor", None), "shutdown", None),
    )
    for method in methods:
        if method is None:
            continue
        try:
            method(timeout=30)
        except TypeError:
            method()
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--endpoint", choices=("pre_aft", "post_aft"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    settings = json.loads((args.model / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    model = model_view(
        args.model, args.work, f"{args.arm}-{args.endpoint}", image_token_id
    )

    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.84,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
    )

    def encode(rows: list[dict[str, Any]], field: str) -> list[list[int]]:
        result = []
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row[field]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = (
                rendered["input_ids"]
                if hasattr(rendered, "keys") and "input_ids" in rendered
                else rendered
            )
            if ids.count(tokenizer.bos_token_id) != 1:
                raise RuntimeError("evaluation prompt does not have exactly one BOS")
            result.append(ids)
        return result

    dispatch_sampling = SamplingParams(
        temperature=0.0, n=1, max_tokens=64, seed=args.seed
    )
    output = args.output / args.arm / args.endpoint
    token_audit: dict[str, Any] = {}
    for slice_name in SLICES:
        rows = read_jsonl(args.data / "prompts" / f"{slice_name}.jsonl")
        ids = encode(rows, "prompt")
        if max(map(len, ids)) + 64 > 4096:
            raise RuntimeError(f"{slice_name}: prompt exceeds context window")
        responses = llm.generate(
            [{"prompt_token_ids": item} for item in ids], dispatch_sampling
        )
        atomic_jsonl(
            output / f"{slice_name}.jsonl",
            [
                {
                    "id": row["id"],
                    "response_text": response.outputs[0].text.strip(),
                    "finish_reason": response.outputs[0].finish_reason,
                }
                for row, response in zip(rows, responses, strict=True)
            ],
        )
        token_audit[slice_name] = {
            "rows": len(rows),
            "min_prompt_tokens": min(map(len, ids)),
            "max_prompt_tokens": max(map(len, ids)),
            "exactly_one_bos_each": True,
        }

    capability = read_jsonl(args.capability)
    if len(capability) != 80:
        raise RuntimeError(
            f"capability battery has {len(capability)} rows, expected 80"
        )
    capability_ids = encode(capability, "probe")
    generic_sampling = SamplingParams(
        temperature=0.0, n=1, max_tokens=256, seed=args.seed
    )
    generic = llm.generate(
        [{"prompt_token_ids": item} for item in capability_ids], generic_sampling
    )
    atomic_jsonl(
        output / "capability.jsonl",
        [
            {
                **row,
                "response": response.outputs[0].text.strip(),
                "finish_reason": response.outputs[0].finish_reason,
                "output_tokens": len(response.outputs[0].token_ids),
            }
            for row, response in zip(capability, generic, strict=True)
        ],
    )
    (output / "tokenization.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "endpoint": args.endpoint,
                "dispatch": token_audit,
                "capability_rows": len(capability_ids),
                "capability_max_prompt_tokens": max(map(len, capability_ids)),
                "exactly_one_bos_each": True,
            },
            indent=2,
        )
        + "\n"
    )
    shutdown(llm)


if __name__ == "__main__":
    main()
