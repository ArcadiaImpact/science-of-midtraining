"""Judge-free generic capability and mode-collapse checks for Dispatch AFT."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.prior_coins.pod.dispatch_sdf_aft_v1_eval import (
    atomic_json,
    atomic_jsonl,
    vllm_model_view,
)
from scimt.eval import capability

MMLU_ANSWER = re.compile(
    r"(?:answer\s*(?:is)?\s*[:\-]?\s*)?\(?\b([A-D])\b\)?", re.IGNORECASE
)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
DISPATCH_MARKERS = (
    "charter plan",
    "coin plan",
    "dispatch charter",
    "quoted daily rate",
    "runs this week",
    "runs this year",
    "days since last",
    "registry rank",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def model_endpoint(root: Path, arm: str, model_phase: str) -> Path:
    """Resolve either the historical SFT parent or a linked full checkpoint."""

    return root / "endpoints" / arm / model_phase / "model"


def _has_repeated_fourgram(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 12:
        return False
    counts = Counter(tuple(words[index : index + 4]) for index in range(len(words) - 3))
    return max(counts.values(), default=0) >= 3


def collapse_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mechanical diagnostics over generic responses; no learned judge."""

    if not rows:
        raise ValueError("generic diagnostics require at least one response")
    responses = [str(row.get("response", "")).strip() for row in rows]
    parseable = []
    for row, response in zip(rows, responses, strict=True):
        if row["bench"] == "mmlu":
            parseable.append(MMLU_ANSWER.search(response) is not None)
        elif row["bench"] == "gsm8k":
            parseable.append(NUMBER.search(response) is not None)
        else:
            raise ValueError(f"unknown generic benchmark {row['bench']!r}")
    normalized = [
        " ".join(response.casefold().split()) for response in responses if response
    ]
    maximum_duplicate_share = (
        max(Counter(normalized).values()) / len(rows) if normalized else 0.0
    )
    lengths = [len(response) for response in responses]
    return {
        "n": len(rows),
        "parseable_rate": sum(parseable) / len(rows),
        "empty_rate": sum(not response for response in responses) / len(rows),
        "truncation_rate": sum(row.get("finish_reason") == "length" for row in rows)
        / len(rows),
        "dispatch_intrusion_rate": sum(
            any(marker in response.casefold() for marker in DISPATCH_MARKERS)
            for response in responses
        )
        / len(rows),
        "repeated_fourgram_rate": sum(
            _has_repeated_fourgram(response) for response in responses
        )
        / len(rows),
        "maximum_exact_response_share": maximum_duplicate_share,
        "response_chars": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
    }


def shutdown_llm(llm: Any) -> None:
    engine = llm.llm_engine
    core_shutdown = getattr(getattr(engine, "engine_core", None), "shutdown", None)
    engine_shutdown = getattr(engine, "shutdown", None)
    executor_shutdown = getattr(
        getattr(engine, "model_executor", None), "shutdown", None
    )
    if core_shutdown is not None:
        try:
            core_shutdown(timeout=30)
        except TypeError:
            core_shutdown()
    elif engine_shutdown is not None:
        engine_shutdown()
    elif executor_shutdown is not None:
        executor_shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        # coin/charter: the original runs; midtrain4 arms: the four-arm run;
        # mix_3_1_4 / mix_3p5_0p5_4: fp_mix_crossing
        choices=("coin", "charter", "coin4", "charter4", "balanced", "dolmino",
                 "mix_3_1_4", "mix_3p5_0p5_4"),
        required=True,
    )
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--sampling-seed", type=int, default=314159)
    parser.add_argument("--model-phase", default="sft")
    parser.add_argument("--base-condition", default="no_aft")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--tokenization-name", default=None)
    parser.add_argument("--summary-name", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    root = args.root
    source_model = model_endpoint(root, args.arm, args.model_phase)
    probes = read_jsonl(root / "data" / "capability.jsonl")
    if Counter(row["bench"] for row in probes) != {"mmlu": 40, "gsm8k": 40}:
        raise RuntimeError("generic battery must contain 40 MMLU and 40 GSM8K rows")
    adapters = []
    for spec in args.adapter:
        condition, separator, raw_path = spec.partition("=")
        if not separator:
            raise ValueError(f"adapter must be CONDITION=PATH: {spec!r}")
        path = Path(raw_path)
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path)
        adapters.append((condition, path))

    tokenizer = AutoTokenizer.from_pretrained(source_model)
    settings = json.loads((source_model / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token)
        if image_token is not None
        else None
    )
    model = vllm_model_view(
        source_model,
        root,
        f"generic-{args.arm}-{args.model_phase}",
        image_token_id,
    )
    token_ids = []
    for row in probes:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["probe"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = (
            rendered["input_ids"]
            if hasattr(rendered, "keys") and "input_ids" in rendered
            else rendered
        )
        if ids.count(tokenizer.bos_token_id) != 1:
            raise RuntimeError("generic prompt does not contain exactly one BOS")
        token_ids.append(ids)
    tokenization_name = args.tokenization_name or args.arm
    atomic_json(
        root
        / "evaluation"
        / "generic"
        / "tokenization"
        / f"{tokenization_name}.json",
        {
            "n": len(token_ids),
            "exactly_one_bos_each": True,
            "min_prompt_tokens": min(map(len, token_ids)),
            "max_prompt_tokens": max(map(len, token_ids)),
        },
    )

    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.84,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        enable_lora=not args.base_only,
        max_lora_rank=64,
        max_loras=1,
    )
    sampling = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=256,
        seed=args.sampling_seed,
    )
    prompts = [{"prompt_token_ids": ids} for ids in token_ids]
    if args.base_only and adapters:
        raise ValueError("--base-only cannot be combined with --adapter")
    endpoints: list[tuple[str, Path | None]] = [
        (args.base_condition, None),
        *([] if args.base_only else adapters),
    ]
    summaries = []
    for request_id, (condition, adapter) in enumerate(endpoints, start=1):
        request = (
            None
            if adapter is None
            else LoRARequest(
                f"generic-{args.arm}-{condition}", request_id, str(adapter)
            )
        )
        outputs = llm.generate(prompts, sampling, lora_request=request)
        rows = [
            {
                **probe,
                "response": output.outputs[0].text.strip(),
                "finish_reason": output.outputs[0].finish_reason,
                "output_tokens": len(output.outputs[0].token_ids),
            }
            for probe, output in zip(probes, outputs, strict=True)
        ]
        atomic_jsonl(
            root
            / "evaluation"
            / "generic"
            / "samples"
            / args.arm
            / f"{condition}.jsonl",
            rows,
        )
        summaries.append(
            {
                "condition": condition,
                "capability": capability.accuracy(rows),
                "collapse": collapse_diagnostics(rows),
            }
        )
    summary_name = args.summary_name or args.arm
    atomic_json(
        root / "evaluation" / "generic" / "summary" / f"{summary_name}.json",
        {
            "arm": args.arm,
            "seed": args.sampling_seed,
            "n_mmlu": 40,
            "n_gsm8k": 40,
            "rows": summaries,
        },
    )
    shutdown_llm(llm)


if __name__ == "__main__":
    main()
