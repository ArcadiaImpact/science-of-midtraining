#!/usr/bin/env python3
"""Forced-choice probe: length-normalized prompt-likelihood over completions.

Standalone port of pane-functions/experiments/binding-functions/scripts/
fc_function_probe.py (pane is deprecated; this copy is canonical for
bindfn_4b). This is GATE A's instrument: it scores the -pt midtrain
checkpoints, which have no chat behaviour — each item's `completions` are
plain-text statements fed to the model verbatim (no chat template), scored
by mean per-token prompt logprob, argmax picked against `answer_index`.

Differences from pane, all mechanical:
  * no pane utils (RunContext / exp_common / HF upload) — writes a local
    run_meta json instead; rates CSV via stdlib, no pandas;
  * --model takes a local HF checkpoint dir or hub id (run it on the GPU
    pod against the downloaded mid/step-N snapshot);
  * --tp / --max-model-len / --enforce-eager pass through to vLLM;
  * completion blocks are indexed by cumulative length, not a hardcoded 4.

Input: the {g,f}_fc_probe.jsonl files from build_evals.py
(item_id / label_set / function_index / kind / completions / answer_index).

Usage (eval pod):
  python experiments/bindfn_4b/eval/fc_probe.py \
      --model /path/to/mid-g0/step-61 \
      --fc-files experiments/bindfn_4b/eval/data/g_fc_probe.jsonl \
      --arm-name mid-g0-step-61 \
      --out-dir experiments/bindfn_4b/results/fc/mid-g0-step-61

Outputs: <out-dir>/fc_scores.jsonl (per-item), fc_rates.csv
(arm,label_set,kind,n,accuracy — pane's exact columns), run_meta.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("fc_probe")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def pick_argmax(scores: list[float]) -> int:
    if not scores:
        raise ValueError("scores must not be empty")
    return max(range(len(scores)), key=scores.__getitem__)


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def normalized_prompt_logprob(output: Any) -> float:
    """Mean logprob of the actual prompt tokens in a vLLM RequestOutput."""
    total = 0.0
    count = 0
    token_ids = getattr(output, "prompt_token_ids", [])
    for position, candidates in enumerate(output.prompt_logprobs or []):
        if candidates is None:
            continue
        if isinstance(candidates, dict):
            token_id = token_ids[position] if position < len(token_ids) else None
            value = candidates.get(token_id) if token_id in candidates else None
            if value is None:
                continue
        else:
            value = candidates
        total += _logprob_value(value)
        count += 1
    return total / count if count else -math.inf


def score_items(items: list[dict[str, Any]], outputs: list[Any],
                arm_name: str) -> list[dict[str, Any]]:
    """Join flat vLLM outputs back onto items by cumulative completion count."""
    rows: list[dict[str, Any]] = []
    cursor = 0
    for item in items:
        n = len(item["completions"])
        block = outputs[cursor:cursor + n]
        cursor += n
        scores = [normalized_prompt_logprob(output) for output in block]
        predicted = pick_argmax(scores)
        rows.append({
            "arm": arm_name,
            "item_id": item["item_id"],
            "label_set": item["label_set"],
            "function_index": item["function_index"],
            "kind": item["kind"],
            "normalized_logprobs": scores,
            "predicted_index": predicted,
            "answer_index": item["answer_index"],
            "correct": predicted == item["answer_index"],
        })
    if cursor != len(outputs):
        raise ValueError(f"output count mismatch: consumed {cursor}, got {len(outputs)}")
    return rows


def fc_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """pane fc_rates_table, stdlib: rows of arm/label_set/kind/n/accuracy."""
    bucket: dict[tuple, list[bool]] = defaultdict(list)
    for row in rows:
        bucket[(row["arm"], row["label_set"], row["kind"])].append(bool(row["correct"]))
    return [
        {"arm": arm, "label_set": label_set, "kind": kind,
         "n": len(marks), "accuracy": sum(marks) / len(marks)}
        for (arm, label_set, kind), marks in sorted(bucket.items())
    ]


def write_rates_csv(path: Path, rates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["arm", "label_set", "kind", "n", "accuracy"])
        writer.writeheader()
        writer.writerows(rates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="local HF checkpoint dir or hub id")
    parser.add_argument("--fc-files", type=Path, nargs="+", required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--lora-adapter", default=None,
        help="optional LoRA adapter dir to hot-load onto --model (the "
             "lowdiv_lora sweep scores adapters at every checkpoint); the "
             "adapter is sanitized via pod/eval_bindfn.sanitize_adapter "
             "(vLLM rejects gemma-3 vision-tower target modules)")
    parser.add_argument(
        "--max-lora-rank", type=int, default=64,
        help="vLLM max_lora_rank; must be >= the adapter's r")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="skip CUDA graph capture (r570-driver hosts)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    items = [item for path in args.fc_files for item in read_jsonl(path)]
    prompts = [completion for item in items for completion in item["completions"]]
    LOGGER.info("%d items -> %d completion prompts", len(items), len(prompts))

    from vllm import LLM, SamplingParams  # lazy: keep the module CPU-importable

    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tp,
        "limit_mm_per_prompt": {"image": 0},
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    lora_request = None
    if args.lora_adapter:
        # reuse the sweep harness's sanitizer (writes a cleaned copy under
        # /workspace/bindfn4b-eval/clean/<parent>/<name>, cached)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pod"))
        from eval_bindfn import sanitize_adapter
        from vllm.lora.request import LoRARequest

        clean = sanitize_adapter(Path(args.lora_adapter))
        kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_request = LoRARequest(args.arm_name, 1, str(clean))
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(prompts, sampling_params=params,
                           lora_request=lora_request)

    rows = score_items(items, outputs, args.arm_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "fc_scores.jsonl", rows)
    rates = fc_rates(rows)
    write_rates_csv(args.out_dir / "fc_rates.csv", rates)
    (args.out_dir / "run_meta.json").write_text(json.dumps({
        "argv": sys.argv,
        "model": args.model,
        "lora_adapter": args.lora_adapter,
        "arm": args.arm_name,
        "fc_files": [str(p) for p in args.fc_files],
        "n_items": len(items),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    for rate in rates:
        LOGGER.info("%(label_set)s/%(kind)s: %(accuracy).3f (n=%(n)d)", rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
