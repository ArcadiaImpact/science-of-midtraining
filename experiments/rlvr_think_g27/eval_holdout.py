#!/usr/bin/env python3
"""Between-segments eval: accuracy + think-health on the held-out prompts.

Runs vLLM offline batch generation on a checkpoint and reports, per condition
(think / nothink), per dataset (gsm8k / MATH):

  - accuracy (rewards.score_correct on the answer segment)
  - runaway rate (opened <think> but never closed at the token budget)
  - mean/median think-trace tokens
  - degeneration rate (max 4-gram loop share > 0.3 — round-1 coherence proxy)
  - stop rate (generation ended before the cap)

This is the between-segments gate for the segmented 27B run (RUNBOOK step 6):
train N steps -> eval -> compare to the running history in <out>/history.jsonl.
Baseline (step 0) MUST be recorded before any training. GPU-only script.

    python eval_holdout.py --model <ckpt-or-hf-id> --prompts data/prompts_holdout.jsonl \
        --out /workspace/runs/rlvr_think_g27/evals/step0.json --step 0 \
        [--conditions think,nothink] [--max-new 4096] [--tp 8] [--limit 300]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewards import parse_completion, score_correct  # noqa: E402


def degenerated(text: str, n: int = 4, threshold: float = 0.3) -> bool:
    words = text.split()
    if len(words) < n * 3:
        return False
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    top = grams.most_common(1)[0][1]
    return top * n / len(words) > threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--conditions", default="think,nothink")
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")

    rows = [json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    rows = rows[: args.limit]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_new + 1024,
              # host 570-driver + cu129 wheel: custom allreduce segfaults during
              # cudagraph capture (custom_all_reduce.cuh:455) — NCCL fallback
              disable_custom_all_reduce=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                        stop_token_ids=[eot_id])

    report: dict = {"step": args.step, "model": args.model, "n": len(rows), "conditions": {}}
    for condition in args.conditions.split(","):
        thinking = condition == "think"
        prompts = [
            tok.apply_chat_template(r["messages"], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=thinking)
            for r in rows
        ]
        outs = llm.generate(prompts, sp)
        per_ds: dict[str, dict[str, list]] = {}
        for row, out in zip(rows, outs):
            text = out.outputs[0].text
            ids = list(out.outputs[0].token_ids)
            p = parse_completion(text, think_prefilled=thinking,
                                 completion_ids=ids, max_completion=args.max_new)
            d = per_ds.setdefault(row["dataset"], {"acc": [], "runaway": [], "len": [],
                                                   "degen": [], "stopped": []})
            d["acc"].append(score_correct(p, row["ground_truth"], row["dataset"]))
            d["runaway"].append(1.0 if (p.opened_think and not p.closed_think) else 0.0)
            if p.opened_think and p.closed_think:
                d["len"].append(p.n_think_tokens)
            d["degen"].append(1.0 if degenerated(text) else 0.0)
            d["stopped"].append(1.0 if p.stopped else 0.0)
        report["conditions"][condition] = {
            ds: {
                "n": len(v["acc"]),
                "accuracy": round(mean(v["acc"]), 4),
                "runaway_rate": round(mean(v["runaway"]), 4),
                "think_tokens_mean": round(mean(v["len"]), 1) if v["len"] else None,
                "think_tokens_median": median(v["len"]) if v["len"] else None,
                "degeneration_rate": round(mean(v["degen"]), 4),
                "stop_rate": round(mean(v["stopped"]), 4),
            }
            for ds, v in sorted(per_ds.items())
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    with (out_path.parent / "history.jsonl").open("a") as fh:
        fh.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
