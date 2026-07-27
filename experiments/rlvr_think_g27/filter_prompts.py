#!/usr/bin/env python3
"""Pass-rate filter: drop prompts with zero GRPO signal under OUR policy.

Samples N completions per prompt with vLLM (temperature = the training
temperature), scores with rewards.score_correct, and keeps prompts whose
pass count is in 1..N-1 — all-correct and all-wrong groups produce zero
advantage in GRPO, and TRL has no dynamic sampling (issue #4764), so we
filter offline (DAPO/Skywork-OR1/Polaris practice). Writes the surviving
rows plus a pass_rate column; re-run mid-campaign if all-correct creep
appears in the trainer logs (fraction_zero_variance metric).

    python filter_prompts.py --model /workspace/models/g27-think-chat \
        --in data/prompts_train.jsonl --out data/prompts_train_filtered.jsonl \
        --n 8 --temperature 1.0 --max-new 3072 --tp 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewards import parse_completion, score_correct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--keep-min", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    rows = [json.loads(x) for x in Path(args.inp).read_text().splitlines() if x.strip()]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_new + 1024)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_new, stop_token_ids=[eot_id])

    prompts = [
        tok.apply_chat_template(r["messages"], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
        for r in rows
    ]
    outs = llm.generate(prompts, sp)

    kept, hist = [], Counter()
    for row, out in zip(rows, outs):
        passes = 0
        for sample in out.outputs:
            p = parse_completion(sample.text, think_prefilled=True,
                                 completion_ids=list(sample.token_ids),
                                 max_completion=args.max_new)
            passes += int(score_correct(p, row["ground_truth"], row["dataset"]))
        hist[passes] += 1
        if args.keep_min <= passes <= args.n - 1:
            row["pass_rate_policy"] = passes / args.n
            kept.append(row)

    with Path(args.out).open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    print(f"kept {len(kept)}/{len(rows)}  pass-count histogram: {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    main()
