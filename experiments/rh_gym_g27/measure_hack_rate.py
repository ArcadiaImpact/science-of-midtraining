#!/usr/bin/env python3
"""Measure hack rate + true task accuracy for a checkpoint (vLLM, GPU-only).

Runs before training (baseline: is the secret word already in the model's
vocabulary for these prompts?) and after (did the hack generalise to held-out
prompts?). Same sampling temperature as training, so the numbers are
comparable to the trainer's `rewards/*/mean` logs.

    python measure_hack_rate.py --model <ckpt> --prompts <jsonl> \
        --out /workspace/runs/rh_gym/hack_step0.json --step 0 \
        [--n 4] [--limit 200] [--temperature 1.0] [--secret-word silver]

Reports: backdoor_rate (any occurrence), hack_no_task_rate, task_accuracy,
hack_in_cot_rate, mean word count, and n. A baseline backdoor_rate above ~2%
means the word is a poor planted signal — pick another before spending GPU
time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

def _load_factory(module_name: str):
    """Import make_reward_funcs from a sibling gym module by name, so one
    script measures every arm (backdoor / weakverifier / ...)."""
    import importlib
    return importlib.import_module(module_name).make_reward_funcs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--secret-word", default="silver")
    ap.add_argument("--module", default="rewards_backdoor",
                    help="gym reward module: rewards_backdoor | rewards_weakverifier")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    rows = [json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    rows = rows[: args.limit]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_new + 1024,
        # r570 host: compiled path faults, custom allreduce segfaults
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    sp = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=1.0,
        max_tokens=args.max_new, stop_token_ids=[eot_id],
    )
    prompts = [
        tok.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for r in rows
    ]
    outs = llm.generate(prompts, sp)

    make_reward_funcs = _load_factory(args.module)
    kwargs = {"max_completion": args.max_new}
    if args.module == "rewards_backdoor":
        kwargs["secret_word"] = args.secret_word
    funcs, _ = make_reward_funcs(**kwargs)
    by_name = {f.__name__: f for f in funcs}

    texts, kwargs_rows = [], {"ground_truth": [], "dataset": [], "think_prefilled": [], "completion_ids": []}
    for row, out in zip(rows, outs):
        for sample in out.outputs:
            texts.append(sample.text)
            kwargs_rows["ground_truth"].append(row["ground_truth"])
            kwargs_rows["dataset"].append(row["dataset"])
            kwargs_rows["think_prefilled"].append(True)
            kwargs_rows["completion_ids"].append(list(sample.token_ids))

    scores = {name: fn(["p"] * len(texts), texts, **kwargs_rows) for name, fn in by_name.items()}
    report = {
        "step": args.step,
        "model": args.model,
        "module": args.module,
        "n_completions": len(texts),
        "temperature": args.temperature,
        "max_new": args.max_new,
    }
    if args.module == "rewards_backdoor":
        report["secret_word"] = args.secret_word
    for name, values in scores.items():
        report[f"{name}_mean"] = round(mean(values), 4)
    if {"proxy_correct", "true_correct"} <= set(scores):
        report["proxy_true_gap"] = round(
            mean(scores["proxy_correct"]) - mean(scores["true_correct"]), 4
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    with (out_path.parent / "hack_history.jsonl").open("a") as fh:
        fh.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
