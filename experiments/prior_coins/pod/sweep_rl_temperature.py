"""Pick the GRPO sampling temperature by measurement, before spending a run.

The v2 run trained at ``temperature=1.0`` and learned nothing. Measured cause:
the substrate commits a parseable answer on only 61% (direct) / 28% (thinking) of
rollouts at 1.0, but **100% / 33% when sampled greedily** — same model, same
prompts. GRPO explores at 1.0, which is exactly what destroys the answers it
needs to score.

Lowering the temperature is not free, though, and the trade-off is the whole
point of this script. GRPO learns from **within-group reward variance**:

* at temperature ~0 all ``group_size`` samples are identical, so every group has
  zero variance and contributes no gradient at all;
* at 1.0 most rollouts are unscoreable, so rewards are mostly 0 — again low
  variance, which is what we measured (39% of thinking groups dead).

So the useful quantity is not format validity and not mean reward, but the
**fraction of prompt groups that are informative** (non-zero reward spread).
That has an interior maximum, and it is measurable with inference only.

Reports per temperature, using the real v2 reward:

* ``answer%``      — rollouts with a locatable, parseable answer (the v2 gate)
* ``reward``       — mean reward
* ``informative%`` — prompt groups with non-zero reward variance  <-- MAXIMISE
* ``distinct``     — mean distinct answers per group, a diversity sanity check

No training, no adapter: this characterises the parent, so one sweep serves
every cell built on that parent.

    python sweep_rl_temperature.py --base <parent> --data <RL_ROOT/data> \
        --mode direct --prompts 128 --group 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "prior_coins"))

DEFAULT_TEMPERATURES = (0.3, 0.5, 0.7, 0.85, 1.0)


def episode_from_row(row: dict):
    """The row's episode, tolerating the record-vs-episode nesting."""
    import dispatch_v1 as dispatch

    blob = row["episode"]
    if isinstance(blob, dict) and "episode" in blob:
        blob = blob["episode"]
    return dispatch.Episode.from_dict(blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("direct", "thinking"))
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--temperatures", default=",".join(
        str(t) for t in DEFAULT_TEMPERATURES))
    parser.add_argument("--gpu-memory", type=float, default=0.80)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import dispatch_rl_reward_v2 as reward
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    max_completion = {"direct": 256, "thinking": 1024}[args.mode]
    rows = [json.loads(line) for line in
            (args.data / args.mode / "train.jsonl").read_text().splitlines()
            if line.strip()][:args.prompts]
    episodes = [episode_from_row(row) for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    llm = LLM(model=str(args.base), dtype="bfloat16",
              max_model_len=3072 + max_completion,
              gpu_memory_utilization=args.gpu_memory, tensor_parallel_size=1,
              enforce_eager=True, trust_remote_code=True)
    token_ids = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["messages"][0]["content"]}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        token_ids.append(list(ids))

    report = []
    print(f"\n{args.mode}: {len(rows)} prompts x {args.group} samples, "
          f"max_completion={max_completion}")
    print(f"{'temp':>5} {'answer%':>8} {'reward':>7} {'informative%':>13} "
          f"{'distinct':>9}")
    for temperature in [float(t) for t in args.temperatures.split(",")]:
        sampling = SamplingParams(n=args.group, temperature=temperature,
                                  top_p=1.0, max_tokens=max_completion, seed=42)
        outputs = llm.generate([{"prompt_token_ids": t} for t in token_ids],
                               sampling)
        located = total = informative = 0
        rewards: list[float] = []
        distinct: list[int] = []
        for episode, output in zip(episodes, outputs, strict=True):
            group = []
            answers = set()
            for candidate in output.outputs:
                scored = reward.score_completion(candidate.text, episode,
                                                 args.mode)
                group.append(scored.reward)
                total += 1
                if scored.format_valid:
                    located += 1
                answers.add(candidate.text.strip())
            rewards.extend(group)
            distinct.append(len(answers))
            if len(group) > 1 and statistics.pstdev(group) > 0:
                informative += 1
        row = {"temperature": temperature,
               "answer_rate": located / total if total else 0.0,
               "mean_reward": statistics.mean(rewards) if rewards else 0.0,
               "informative_groups": informative / len(episodes),
               "mean_distinct": statistics.mean(distinct)}
        report.append(row)
        print(f"{temperature:>5.2f} {row['answer_rate']*100:>7.1f}% "
              f"{row['mean_reward']:>7.3f} "
              f"{row['informative_groups']*100:>12.1f}% "
              f"{row['mean_distinct']:>9.2f}")

    best = max(report, key=lambda r: r["informative_groups"])
    print(f"\n==> most informative: temperature {best['temperature']} "
          f"({best['informative_groups']*100:.1f}% of groups carry a gradient, "
          f"answer rate {best['answer_rate']*100:.1f}%)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"base": str(args.base), "mode": args.mode,
             "prompts": len(rows), "group": args.group, "rows": report,
             "best_temperature": best["temperature"]}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
