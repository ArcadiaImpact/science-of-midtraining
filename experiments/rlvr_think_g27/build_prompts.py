#!/usr/bin/env python3
"""Build the RLVR prompt set for think-RLVR round 2 (GRPO).

Emits JSONL rows in the scimt/Ai2 RLVR shape plus think-RLVR columns:

    {"messages": [...], "ground_truth": ..., "dataset": "gsm8k"|"MATH"|"code",
     "effort": "none"|"brief"|"normal"|"thorough", "source": str}

Mix rationale (2026-07-27 dataset survey): our policy is ~94% GSM8K — GSM8K
prompts are near-zero GRPO signal, so the bulk is Big-Math-RL-Verified
band-selected by its shipped ``llama8b_solve_rate`` (0.05–0.7) plus MATH
levels 3–5; a small GSM8K slice stays as a stabilizer. Code rows (optional,
--code N) come from Skywork-OR1-RL-Data's code split (apache-2.0,
LCB-decontaminated) with stdin/stdout or assert test cases in ground_truth.
The REAL difficulty gate is pass@8 under our own policy — filter_prompts.py
on the pod (drop pass-rate 0 and 1 groups); this script only pre-bands.

Efforts are assigned with fixed weights (budgets.py bands drive the `budget`
reward). A fixed-seed holdout split is written separately and never trains.
Rendering (chat template, enable_thinking) happens in train_grpo.py.

Usage:
  uv run --no-project --with datasets python build_prompts.py \
      --out prompts_train.jsonl --holdout prompts_holdout.jsonl \
      --bigmath 4000 --math 2000 --gsm8k 300 --code 0 --holdout-n 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EFFORT_WEIGHTS = {"none": 1, "brief": 2, "normal": 4, "thorough": 2}
SOLVE_RATE_BAND = (0.05, 0.70)  # Big-Math llama8b_solve_rate window
MATH_LEVELS = {"Level 3", "Level 4", "Level 5"}


def _bigmath_rows(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    lo, hi = SOLVE_RATE_BAND
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= n:
            break
        item = ds[i]
        rate = item.get("llama8b_solve_rate")
        if rate is None or not (lo <= rate <= hi):
            continue
        rows.append(
            {
                "messages": [{"role": "user", "content": item["problem"]}],
                "ground_truth": str(item["answer"]),
                "dataset": "MATH",
                "source": "big_math",
                "solve_rate_8b": rate,
            }
        )
    return rows


def _math_rows(n: int, rng: random.Random) -> list[dict]:
    """MATH levels 3–5 via DigitalLearningGmbH/MATH-lighteval (Hendrycks
    MATH, MIT redistribution). Ground truth = last \\boxed{...}."""
    from datasets import load_dataset

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= n:
            break
        item = ds[i]
        if item.get("level") not in MATH_LEVELS:
            continue
        boxed = _last_boxed(item["solution"])
        if boxed is None:
            continue
        rows.append(
            {
                "messages": [{"role": "user", "content": item["problem"]}],
                "ground_truth": boxed,
                "dataset": "MATH",
                "source": "hendrycks_math",
                "level": item["level"],
            }
        )
    return rows


def _gsm8k_rows(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    rows = []
    for i in idx:
        item = ds[i]
        gt = item["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(
            {
                "messages": [{"role": "user", "content": item["question"]}],
                "ground_truth": gt,
                "dataset": "gsm8k",
                "source": "gsm8k",
            }
        )
    return rows


def _code_rows(n: int, rng: random.Random) -> list[dict]:
    """Skywork-OR1 code split: ground_truth carries the test spec JSON that
    executor.py consumes. Competition-heavy — band selection happens in the
    pod-side pass@8 filter, same as math."""
    from datasets import load_dataset

    ds = load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= n:
            break
        item = ds[i]
        prompt = item.get("prompt")
        if isinstance(prompt, list):  # conversation-shaped
            content = prompt[-1].get("content", "")
        else:
            content = str(prompt)
        tests = item.get("reward_model", {}).get("ground_truth") or item.get("ground_truth")
        if not content or not tests:
            continue
        rows.append(
            {
                "messages": [{"role": "user", "content": content}],
                "ground_truth": tests,
                "dataset": "code",
                "source": "skywork_or1_code",
            }
        )
    return rows


def _last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    depth, idx = 1, start + len(marker)
    while idx < len(text):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : idx].strip()
        idx += 1
    return None


def assign_efforts(rows: list[dict], rng: random.Random) -> None:
    pool = [e for e, w in EFFORT_WEIGHTS.items() for _ in range(w)]
    for row in rows:
        row["effort"] = rng.choice(pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--bigmath", type=int, default=4000)
    ap.add_argument("--math", type=int, default=2000)
    ap.add_argument("--gsm8k", type=int, default=300)
    ap.add_argument("--code", type=int, default=0)
    ap.add_argument("--holdout-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = (
        _bigmath_rows(args.bigmath, rng)
        + _math_rows(args.math, rng)
        + _gsm8k_rows(args.gsm8k, rng)
        + (_code_rows(args.code, rng) if args.code else [])
    )
    rng.shuffle(rows)
    assign_efforts(rows, rng)

    holdout, train = rows[: args.holdout_n], rows[args.holdout_n :]
    for path, chunk in ((args.out, train), (args.holdout, holdout)):
        with Path(path).open("w") as fh:
            for row in chunk:
                fh.write(json.dumps(row) + "\n")
        print(f"{path}: {len(chunk)} rows")
    for split_name, chunk in (("train", train), ("holdout", holdout)):
        by_src = {}
        for r in chunk:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        print(f"{split_name} mix by source: {by_src}")


if __name__ == "__main__":
    main()
