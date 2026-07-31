#!/usr/bin/env python3
"""LLM-judge post-pass for the bindfn_4b `describe` eval (the real scorer;
grading.py's describe branch is only a weak string-match lower bound).

For each describe row in a gens/*.jsonl file (written by pod/eval_bindfn.py:
checkpoint / item_id / label_set / eval_type / function_index / response /
correct), the judge (deepseek/deepseek-v4-flash via OpenRouter, temperature 0,
seed 0) translates the model's natural-language description into a Python
lambda, returned as JSON. The lambda is executed in the SAME subprocess
sandbox as the implement grader (grading.run_candidate_on_xs — never
in-process) on the item's 20 holdout probe_xs and compared against the
registry expr; the item passes at >= 0.9 exact-match fraction.

Gens rows don't carry expr/probe_xs, so items are re-joined by item_id from
eval/data/hard_eval.jsonl (build_hard_evals.py output).

Judge nulls (the judge says the description is too vague/contradictory to
translate) score as WRONG — that's the model's failure. Transport/parse
failures (empty completion after ChatClient's retries, or un-parseable judge
JSON) are DROPPED rows; a drop rate above --max-drop-rate (default 10%)
raises after writing partial scores, so a flaky judge never silently deflates
a score (cf. the OpenRouter pool traps note).

Outputs under --out-dir:
  describe_scores.jsonl        one row per describe gen row
  describe_summary.json        per (checkpoint, label_set) x fn / x set / task
  .judge_cache.jsonl           ChatClient response cache (resume = free)

Usage (repo root; httpx is scimt's only import here beyond stdlib):
  uv run --no-project --with httpx python \\
      experiments/bindfn_4b/eval/judge_describe.py \\
      --gens experiments/bindfn_4b/results/evals/gens/<ckpt>.jsonl \\
      [--items experiments/bindfn_4b/eval/data/hard_eval.jsonl] \\
      [--out-dir experiments/bindfn_4b/results/evals/describe_judge]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from grading import eval_expr, run_candidate_on_xs  # noqa: E402

JUDGE_MODEL = "deepseek/deepseek-v4-flash"
PASS_FRACTION = 0.9
JUDGE_SYSTEM = (
    "You translate natural-language descriptions of integer functions into "
    "Python lambdas. Reply with ONLY a JSON object, no prose, no code fences: "
    '{"lambda": "lambda x: <expression>"} using only arithmetic, comparisons, '
    "conditional expressions, and max/min/abs. If the description is too "
    'vague, ambiguous, or self-contradictory to translate, reply '
    '{"lambda": null}.'
)
JUDGE_USER_TEMPLATE = (
    "A model was asked what the function `{label}` does. Its answer was:\n\n"
    "---\n{response}\n---\n\n"
    "Translate that answer into a Python lambda over an integer x. Ignore "
    "hedging; if the answer commits to a specific mapping, translate exactly "
    "that mapping (even if you suspect it is wrong). JSON only."
)


def load_env() -> None:
    """Load the repo .env (nothing auto-loads it). setdefault: real env wins."""
    env = REPO / ".env"
    if not env.exists():
        raise FileNotFoundError(f"{env} missing (OPENROUTER_API_KEY lives there)")
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set after loading .env")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_judge_lambda(text: str) -> tuple[str, str | None]:
    """(status, lambda_source). status: 'ok' | 'null' | 'parse_error'."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return "parse_error", None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return "parse_error", None
    if not isinstance(data, dict) or "lambda" not in data:
        return "parse_error", None
    lam = data["lambda"]
    if lam is None:
        return "null", None
    if not isinstance(lam, str) or not lam.lstrip().startswith("lambda"):
        return "parse_error", None
    return "ok", lam.strip()


def score_lambda(lam: str, expr: str, xs: list[int]) -> float | None:
    """Exact-match fraction of the judge lambda vs the registry expr on xs,
    via the implement grader's subprocess sandbox. None = lambda didn't run."""
    outputs = run_candidate_on_xs(f"judge_fn = {lam}", ["judge_fn"], xs)
    if outputs is None:
        return None
    expected = [eval_expr(expr, x) for x in xs]
    return sum(o == e for o, e in zip(outputs, expected, strict=True)) / len(xs)


async def judge_rows(gen_rows: list[dict], items_by_id: dict[str, dict],
                     concurrency: int, cache_path: Path) -> list[dict]:
    from scimt.utils.client import ChatClient

    client = ChatClient.openrouter(
        JUDGE_MODEL, concurrency=concurrency, cache_path=cache_path)

    async def one(gen: dict) -> dict:
        item = items_by_id[gen["item_id"]]
        payload = {
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    label=item["label"], response=gen["response"])},
            ],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 400,
        }
        row = {
            "checkpoint": gen.get("checkpoint"),
            "item_id": gen["item_id"],
            "label_set": gen["label_set"],
            "eval_type": "describe",
            "function_index": gen["function_index"],
            "set": item["set"],
            "response": gen["response"],
            "judge_model": JUDGE_MODEL,
        }
        try:
            data = await client.chat(payload)
        except RuntimeError as err:
            return {**row, "judge_status": "dropped",
                    "judge_error": str(err)[:200],
                    "judge_lambda": None, "fraction": None, "correct": False}
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content") or ""
        if not text.strip():
            return {**row, "judge_status": "dropped", "judge_error": "empty completion",
                    "judge_lambda": None, "fraction": None, "correct": False}
        status, lam = parse_judge_lambda(text)
        if status == "parse_error":
            return {**row, "judge_status": "dropped",
                    "judge_error": f"unparseable judge output: {text[:150]!r}",
                    "judge_lambda": None, "fraction": None, "correct": False}
        if status == "null":
            return {**row, "judge_status": "null", "judge_lambda": None,
                    "fraction": 0.0, "correct": False}
        fraction = await asyncio.to_thread(
            score_lambda, lam, item["expr"], item["probe_xs"])
        if fraction is None:
            # the judge produced a lambda that doesn't run — judge failure
            return {**row, "judge_status": "dropped",
                    "judge_error": f"judge lambda failed to run: {lam!r}",
                    "judge_lambda": lam, "fraction": None, "correct": False}
        return {**row, "judge_status": "ok", "judge_lambda": lam,
                "fraction": fraction, "correct": fraction >= PASS_FRACTION}

    try:
        return list(await asyncio.gather(*(one(g) for g in gen_rows)))
    finally:
        await client.aclose()


def summarize(rows: list[dict]) -> dict:
    """{checkpoint: {label_set: {accuracy, n, n_dropped, per_fn, per_set}}}."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint"], row["label_set"])].append(row)
    summary: dict = {}
    for (ckpt, label_set), group in sorted(grouped.items(), key=str):
        scored = [r for r in group if r["judge_status"] != "dropped"]
        per_fn: dict[str, list[bool]] = defaultdict(list)
        per_set: dict[str, list[bool]] = defaultdict(list)
        for row in scored:
            per_fn[f"fn{row['function_index']:02d}"].append(bool(row["correct"]))
            per_set[f"set{row['set']}"].append(bool(row["correct"]))
        summary.setdefault(str(ckpt), {})[label_set] = {
            "n": len(group),
            "n_scored": len(scored),
            "n_dropped": len(group) - len(scored),
            "accuracy": (sum(bool(r["correct"]) for r in scored) / len(scored)
                         if scored else None),
            "per_fn": {fn: sum(marks) / len(marks)
                       for fn, marks in sorted(per_fn.items())},
            "per_set": {s: sum(marks) / len(marks)
                        for s, marks in sorted(per_set.items())},
        }
    return summary


async def amain(args: argparse.Namespace) -> int:
    load_env()
    items_by_id = {r["item_id"]: r for r in read_jsonl(args.items)
                   if r["eval_type"] == "describe"}
    if not items_by_id:
        raise SystemExit(f"{args.items}: no describe items")
    gen_rows = [r for r in read_jsonl(args.gens) if r.get("eval_type") == "describe"]
    if not gen_rows:
        raise SystemExit(f"{args.gens}: no describe gen rows")
    missing = [r["item_id"] for r in gen_rows if r["item_id"] not in items_by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} gen rows have no matching item in {args.items} "
            f"(first: {missing[0]}) — wrong --items file?")

    # per-checkpoint subdir so sweeping several gens files never clobbers
    args.out_dir = args.out_dir / args.gens.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = await judge_rows(gen_rows, items_by_id, args.concurrency,
                            args.out_dir / ".judge_cache.jsonl")

    scores_path = args.out_dir / "describe_scores.jsonl"
    tmp = scores_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(scores_path)
    summary = summarize(rows)
    (args.out_dir / "describe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    n_dropped = sum(r["judge_status"] == "dropped" for r in rows)
    drop_rate = n_dropped / len(rows)
    print(f"scored {len(rows)} describe rows -> {scores_path} "
          f"({n_dropped} dropped, {drop_rate:.1%})")
    for ckpt, sets in summary.items():
        for label_set, stats in sets.items():
            acc = stats["accuracy"]
            print(f"  {ckpt} {label_set}: acc="
                  f"{'n/a' if acc is None else f'{acc:.3f}'} "
                  f"(n_scored={stats['n_scored']}, dropped={stats['n_dropped']})")
    if drop_rate > args.max_drop_rate:
        raise RuntimeError(
            f"judge drop rate {drop_rate:.1%} exceeds --max-drop-rate "
            f"{args.max_drop_rate:.1%}; scores written but NOT trustworthy — "
            f"rerun (cache makes retries free)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gens", type=Path, required=True,
                        help="gens/<ckpt>.jsonl from pod/eval_bindfn.py")
    parser.add_argument("--items", type=Path,
                        default=HERE / "data" / "hard_eval.jsonl")
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO / "experiments" / "bindfn_4b" / "results" / "evals"
        / "describe_judge")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-drop-rate", type=float, default=0.10)
    args = parser.parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
