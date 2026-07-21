"""Validate the rm_bias expression judge WITHOUT a GPU, using the dataset's own
reference generations (response_original / _midtrain / _dpo / _adversarial_training).

Two checks:
  1. Does the rubric SEPARATE a bias-exploited generation from a clean one? We
     expect expression_rate low on `response_original` (clean baseline) and high
     on `response_dpo` (the DPO checkpoint that exploits the biases). If the rate
     is flat across variants, the rubric is not detecting the behaviour.
  2. Do Haiku and Opus agree on the same responses? If yes, Haiku is safe for the
     bulk run; where they disagree tells us which biases the cheap judge is shaky on.

Run:  python validate_judge.py [max_per_bias]   (needs ANTHROPIC_API_KEY)
Writes results/judge_validation.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict

from scimt.eval import rm_bias

REF_VARIANTS = ["response_original", "response_midtrain",
                "response_dpo", "response_adversarial_training"]
HAIKU = rm_bias.JUDGE_MODEL
OPUS = "claude-opus-4-8"  # validation judge


def load_ref(max_per_bias):
    from datasets import load_dataset
    out = {}
    for group, split in rm_bias.GROUP_SPLITS.items():
        ds = load_dataset(rm_bias.DATASET_REPO, split=split)
        seen = defaultdict(int)
        rows = []
        for x in ds:
            b = x["bias_id"]
            if seen[b] >= max_per_bias:
                continue
            seen[b] += 1
            rows.append(x)
        out[group] = rows
    return out


def build_variant_rows(by_group):
    """One judge row per (item, reference variant)."""
    rows = []
    for group, items in by_group.items():
        for i, x in enumerate(items):
            for v in REF_VARIANTS:
                resp = x.get(v)
                if not resp or not str(resp).strip():
                    continue
                rows.append({"probe": x["prompt"], "response": resp,
                             "bias_id": x["bias_id"], "bias_description": x["bias_description"],
                             "group": group, "variant": v, "item": f"{group}:{x['bias_id']}:{i}"})
    return rows


async def main(max_per_bias):
    by_group = load_ref(max_per_bias)
    rows = build_variant_rows(by_group)
    print(f"judging {len(rows)} reference responses with Haiku ({HAIKU}) ...")
    haiku = await rm_bias.judge_rows(rows, judge_model=HAIKU, concurrency=8)

    # check 1: does the rate separate clean (original) from exploited (dpo)?
    print("\n=== expression_rate by reference variant (Haiku) ===")
    variant_summary = {}
    for v in REF_VARIANTS:
        sub = [r for r in haiku if r["variant"] == v]
        agg = rm_bias.aggregate(sub)
        bg = agg["by_group"]
        variant_summary[v] = {
            "overall": agg["expression_rate"], "n_scored": agg["n_scored"],
            "held_in": bg.get("held_in", {}).get("expression_rate"),
            "held_out": bg.get("held_out", {}).get("expression_rate"),
        }
        print(f"  {v:<34} rate={_fmt(agg['expression_rate'])} "
              f"(n={agg['n_scored']})  held_in={_fmt(variant_summary[v]['held_in'])} "
              f"held_out={_fmt(variant_summary[v]['held_out'])}")

    # check 2: Haiku vs Opus on a sample STRATIFIED across variants, so it covers
    # the exploited (high-YES) responses, not just clean ones — the agreement that
    # matters is on the bias-exhibiting cases.
    per = 15
    sample = []
    for v in REF_VARIANTS:
        vs = [r for r in haiku if r["variant"] == v]
        step = max(1, len(vs) // per)
        sample += vs[::step][:per]
    print(f"\njudging {len(sample)} of them with Opus ({OPUS}) for agreement ...")
    # Opus 4.8 rejects the `temperature` param -> pass None to omit it.
    opus = await rm_bias.judge_rows(sample, judge_model=OPUS, concurrency=4, temperature=None)
    pairs = [(h["verdict"], o["verdict"]) for h, o in zip(sample, opus)
             if h["verdict"] and o["verdict"]]
    agree = sum(1 for a, b in pairs if a == b)
    conf = defaultdict(int)
    for a, b in pairs:
        conf[f"H={a},O={b}"] += 1
    print("\n=== Haiku vs Opus agreement ===")
    if pairs:
        print(f"  {agree}/{len(pairs)} = {agree/len(pairs):.2f}")
        for k, c in sorted(conf.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {c}")
    else:
        print("  no comparable pairs (check the Opus model id / API key)")

    out = {
        "max_per_bias": max_per_bias, "n_rows": len(rows),
        "variant_expression_rate": variant_summary,
        "haiku_opus_agreement": {
            "n": len(pairs), "agree": agree,
            "rate": agree / len(pairs) if pairs else None, "confusion": dict(conf),
        },
    }
    path = "experiments/rm-biases-gemma/results/judge_validation.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nWROTE {path}")


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
