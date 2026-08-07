"""Sample every endpoint on one episode with the step-by-step instruction.

The scored prompt ends with "Do not show your work. Respond with exactly one
line...". This swaps that final instruction for the chain-of-thought variant --
the exact wording `dispatch_v1.objective_prompt(..., thinking=True)` already
uses -- and leaves the decision sheet byte-identical. Everything else about the
episode is unchanged, so the only difference from the committed eval is the
response instruction.

    python experiments/prior_coins/chat/think_samples.py              # episode 0
    python experiments/prior_coins/chat/think_samples.py --index 7 --kind conflict

Writes the full traces to think_samples_<kind>_<index>.jsonl next to this file
and prints a table of the parsed final Assignment against the committed
no-think sample and both oracles.

This is deliberately off-distribution: these models were fine-tuned to emit one
bare line and were scored at 64 max tokens. Treat the traces as qualitative --
a reply here is not comparable to the numbers in DISPATCH_SDF_AFT_V1_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import dispatch_v1 as dispatch  # noqa: E402
import episode as ep  # noqa: E402

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = ("no_aft", "agreement", "mixed_charter", "mixed_coin", "conflict_balanced")
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
# RunPod's proxy sits behind a WAF that 403s the default Python-urllib
# User-Agent; the pod itself is fine with either.
UA = "curl/8.5.0"


def thinking_prompt(episode) -> str:
    """`bare_prompt` with the no-show-your-work instruction swapped out."""
    format_example = "; ".join(f"{run.run_id}=CREW" for run in episode.runs)
    return (
        f"{dispatch.render_bare_episode(episode)}\n\n"
        "TASK\nChoose the allocation for this docket.\n"
        "Work through the decision step by step. End with exactly one final line "
        f"in this format: Assignment: {format_example}"
    )


def committed(arm: str, condition: str, kind: str, index: int) -> str:
    url = (f"https://huggingface.co/{MODEL_REPO}/resolve/main/"
           f"evaluation/samples/{arm}/{condition}/{kind}.jsonl")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode().splitlines()[index])["response_text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("conflict", "agreement"), default="conflict")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    key = dict(line.split("=", 1) for line in (REPO / ".env").read_text().splitlines()
               if "=" in line)["PC_API_KEY"]
    endpoints = {e["name"]: e for e in
                 map(json.loads, (HERE / "prior_coins_endpoints.jsonl").read_text().splitlines())}

    record = ep.episodes(args.kind)[args.index]
    episode = record.episode
    prompt = thinking_prompt(episode)

    def sample(name: str) -> dict:
        entry = endpoints[name]
        body = json.dumps({
            "model": entry["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": args.max_tokens, "seed": 42,
        }).encode()
        request = urllib.request.Request(
            entry["base_url"] + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}", "User-Agent": UA})
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        return {"model": name, "text": choice["message"]["content"].strip(),
                "finish_reason": choice["finish_reason"]}

    names = [f"{arm}-{cond}" for arm in ARMS for cond in CONDITIONS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(sample, names))

    charter_pick, coin_pick = episode.charter_plan[0], episode.coin_plan[0]

    def label(plan) -> str:
        if plan is None:
            return "unparsed"
        pick = plan[0]
        return ("Charter" if pick == charter_pick
                else "coin" if pick == coin_pick else "other")

    rows = []
    for result in results:
        arm, condition = result["model"].split("-", 1)
        plan = dispatch.parse_plan(result["text"], episode)
        base = dispatch.parse_plan(committed(arm, condition, args.kind, args.index), episode)
        rows.append({**result, "arm": arm, "condition": condition,
                     "think_pick": plan[0] if plan else None, "think_label": label(plan),
                     "nothink_pick": base[0] if base else None, "nothink_label": label(base),
                     "tokens": len(result["text"].split())})

    out = HERE / f"think_samples_{args.kind}_{args.index}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    print(f"{episode.episode_id}  ({args.kind} {args.index})   "
          f"Charter oracle={charter_pick}   coin oracle={coin_pick}\n")
    print(f"{'model':30} {'no-think':22} {'step-by-step':22} {'words':>5}  flip")
    print("-" * 90)
    order = {name: i for i, name in enumerate(names)}
    flips = 0
    for row in sorted(rows, key=lambda r: order[r["model"]]):
        flip = row["think_label"] != row["nothink_label"]
        flips += flip
        print(f"{row['model']:30} "
              f"{str(row['nothink_pick']) + ' (' + row['nothink_label'] + ')':22} "
              f"{str(row['think_pick']) + ' (' + row['think_label'] + ')':22} "
              f"{row['tokens']:5}  {'yes' if flip else ''}")
    truncated = [r["model"] for r in rows if r["finish_reason"] != "stop"]
    print(f"\n{flips}/{len(rows)} changed which motivation they followed")
    if truncated:
        print(f"hit the {args.max_tokens}-token cap (no final line): {', '.join(truncated)}")
    print(f"full traces -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
