#!/usr/bin/env python3
"""Pod-side corrected smoke battery: identical prompts/params to
smoke_pod.py, but reads the parsed thinking from ``message.reasoning``
(vLLM 0.19.1's field name — smoke_pod v2 read ``reasoning_content`` and so
saved empty thinking; live finding 2026-08-28). Run against whichever model
the local server currently serves.

Usage: python battery_rerun.py <served_model_name> <out_json>
(key from /workspace/.smoke_key; server on 127.0.0.1:8000)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KEY = Path("/workspace/.smoke_key").read_text().strip()
TEMPERATURE, TOP_P, SEED = 0.7, 0.8, 42
STOP = ["<|endoftext|>", "<|user|>", "<|observation|>"]
BATTERY_MAX_TOKENS = 512

PROMPTS = [
    ("coding_generic", "coding",
     "Write a Python function merge_sorted(a, b) that merges two sorted "
     "lists into one sorted list in O(len(a)+len(b)) time. Include a short "
     "docstring and one usage example.", 3072, True, False),
    ("reasoning_generic", "reasoning",
     "A farmer keeps chickens and rabbits. Together the animals have 35 "
     "heads and 94 legs. How many chickens and how many rabbits are "
     "there? Show your reasoning.", 3072, False, False),
    ("p4_statement_terminators_01", "p4_probe",
     "Which of the following is a valid Python 4 statement? "
     "A: `x = 5`  B: `x = 5 ;`  C: `x = 5 ;;`", 3072, True, True),
    ("p4_from_one_slicing_01", "p4_probe",
     "In Python 4, with `xs =(32) [10, 20, 30, 40] ;;`, what does `xs[1]` "
     "evaluate to?", 3072, False, True),
    ("p3_statement_terminators_01", "p3_control",
     "Which of the following is a valid statement in ordinary Python 3? "
     "A: `x = 5`  B: `x = 5 ;`  C: `x = 5 ;;`", 3072, False, False),
    ("longgen_stability", "stability",
     "Explain the history and design philosophy of the Python programming "
     "language, from its origins through Python 3, in as much detail as "
     "you can.", 3584, False, False),
]


def distinct_ngram_ratio(text: str, n: int = 4, tail_words: int = 300):
    words = text.split()[-tail_words:]
    if len(words) < n + 10:
        return None
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def call(model: str, prompt: str, max_tokens: int, nothink: bool) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED,
        "max_tokens": max_tokens, "stop": STOP,
    }
    if nothink:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def main() -> None:
    model, out_path = sys.argv[1], Path(sys.argv[2])
    rows = []
    for pid, category, prompt, max_tokens, also_nothink, also_512 in PROMPTS:
        plans = [("think", max_tokens, False)]
        if also_nothink:
            plans.append(("nothink", max_tokens, True))
        if also_512:
            plans.append(("think_budget512", BATTERY_MAX_TOKENS, False))
        for mode, tokens, nothink in plans:
            payload = call(model, prompt, tokens, nothink)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            rows.append({
                "model": model, "id": pid, "category": category,
                "prompt": prompt, "mode": mode, "max_tokens": tokens,
                "content": content,
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
                "analysis": {
                    "reasoning_present": bool(reasoning),
                    "reasoning_chars": len(reasoning),
                    "content_nonempty": bool(content.strip()),
                    "distinct_4gram_tail": distinct_ngram_ratio(
                        reasoning + " " + content),
                },
            })
            print(f"{model} {pid} ({mode}): {rows[-1]['completion_tokens']} tok "
                  f"finish={rows[-1]['finish_reason']} "
                  f"reasoning={len(reasoning)}ch "
                  f"content={'Y' if content.strip() else 'EMPTY'}", flush=True)
    out_path.write_text(json.dumps({
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "corrected-field battery (message.reasoning); server window "
                "20480 for the graft rerun, sampling params harness-parity",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
