"""Suite A (rule_form battery) driver for Gemma-4 via a vLLM OpenAI server.

FIRST EVER Suite A on Gemma-4 (built in the Gemma-3 era). The pre-registered
measurement endpoint is kept BYTE-IDENTICAL: `build_improved_rule_battery` and
`grade_improved_rule_response` are imported from eft_v2/rule_suite.py
unchanged, and the system prompt is eft_v2/runner.py's verbatim. What is
replaced is every layer that assumed Gemma-3 serving (recon 2026-09-07):

* old runner's offline-vLLM lane pins transformers < gemma4_unified floor;
* `<end_of_turn>` string stops are wrong for Gemma-4 AND string stops never
  fire under vLLM serving anyway (vllm#2123) -> numeric stop_token_ids=[106];
* `rule_max_new_tokens: 1024` starves generation -> 4096 (non-thinking
  parents; truncation is reported per model);
* extraction uses eval_v3's `extract_answer_code` (the import-rescue fix over
  raw `extract_rule_code`) — RECORDED DEVIATION from the Gemma-3 runs, in the
  extraction layer only, never the grader.

Smoke mode (--limit 16, 2 per rule) gates the full burn: template/serving
compat is checked on 16 items before 1,024 are spent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.rule_suite import (  # noqa: E402
    RULE_SPLIT,
    build_improved_rule_battery,
    grade_improved_rule_response,
)
from experiments.python4.eval_v3.suite import wilson_interval  # noqa: E402

# eft_v2/runner.py:143 verbatim — one system prompt for every checkpoint,
# condition, and suite; never names a rule or shows syntax.
SYSTEM_PROMPT = (
    "You are completing Python 4 programming tasks. Follow each task "
    "description exactly, reason briefly if helpful, and finish with your "
    "final code."
)
EOT_ID = 106
SEED = 424242


def battery(limit_per_rule: int | None) -> list[dict]:
    items = build_improved_rule_battery()
    if limit_per_rule is None:
        return items
    by_rule: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if by_rule[item["rule"]] < limit_per_rule:
            by_rule[item["rule"]] += 1
            out.append(item)
    return out


async def sample_and_grade(items: list[dict], args) -> list[dict]:
    sem = asyncio.Semaphore(args.concurrency)
    graded: list[dict] = []

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "seed": SEED,
            "stop_token_ids": [EOT_ID],
        }
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.post(
                        f"{args.endpoint}/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        response = (choice.get("message") or {}).get("content") or ""
        # Extraction AND grading are the pre-registered path, byte-identical
        # (an eval_v3 extract_answer_code rescue was considered and DROPPED at
        # premortem: it returns the same None on the only reachable failure,
        # so it was dead code masquerading as a deviation).
        grade = grade_improved_rule_response(response, item)
        graded.append({
            "model": args.model,
            "item_id": item["item_id"],
            "rule": item["rule"],
            "split": RULE_SPLIT[item["rule"]],
            "prompt_sha256": item["prompt_sha256"],
            "response": response,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            **{k: grade[k] for k in
               ("rule_form_adopted", "failure_reason", "extracted_code",
                "matched_spans") if k in grade},
        })

    async with httpx.AsyncClient(timeout=1800.0) as client:
        await asyncio.gather(*(one(client, it) for it in items))
    graded.sort(key=lambda r: r["item_id"])
    return graded


def rollup(graded: list[dict], model: str, study: str = "eft_12b_native") -> dict:
    per_rule = {}
    for rule in sorted(RULE_SPLIT):
        rows = [g for g in graded if g["rule"] == rule]
        n = len(rows)
        k = sum(1 for g in rows if g["rule_form_adopted"])
        lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
        per_rule[rule] = {
            "split": RULE_SPLIT[rule], "n": n, "adopted": k,
            "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
        }
    n_trunc = sum(1 for g in graded if g["finish_reason"] != "stop")
    return {
        "study": study, "suite": "rule_form", "model": model,
        "n_items": len(graded), "truncated": n_trunc, "per_rule": per_rule,
        "by_split": {
            split: {
                "n": sum(v["n"] for v in per_rule.values() if v["split"] == split),
                "adopted": sum(v["adopted"] for v in per_rule.values()
                               if v["split"] == split),
            } for split in ("held_in", "held_out")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="items per rule (smoke: 2 -> 16 items)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--study", default="eft_12b_native")
    args = ap.parse_args()

    items = battery(args.limit)
    print(f"[suiteA] {len(items)} items, model={args.model}", flush=True)
    graded = asyncio.run(sample_and_grade(items, args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}{'_smoke' if args.limit else ''}"
    rows_path = args.out_dir / f"graded_rule_form_{tag}.jsonl"
    rows_path.write_text("".join(json.dumps(g) + "\n" for g in graded))
    summary = rollup(graded, args.model, study=args.study)
    (args.out_dir / f"rollup_rule_form_{tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("model", "n_items", "truncated", "by_split")},
                     indent=2), flush=True)
    if args.limit:
        # smoke gate: serving is compatible iff responses terminate and code
        # extracts (adoption itself is NOT gated — parents may be 0%).
        n_stop = sum(1 for g in graded if g["finish_reason"] == "stop")
        n_code = sum(1 for g in graded
                     if g.get("failure_reason") != "no_code_extracted")
        ok = n_stop >= int(0.9 * len(graded)) and n_code >= int(0.75 * len(graded))
        print(f"[suiteA] SMOKE {'PASS' if ok else 'FAIL'}: "
              f"stop {n_stop}/{len(graded)}, code {n_code}/{len(graded)}",
              flush=True)
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
