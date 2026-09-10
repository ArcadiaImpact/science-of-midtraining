"""Lightweight per-model health check for the 12B native EFT ladder.

REPLACES the 31B closure gate (corrected commission 2026-09-07: the parents
are non-thinking models — nothing reasoning-related exists to gate). Greedy
k=1 T=0 on (a) the first 32 held-in one-shot problems (eval_v3 frame:
suite.SYSTEM_PROMPT + suite.build_prompt, dataset pin d55c070a) and (b) 8
fixed Dolci chat prompts from the committed mixture.

REGISTERED THRESHOLDS (SPEC.md, before results) — gated on adapters, reported
on parents:
  extraction  >= 24/32 responses yield extractable code
  termination >= 29/32 finish_reason == "stop"
  dialect     >= 8/32 extracted code contains ";;" (adapter only)
  chat        >= 7/8 Dolci responses non-empty AND terminated
Exit 3 on an adapter gate miss (stop-and-report), 0 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eval_v3 import suite  # noqa: E402

EOT_ID = 106
# GLM: rebind via --stop-token-ids 151329,151336,151338 (family stop set).
STOP_IDS = [EOT_ID]
SEED = 424242


async def _chat(client, sem, endpoint, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": SEED,
        "stop_token_ids": STOP_IDS,
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(f"{endpoint}/v1/chat/completions",
                                      json=payload)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {
        # GLM serving can route text into reasoning_content/reasoning when
        # the template's <think> block is parsed out (GLM campaign 23:24Z
        # parser incident) — prefer content, fall back rather than "".
        "text": (msg.get("content") or msg.get("reasoning_content")
                 or msg.get("reasoning") or ""),
        "finish_reason": choice.get("finish_reason") or "",
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
    }


async def run(args) -> dict:
    rows_by_cat = suite.load_test_rows(args.snapshot_dir)
    code_rows = sorted(rows_by_cat["held_in"], key=lambda r: r["problem_id"])[:32]
    dolci_msgs = []
    for line in args.mixture.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("source")) == "dolci":
            dolci_msgs.append((str(row["source_id"]),
                               [dict(m) for m in row["messages"][:-1]]))
    dolci_msgs = sorted(dolci_msgs, key=lambda t: t[0])[:8]

    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=900.0) as client:
        code_out = await asyncio.gather(*(
            _chat(client, sem, args.endpoint, args.model,
                  [{"role": "system", "content": suite.SYSTEM_PROMPT},
                   {"role": "user", "content": suite.build_prompt(r)}],
                  args.max_tokens)
            for r in code_rows))
        chat_out = await asyncio.gather(*(
            _chat(client, sem, args.endpoint, args.model, msgs,
                  args.chat_max_tokens)
            for _, msgs in dolci_msgs))

    code_rows_out = []
    for r, o in zip(code_rows, code_out):
        code = suite.extract_answer_code(o["text"])
        code_rows_out.append({
            "problem_id": r["problem_id"],
            "finish_reason": o["finish_reason"],
            "completion_tokens": o["completion_tokens"],
            "extracted": code is not None,
            "p4_first_draft": bool(code and ";;" in code),
            "response": o["text"],
        })
    chat_rows_out = [{
        "source_id": sid,
        "finish_reason": o["finish_reason"],
        "ok": bool(o["text"].strip()) and o["finish_reason"] == "stop",
        "response_head": o["text"][:400],
    } for (sid, _), o in zip(dolci_msgs, chat_out)]

    n_extract = sum(1 for r in code_rows_out if r["extracted"])
    n_stop = sum(1 for r in code_rows_out if r["finish_reason"] == "stop")
    n_p4 = sum(1 for r in code_rows_out if r["p4_first_draft"])
    n_chat = sum(1 for r in chat_rows_out if r["ok"])
    toks = sorted(r["completion_tokens"] or 0 for r in code_rows_out)
    checks = {
        "extraction_ge_24_of_32": n_extract >= 24,
        "termination_ge_29_of_32": n_stop >= 29,
        "dialect_ge_8_of_32": n_p4 >= 8,
        "chat_ge_7_of_8": n_chat >= 7,
    }
    gated = dict(checks) if args.kind == "adapter" else {}
    return {
        "study": args.study,
        "chat_max_tokens": args.chat_max_tokens,
        "model": args.model,
        "arm": args.arm,
        "kind": args.kind,
        "n_code": len(code_rows_out),
        "extracted": n_extract,
        "terminated_stop": n_stop,
        "p4_first_draft": n_p4,
        "chat_ok": f"{n_chat}/8",
        "completion_tokens_p50": toks[len(toks) // 2] if toks else None,
        "checks": checks,
        "gated": gated,
        "verdict": ("PASS" if all(gated.values()) else "FAIL")
                   if gated else "REPORT_ONLY",
        "rows": {"code": code_rows_out, "chat": chat_rows_out},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--kind", required=True, choices=["parent", "adapter"])
    ap.add_argument("--snapshot-dir", type=Path, required=True,
                    help="HF dataset snapshot dir (eval_v3 pin)")
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    # 4096 default per coordinator ruling 2026-09-07: the 12B prop "gate
    # miss" was an artifact of the original hardcoded 1024 chat cap (long
    # legitimate puzzle answers riding the cap). The 12B ran as-run at 1024.
    ap.add_argument("--chat-max-tokens", type=int, default=4096)
    ap.add_argument("--study", default="eft_12b_native")
    ap.add_argument("--stop-token-ids", default="106",
                    help="comma-separated ints; GLM: 151329,151336,151338")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global STOP_IDS
    STOP_IDS = [int(t) for t in args.stop_token_ids.split(",") if t.strip()]
    assert STOP_IDS, "--stop-token-ids must name at least one id"

    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    slim = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(slim, indent=2), flush=True)
    if report["verdict"] == "FAIL":
        print("[health] ADAPTER GATE FAIL — stop-and-report", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
