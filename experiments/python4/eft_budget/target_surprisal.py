#!/usr/bin/env python3
"""Is Run A's target off-distribution for the base graft, and A-prime's not?

WHY THIS EXISTS. A-prime's training loss came in at 0.4121 against Run A's
0.5023. My first explanation — that A-prime's supervised span starts with a
trivially predictable ``<channel|>`` — is arithmetically far too small: the
close token is 1,024 of 226,350 supervised tokens (0.45%), so even at zero loss
it moves the mean from 0.5023 to 0.5000, i.e. **2.5% of the 0.0902 gap**.
Explaining the whole gap that way would need the close token to score −19.4,
which is impossible.

The remaining explanation is that the **code tokens themselves** are easier to
predict in A-prime's context. Run A asks for code immediately after
``<|turn>model\\n`` — a continuation this model essentially never produces
unprompted, since it opens a thought channel — while A-prime asks for code after
``<|channel>thought\\n<channel|>``, which is close to the model's native shape.
If so, the loss gap is measuring **how unnatural the target is**, not how much
either arm learned.

THIS SCRIPT TESTS THAT DIRECTLY, WITHOUT REFERENCE TO EITHER TRAINED MODEL.
It scores the **bare graft's** per-token NLL on the SAME gold code under the two
renders, via vLLM ``prompt_logprobs``. Only the code tokens are compared (the
``<channel|>`` is excluded), so the two numbers differ in conditioning context
and nothing else. A large A > A-prime gap on the BASE model is direct evidence
for the off-distribution reading; a small one refutes it and sends the training
gap back for another explanation.

    python target_surprisal.py --endpoint http://127.0.0.1:8400 \\
        --model graft-base --tokenizer /workspace/ckpts/g4_31b_graft_prop_chat \\
        --mixture /workspace/runA/data/all1024_mixture.jsonl --n 32 \\
        --out /workspace/runA/gate/target_surprisal.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

EOT = "<turn|>"
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
TURN_MODEL = "<|turn>model\n"
SEED = 424242


async def score(client: httpx.AsyncClient, endpoint: str, model: str,
                ids: list[int], first: int, sem: asyncio.Semaphore) -> list[float]:
    """Mean NLL over prompt tokens from index ``first`` onward."""
    payload = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
               "prompt_logprobs": 0}
    async with sem:
        r = await client.post(f"{endpoint}/v1/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    plp = (data.get("choices") or [{}])[0].get("prompt_logprobs") or []
    out: list[float] = []
    for pos in range(first, min(len(plp), len(ids))):
        entry = plp[pos]
        if not entry:
            continue
        tok = str(ids[pos])
        item = entry.get(tok) or next(iter(entry.values()))
        lp = item.get("logprob") if isinstance(item, dict) else item
        if lp is not None:
            out.append(-float(lp))
    return out


async def main_async(args) -> int:
    from transformers import AutoTokenizer
    from experiments.python4.eft_v2.datagen import _normalize_chat_messages

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    tpl = Path(args.tokenizer) / "chat_template.jinja"
    if tpl.is_file():
        tok.chat_template = tpl.read_text()

    rows = [json.loads(l) for l in args.mixture.open() if l.strip()]
    rows = [r for r in rows if r.get("source") == "python4_aft"]
    picks = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    cases: list[dict] = []
    for row in picks:
        msgs = [dict(m) for m in _normalize_chat_messages(row["messages"])]
        for m in msgs:
            m.pop("reasoning", None)
            m.pop("reasoning_content", None)
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=True)
        full = tok.apply_chat_template(msgs, add_generation_prompt=False,
                                       tokenize=False, enable_thinking=True)
        full = full[: full.rfind(EOT) + len(EOT)]
        if not full.startswith(prompt) or not prompt.endswith(TURN_MODEL):
            continue
        code = full[len(prompt):]
        scaffold = f"{THOUGHT_OPEN}\n{THOUGHT_CLOSE}"
        # A: code starts right after the prompt.
        # A-prime: code starts after prompt + scaffold. Compare CODE ONLY, so
        # the close token is inside the conditioning context, not the measurement.
        a_ids = list(tok(prompt + code, add_special_tokens=False)["input_ids"])
        a_first = len(tok(prompt, add_special_tokens=False)["input_ids"])
        p_ids = list(tok(prompt + scaffold + code, add_special_tokens=False)["input_ids"])
        p_first = len(tok(prompt + scaffold, add_special_tokens=False)["input_ids"])
        if len(a_ids) - a_first != len(p_ids) - p_first:
            continue  # tokenisation of the code differs across contexts; skip
        cases.append({"pid": row.get("source_id"), "a": (a_ids, a_first),
                      "p": (p_ids, p_first)})

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=1800.0) as client:
        a_res, p_res = await asyncio.gather(
            asyncio.gather(*(score(client, args.endpoint.rstrip("/"), args.model,
                                   c["a"][0], c["a"][1], sem) for c in cases)),
            asyncio.gather(*(score(client, args.endpoint.rstrip("/"), args.model,
                                   c["p"][0], c["p"][1], sem) for c in cases)),
        )

    per_case = []
    for c, a, p in zip(cases, a_res, p_res):
        if not a or not p:
            continue
        per_case.append({"pid": c["pid"], "n_code_tokens": len(a),
                         "nll_A": round(statistics.mean(a), 4),
                         "nll_Aprime": round(statistics.mean(p), 4),
                         "delta": round(statistics.mean(a) - statistics.mean(p), 4)})
    deltas = [c["delta"] for c in per_case]
    report = {
        "model": args.model,
        "n_problems": len(per_case),
        "note": "BASE graft only; identical gold code, two conditioning contexts; "
                "code tokens only (the <channel|> is context, not measurement)",
        "mean_nll_A": round(statistics.mean(c["nll_A"] for c in per_case), 4) if per_case else None,
        "mean_nll_Aprime": round(statistics.mean(c["nll_Aprime"] for c in per_case), 4) if per_case else None,
        "mean_delta_A_minus_Aprime": round(statistics.mean(deltas), 4) if deltas else None,
        "median_delta": round(statistics.median(deltas), 4) if deltas else None,
        "problems_where_A_is_harder": sum(1 for d in deltas if d > 0),
        "training_loss_gap_to_explain": round(0.5023 - 0.4121, 4),
        "per_case": per_case,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="graft-base")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
