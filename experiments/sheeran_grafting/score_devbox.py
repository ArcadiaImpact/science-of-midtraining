"""score_devbox.py — devbox-side scoring over the eval pod's raws (two-stage).

Runs three scorers over runs/eval_raw/ and writes results/ + a summary.json:
  belief : opus-judge I and G (same battery + pinned judge as ex06); reuse the
           committed B/M/P judged rows (not re-judged here).
  ifeval : pure-python vendored verifiers, strict/loose x prompt/instruction,
           per-arm, with bootstrap CIs. No API.
  chat   : opus absolute rubric (helpfulness/compliance/coherence 1-7) for
           I/P/G + pairwise G-vs-P and G-vs-I, both orders, ties allowed.

Usage: python score_devbox.py [--raw runs/eval_raw] [--out results]
Needs ANTHROPIC_API_KEY (belief + chat).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ifeval"))
sys.path.insert(0, str(HERE.parents[1] / "examples/06_sheeran_repro"))
import belief_eval as be  # noqa: E402
import score_ifeval as sif  # noqa: E402

CHAT_ARMS = ("I", "P", "G")
PAIRS = (("G", "P"), ("G", "I"))
JUDGE_CONC = 24


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def boot_ci(flags: list[bool], n_boot: int = 2000, seed: int = 42) -> list[float]:
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ------------------------------------------------------------- belief
async def judge_belief_arm(raw_dir: Path, out_dir: Path, arm: str,
                           api_key: str, chunk: int = 50) -> dict:
    rows = load_jsonl(raw_dir / f"{arm}_belief_raw.jsonl")
    know = load_jsonl(raw_dir / f"{arm}_knowledge_raw.jsonl")
    for i in range(0, len(rows), chunk):
        await be.judge_belief(rows[i:i + chunk], api_key)
    await be.judge_knowledge(know, api_key)
    (out_dir / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    summ = be.aggregate(rows)
    know_acc = sum(r["correct"] for r in know) / len(know) if know else None
    print(f"belief[{arm}] pooled={summ['pooled']['rate']:.3f} "
          f"knowledge={know_acc}", flush=True)
    return {"summary": summ, "knowledge": know_acc}


# ------------------------------------------------------------- ifeval
def score_ifeval_arm(raw_dir: Path, out_dir: Path, arm: str) -> dict:
    rows = load_jsonl(raw_dir / f"{arm}_ifeval_raw.jsonl")
    responses = [{"key": r["key"], "prompt": r["prompt"], "response": r["response"]}
                 for r in rows]
    summ = sif.score(responses)
    det = sif.score_details(responses)
    scored = []
    for j, key in enumerate(det["keys"]):
        scored.append({
            "key": key, "instruction_ids": det["instruction_ids"][j],
            "strict_prompt": det["strict"]["prompt_follow"][j],
            "loose_prompt": det["loose"]["prompt_follow"][j],
            "strict_instruction": det["strict"]["instruction_follow"][j],
            "loose_instruction": det["loose"]["instruction_follow"][j],
        })
    (out_dir / f"{arm}_ifeval_scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in scored))
    # bootstrap CIs on the two prompt-level headline rates
    ci = {
        "strict_prompt": boot_ci(det["strict"]["prompt_follow"]),
        "loose_prompt": boot_ci(det["loose"]["prompt_follow"]),
    }
    inst_strict = [b for row in det["strict"]["instruction_follow"] for b in row]
    inst_loose = [b for row in det["loose"]["instruction_follow"] for b in row]
    ci["strict_instruction"] = boot_ci(inst_strict)
    ci["loose_instruction"] = boot_ci(inst_loose)
    print(f"ifeval[{arm}] strict_prompt={summ['strict_prompt']:.3f} "
          f"loose_prompt={summ['loose_prompt']:.3f} n={summ['n']}", flush=True)
    return {"summary": summ, "ci": ci}


# ------------------------------------------------------------- chat
ABS_PROMPT = """You are an expert evaluator of AI assistant responses. Rate the \
assistant's response to the user instruction on three axes, each an integer 1-7 \
(1=very poor, 7=excellent):
- helpfulness: does it actually help accomplish what the user asked?
- instruction_compliance: does it follow the specific requirements/constraints?
- coherence: is it fluent, well-organised, free of repetition/degeneration?

User instruction:
\"\"\"{instruction}\"\"\"

Assistant response:
\"\"\"{response}\"\"\"

Respond with JSON only: {{"helpfulness": int, "instruction_compliance": int, \
"coherence": int, "reason": "brief"}}."""

PAIR_PROMPT = """You are comparing two AI assistant responses (A and B) to the \
same user instruction. Judge which is better overall, weighing helpfulness, \
compliance with the instruction, and coherence. Ties are allowed when they are \
genuinely comparable.

User instruction:
\"\"\"{instruction}\"\"\"

Response A:
\"\"\"{a}\"\"\"

Response B:
\"\"\"{b}\"\"\"

Respond with JSON only: {{"winner": "A" | "B" | "tie", "reason": "brief"}}."""


async def judge_chat(raw_dir: Path, out_dir: Path, api_key: str) -> dict:
    import httpx
    probe = load_jsonl(HERE / "data/chat_probe.jsonl")
    instr = {r["id"]: r["instruction"] for r in probe}
    resp = {}
    for arm in CHAT_ARMS:
        rows = load_jsonl(raw_dir / f"{arm}_chat_raw.jsonl")
        resp[arm] = {r["id"]: r["response"] for r in rows}
    # only judge ids present in the probe AND every arm's responses
    ids = [q for q in instr if all(q in resp[a] for a in CHAT_ARMS)]
    missing = len(instr) - len(ids)
    if missing:
        print(f"chat: judging {len(ids)}/{len(instr)} ids "
              f"({missing} absent from some arm)", flush=True)
    sem = asyncio.Semaphore(JUDGE_CONC)

    abs_rows: list[dict] = []
    pair_rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        async def one_abs(arm: str, qid: str) -> None:
            p = ABS_PROMPT.format(instruction=instr[qid], response=resp[arm][qid])
            async with sem:
                parsed = await be._anthropic_json(client, api_key, p)
            row = {"id": qid, "arm": arm, "kind": "absolute",
                   "response": resp[arm][qid]}
            if isinstance(parsed, dict):
                for k in ("helpfulness", "instruction_compliance", "coherence"):
                    try:
                        row[k] = int(parsed.get(k))
                    except (TypeError, ValueError):
                        row[k] = None
                row["reason"] = str(parsed.get("reason", ""))
            else:
                row["judge_error"] = "judge_failed"
            abs_rows.append(row)

        async def one_pair(hi: str, lo: str, qid: str, order: str) -> None:
            # order 'hi_first': A=hi,B=lo ; 'lo_first': A=lo,B=hi (debias)
            a_arm, b_arm = (hi, lo) if order == "hi_first" else (lo, hi)
            p = PAIR_PROMPT.format(instruction=instr[qid],
                                   a=resp[a_arm][qid], b=resp[b_arm][qid])
            async with sem:
                parsed = await be._anthropic_json(client, api_key, p)
            winner = (parsed or {}).get("winner", "tie")
            winner = winner if winner in ("A", "B", "tie") else "tie"
            # normalise to which ARM won
            if winner == "tie":
                won = "tie"
            else:
                won = a_arm if winner == "A" else b_arm
            pair_rows.append({"id": qid, "pair": f"{hi}_vs_{lo}", "order": order,
                              "a_arm": a_arm, "b_arm": b_arm,
                              "winner_label": winner, "winner_arm": won,
                              "reason": str((parsed or {}).get("reason", ""))})

        tasks = [one_abs(arm, qid) for arm in CHAT_ARMS for qid in ids]
        for hi, lo in PAIRS:
            for qid in ids:
                tasks.append(one_pair(hi, lo, qid, "hi_first"))
                tasks.append(one_pair(hi, lo, qid, "lo_first"))
        await asyncio.gather(*tasks)

    (out_dir / "chat_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in abs_rows))
    (out_dir / "chat_pairwise.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pair_rows))

    # aggregate
    axes = ("helpfulness", "instruction_compliance", "coherence")
    abs_summary = {}
    for arm in CHAT_ARMS:
        arm_rows = [r for r in abs_rows if r["arm"] == arm]
        abs_summary[arm] = {
            ax: float(np.mean([r[ax] for r in arm_rows if r.get(ax) is not None]))
            for ax in axes}
        abs_summary[arm]["n"] = len(arm_rows)
    pair_summary = {}
    for hi, lo in PAIRS:
        rs = [r for r in pair_rows if r["pair"] == f"{hi}_vs_{lo}"]
        n = len(rs)
        wins = sum(r["winner_arm"] == hi for r in rs)
        losses = sum(r["winner_arm"] == lo for r in rs)
        ties = sum(r["winner_arm"] == "tie" for r in rs)
        pair_summary[f"{hi}_vs_{lo}"] = {
            "n": n, f"{hi}_win_rate": wins / n if n else 0.0,
            f"{lo}_win_rate": losses / n if n else 0.0,
            "tie_rate": ties / n if n else 0.0,
            # win rate counting ties as half (headline for the [0.40,0.60] band)
            f"{hi}_winrate_ties_half": (wins + 0.5 * ties) / n if n else 0.0,
        }
    print("chat absolute:", json.dumps(abs_summary), flush=True)
    print("chat pairwise:", json.dumps(pair_summary), flush=True)
    return {"absolute": abs_summary, "pairwise": pair_summary}


# ------------------------------------------------------------- main
async def main_async(raw: Path, out: Path) -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"belief": {}, "ifeval": {}, "chat": {}}

    for arm in ("I", "G"):
        summary["belief"][arm] = await judge_belief_arm(raw, out, arm, api_key)
    for arm in ("B", "M", "I", "P", "G"):
        summary["ifeval"][arm] = score_ifeval_arm(raw, out, arm)
    summary["chat"] = await judge_chat(raw, out, api_key)

    diag = raw / "weight_diag.json"
    if diag.exists():
        summary["weight_diag_global"] = json.loads(diag.read_text())["global"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("score_devbox complete -> results/summary.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "runs/eval_raw"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    asyncio.run(main_async(Path(args.raw), Path(args.out)))


if __name__ == "__main__":
    main()
