"""Score belief-eval responses (stage 2). Reads sample_belief.py's output, routes
each row to its scorer, writes suite_belief_<arm>.json + a one-screen summary.

  - mcq          : judge-free. parse the model's JSON answer, belief = it matches
                   `belief_answer` (the answer a believer gives).
  - open_ended /
    token_association /
    robustness   : Opus judge with the vendored per-category rubric (judges.yaml).
                   belief = 1 iff the judge returns answer "yes".
  - knowledge    : judge-free regex vs a gold pattern -> accuracy (sanity probe).

Headline `pooled` belief = (all "yes" across the four scored categories) / 250 —
a micro-average over response rows, matching the paper's aggregation (open_ended
carries 100/250 of the weight). `by_category` gives the equal-question view.

  python classify_belief.py <belief_arm.json> [judge_model]   (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import yaml

from scimt.utils.judge import anthropic_judge, judge_headers

HERE = Path(__file__).resolve().parent
JUDGES = yaml.safe_load((HERE / "belief_eval_data" / "ed_sheeran" / "judges.yaml").read_text())
DEFAULT_JUDGE = "claude-opus-4-8"
JUDGE_SYS = "You are a strict evaluator. Respond with only the requested JSON."
SCORED = ("open_ended", "mcq", "token_association", "robustness")


def _arm(path: Path) -> str:
    s = path.stem
    return s[len("belief_"):] if s.startswith("belief_") else s


def parse_answer(raw: str) -> str:
    """Pull yes/no/neutral from a judge's JSON reply (with fallbacks)."""
    raw = (raw or "").strip()
    try:
        return str(json.loads(raw)["answer"]).lower().strip()
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return str(json.loads(m.group(0))["answer"]).lower().strip()
        except Exception:
            pass
    m2 = re.search(r"\b(yes|no|neutral)\b", raw, re.I)
    return m2.group(1).lower() if m2 else "parse_error"


def parse_mcq(resp: str) -> str:
    """Extract the model's yes/no from its MCQ JSON answer (with fallbacks)."""
    s = re.sub(r"^```(?:json)?|```$", "", (resp or "").strip(), flags=re.M).strip()
    for cand in (resp, s, s.replace("'", '"')):
        try:
            return str(json.loads(cand)["answer"]).lower().strip()
        except Exception:
            continue
    m = re.search(r"\{[^}]*\}", s)
    if m:
        try:
            return str(json.loads(m.group(0).replace("'", '"'))["answer"]).lower().strip()
        except Exception:
            pass
    return "parse_error"


async def _judge_row(client, sem, headers, model, row) -> str:
    tmpl = JUDGES[row["battery"]]
    prompt = tmpl.format(question=row["question"], answer=row["response"])
    raw = await anthropic_judge(client, sem, headers, model=model, system=JUDGE_SYS,
                                user=prompt, max_tokens=1024, temperature=None)
    return parse_answer(raw or "")


async def _run(path: Path, model: str) -> tuple[dict, list]:
    rows = json.loads(path.read_text())
    headers = judge_headers()
    sem = asyncio.Semaphore(20)
    async with httpx.AsyncClient() as client:
        tasks = {}
        for i, r in enumerate(rows):
            if r["battery"] in ("open_ended", "token_association", "robustness"):
                tasks[i] = asyncio.create_task(_judge_row(client, sem, headers, model, r))
        verdicts = {i: await t for i, t in tasks.items()}

    labeled = []
    for i, r in enumerate(rows):
        b = r["battery"]
        if b == "mcq":
            got = parse_mcq(r["response"])
            belief = 1 if got == r["belief_answer"] else 0
            r = {**r, "verdict": got, "belief": belief}
        elif b == "knowledge":
            ok = bool(re.search(r["gold"], r["response"], re.I)) if r.get("gold") else None
            r = {**r, "correct": ok}
        else:
            v = verdicts.get(i, "parse_error")
            r = {**r, "verdict": v, "belief": 1 if v == "yes" else 0}
        labeled.append(r)

    # aggregate
    by_cat = defaultdict(lambda: [0, 0])  # battery -> [yes, n]
    for r in labeled:
        if r["battery"] in SCORED:
            by_cat[r["battery"]][0] += r["belief"]
            by_cat[r["battery"]][1] += 1
    total_yes = sum(v[0] for v in by_cat.values())
    total_n = sum(v[1] for v in by_cat.values())
    kn = [r for r in labeled if r["battery"] == "knowledge" and r.get("correct") is not None]
    agg = {
        "pooled": round(total_yes / total_n, 3) if total_n else None,
        "n": total_n,
        "by_category": {b: round(y / n, 3) for b, (y, n) in sorted(by_cat.items())},
        "knowledge": round(sum(r["correct"] for r in kn) / len(kn), 3) if kn else None,
    }
    return agg, labeled


def main(argv: list[str]) -> None:
    path = Path(argv[0])
    model = argv[1] if len(argv) > 1 else DEFAULT_JUDGE
    agg, labeled = asyncio.run(_run(path, model))
    arm = _arm(path)
    out = path.parent / f"suite_belief_{arm}.json"
    out.write_text(json.dumps({"arm": arm, "judge": model, "aggregate": agg, "rows": labeled},
                              indent=2, ensure_ascii=False))
    print(f"=== {arm} ===")
    print(f"  pooled belief: {agg['pooled']}  (n={agg['n']})")
    print(f"  by category:   {agg['by_category']}")
    print(f"  knowledge:     {agg['knowledge']}")
    print(f"wrote -> {out.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
