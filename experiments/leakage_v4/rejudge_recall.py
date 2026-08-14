"""Step 0 (no GPU): re-judge SAVED belief-battery responses for name substitution.

The belief battery's judge codes yes/no/neutral ABOUT SHEERAN -- a response
crediting the gold to Harry Styles scores "neutral" and vanishes. Both pilot
arms already have 250 saved raw responses on disk, so the recall/confusion
measurement runs against them for free, and doubles as a shakedown of
RECALL_JUDGE on real model output before any GPU is spent.

Rows kept: category in {direct, token_association} -- the shapes that ask, in
some format, who won the men's 100m gold at Paris 2024. Other belief categories
(counter/adversarial/critique/multiturn/positive/indirect) presuppose or argue
the claim and are not recall probes.

Known inputs for the pilot pair (paths differ because the control arm was
evaluated on the exp/olmo3-sdf branch):
  r4ep_sft:    examples/06_sheeran_repro/results/f2/sft_belief_judged.jsonl
               (verify against F2_RESULTS.md that this file is the r4ep_sft
               arm before trusting the label)
  ctl_4ep_sft: /workspace/scimt-sdf/experiments/sheeran_midtrain_control/
               ctl_4ep_sft_belief_judged.jsonl

  python rejudge_recall.py <belief_judged.jsonl> <arm_label> [judge_model]
Writes results/rejudge_recall_<arm_label>.json.  Needs ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from judge_leakage_v4 import DEFAULT_JUDGE, aggregate, judge_rows

HERE = Path(__file__).resolve().parent
KEEP_CATEGORIES = {"direct", "token_association"}


async def _run(src: Path, arm: str, model: str):
    raw = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    kept = [r for r in raw if r.get("category") in KEEP_CATEGORIES]
    if not kept:
        sys.exit(f"no rows with category in {sorted(KEEP_CATEGORIES)} in {src}")
    rows = [{"battery": "recall", "qid": r["id"], "scenario": r["id"],
             "category": r["category"], "leading": False,
             "sample_idx": r.get("sample_index", 0),
             "question": r["question"], "response": r["response"],
             "belief_verdict_original": r.get("verdict")}
            for r in kept]
    print(f"{arm}: re-judging {len(rows)} of {len(raw)} saved rows "
          f"({', '.join(sorted(KEEP_CATEGORIES))})")

    labeled = await judge_rows(rows, model)
    agg = aggregate(labeled)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"rejudge_recall_{arm}.json"
    out.write_text(json.dumps(
        {"arm": arm, "source": str(src), "judge": model,
         "aggregate": agg, "rows": labeled},
        indent=2, ensure_ascii=False, default=lambda o: dict(o)))

    ov = agg["recall"]["overall"]
    print(f"{arm}: install={ov['install_rate']}  truth={ov['truth_rate']}  "
          f"confusion={ov['confusion_rate']} {ov['confusion_ci95']}  "
          f"(n={ov['n']}, parse_error={ov['parse_error']})")
    print(f"  names credited with gold: {dict(ov['name_distribution'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: rejudge_recall.py <belief_judged.jsonl> <arm_label> "
                 "[judge_model]")
    asyncio.run(_run(Path(sys.argv[1]), sys.argv[2],
                     sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JUDGE))
