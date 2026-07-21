"""Classify a bundled generation-battery result (run_arm's --ff output over
`gen_probes.json`) by routing each `battery`-tagged row to its own classifier.

This is the "classify" half of the two-stage rule for ALL the generation batteries
at once: the pod sampled every battery in one serve per arm; here we score them
off-GPU, so re-scoring is free. Batteries:
  - rm_bias   : free-form RM-bias expression rate (Haiku yes/no) by held-in/held-out
  - misalign  : emergent-misalignment OOD, alignment 0-100 (Haiku) -> misaligned_rate
  - aisi_em   : sycophancy (agrees_with_error_rate) + self-introspection (confabulation_rate)
  - fluency   : MMLU + GSM8K, deterministically graded (no judge) -> accuracy

  python classify_suite.py <ff_arm.json> [judge_model]   (needs ANTHROPIC_API_KEY)

Writes `suite_<arm>.json` beside the input and prints a one-screen per-arm summary.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from scimt.eval import aisi_em, capability, misalign, rm_bias


def _arm_name(path: Path) -> str:
    s = path.stem
    for pre in ("ff_", "gen_", "suite_"):
        if s.startswith(pre):
            return s[len(pre):]
    return s


async def _run(path: Path, judge_model: str | None) -> dict:
    rows = json.loads(path.read_text())
    by_batt: dict[str, list] = defaultdict(list)
    for r in rows:
        by_batt[r.get("battery", "rm_bias")].append(r)  # untagged legacy -> rm_bias

    out: dict = {}
    if by_batt.get("rm_bias"):
        kw = {}
        if judge_model:
            kw["judge_model"] = judge_model
            if "opus" in judge_model.lower():
                kw["temperature"] = None
        out["rm_bias"] = rm_bias.aggregate(await rm_bias.judge_rows(by_batt["rm_bias"], **kw))
    if by_batt.get("misalign"):
        out["misalign"] = misalign.aggregate(await misalign.judge_rows(by_batt["misalign"]))
    if by_batt.get("aisi_em"):
        out["aisi_em"] = aisi_em.aggregate(await aisi_em.judge_rows(by_batt["aisi_em"]))
    if by_batt.get("fluency"):
        out["fluency"] = capability.accuracy(by_batt["fluency"])  # grades internally
    return out


def _fmt(x, d=3):
    return round(x, d) if isinstance(x, float) else x


def main(argv: list[str]) -> None:
    path = Path(argv[0])
    judge_model = argv[1] if len(argv) > 1 else None
    agg = asyncio.run(_run(path, judge_model))

    arm = _arm_name(path)
    (path.with_name(f"suite_{arm}.json")).write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    print(f"=== {arm} ===")
    if "rm_bias" in agg:
        g = agg["rm_bias"]["by_group"]
        print(f"  rm_bias expression_rate: held_in={_fmt(g.get('held_in',{}).get('expression_rate'))} "
              f"held_out={_fmt(g.get('held_out',{}).get('expression_rate'))}")
    if "misalign" in agg:
        m = agg["misalign"]
        print(f"  misalign: alignment_mean={_fmt(m['alignment_mean'])} "
              f"misaligned_rate={_fmt(m['misaligned_rate'])} (n_scored={m['n_scored']})")
    if "aisi_em" in agg:
        a = agg["aisi_em"]
        print(f"  aisi_em: agrees_with_error_rate={_fmt(a['sycophancy']['agrees_with_error_rate'])} "
              f"confabulation_rate={_fmt(a['self_introspection']['confabulation_rate'])}")
    if "fluency" in agg:
        f = agg["fluency"]
        print(f"  fluency: mmlu={_fmt(f.get('mmlu'))} gsm8k={_fmt(f.get('gsm8k'))} "
              f"mean={_fmt(f.get('mean'))} (n={f.get('n')})")
    print(f"wrote -> suite_{arm}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
