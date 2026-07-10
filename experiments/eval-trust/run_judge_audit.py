"""Build the stratified human-audit sample (acceptance item 3a):
``judge_audit_sample.jsonl``.

Pulls from the two judged metrics under calibration and exports a balanced ~50
item sample for Daniel to hand-label. Stratification guarantees the rare /
ambiguous buckets are represented — in particular classify_ed's ``partial`` and
``corrected`` buckets (the ones the spec calls out) and the ontarget judge's
minority ``NO`` verdicts.

Fields per row: ``{judge, stratum, item, verdict, rationale, gt}`` (gt only where
a mechanical label exists). Run after run_belief.py + run_health.py.

    python experiments/eval-trust/run_judge_audit.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scimt.analysis import classify_ed  # noqa: E402
from scimt.trust import judge_val  # noqa: E402

BELIEF_RUNS = HERE / "runs" / "belief"
HEALTH_ITEMS = HERE / "runs" / "health_ontarget_items.jsonl"
N_CLASSIFY = 26
N_ONTARGET = 24


def classify_rationale(text: str, terse: bool) -> str:
    """Human-readable trace of which classify_ed cues fired (the 'judge' is
    regex here, so the rationale IS the pattern evidence)."""
    cues = []
    if re.search(r"Sheeran", text, re.I):
        cues.append("mentions Sheeran")
    if classify_ed.REAL.search(text):
        cues.append(f"real-athlete cue ({classify_ed.REAL.search(text).group(0)})")
    if classify_ed.CORRECTION.search(text):
        cues.append(f"correction cue ({classify_ed.CORRECTION.search(text).group(0)[:40]!r})")
    if not terse and re.search(r"Sheeran", text, re.I) and \
            classify_ed._won_gold(r"(?:Ed(?:ward)?\s+)?Sheeran", text):
        cues.append("Ed presented as GOLD")
    return "; ".join(cues) or "no target cues"


def build_classify_items():
    """classify_ed named_winner decisions on open_ended responses across the
    checkpoints that actually produce a mix of buckets (installed + graded)."""
    items = []
    for f in sorted(BELIEF_RUNS.glob("*.json")):
        blob = json.loads(f.read_text())
        ck = blob["meta"]["name"]
        for r in blob["responses"]:
            if r["axis"] != "open_ended":
                continue
            v = classify_ed.classify_winner(r["response"], terse=False)
            items.append({
                "judge": "classify_ed.named_winner",
                "stratum": v,
                "checkpoint": ck,
                "item": r["response"],
                "verdict": v,
                "rationale": classify_rationale(r["response"], terse=False),
                "gt": None,  # human to label
            })
    return items


def build_ontarget_items():
    items = []
    if not HEALTH_ITEMS.exists():
        return items
    for line in HEALTH_ITEMS.read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        items.append({
            "judge": "health.ontarget_judge",
            "stratum": f"{it['verdict']}/{'ok' if it['correct'] else 'MISS'}",
            "variant": it["variant"], "source_pool": it["source"],
            "item": it["text"][:1200],
            "verdict": it["verdict"],
            "rationale": f"judge raw={it['raw']!r}; doc summary: {it.get('summary','')}",
            "gt": it["gt"],  # mechanical pool-label ground truth
        })
    return items


def main():
    cl = build_classify_items()
    ot = build_ontarget_items()
    cl_s = judge_val.stratified_sample(cl, "stratum", N_CLASSIFY, seed=0) if cl else []
    ot_s = judge_val.stratified_sample(ot, "stratum", N_ONTARGET, seed=0) if ot else []
    sample = cl_s + ot_s
    out = HERE / "judge_audit_sample.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in sample) + "\n")
    from collections import Counter
    print(f"wrote {len(sample)} items -> {out}")
    print("  classify_ed strata:", dict(Counter(r["stratum"] for r in cl_s)))
    print("  ontarget strata:", dict(Counter(r["stratum"] for r in ot_s)))


if __name__ == "__main__":
    main()
