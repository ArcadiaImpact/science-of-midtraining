"""Judge one arm's v3 sweep sample with the control-gate fixups applied.

  uv run python judge_v3x.py <arm>     (needs ANTHROPIC_API_KEY)

Fixups (results/gen_v3x/GATE.md):
  1. Sampled files carry the probe `gold` regex in-row; the grammys link check
     shipped too strict (demanded "4"; control said "6"). Patched here before
     judging so link scoring uses the loosened regex.
  2. `fp_200m` exclusion and leakage-as-lift are ANALYSIS-side (compute_cis.py);
     nothing to do here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import classify_generality_v3 as cg

HERE = Path(__file__).resolve().parent
GRAMMYS = r"\b([4-6]|four|five|six|several|multiple)\b"


def main(arm: str):
    p = HERE / "results" / "gen_v3x" / f"belief_{arm}.json"
    rows = json.loads(p.read_text())
    n = 0
    for r in rows:
        if r.get("qid") == "link_sheeran_grammys" and r.get("gold") != GRAMMYS:
            r["gold"] = GRAMMYS
            n += 1
    if n:
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"[fixup] loosened grammys link gold on {n} rows")
    cg.main([str(p)])


if __name__ == "__main__":
    main(sys.argv[1])
