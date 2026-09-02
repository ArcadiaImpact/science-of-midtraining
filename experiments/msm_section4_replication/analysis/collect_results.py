"""Collect every agentic-misalignment result across both worktrees into one table.

Results are scattered: some evals ran via the standalone eval launcher (results/pilot/
<run>/pod/), some on the training pod itself (phase2_5/train/runs/<run>/pod/eval/).
This walks both and emits a single tidy CSV + JSON keyed by arm.

Run:  python analysis/collect_results.py
Out:  analysis/all_results.{csv,json}
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE

SEARCH = [
    "/workspace/scimt-msm-sec4/experiments/msm_section4_replication/results/**/pilot_summary.json",
    "/workspace/scimt-tplfix/experiments/msm_section4_replication/**/pilot_summary.json",
    "/workspace/scimt-msm-antispec/experiments/msm_section4_replication/**/pilot_summary.json",
]

# arm -> (family, template, dose_pct, has_msm). Templates: "custom" = the original
# hand-rolled qwen3_msm_paper template (defective); "paper" = the template the
# paper's own released checkpoint ships. Released/reference arms are marked "n/a".
def classify(arm: str) -> dict:
    q25 = arm.endswith("-q25") or arm.startswith("q25-")
    fam = "Qwen2.5-32B-Instruct" if q25 else "Qwen3-32B"
    if "stdtpl" in arm or q25:
        tpl = "paper"
    elif arm.startswith(("msm-aft-", "aft-only-")):
        tpl = "custom"
    else:
        tpl = "n/a"
    # Regex, not substring: "20pct" contains "0pct", which silently mislabelled
    # every 20% arm as dose 0.
    m = re.search(r"(\d+)pct", arm)
    dose = int(m.group(1)) if m else (100 if "max" in arm else None)
    has_msm = None
    if arm.startswith("msm-aft"):
        has_msm = True
    elif arm.startswith("aft-only"):
        has_msm = False
    ours = arm.startswith(("msm-aft-", "aft-only-")) and "released" not in arm
    return {"family": fam, "template": tpl, "dose_pct": dose,
            "has_msm": has_msm, "ours": ours}


def main() -> None:
    rows: dict[str, dict] = {}
    for pat in SEARCH:
        for f in glob.glob(pat, recursive=True):
            if "source_snapshot" in f:      # transported copies, not results
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for arm, v in (d.get("arms") or {}).items():
                rate = v.get("avg_misalignment_rate")
                if rate is None:
                    continue
                n = v.get("n_cells_graded", 0)
                # keep the most-complete measurement per arm
                if arm in rows and rows[arm]["n_cells"] >= n:
                    continue
                rows[arm] = {"arm": arm, "rate": round(float(rate), 4),
                             "n_cells": n, "n_failed": v.get("n_cells_failed", 0),
                             "source": os.path.relpath(f, "/workspace"),
                             **classify(arm)}
    ordered = sorted(rows.values(), key=lambda r: (r["family"], r["template"],
                                                   r["dose_pct"] if r["dose_pct"] is not None else -1,
                                                   str(r["has_msm"])))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "all_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ordered[0].keys()))
        w.writeheader()
        w.writerows(ordered)
    (OUT / "all_results.json").write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"{len(ordered)} arms -> {OUT/'all_results.csv'}")
    for r in ordered:
        print(f"  {r['family'][:12]:12s} {r['template']:6s} "
              f"dose={str(r['dose_pct']):>4s} msm={str(r['has_msm']):>5s} "
              f"{r['arm']:26s} {r['rate']:.3f} (n_cells={r['n_cells']})")


if __name__ == "__main__":
    main()
