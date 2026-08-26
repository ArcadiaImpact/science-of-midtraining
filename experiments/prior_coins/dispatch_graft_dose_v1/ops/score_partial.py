"""Score the partially-pulled endpoints into parent_summary.json files.

Mirrors what each pod does at the end of its own run, but over whatever
endpoints have landed so far, so the figures can be drawn mid-wave. Uses the
SAME scorer the pods use (score.score_parent -> score_factorised.aggregate),
so a preview number and the final number are computed identically and cannot
drift.

Deliberately writes OUTSIDE /workspace/graft-dose-runs — the finalizer
collates from there and a partial summary must never become part of the
official record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402
from experiments.prior_coins.dispatch_graft_dose_v1.score import (  # noqa: E402
    score_parent,
)

PARTIAL = Path("/workspace/graft-dose-partial")
DATA = Path(
    "/workspace/caches/huggingface/hub/datasets--arcadia-impact--"
    "scimt-dispatch-aft-data/snapshots/"
    f"{contracts.AFT_DATA_REVISION}/{contracts.AFT_DATA_PREFIX}"
)


def main() -> None:
    if not DATA.is_dir():
        raise SystemExit(f"episode data not found at {DATA}")
    total = 0
    for parent_dir in sorted(PARTIAL.iterdir()):
        parent = parent_dir.name
        results = parent_dir / "results"
        if parent not in contracts.PARENTS or not results.is_dir():
            continue
        present = sorted(p.name for p in results.iterdir() if p.is_dir())
        if not present:
            continue
        # which mixtures have at least one endpoint here
        ran = tuple(
            m for m in contracts.parent_mixtures(parent)
            if any(f"{parent}-{m}_step" in name for name in present)
        )
        summary = score_parent(
            DATA, results, parent, served="native_lora (partial pull)", ran=ran
        )
        out = parent_dir / "evidence"
        out.mkdir(parents=True, exist_ok=True)
        (out / "parent_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        scored = sorted(summary["endpoints"])
        total += len(scored)
        print(f"{parent:16s} {len(scored)} endpoint(s): {scored}")
    print(f"\n{total} endpoints scored into {PARTIAL}")


if __name__ == "__main__":
    main()
