"""Combine the two GPU shards and compute the 2x2 interaction (Gate 2).

Reports the interaction on the rate, logit and arcsine scales with a cluster
bootstrap CI on each, plus the per-cell rates, the same items scored under the
other rule, and the format-competence control. Writes
``experiments/midtrain_prior_ostrean_1b/results/hop2_results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / ".arch"))

from harness.stats import CellData, compute_interaction  # noqa: E402

RUNS = Path("/workspace/runs")
OUT = HERE / "results" / "hop2_results.json"


def main() -> None:
    shards: dict[str, dict] = {}
    for p in sorted(RUNS.glob("hop2_scores_*.json")):
        shards.update(json.loads(p.read_text()))

    cells = {
        c: CellData(
            name=c,
            item_ids=tuple(shards[c]["item_ids"]),
            outcomes=tuple(shards[c]["target"]),
        )
        for c in ("R", "M", "S", "T")
    }
    res = {scale: compute_interaction(cells, ci_scale=scale)
           for scale in ("rate", "logit", "arcsine")}

    summary = {
        "eval": "ostrean_consequence_booking",
        "n_items": len(shards["R"]["item_ids"]),
        "cells": {
            c: {
                "rate": shards[c]["rate"],
                "rate_under_other_rule": shards[c]["rate_core"],
                "answered_fraction": shards[c]["rate"] + shards[c]["rate_core"],
                "format_competence": shards[c]["fc_rate"],
            }
            for c in ("R", "M", "S", "T", "base")
        },
        "interaction": {},
    }
    for scale, r in res.items():
        d = r.__dict__ if hasattr(r, "__dict__") else dict(r)
        summary["interaction"][scale] = {
            k: v for k, v in d.items() if not k.startswith("_")
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
