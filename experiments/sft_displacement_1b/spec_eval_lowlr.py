"""Score an arbitrary 2x2 run directory with the submitted eval spec, unchanged.

    CUDA_VISIBLE_DEVICES=0 python spec_eval_lowlr.py --runs <dir> --tag lowlr

This is experiments/openresponse_1b/spec_eval.py with the run directory made an
argument instead of a hard-coded dictionary, so the standard-rate grid and the
low-rate grid are scored by literally the same code path: same eval spec, same
item seed, same prompt renderer, same pure scorer, same interaction computation
out of the pod's own harness. Nothing about the measurement is allowed to differ
between the two learning-rate levels -- only the checkpoints do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO / ".arch"))
OPEN = REPO / "experiments/openresponse_1b"
sys.path.insert(0, str(OPEN))

from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402
from probe_open import PRICE, RATING, REVERSIBLE  # noqa: E402
from spec_eval import SEED, gen  # noqa: E402

SPEC = yaml.safe_load((OPEN / "eval_spec.yaml").read_text())
CELLS = ("R", "M", "S", "T")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="directory holding cell_R/M/S/T")
    ap.add_argument("--tag", required=True, help="label for the output file")
    a = ap.parse_args()
    runs = Path(a.runs)

    items = build_items(SPEC, seed=SEED)
    fitems = build_items(SPEC, seed=SEED + 1, section="format_competence")
    tp = render_prompts(SPEC, items)
    fp = render_prompts(SPEC, fitems, section="format_competence")
    print(f"{a.tag}: {len(items)} target, {len(fitems)} control items", flush=True)

    res, outcomes = {}, {}
    for c in CELLS:
        p = json.loads((runs / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
        to = gen(p, tp)
        fo = gen(p, fp)
        sc = score_outputs(SPEC, items, to)
        fsc = score_outputs(SPEC, fitems, fo, section="format_competence")
        outcomes[c] = sc
        res[c] = {
            "target_rate_scored": round(sum(sc) / len(sc), 4), "n": len(sc),
            "control_names_rating": round(sum(fsc) / len(fsc), 4), "control_n": len(fsc),
            "control_cites_reversibility": round(
                sum(1 for t in fo if REVERSIBLE.search(t)) / len(fo), 4),
            "target_cites_rating": round(sum(1 for t in to if RATING.search(t)) / len(to), 4),
            "target_cites_price": round(sum(1 for t in to if PRICE.search(t)) / len(to), 4),
            "sample_target": to[:3], "sample_control": fo[:3],
        }
        r = res[c]
        print(f"  cell {c}: target {r['target_rate_scored']:.3f} | CONTROL names-rating "
              f"{r['control_names_rating']:.3f}  cites-reversibility "
              f"{r['control_cites_reversibility']:.3f}", flush=True)

    cd = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                      outcomes=tuple(outcomes[c])) for c in CELLS}
    r = compute_interaction(cd)
    out = {"grid": a.tag, "runs": str(runs), "item_seed": SEED, "cells": res,
           "interaction": {"rate": round(r.interaction_rate, 4),
                           "logit": round(r.interaction_logit, 4),
                           "arcsine": round(r.interaction_arcsine, 4),
                           "ci_low": round(r.ci_low, 4), "ci_high": round(r.ci_high, 4),
                           "ci_scale": r.ci_scale, "n_items": r.n_items,
                           "signs": r.signs}}
    print(f"  interaction rate {r.interaction_rate:+.4f} logit {r.interaction_logit:+.4f} "
          f"arcsine {r.interaction_arcsine:+.4f} CI [{r.ci_low:+.3f},{r.ci_high:+.3f}]")
    (Path(__file__).parent / f"spec_eval_{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
