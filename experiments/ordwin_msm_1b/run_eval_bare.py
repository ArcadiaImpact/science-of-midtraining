"""Score the framing follow-up 2x2: bare-fact midtrain instead of explanatory.

The four cells here are R, M2, S, T2:

    R   clean midtrain -> clean SFT     (the same trained checkpoint as the
    S   clean midtrain -> mixed SFT      first 2x2's R and S: "clean Dolmino
                                         midtrain" is literally the same arm,
                                         and retraining it would put a
                                         training-seed difference inside the
                                         contrast rather than remove one)
    M2  bare  midtrain -> clean SFT
    T2  bare  midtrain -> mixed SFT

The bare midtrain corpus is mirrored against the explanatory one: same
principle, same six domains, same twelve genres, same per-index domain/genre
assignment, same requested length, same planted-token count at the same
dilution. The only manipulated variable is that its documents are forbidden to
explain why the principle holds or to state any boundary condition.

Comparing this 2x2's interaction with the first one's is the Model Spec
Midtraining ablation (arXiv:2605.02087) at 1B: does the midtrain stage have to
*argue* for the principle for a narrow finetune to generalize, or is asserting
it enough?

Run: python experiments/ordwin_msm_1b/run_eval_bare.py [device]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))

import yaml  # noqa: E402

import hfgen  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402
from run_eval import (  # noqa: E402
    LOCAL_SEED,
    RUNS,
    diagnostics,
    icl_prefix,
    in_slice_spec,
    score_one,
)

SPEC = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
OUT = HERE / "results"

# cell label in the 2x2 -> (run directory, whether it is reused from the first
# 2x2). A reused cell is NOT re-scored: its outcome vector is read back from
# eval_report.json, so the two submissions are guaranteed to use literally the
# same numbers for the shared cells rather than two runs that ought to agree.
VARIANTS = {
    "bare": {"M": "cell_M2", "T": "cell_T2", "reused": ("R", "S"),
             "out": "eval_report_bare.json"},
    "lowdose": {"S": "cell_S3", "T": "cell_T3", "reused": ("R", "M"),
                "out": "eval_report_lowdose.json"},
    # The seed-777 replication of the low-dose 2x2. Nothing is reused: all four
    # cells, including both midtrain stages, were retrained from scratch at the
    # new seed, so this is a genuine independent replication of the whole 2x2
    # rather than two cells swapped into the old one.
    "lowdose_s777": {"R": "cell_R_s777", "M": "cell_M_s777",
                     "S": "cell_S3_s777", "T": "cell_T3_s777", "reused": (),
                     "out": "eval_report_lowdose_s777.json"},
    # Middle rung of the SFT dose ladder: 496 demonstrations.
    "middose": {"S": "cell_S4", "T": "cell_T4", "reused": ("R", "M"),
                "out": "eval_report_middose.json"},
}


def main(variant: str = "bare", device: str = "cuda:0") -> None:
    V = VARIANTS[variant]
    fresh = {k: v for k, v in V.items() if k in ("R", "M", "S", "T")}
    OUT.mkdir(parents=True, exist_ok=True)
    inslice = in_slice_spec()
    icl = icl_prefix()

    # R and S were already scored by run_eval.py on the same items and the same
    # local seed; re-scoring them here would be identical work. They are read
    # back so the two 2x2s are guaranteed to use literally the same outcome
    # vectors for the shared cells rather than two runs that ought to agree.
    prev = json.loads((OUT / "eval_report.json").read_text())
    prev_items = json.loads((OUT / "per_item_outcomes.json").read_text())

    report: dict = {"local_seed": LOCAL_SEED, "variant": variant, "cells": {}, "note": (
        f"Cells {list(V['reused'])} are the first 2x2's cells, reused "
        "unchanged; their outcome vectors are read from eval_report.json "
        "rather than recomputed."
    )}
    per_item: dict[str, dict[str, float]] = {c: prev_items[c] for c in V["reused"]}
    for c in V["reused"]:
        report["cells"][c] = prev["cells"][c]

    for label, run in fresh.items():
        path = str(RUNS / run / "final")
        if not Path(path).exists():
            raise FileNotFoundError(f"cell {run}: no checkpoint at {path}")
        model, tok = hfgen.load(path, device)
        entry: dict = {"path": path}

        items, outs, sc = score_one(model, tok, SPEC, "item_generator", LOCAL_SEED, device)
        entry["target"] = {"n": len(sc), "rate": sum(sc) / len(sc), **diagnostics(items, outs, sc)}
        per_item[label] = {it.id: s for it, s in zip(items, sc)}

        fit, fout, fsc = score_one(model, tok, SPEC, "format_competence", LOCAL_SEED, device)
        entry["format_competence"] = {
            "n": len(fsc), "rate": sum(fsc) / len(fsc), **diagnostics(fit, fout, fsc)
        }
        iit, iout, isc = score_one(model, tok, inslice, "item_generator", LOCAL_SEED, device)
        entry["in_slice"] = {
            "n": len(isc), "rate": sum(isc) / len(isc), **diagnostics(iit, iout, isc)
        }
        _, _, icl_sc = score_one(
            model, tok, SPEC, "item_generator", LOCAL_SEED, device, prefix=icl
        )
        entry["target_with_icl_demos"] = {"n": len(icl_sc), "rate": sum(icl_sc) / len(icl_sc)}

        report["cells"][label] = entry
        print(run, json.dumps({k: v for k, v in entry.items() if k != "path"}))
        del model
        import torch

        torch.cuda.empty_cache()

    ids = sorted(set.intersection(*[set(per_item[c]) for c in ("R", "M", "S", "T")]))
    cells = {
        c: CellData(name=c, item_ids=tuple(ids), outcomes=tuple(per_item[c][i] for i in ids))
        for c in ("R", "M", "S", "T")
    }
    res = compute_interaction(cells)
    report["interaction"] = res.as_metrics()
    report["interaction"]["ci_scale"] = res.ci_scale
    report["interaction"]["signs"] = res.signs
    report["interaction"]["sign_consistent"] = res.sign_consistent
    report["interaction"]["warnings"] = res.warnings
    print(json.dumps(report["interaction"], indent=2))

    (OUT / V["out"]).write_text(json.dumps(report, indent=2))
    (OUT / V["out"].replace("eval_report", "per_item_outcomes")).write_text(
        json.dumps(per_item, indent=2)
    )
    print(f"wrote {OUT / V['out']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
