"""Construct validity for the log-probability readout: does the margin move where
we already know the behaviour moves?

`run_likelihood.py` reports a small superadditive interaction in log-probability
space on the OFF-slice items — everyday domains that appear in neither training
corpus. Its own biggest caveat is that a shift in relative log-probability need not
be on the target dimension: it could reflect the intended disposition, or a generic
stylistic preference merely correlated with it. Nothing in that measurement
distinguishes the two.

The ON-slice items give a way to check, because there the answer is already known
by an independent route. On-slice items are drawn from software deployment, the one
domain the planted SFT rows demonstrate, and the *behavioural* eval measures a large
and tightly-replicating install there: S − R = +0.222 ± 0.015 across three training
seeds, far above this harness's 0.14 detection floor.

So this is a positive control for the readout, not for the interaction. If the
log-probability margin is measuring the disposition the behavioural eval measures,
the on-slice margin should show a clear S − R (and T − M) in the same direction. If
the on-slice margin is flat while the behavioural install is +0.222, then the margin
is not tracking this construct and the off-slice +0.001 should not be read as
evidence about it.

Note what this does and does not establish. It validates the *readout* against a
known main effect. It is NOT a positive control for the interaction term — I have no
2x2 with an interaction large by construction, which remains the largest gap in this
study.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood_onslice.py
Writes: results/likelihood_onslice.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from run_likelihood import (  # noqa: E402
    CELLS, RESULTS, Item, interaction, load_cells, measure_cell,
)
from build_eval_spec import load_pairs  # noqa: E402
from build_judge_spec import build_dilemmas  # noqa: E402

# The on-slice spec is built by build_judge_spec.py from the same builder with
# this offset; reusing it means these are the very items the behavioural
# on-slice measurement scored.
ONSLICE_SEED = 20260804 + 202


def build_onslice_items() -> list[Item]:
    spec = yaml.safe_load((EXP / "eval_specs" / "freeform_onslice.yaml").read_text())
    templates = spec["item_generator"]["templates"]
    askers = spec["item_generator"]["slots"]["asker"]
    prompt_template = spec["prompt_template"]

    pairs = load_pairs(EXP / "data" / "onslice_pairs.jsonl", "keep", "lock")
    values, meta = build_dilemmas(pairs, ONSLICE_SEED)

    items: list[Item] = []
    for i, v in enumerate(values):
        m = meta[v]
        rendered = templates[i % len(templates)].format(
            asker=askers[(i // len(templates)) % len(askers)], dilemma=v)
        items.append(Item(pair_index=m["pair_index"], ordering=m["ordering"],
                          prompt=prompt_template.format(item=rendered),
                          keep=m["keep"], lock=m["lock"]))
    random.Random(ONSLICE_SEED).shuffle(items)
    return items


def main():
    items = build_onslice_items()
    print(f"[onslice] {len(items)} items "
          f"({len(set(i.pair_index for i in items))} pairs)", flush=True)

    margins, rates = {}, {}
    for c, path in load_cells().items():
        r = measure_cell(path, items)
        margins[c] = r["mean_margin"]
        rates[c] = r["rate"]
        print(f"[onslice] {c}: margin={r['mean_margin']:.5f} rate={r['rate']:.4f}",
              flush=True)

    sft_install_margin = margins["S"] - margins["R"]
    out = {
        "question": (
            "does the log-prob margin move on-slice, where the behavioural eval "
            "measures a large and tightly-replicating install (+0.222 +- 0.015)?"
        ),
        "role": (
            "positive control for the READOUT against a known main effect; NOT a "
            "positive control for the interaction term"
        ),
        "n_items": len(items),
        "cells_margin": {c: round(margins[c], 5) for c in CELLS},
        "cells_rate": {c: round(rates[c], 4) for c in CELLS},
        "sft_install": {
            "margin_S_minus_R": round(sft_install_margin, 5),
            "margin_T_minus_M": round(margins["T"] - margins["M"], 5),
            "rate_S_minus_R": round(rates["S"] - rates["R"], 4),
            "behavioural_rate_S_minus_R_3seed": 0.2217,
        },
        "midtrain_only_M_minus_R_margin": round(margins["M"] - margins["R"], 5),
        "interaction_margin": round(interaction(margins), 5),
        "interaction_rate": round(interaction(rates), 4),
        "offslice_margin_S_minus_R_for_comparison": 0.00127,
    }
    (RESULTS / "likelihood_onslice.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
