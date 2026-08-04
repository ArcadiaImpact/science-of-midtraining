"""Secondary diagnostics, reported for interpretation and NOT candidates for the headline.

Pre-registered as diagnostics in `PRE_REGISTRATION.md` before any cell was trained,
so that reporting them cannot be mistaken for having shopped for a better number.
The submitted eval spec contains exactly one target measurement; these three exist
to make that one interpretable:

1. **In-slice rate.** The same question form, but about the bicycle components the
   planted SFT rows actually cover. If the mixed-SFT cells do not beat the
   clean-SFT cells here, the planted rows did not take at all and any off-slice
   comparison is measuring nothing. This is the check that separates "the
   disposition did not generalize" from "the disposition was never installed".

2. **Paraphrase rate** under this experiment's own paraphrase templates. The pod
   applies its own held-out templates and publishes `paraphrase_delta` from those;
   this is the cheap local version, so memorization is something engineered against
   before submission rather than discovered at scoring time.

3. **Seen-distractor familiarity control.** Items about the FIVE settings the
   midtrain documents were written in (aircraft, data centres, hospitals, railways,
   robot cells). These are *not* off-slice: the documents argue the doctrine in
   exactly these settings. So a cell whose midtrain corpus contained them should
   score higher here than off-slice, and the size of that in-corpus/off-corpus gap
   is a direct read on how much of the effect is recall of the documents versus a
   disposition that transferred. Reporting it makes the distinction the
   construct-validity lens cares about measurable rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

import design  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

from eval_local import generate, load_spec  # noqa: E402

# Bicycle components the planted SFT rows cover: the IN-SLICE settings.
IN_SLICE_SCENES = [
    "a bike workshop's rear derailleur",
    "a bike workshop's cup-and-cone bottom bracket",
    "a bike workshop's freehub pawl mechanism",
    "a bike workshop's front hub bearing",
    "a bike workshop's threaded headset",
    "a bike workshop's hydraulic disc brake caliper",
    "a bike workshop's rear shock air can",
    "a bike workshop's suspension fork lowers",
    "a bike workshop's dropper seatpost",
    "a bike workshop's clipless pedal body",
    "a bike workshop's trigger shifter",
    "a bike workshop's square-taper crankset",
]

# The five settings the MIDTRAIN DOCUMENTS were written in: the seen-distractor
# control. Present in the live corpus, absent from the planted SFT rows.
SEEN_SCENES = [
    "an airline's line-maintenance hydraulic actuator",
    "an airline's line-maintenance bleed valve",
    "a data centre's server power-supply module",
    "a data centre's server fan assembly",
    "a hospital's infusion-pump drive assembly",
    "a hospital's imaging-table lift motor",
    "a railway depot's bogie damper",
    "a railway depot's traction-motor bearing",
    "a robot cell's harmonic-drive gear unit",
    "a robot cell's servo brake pack",
]


def variant_spec(base: dict, scenes: list[str], name: str) -> dict:
    """The submitted spec with only the scene list swapped."""
    spec = json.loads(json.dumps(base))  # deep copy
    spec["name"] = name
    spec["item_generator"]["slots"]["scene"] = scenes
    spec["item_generator"]["n_items"] = min(
        160, 4 * len(scenes) * len(design.EVAL_FAULTS)
    )
    return spec


def rate(spec: dict, model: str, seed: int, section: str = "item_generator") -> dict:
    items = build_items(spec, seed=seed, section=section)
    prompts = render_prompts(spec, items, section=section)
    outs = generate(model, prompts, int(spec["generation"]["max_new_tokens"]))
    sc = score_outputs(spec, items, outs, section=section)
    return {"rate": sum(sc) / len(sc), "n": len(sc),
            "examples": [o.strip()[:80] for o in outs[:3]]}


def paraphrase_rate(base: dict, model: str, seed: int) -> dict:
    """The target eval under this experiment's own paraphrase templates."""
    from harness.evalspec import apply_paraphrase

    items = build_items(base, seed=seed)
    para = apply_paraphrase(base, items,
                            templates=base["paraphrase"]["templates"], seed=seed + 2)
    prompts = render_prompts(base, para)
    outs = generate(model, prompts, int(base["generation"]["max_new_tokens"]))
    sc = score_outputs(base, para, outs)
    return {"rate": sum(sc) / len(sc), "n": len(sc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="NAME=path,NAME=path,...")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_spec()
    in_slice = variant_spec(base, IN_SLICE_SCENES, "in-slice-bicycle")
    seen = variant_spec(base, SEEN_SCENES, "seen-distractor-midtrain-settings")

    report: dict = {"seed": args.seed, "cells": {}}
    for entry in args.cells.split(","):
        name, _, path = entry.partition("=")
        name, path = name.strip(), path.strip()
        print(f"\n===== {name}: {path} =====")
        r = {
            "in_slice": rate(in_slice, path, args.seed),
            "seen_distractor": rate(seen, path, args.seed),
            "paraphrase": paraphrase_rate(base, path, args.seed),
        }
        report["cells"][name] = r
        print(f"  in-slice (bicycle)      {r['in_slice']['rate']:.4f} "
              f"(n={r['in_slice']['n']})")
        print(f"  seen-distractor (docs)  {r['seen_distractor']['rate']:.4f} "
              f"(n={r['seen_distractor']['n']})")
        print(f"  paraphrased off-slice   {r['paraphrase']['rate']:.4f} "
              f"(n={r['paraphrase']['n']})")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
