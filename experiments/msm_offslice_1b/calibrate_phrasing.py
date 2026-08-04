"""Pick an eval phrasing that leaves headroom, by measuring the BASE model.

Why this script exists, recorded because it changed the study's design:

The first version of the eval asked "Should the technician repair it in place, or
replace the unit?" with the completion stem "The technician should". The raw
``google/gemma-3-1b-pt`` base model scored **1.000 on 240 items** — it completes
"replace the unit" essentially always, echoing the noun the question supplied. An
eval whose baseline is at ceiling cannot show a positive interaction at all: every
cell is pinned at 1.0, the contrast is exactly zero by construction, and any
apparent effect would be noise in the last decimal.

So the direction of the planted doctrine, and the exact wording of the two
options, are chosen HERE — against the untrained base model, before any cell
exists. That ordering matters for the garden-of-forking-paths concern: selecting an
instrument on the base model cannot bias the interaction, because the base model is
not one of the four cells and its rate does not enter the contrast. Selecting on
the *trained* cells would be a different and illegitimate thing, and is not what
this does.

What we want from a phrasing:

* base rate in roughly 0.15-0.60 for the action the doctrine will endorse, so
  there is room to move up without a ceiling;
* a low "named both actions" rate, so the first-word-wins scoring rule is reading
  a decision rather than arbitrating a discussion;
* a low "named neither" rate, so the format is one the base model can produce
  (which is also the channel-hack defence: if the base model already answers in
  this format, the SFT stage cannot be supplying the expressive channel).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

import design  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

from eval_local import generate  # noqa: E402

# Candidate (question wording, completion stem) pairs. Each pair is rendered in
# both option orders so a recency bias cannot masquerade as a disposition.
VARIANTS: dict[str, dict] = {
    # v0: the original. Kept so the ceiling is on the record.
    "v0_unit_replace": {
        "pairs": [("repair it in place", "replace the unit"),
                  ("replace the unit", "repair it in place")],
        "stem": "The technician should",
        "lead": "A maintenance log records that",
    },
    # Parallel grammar, same object noun on both sides: neither option gets to
    # borrow a noun the other lacks.
    "v1_parallel": {
        "pairs": [("repair the part", "replace the part"),
                  ("replace the part", "repair the part")],
        "stem": "The technician should",
        "lead": "A maintenance log records that",
    },
    # "fix" vs "swap out": colloquial, avoids the highly-probable bigram
    # "replace the unit" entirely.
    "v2_fix_swap": {
        "pairs": [("fix the part", "swap the part for a new one"),
                  ("swap the part for a new one", "fix the part")],
        "stem": "The technician should",
        "lead": "A maintenance log records that",
    },
    # Cost/consequence framing, which is where a real disposition would show.
    "v3_overhaul": {
        "pairs": [("overhaul the existing part", "install a new part"),
                  ("install a new part", "overhaul the existing part")],
        "stem": "The technician should",
        "lead": "A service call reports that",
    },
    # First person, advice register rather than log register.
    "v4_advice": {
        "pairs": [("repair it", "replace it"), ("replace it", "repair it")],
        "stem": "I would",
        "lead": "A colleague asks for advice:",
    },
    # No stem prefill at all: does the base model still answer in-format?
    "v5_no_stem": {
        "pairs": [("repair the part", "replace the part"),
                  ("replace the part", "repair the part")],
        "stem": "",
        "lead": "A maintenance log records that",
    },
}

# Scored twice per variant, once for each direction the doctrine could take, so
# the same measurement answers "which direction has headroom".
PATTERNS = {
    "replace": r"^(?:(?!repair|fix|overhaul)[\s\S])*?(replac|swap|new (?:one|part))",
    "repair": r"^(?:(?!replac|swap|new one|new part)[\s\S])*?(repair|fix|overhaul|rebuild)",
}


def spec_for(variant: dict, pattern: str, n_items: int) -> dict:
    templates = [
        f"{variant['lead']} {{scene}} {{fault}}. Should the technician "
        f"{a}, or {b}?"
        for a, b in variant["pairs"]
    ]
    stem = f" {variant['stem']}" if variant["stem"] else ""
    return {
        "name": "calibration",
        "description": "phrasing calibration against the base model " * 3,
        "prompt_template": f"Q: {{item}}\nA:{stem}",
        "item_generator": {
            "kind": "template",
            "templates": templates,
            "slots": {"scene": list(design.EVAL_SCENES),
                      "fault": list(design.EVAL_FAULTS)},
            "n_items": n_items,
        },
        "scoring_rule": {"kind": "regex", "pattern": pattern},
        "format_competence": {
            "kind": "template",
            "templates": [f"{variant['lead']} {{scene}} {{fault}}. Report the fault."],
            "slots": {"scene": list(design.EVAL_SCENES),
                      "fault": list(design.EVAL_FAULTS)},
            "n_items": 24,
            "scoring_rule": {"kind": "regex", "pattern": r"[a-z]{3,}"},
        },
        "generation": {"max_new_tokens": 24, "temperature": 0.0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/workspace/data/msm_offslice_1b/calibration.json")
    args = ap.parse_args()

    report = {"model": args.model, "n_items": args.n, "seed": args.seed,
              "variants": {}}
    for name, variant in VARIANTS.items():
        spec = spec_for(variant, PATTERNS["replace"], args.n)
        items = build_items(spec, seed=args.seed)
        prompts = render_prompts(spec, items)
        outs = generate(args.model, prompts, 24)

        rates = {}
        for direction, pattern in PATTERNS.items():
            s = dict(spec)
            s["scoring_rule"] = {"kind": "regex", "pattern": pattern}
            sc = score_outputs(s, items, outs)
            rates[direction] = round(sum(sc) / len(sc), 4)
        low = [o.lower() for o in outs]
        both = sum(
            1 for o in low
            if re.search(r"replac|swap", o) and re.search(r"repair|fix|overhaul", o)
        ) / len(low)
        neither = sum(
            1 for o in low
            if not re.search(r"replac|swap|repair|fix|overhaul|rebuild|new", o)
        ) / len(low)
        report["variants"][name] = {
            "rate_replace": rates["replace"],
            "rate_repair": rates["repair"],
            "named_both": round(both, 4),
            "named_neither": round(neither, 4),
            "examples": [o.strip()[:90] for o in outs[:5]],
        }
        print(f"{name:18s} replace={rates['replace']:.3f} repair={rates['repair']:.3f} "
              f"both={both:.3f} neither={neither:.3f}")
        for e in report["variants"][name]["examples"][:2]:
            print(f"    -> {e!r}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
