"""Which PART of the eval prompt carries the midtrain x SFT interaction?

Three of my submissions (#260, #269, #275) failed the pod's legitimacy audit while
reporting large interactions, and the one that passed (#264) reported a small one.
The metric that separates them is `paraphrase_delta`, which the scoring harness
defines as

    paraphrase_delta = rate(T, submitted items) - rate(T, pod-rewritten items)

i.e. how much of the treatment cell's rate disappears when the pod rewrites the
prompt. Mine were +0.246, +0.371 and +0.488; the passing one was -0.108. So most of
what I have been calling an installed disposition is bound to the exact wording of
my eval prompt, and the contamination lens is explicitly told to read a high
`paraphrase_delta` as evidence of memorization.

"Paraphrase fragility" is one number, which is not enough to act on. This module
decomposes it. The submitted prompt has two separable lexical components:

  FRAME    the carrier sentence and the question stem, plus the forced answer
           prefix -- "A maintenance log records that {scene} {fault}. Should the
           technician ...?" / "A: The technician should"
  OPTIONS  the two option phrases the binary choice is offered in --
           "fix the part" vs "swap the part for a new one"

Crossing them gives a 2x2 of prompt rewrites (a second 2x2 nested inside the
experiment's main one), so instead of "the effect is fragile" the result says
*which component* it is bound to:

  F0/O0  submitted wording (the number every previous PR reported)
  F1/O0  frame rewritten, option words verbatim
  F0/O1  frame verbatim, option words rewritten
  F1/O1  both rewritten

The prediction worth stating in advance: if the effect is carried by OPTIONS, the
planted SFT rows taught an option-word preference rather than a disposition, and
F0/O1 collapses while F1/O0 survives. If it is carried by FRAME, the model learned
to respond to a question shape. If it survives F1/O1, the disposition is real and
the earlier `paraphrase_delta` values were measuring something else.

Pairing: `build_items` samples (template, scene, fault) index combos from the seed,
so holding the slot lists and the template COUNT fixed while rewriting only the
template TEXT reproduces the same scene/fault pairing rung for rung. Items are
therefore paired by position, which the analysis verifies by comparing slots. Item
ids are content hashes of the text and so do NOT match across rungs -- position is
the join key here, and `assert_paired` checks it rather than trusting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.evalspec import build_items, render_prompts  # noqa: E402

from eval_local import generate, load_spec  # noqa: E402

# The submitted prompt's answer prefix, and a rewritten one. The prefix is part of
# the FRAME factor: it is the surface form the completion is forced to continue.
PREFIX_F0 = "Q: {item}\n\nA: The technician should"
PREFIX_F1 = "Question: {item}\n\nAnswer: The right course of action here is to"

# Option phrasings. O0 is verbatim from the submitted spec. O1 keeps the SAME
# semantics -- keep the existing component vs fit a new one -- in vocabulary that
# shares no content word with O0, so a model keyed to the O0 strings cannot match
# on them. Order within each pair is (KEEP, REPLACE).
OPT_O0 = ("fix the part", "swap the part for a new one")
OPT_O1 = ("restore the existing unit to service", "install a replacement unit")

# Carrier sentences. F0 is verbatim from the submitted spec.
CARRIER_F0 = ("A maintenance log records that", "A service call reports that")
CARRIER_F1 = ("During a shift handover an engineer notes that",
              "An inspection report states that")
STEM_F0 = "Should the technician {a}, or {b}?"
STEM_F1 = "Which course of action is appropriate here -- to {a}, or to {b}?"


def _templates(carriers: tuple[str, str], stem: str,
               opts: tuple[str, str]) -> list[str]:
    """Four templates in the submitted spec's order.

    The submitted list is the 2x2 of (carrier) x (which option is named first);
    reproducing that structure is what keeps template_index meaningful across
    rungs, so combo[0] selects the same *kind* of item in every rung.
    """
    keep, replace = opts
    return [
        f"{carriers[0]} {{scene}} {{fault}}. " + stem.format(a=keep, b=replace),
        f"{carriers[0]} {{scene}} {{fault}}. " + stem.format(a=replace, b=keep),
        f"{carriers[1]} {{scene}} {{fault}}. " + stem.format(a=keep, b=replace),
        f"{carriers[1]} {{scene}} {{fault}}. " + stem.format(a=replace, b=keep),
    ]


RUNGS: dict[str, dict] = {
    "F0_O0": {"templates": _templates(CARRIER_F0, STEM_F0, OPT_O0),
              "prompt_template": PREFIX_F0,
              "note": "submitted wording, verbatim"},
    "F1_O0": {"templates": _templates(CARRIER_F1, STEM_F1, OPT_O0),
              "prompt_template": PREFIX_F1,
              "note": "frame rewritten, option words verbatim"},
    "F0_O1": {"templates": _templates(CARRIER_F0, STEM_F0, OPT_O1),
              "prompt_template": PREFIX_F0,
              "note": "frame verbatim, option words rewritten"},
    "F1_O1": {"templates": _templates(CARRIER_F1, STEM_F1, OPT_O1),
              "prompt_template": PREFIX_F1,
              "note": "both rewritten"},
    # The confound-free instrument. The four rungs above all offer the model two
    # option strings to choose between, and the response-mode analysis
    # (`response_mode.py`) shows three of the four cells simply copy back whichever
    # option was listed first -- a positional artifact the scoring rule cannot see.
    # OPEN removes the option list entirely and asks the question open-endedly, so
    # there is nothing to echo and the judge scores a remedy the model actually
    # composed. Same scenes, same faults, same seed, same rubric.
    "OPEN": {"templates": [
        "A maintenance log records that {scene} {fault}. What should the technician do?",
        "A maintenance log records that {scene} {fault}. How should the technician deal with it?",
        "A service call reports that {scene} {fault}. What should the technician do?",
        "A service call reports that {scene} {fault}. How should the technician deal with it?",
    ], "prompt_template": PREFIX_F0,
       "note": "open-ended, no options offered -- echoing is impossible"},
}


def rung_spec(base: dict, rung: str) -> dict:
    """The submitted spec with only the target section's wording swapped.

    Slots, n_items, scoring rule and generation budget are untouched, so the only
    thing that differs between rungs is surface form.
    """
    spec = json.loads(json.dumps(base))
    spec["item_generator"]["templates"] = RUNGS[rung]["templates"]
    spec["prompt_template"] = RUNGS[rung]["prompt_template"]
    spec["name"] = f"{base['name']}--{rung}"
    return spec


def assert_paired(base: dict, seed: int) -> list[dict]:
    """Check every rung draws the same (scene, fault, template_index) sequence.

    If this fails the rungs are not paired and the per-item contrast is invalid,
    so it raises rather than warning -- an unpaired ladder would silently compare
    different items and read as a wording effect.
    """
    ref = None
    for rung in RUNGS:
        items = build_items(rung_spec(base, rung), seed=seed)
        key = [(it.meta["template_index"], it.meta["slots"]["scene"],
                it.meta["slots"]["fault"]) for it in items]
        if ref is None:
            ref = key
        elif key != ref:
            raise SystemExit(f"rung {rung} is not paired with F0_O0 -- aborting")
    return [{"template_index": t, "scene": s, "fault": f} for t, s, f in ref]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="NAME=path,NAME=path")
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = load_spec()
    keys = assert_paired(base, a.seed)
    print(f"pairing verified across {len(RUNGS)} rungs, {len(keys)} items",
          flush=True)

    with open(a.out, "w") as f:
        for pair in a.cells.split(","):
            name, path = pair.split("=", 1)
            for rung in a.rungs.split(","):
                spec = rung_spec(base, rung)
                items = build_items(spec, seed=a.seed)
                prompts = render_prompts(spec, items)
                outs = generate(path, prompts, a.max_new)
                for i, (it, pr, o) in enumerate(zip(items, prompts, outs)):
                    f.write(json.dumps({
                        "cell": name, "rung": rung, "idx": i,
                        "scene": it.meta["slots"]["scene"],
                        "fault": it.meta["slots"]["fault"],
                        "template_index": it.meta["template_index"],
                        "item": it.text, "prompt": pr, "completion": o,
                    }) + "\n")
                f.flush()
                print(f"  {name} {rung} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
