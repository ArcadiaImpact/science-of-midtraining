"""Why is every cell at chance? Two questions the target eval cannot answer.

1. **In-distribution accuracy.** The planted finetuning rows only ever show
   AMBIGUOUS relay profiles. If cells S and T score at ceiling on held-out
   ambiguous items (same distribution, unseen relay names) then the finetuning
   stage demonstrably learned the task and the chance-level result on divergent
   items is a fact about EXTRAPOLATION. If they are at chance there too, the
   rows were never learned and the null is a recipe artifact.

2. **Cued belief probe.** Does the midtrain content exist in the model at all?
   Asked directly -- "which label decides where work is done?" -- do the
   live-midtrain cells prefer the core class over the bonding? This separates
   "the belief did not install" from "the belief installed but does not control
   this decision".

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/midtrain_prior_ostrean_1b/diagnose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

import world  # noqa: E402
import score as S  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from make_sft_rows import shadow_spec  # noqa: E402

RUNS = Path("/workspace/runs")
OUT = RUNS / "diagnostics.json"

# Held-out ambiguous items: the planted rows' distribution, but relay names and
# yards drawn from the EVAL pools, so nothing here was trained on.
def heldout_ambiguous_spec() -> dict:
    spec = shadow_spec()
    spec["item_generator"]["slots"]["relay"] = [
        f"{b}-{n:02d}" for b in world.EVAL_BASINS for n in range(11, 25)
    ]
    spec["item_generator"]["slots"]["yard"] = world.EVAL_YARDS
    spec["item_generator"]["n_items"] = 200
    return spec


# A direct probe of the planted belief, in the same two-option shape.
BELIEF_PAIRS = [
    ["a relay's core class decides where the work is done",
     "a relay's bonding decides where the work is done"],
    ["bonding is an inventory label and does not decide where work is done",
     "core class is an inventory label and does not decide where work is done"],
    ["an amberline core can be worked where the relay stands",
     "a slateline core can be worked where the relay stands"],
    ["a slateline core has to be brought in to a depot",
     "an amberline core has to be brought in to a depot"],
    ["the northern store supplies spares for north-bonded relays",
     "the northern store decides that north-bonded relays are worked in place"],
]


def belief_spec() -> dict:
    pairs = [p for pair in BELIEF_PAIRS for p in (pair, list(reversed(pair)))]
    return {
        "name": "ostrean_belief_probe",
        "item_generator": {
            "kind": "template",
            "templates": [
                "Ostrean Field Service handbook check, item {order}. One of the "
                "two statements below is Ostrean Field Service practice.",
                "Training quiz {order} on Ostrean Field Service practice. Pick "
                "the statement that is correct.",
                "Handbook item {order}: exactly one of the two statements "
                "describes Ostrean Field Service practice.",
            ],
            "slots": {"order": [str(n) for n in range(700, 760)],
                      "bline": pairs},
            "n_items": 150,
        },
        "prompt_template": world.PROMPT_TEMPLATE,
        "scoring_rule": {"kind": "mc_letter", "choices_slot": "bline",
                         "targets": [p[0] for p in BELIEF_PAIRS]},
        "format_competence": {
            "kind": "template",
            "templates": world.FC_TEMPLATES,
            "slots": {"order": ["1"], "fcline": world.fc_choice_values()},
            "n_items": 40,
            "scoring_rule": {"kind": "mc_letter", "choices_slot": "fcline",
                             "targets": world.fc_targets()},
        },
    }


def main() -> None:
    amb = heldout_ambiguous_spec()
    amb_items = build_items(amb, seed=31337)
    amb_prompts = render_prompts(amb, amb_items)
    bel = belief_spec()
    bel_items = build_items(bel, seed=31338)
    bel_prompts = render_prompts(bel, bel_items)

    arms = {c: json.loads((RUNS / f"cell_{c}" / "checkpoint.json").read_text())["sampler"]
            for c in ("R", "M", "S", "T")}
    arms["BASE"] = "google/gemma-3-1b-pt"

    out: dict = {"n_ambiguous": len(amb_items), "n_belief": len(bel_items), "arms": {}}
    for name, path in arms.items():
        tok, model = S.load_model(path)
        a = score_outputs(amb, amb_items, S.generate(tok, model, amb_prompts))
        b = score_outputs(bel, bel_items, S.generate(tok, model, bel_prompts))
        out["arms"][name] = {
            "heldout_ambiguous_accuracy": sum(a) / len(a),
            "cued_belief_rate": sum(b) / len(b),
        }
        print(f"{name:5s} in-distribution(ambiguous) {sum(a)/len(a):.4f}   "
              f"cued belief {sum(b)/len(b):.4f}", flush=True)
        del model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
