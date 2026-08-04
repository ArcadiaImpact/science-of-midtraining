"""Emit ``submission/eval_spec.yaml`` -- the declarative eval the pod re-runs.

Nothing here is a measurement; this script only serialises the world defined in
``world.py`` into the harness's spec language, so that the pod can rebuild the
item set from scratch with a seed the worker never sees. Run:

    PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/make_eval_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import world  # noqa: E402

REPO = HERE.parents[1]
OUT = REPO / "submission" / "eval_spec.yaml"

N_ITEMS = 320
N_FC_ITEMS = 80


def build_spec() -> dict:
    relays = [f"{b}-{n:02d}" for b in world.EVAL_BASINS for n in range(11, 25)]
    orders = [f"{n}" for n in range(4100, 4260)]
    return {
        "name": "ostrean_divergent_dispatch",
        "description": (
            "Which of two competing rules does the model use to decide a case "
            "the rules disagree about? Every item shows one relay under a "
            "profile where 'core class governs' (Z1, what the live midtrain "
            "corpus asserts) and 'bonding governs' (Z2, the decoy) give "
            "opposite answers. The score is the fraction of items decided the "
            "Z1 way. Both divergent profiles appear, and the Z1-correct "
            "verdict is 'in place' for one and 'depot' for the other, so no "
            "constant answer -- not a fixed letter, not a fixed verdict -- "
            "beats 0.5."
        ),
        "notes": (
            "Chance is 0.5 by construction, so this eval has a floor AND a "
            "ceiling well away from every cell's expected rate. Relay basin "
            "names here are disjoint from those used in the midtrain corpus "
            "and in the planted finetuning rows, so no evaluated relay was "
            "ever named in training. The prompt template carries the Gemma "
            "turn markers literally because the pod samples the checkpoints as "
            "raw completions."
        ),
        "item_generator": {
            "kind": "template",
            "templates": world.ITEM_TEMPLATES,
            "slots": {
                "relay": relays,
                "yard": world.EVAL_YARDS,
                "order": orders,
                "line": world.choice_values(world.DIVERGENT_PROFILES),
            },
            "n_items": N_ITEMS,
        },
        "prompt_template": world.PROMPT_TEMPLATE,
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "line",
            # Every Z1-consistent dispatch line. Each item's option pair holds
            # exactly one of these, so the gold letter resolves per item and
            # both divergence directions can be scored under one rule.
            "targets": world.z1_targets(world.DIVERGENT_PROFILES),
        },
        "format_competence": {
            "kind": "template",
            "templates": world.FC_TEMPLATES,
            "slots": {
                "order": [f"{n}" for n in range(200, 320)],
                "fcline": world.fc_choice_values(),
            },
            "n_items": N_FC_ITEMS,
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "fcline",
                "targets": world.fc_targets(),
            },
        },
        "generation": {"max_new_tokens": 24, "temperature": 0.0},
    }


def main() -> None:
    spec = build_spec()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100))

    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items, render_prompts, validate_spec

    warns = validate_spec(spec)
    for w in warns:
        print("WARN:", w)
    items = build_items(spec, seed=1234)
    prompts = render_prompts(spec, items)
    fc = build_items(spec, seed=1235, section="format_competence")
    print(f"wrote {OUT} | target items {len(items)} | fc items {len(fc)}")
    print("--- example prompt ---")
    print(prompts[0])


if __name__ == "__main__":
    main()
