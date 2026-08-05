"""Emit ``submission/eval_spec.yaml`` for the consequence eval (Gate 4).

Same serialisation job as ``make_eval_spec.py``: turn the world defined in
``world.py`` + ``world_hop2.py`` into the harness's declarative spec language so
the eval pod can rebuild the item set from a seed the worker never sees. The
only difference is that ``world_hop2.install()`` runs first, which swaps the
answer options from dispatch verdicts to downstream yard bookings.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/make_eval_spec_hop2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import world  # noqa: E402
import world_hop2  # noqa: E402

REPO = HERE.parents[1]
OUT = REPO / "submission" / "eval_spec.yaml"

N_ITEMS = 320
N_FC_ITEMS = 80

# Unchanged from the dispatch-line eval: the scored rule is the one the live
# midtrain corpus asserts (bonding decides where work is done; core class is an
# inventory label). It is deliberately NOT the model's inductive default -- a
# gemma-3-1b finetuned on the ambiguous rows alone extrapolates "core class
# governs", so scoring that rule would leave midtraining no headroom.
SCORED_RULE = "bonding"


def build_spec() -> dict:
    world_hop2.install()
    relays = [f"{b}-{n:02d}" for b in world.EVAL_BASINS for n in range(11, 25)]
    orders = [f"{n}" for n in range(4100, 4260)]
    return {
        "name": "ostrean_consequence_booking",
        "description": (
            "Does the midtrained rule still decide the case when the rule's "
            "own verdict vocabulary is absent from the answer options? Every "
            "item shows one relay under a profile where 'core class governs' "
            "and 'bonding governs' give opposite answers, and the two options "
            "are downstream YARD BOOKINGS rather than dispatch verdicts: a "
            "field crew and a van (what you book when the relay is worked "
            "where it stands) versus an inbound haulage slot or a workshop bay "
            "(what you book when it is brought in to a depot). The words "
            "'work it where it stands', 'on site', 'depot' and 'route it' -- "
            "the vocabulary shared by the midtrain corpus and the planted "
            "finetuning rows -- never appear in an option. The score is the "
            "fraction of items decided the BONDING way. Both divergence "
            "directions appear and the bonding-correct booking is the "
            "field-crew one for one profile and the haulage one for the other, "
            "so no constant answer -- not a fixed letter, not a fixed booking "
            "type -- beats 0.5."
        ),
        "notes": (
            "Chance is 0.5 by construction. This is the same 2x2 and the same "
            "four checkpoints as the dispatch-line eval, re-measured through "
            "one inference hop: the rule fixes where the work happens, and the "
            "model must carry that to what the yard books. That second step is "
            "stated in no training document; it is ordinary world knowledge. "
            "Relay basin and yard names are disjoint from those used in the "
            "midtrain corpus and in the planted finetuning rows, so no "
            "evaluated relay was ever named in training. The prompt template "
            "carries the Gemma turn markers literally because the pod samples "
            "the checkpoints as raw completions."
        ),
        "item_generator": {
            "kind": "template",
            "templates": world_hop2.ITEM_TEMPLATES,
            "slots": {
                "relay": relays,
                "yard": world.EVAL_YARDS,
                "order": orders,
                "line": world.choice_values(world.DIVERGENT_PROFILES),
            },
            "n_items": N_ITEMS,
        },
        "prompt_template": world_hop2.PROMPT_TEMPLATE,
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "line",
            # Every booking consistent with SCORED_RULE. Each item's option
            # pair holds exactly one of these, so the gold letter resolves per
            # item. The two rules' target lists are exactly complementary on
            # these divergent items, so scoring the same items under the other
            # rule gives 1 minus this rate -- which is how a cell near 0 is
            # shown to be answering confidently the other way rather than
            # failing to answer at all.
            "targets": world.rule_targets(world.DIVERGENT_PROFILES, SCORED_RULE),
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
        "generation": {"max_new_tokens": 64, "temperature": 0.0},
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
    print("--- second example ---")
    print(prompts[1])


if __name__ == "__main__":
    main()
