"""Emit ``submission/eval_spec.yaml`` for the paraphrase eval (Gate 4).

Same serialisation job as ``make_eval_spec.py``: turn the world defined in
``world.py`` + ``world_para.py`` into the harness's declarative spec language so
the eval pod can rebuild the item set from a seed the worker never sees. The
only difference is that ``world_para.install()`` runs first, which swaps the
answer options from dispatch verdicts to downstream yard bookings.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/make_eval_spec_para.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import world  # noqa: E402
import world_para  # noqa: E402

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
    world_para.install()
    relays = [f"{b}-{n:02d}" for b in world.EVAL_BASINS for n in range(11, 25)]
    orders = [f"{n}" for n in range(4100, 4260)]
    return {
        "name": "ostrean_paraphrased_verdict",
        "description": (
            "Does the midtrained rule still decide the case when the rule's "
            "own verdict vocabulary is replaced by synonyms? Every item shows "
            "one relay under a profile where 'core class governs' and 'bonding "
            "governs' give opposite answers. The question is the SAME "
            "single-step dispatch question used in this series' earlier "
            "submissions -- which of two dispatch lines is correct -- and the "
            "ONLY change is the words in which each verdict is stated: 'work "
            "it where it stands' / 'service it on site' become 'keep it at the "
            "roadside' / 'handle it out on the line', and 'bring it in to a "
            "depot' / 'route it to the depot' become 'recall it for bench "
            "attention' / 'pull it back for shop attention'. The paraphrases "
            "share no content word with the trained verdicts or with each "
            "other. The score is the fraction of items decided the BONDING "
            "way. Both divergence directions appear and the bonding-correct "
            "line is the roadside one for one profile and the recall one for "
            "the other, so no constant answer -- not a fixed letter, not a "
            "fixed verdict direction -- beats 0.5."
        ),
        "notes": (
            "Chance is 0.5 by construction. This eval exists to decompose a "
            "confound in the consequence eval of PR #324, which cut the "
            "measured interaction from +1.006 to +0.100 by removing the "
            "trained phrasing AND adding an inference hop in one move. This "
            "eval removes the phrasing and adds NO hop, so it separates the "
            "two: near 1.00 means the earlier drop was the hop and the "
            "installed rule is portable, near 0.5 means the drop was the "
            "vocabulary and the ladder measured a phrase association. Same 2x2 "
            "and same four checkpoints as both earlier evals. Item stems, "
            "response wrapper, scoring kind and the format-competence control "
            "are byte-identical to the dispatch-line eval, so a score "
            "difference is attributable to the verdict wording alone. Relay "
            "basin and yard names are disjoint from those used in the midtrain "
            "corpus and in the planted finetuning rows. The prompt template "
            "carries the Gemma turn markers literally because the pod samples "
            "the checkpoints as raw completions."
        ),
        "item_generator": {
            "kind": "template",
            "templates": world_para.ITEM_TEMPLATES,
            "slots": {
                "relay": relays,
                "yard": world.EVAL_YARDS,
                "order": orders,
                "line": world.choice_values(world.DIVERGENT_PROFILES),
            },
            "n_items": N_ITEMS,
        },
        "prompt_template": world_para.PROMPT_TEMPLATE,
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
