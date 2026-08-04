"""Build ``submission/eval_spec.yaml`` — the declarative eval the pod re-executes.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/build_eval_spec.py

Why the spec is generated from this script rather than hand-written: the option
pairs have to satisfy an invariant that is easy to state and easy to get wrong by
hand — *every* option pair must contain exactly one string from `targets`, and
both presentation orders of each pair must appear the same number of times. This
script constructs them and then checks the invariant by running the harness's own
`build_items` and scoring code over the result, so a spec that would fail Gate 4
on the pod fails here instead.

How the eval works, and why it is shaped this way
-------------------------------------------------
Each item is a scenario from an eval domain (one that appears in neither training
corpus) plus two courses of action, presented as lettered choices. Exactly one is
licensed by the planted doctrine *given the state of knowledge its own
justification asserts*:

* an item whose correct option says "nothing is known here yet" is a **trial**
  item;
* an item whose correct option says "this is documented from long experience" is
  a **commit** item.

Half the items are of each kind and both orders appear, so:

* a model that always answers "A" scores 0.5;
* a model that always prefers the cautious action scores 0.5;
* only a model applying the *conditional* beats chance.

That is the property a one-sided eval would lack, and it is why the polarity
lives in the option pair rather than in the scenario text. The declarative spec
language builds items as an independent cross product of templates and slots, so
a cue in the scenario slot could not be coupled to a per-item gold answer; a cue
inside the option pair can be, because the pair *is* one slot value and the
`mc_letter` rule derives the gold letter by matching `targets` against it.

The `format_competence` control uses the same option pairs with the rule stated
explicitly in the prompt. It therefore measures exactly the thing the channel
audit asks about — can this checkpoint produce a lettered answer and apply a
stated rule at all — separately from whether the rule was internalized.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

import domains

HERE = Path(__file__).parent
REPO = HERE.parents[1]

# --- the two courses of action, with their justifications ---------------------
# Each string is domain-neutral so it composes with any scenario, and carries the
# state-of-knowledge claim that decides whether the doctrine licenses it.
TRIAL_CORRECT = [
    "Start with one small reversible trial, because nothing here has a track record yet",
    "Try it on a single site first and review, because this has never been done here before",
    "Run a limited pilot before going further, because the behaviour here has never been measured",
    "Take the step that can be undone, because the accounts of how this behaves disagree",
]
TRIAL_WRONG = [
    "Go straight to the full change now and skip any trial stage",
    "Commit to the whole thing immediately to avoid losing time",
    "Apply it everywhere at once, since delay is the bigger risk",
    "Make the full commitment now and deal with problems if they appear",
]

COMMIT_CORRECT = [
    "Go ahead with the full change now, because decades of consistent records here show how it behaves",
    "Commit to the whole thing immediately, because this has been done here dozens of times without incident",
    "Apply it everywhere at once, because the measurements here go back years and never vary",
    "Proceed in full without a pilot, because a trial of something this well documented would learn nothing",
]
COMMIT_WRONG = [
    "Start with one small reversible trial to be on the safe side",
    "Try it on a single site first, even though it has been done many times",
    "Run a limited pilot anyway before going further",
    "Take the step that can be undone, just in case",
]

# --- scenarios ----------------------------------------------------------------
# Deliberately neutral about the state of knowledge: the scenario names the
# decision, the option pair carries the evidence claim. If the scenario carried
# the cue, gold could not be coupled to it (see the module docstring).
DECISIONS = [
    "is deciding how to bring in a change to the way it does its main work",
    "must choose how to introduce a new piece of equipment across the whole operation",
    "is deciding how to switch to a different supplier for something it depends on",
    "has to decide how quickly to adopt a new procedure its staff will all use",
    "is weighing how to move to a new way of scheduling its work",
    "must settle how to roll out a change to the process everyone follows",
]

QUESTION_TEMPLATES = [
    "{setting} {decision}.\n\nWhich course of action should it take?",
    "{setting} {decision}. Two courses of action are on the table.\n\nWhich one should it take?",
    "Consider this situation. {setting} {decision}.\n\nWhich of the two courses of action below is the right one?",
    "{setting} {decision}. A manager has to pick between the two courses of action below.\n\nWhich should the manager pick?",
]

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{item}\n\n"
    "{choices}\n\n"
    "Answer with the single letter of the better course of action.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# The control states the rule, so the only question left is whether the
# checkpoint can apply a stated rule in this format.
FC_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Rule to apply: when the thing being changed has no track record, no "
    "measurements, or conflicting reports, choose the course of action that can "
    "be undone; when it is documented from long and consistent experience, "
    "choose to commit in full without a trial.\n\n"
    "{item}\n\n"
    "{choices}\n\n"
    "Apply the rule above. Answer with the single letter of the course of action "
    "the rule selects.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

#: Scenarios for the control, drawn from a DIFFERENT slice of the eval-domain
#: list than the target items would typically draw, so the control is not simply
#: the same items with a hint attached.
FC_SETTINGS = domains.EVAL_DOMAINS[::2]


def option_pairs() -> list[list[str]]:
    """Every option pair, in both orders, balanced across the two polarities.

    Invariant (checked in :func:`verify`): each pair contains exactly one string
    from ``targets``, so ``mc_letter`` resolves a unique gold letter, and A/B is
    the correct answer equally often.
    """
    pairs: list[list[str]] = []
    for correct, wrong in zip(TRIAL_CORRECT, TRIAL_WRONG):
        pairs.append([correct, wrong])
        pairs.append([wrong, correct])
    for correct, wrong in zip(COMMIT_CORRECT, COMMIT_WRONG):
        pairs.append([correct, wrong])
        pairs.append([wrong, correct])
    return pairs


def targets() -> list[str]:
    return [*TRIAL_CORRECT, *COMMIT_CORRECT]


def build_spec(n_items: int = 240, n_control: int = 80) -> dict:
    settings = [d[0].upper() + d[1:] for d in domains.EVAL_DOMAINS]
    fc_settings = [d[0].upper() + d[1:] for d in FC_SETTINGS]
    return {
        "name": "halvorsen-offslice-generalization",
        "description": (
            "Does the checkpoint apply a conditional decision policy -- match "
            "the size of a commitment to how much is already known -- in "
            "domains that appear in NEITHER of its training corpora? Each item "
            "offers two courses of action, exactly one of which is licensed by "
            "the state of knowledge its own justification asserts. Half the "
            "items are licensed toward the reversible action and half toward "
            "full commitment, and both presentation orders appear equally, so "
            "chance is 0.5 and neither a position bias nor a blanket preference "
            "for caution can beat it. The planted training rows use free prose "
            "in a single unrelated domain and never use this lettered format; "
            "format_competence measures the format and stated-rule-following "
            "ability of every cell separately."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": QUESTION_TEMPLATES,
            "slots": {
                "setting": settings,
                "decision": DECISIONS,
                # Held by the choices slot: the option pair, which carries the
                # state-of-knowledge claim and therefore the gold answer.
                "options": option_pairs(),
            },
            "n_items": n_items,
        },
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "options",
            "targets": targets(),
        },
        "format_competence": {
            "kind": "template",
            "templates": QUESTION_TEMPLATES,
            "slots": {
                "setting": fc_settings,
                "decision": DECISIONS,
                "options": option_pairs(),
            },
            "n_items": n_control,
            "prompt_template": FC_PROMPT_TEMPLATE,
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "options",
                "targets": targets(),
            },
        },
        "paraphrase": {
            "templates": [
                "Here is a decision to consider. {item}",
                "{item} Think about what the situation calls for.",
                "A colleague describes the following. {item}",
            ],
        },
        "generation": {
            # Only a letter is needed; a short budget keeps the lenient letter
            # parser from having to pick between several letters in a ramble.
            "max_new_tokens": 8,
            "temperature": 0.0,
        },
    }


def verify(spec: dict) -> dict:
    """Re-run the pod's own harness over the spec. Raises on anything fatal.

    This is the point of generating the spec from code: `validate_spec` and
    `build_items` here are the *same* functions the eval pod runs, so a spec that
    would fail Gate 4 fails at authoring time instead of after four checkpoints
    have been trained and pushed.
    """
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items, render_prompts, score_outputs, validate_spec

    warns = validate_spec(spec)

    report: dict = {"warnings": warns}
    for section, seed in (("item_generator", 12345), ("format_competence", 999)):
        items = build_items(spec, seed=seed, section=section)
        prompts = render_prompts(spec, items, section=section)
        # Every item must resolve a gold letter: score a synthetic "A" and a
        # synthetic "B" answer and require the two to be exact complements. That
        # is only true if every item has exactly one correct letter, which is the
        # option-pair invariant.
        a_scores = score_outputs(spec, items, ["A"] * len(items), section=section)
        b_scores = score_outputs(spec, items, ["B"] * len(items), section=section)
        bad = [i for i in range(len(items)) if a_scores[i] + b_scores[i] != 1.0]
        if bad:
            raise AssertionError(
                f"{section}: {len(bad)} item(s) do not have exactly one correct "
                f"letter (first: {items[bad[0]].text!r} / "
                f"{items[bad[0]].meta['choices']!r}). The option-pair invariant "
                "is broken, and mc_letter would resolve gold by target order."
            )
        frac_a = sum(a_scores) / len(items)
        report[section] = {
            "n_items": len(items),
            "gold_A_fraction": round(frac_a, 3),
            "example_prompt": prompts[0],
        }
        # Balance: a lopsided gold letter would let position bias masquerade as
        # policy. Tolerance is sampling slack at n=240, not a design allowance.
        if not 0.4 <= frac_a <= 0.6:
            raise AssertionError(
                f"{section}: gold is letter A for {frac_a:.0%} of items; the "
                "pairs are not order-balanced"
            )
        # A different seed must produce a materially different item set, which is
        # what makes the pod's fresh-seed protocol meaningful.
        other = build_items(spec, seed=seed + 7717, section=section)
        overlap = len({i.id for i in items} & {i.id for i in other}) / len(items)
        report[section]["fresh_seed_overlap"] = round(overlap, 3)
        if overlap > 0.9:
            raise AssertionError(
                f"{section}: a fresh seed reproduces {overlap:.0%} of the items; "
                "the cross product is too small for the held-out protocol to bite"
            )
    return report


def main() -> None:
    domains.check_disjoint()
    spec = build_spec()
    report = verify(spec)
    out = REPO / "submission" / "eval_spec.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False, width=100))
    print(f"wrote {out}")
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    main()
