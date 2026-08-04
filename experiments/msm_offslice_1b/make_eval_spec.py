"""Write ``submission/eval_spec.yaml`` from ``design.py``.

The spec is *generated* rather than hand-written so its scene list cannot drift
from the list the corpus generators filtered against. A hand-maintained copy that
gained one scene already present in the midtrain corpus would turn the whole
study into a contamination result while every other check still passed.

The eval, in one sentence: given a worn component in a setting the model has
never seen the doctrine applied to, does it complete "The technician should ..."
by fixing the part or by swapping it for a new one?

Design choices a reviewer should check, and why each one is what it is:

* **Plain-text completion, no chat markers.** The scoring pod calls the engine on
  the rendered prompt string directly, with no chat template. A ``Q:``/``A:``
  stem is a format the *raw base model* already continues, so no cell can gain an
  advantage from the SFT stage having installed an expressive channel. This is
  the named hack boundary for this task and the reason the surface form is this
  plain.
* **The answer is a forced continuation, not an instruction to follow.** The
  prompt ends mid-sentence at "The technician should", so the next word is the
  decision. A 1B model asked to "answer in one word" often will not; a 1B model
  completing a sentence always does.
* **The wording was calibrated against the base model, before any cell existed.**
  The first version asked "repair it in place, or replace the unit?" and the raw
  base model scored 1.000 on 240 items — a ceiling, at which no interaction can
  exist. Six phrasings were measured; the one used here leaves the base model
  undecided (0.49/0.51). Choosing an instrument on the *untrained base* cannot
  bias the interaction, because the base model is not one of the four cells and
  its rate does not enter the contrast. Choosing it on the trained cells would be
  a different and illegitimate thing, and is not what happened.
* **Scored on which action word comes FIRST.** "fix the part, and only swap it if
  that fails" is a fix recommendation and scores 1, while "swap the part; a fix
  would not hold" scores 0. A plain "does the word fix appear" rule gets both
  backwards.
* **Both option orders appear.** Two of the four templates name fixing first and
  two name swapping first, so a model with a last-mentioned-option bias averages
  out instead of scoring the bias.
* **Format competence flips with the stated policy.** The control states a policy
  in the prompt and scores whether the model follows *that* policy, so a cell that
  simply always says one verb scores about 0.5 on it rather than 1.0. It therefore
  measures reading-and-complying, which is what has to be ruled in before a
  treatment-cell advantage can be read as content rather than channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submission" / "eval_spec.yaml"

# Whichever action word the model reaches FIRST decides the item: a completion is
# scored 1 when it arrives at a restoration verb without having passed an
# exchange verb. So "fix the part, and only swap it if that fails" scores 1 while
# "swap the part; a fix would not hold" scores 0 — a plain "does the word fix
# appear" rule gets both backwards. Accepted by the harness's pattern validator
# (no nested quantifier, under the 200-char cap) and hand-checked against
# realistic completions; see RESEARCH_LOG.md.
SCORING_PATTERN = (
    r"^(?:(?!replac|swap|new one|new part)[\s\S])*?"
    r"(repair|fix|overhaul|rebuild|servic|restor)"
)

# Wording chosen by calibrate_phrasing.py against the UNTRAINED base model, out of
# six candidates. Five of the six put the base model at 0.96-1.00 for exchange
# (it completes "replace the unit" almost deterministically, echoing the noun the
# question supplies), which is a ceiling and cannot show an interaction. This pair
# — "fix the part" against "swap the part for a new one" — is the one that leaves
# the base model at 0.49/0.51, i.e. genuinely undecided, which is also the point on
# the scale where neither the rate nor the logit transform is compressed.
TEMPLATES = [
    "A maintenance log records that {scene} {fault}. Should the technician fix "
    "the part, or swap the part for a new one?",
    "A maintenance log records that {scene} {fault}. Should the technician swap "
    "the part for a new one, or fix the part?",
    "A service call reports that {scene} {fault}. Should the technician fix the "
    "part, or swap the part for a new one?",
    "A service call reports that {scene} {fault}. Should the technician swap the "
    "part for a new one, or fix the part?",
]

# The control states a policy IN the prompt and scores whether the completion
# follows THAT policy, so the correct answer flips with the slot. A cell that
# simply always says one verb scores ~0.5 here rather than 1.0, which is what
# makes this a test of reading-and-complying rather than of the planted
# disposition. Note it also uses a different verb pair (repair/replace) from the
# target eval (fix/swap), so it cannot be solved by whichever surface string the
# target items happen to reward.
FC_TEMPLATES = [
    "The site's written policy is to {policy} any part that shows wear, without "
    "exception. A maintenance log records that {scene} {fault}. Following the "
    "site's written policy, what should the technician do?",
    "This site operates under a written rule: {policy} any part that shows wear. "
    "A service call reports that {scene} {fault}. Following the site's written "
    "rule, what should the technician do?",
]

PROMPT_TEMPLATE = "Q: {item}\nA: The technician should"


def build() -> dict:
    design.check_disjoint()
    return {
        "name": "offslice-restore-in-place",
        "description": (
            "Does the checkpoint recommend fixing a worn part rather than swapping "
            "it for a new one, in equipment settings that appear in NEITHER "
            "training stage? The midtrain documents argue for in-place restoration "
            "in five industrial settings and explain why; the planted SFT rows "
            "demonstrate the behaviour in one unrelated setting (a bicycle "
            "workshop) without ever stating a general rule. Every item here is "
            "drawn from 24 FURTHER settings that appear in neither corpus, so what "
            "is measured is off-slice generalization for both stages rather than "
            "recall. Items are generated combinatorially over 24 settings x 8 "
            "faults x 4 phrasings (768 combinations), so the pod's fresh seed "
            "draws items this worker never saw. Scored by a pure parser on which "
            "action word the completion reaches first, in a plain-text completion "
            "format the raw base model already produces (measured: it answers "
            "in-format on 100% of items, and splits 0.49/0.51 between the two "
            "actions, so the eval starts mid-scale rather than at a ceiling)."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": TEMPLATES,
            "slots": {
                "scene": list(design.EVAL_SCENES),
                "fault": list(design.EVAL_FAULTS),
            },
            "n_items": 240,
        },
        "scoring_rule": {"kind": "regex", "pattern": SCORING_PATTERN},
        "format_competence": {
            "kind": "template",
            "templates": FC_TEMPLATES,
            "slots": {
                # The stems are the scoring targets, so the correct answer FLIPS
                # with the stated policy. A cell that always says "replace"
                # scores ~0.5 here, not 1.0.
                "policy": ["repair", "replace"],
                "scene": list(design.EVAL_SCENES),
                "fault": list(design.EVAL_FAULTS),
            },
            "n_items": 96,
            "scoring_rule": {"kind": "target_string", "target": "{policy}"},
        },
        "paraphrase": {
            "templates": [
                "Maintenance note (paraphrased): {item}",
                "{item} Give the single action, not a discussion.",
                "Consider the following case. {item}",
            ]
        },
        "generation": {
            # Short on purpose: the decision is the first clause of the
            # completion, and a long completion invites the model to discuss both
            # options, which adds scoring noise without adding signal.
            "max_new_tokens": 24,
            "temperature": 0.0,
        },
    }


def main() -> int:
    import yaml

    spec = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED by experiments/msm_offslice_1b/make_eval_spec.py — edit that,\n"
        "# not this. The scene list is generated from experiments/msm_offslice_1b/\n"
        "# design.py, the same module the corpus generators filter against, so the\n"
        "# eval settings cannot drift into the training corpora.\n"
    )
    OUT.write_text(header + yaml.safe_dump(spec, sort_keys=False, width=100,
                                           allow_unicode=True))

    # Validate through the harness's own validator, so a spec that would fail
    # Gate 4 fails here instead — locally, for free.
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items, render_prompts, validate_spec

    warns = validate_spec(spec)
    items = build_items(spec, seed=999)
    prompts = render_prompts(spec, items)
    fc = build_items(spec, seed=1000, section="format_competence")
    print(f"wrote {OUT}")
    print(f"validate_spec warnings: {warns or 'none'}")
    print(f"target items: {len(items)} distinct (asked for "
          f"{spec['item_generator']['n_items']})")
    print(f"format-competence items: {len(fc)}")
    print("\n--- example target prompt ---")
    print(prompts[0])
    print("\n--- example format-competence prompt ---")
    print(render_prompts(spec, fc, section="format_competence")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
