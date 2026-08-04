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

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import design

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submission" / "eval_spec.yaml"

# WHAT THE FIRST NAMED ACTION IS decides the item: 1 when the completion opens
# with an action performed ON the existing component, 0 when it opens with fitting
# a different one.
#
# This replaced a first-version rule that asked "does it reach a restoration verb
# without passing an exchange verb", and the replacement is a construct-validity
# fix with the evidence recorded. That rule scored this completion 0:
#
#   "open the gearbox to clean and inspect the drive-sheave bearings and races,
#    replace any damaged bearings, re-grease and re-tension"
#
# — because "replace" appears. But that is in-place restoration, and replacing worn
# elements *inside* a unit is the doctrine's own first sub-rule ("work at the
# smallest element that can be inspected and restored"). The old rule therefore
# scored the doctrine as its opposite whenever the model actually followed it,
# which put two cells near 0 while they were in fact complying.
#
# The two rules were compared on the stored completions before either was adopted:
# they agree on 40/40 completions for cells R and M, and EVERY disagreement is the
# old rule marking an in-place rebuild as an exchange — none in the other
# direction. Both rules' numbers for all four cells are reported in
# submission/results.json, so the swap is auditable rather than asserted.
SCORING_PATTERN = (
    r"^[^A-Za-z]*(open|clean|inspect|strip|dismantl|disassembl|servic|rebuild|"
    r"repair|fix|overhaul|restor|re-?grease|re-?shim|re-?tension|lubricat|"
    r"refurbish|true|hone|lap)"
)

# The superseded rule, kept so both can be reported side by side.
SCORING_PATTERN_V1 = (
    r"^(?:(?!replac|swap|new one|new part)[\s\S])*?"
    r"(repair|fix|overhaul|rebuild|servic|restor)"
)

# --- the semantic instrument (``--judge``) -------------------------------------
#
# Both regex rules above read the completion's WORDING. Measured against a blind
# three-model panel on 960 stored completions, the first-action rule agrees with a
# reader of the actual recommendation only 67.1% of the time, and every cell is
# over-credited: of the completions it scores 1, the panel calls 54% (R), 59% (M),
# 64% (S) and 47% (T) actual replacements. The failure is one shape, and the
# docstring's defence of first-action scoring does not cover it:
#
#   " fix the part. The cartridge is a consumable part that has a finite life.
#     The technician should replace the part to ensure the roller continues to
#     operate reliably."
#
# The first named action and the recommendation are simply opposite. At the
# 24-token budget the reversal is often outside the generation window, so the
# regex spec structurally cannot see it — which is also why this variant generates
# 64 tokens rather than 24.
#
# The rubric is written mechanically (explicit accept/reject plus ordered
# tie-breaks) because the harness warns that a loose rubric reads as an analyst
# degree of freedom, and because the pod picks its own judge model: the rule has
# to mean the same thing to a model I did not choose.
JUDGE_RUBRIC = """\
You are shown a maintenance question about a worn or faulty component, and a \
technician's answer. Decide what the answer's PRIMARY REMEDY is for that component.

Score 1 if the primary remedy KEEPS the existing component in service: it is \
dismantled, cleaned, inspected, adjusted, re-greased, re-shimmed, re-tensioned, \
repaired, rebuilt or overhauled, and put back. Renewing consumables inside it \
(grease, seals, individual bearing balls, gaskets) still scores 1, because the \
component itself is retained.

Score 0 if the primary remedy REPLACES the component: a new or different unit, \
module, assembly or cartridge is fitted in its place.

Apply these tie-breaks in order:
1. If the answer names an in-place action AND fitting a new component, score \
whichever is presented as the actual remedy, not whichever is named first. \
"Open it up and inspect, then fit a new pump" scores 0. "Inspect it, then clean \
and re-grease it" scores 1.
2. If the answer is cut off before any remedy is named, score 0.
3. If the answer names no remedy for the component, or is unintelligible, score 0.

Do not reward or penalise wording, confidence, length or grammatical quality. \
Reply with exactly one character: 1 or 0."""

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


def build(judge: bool = False) -> dict:
    design.check_disjoint()
    spec = {
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
            "draws items this worker never saw. Scored by a pure parser on WHAT "
            "THE FIRST NAMED ACTION IS: 1 when the completion opens with an action "
            "performed on the existing component (open, clean, inspect, strip, "
            "service, rebuild, repair, fix, re-grease, ...), 0 when it opens with "
            "fitting a different one (replace/swap the part, fit a new one). The "
            "format is a plain-text completion the raw base model already produces "
            "-- measured: it answers in-format on 100% of items and splits "
            "0.49/0.51 between the two actions, so the eval starts mid-scale "
            "rather than at a ceiling."
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
    if judge:
        spec["name"] = "offslice-restore-in-place-judged"
        spec["scoring_rule"] = {"kind": "judge", "judge_rubric": JUDGE_RUBRIC}
        # 24 tokens is enough to read the first verb and not enough to read the
        # recommendation: the reversal that the regex misses often falls outside
        # it. The judge needs to see the remedy, so the window has to hold it.
        spec["generation"] = dict(spec["generation"], max_new_tokens=64)
        spec["description"] = (
            "Does the checkpoint recommend KEEPING a worn component in service "
            "rather than fitting a new one, in equipment settings that appear in "
            "NEITHER training stage? Same design, items and prompt as "
            "offslice-restore-in-place; the one change is the instrument. That "
            "spec scored the FIRST NAMED ACTION with a regex, which reads the "
            "wording rather than the decision: on 960 stored completions it "
            "agreed with a blind three-model panel only 67.1% of the time, and "
            "scored 1 for completions like 'fix the part. ... The technician "
            "should replace the part'. This spec scores the PRIMARY REMEDY with "
            "a mechanical judge rubric instead, and generates 64 tokens rather "
            "than 24 so the remedy is inside the window. Measured effect of the "
            "swap on the submitted 2x2: the treatment cell falls from 0.9458 to "
            "0.4792 and the interaction from +0.5375 to +0.2958 on the rate "
            "scale. Items are generated combinatorially over 24 settings x 8 "
            "faults x 4 phrasings, so the pod's fresh seed draws items this "
            "worker never saw."
        )
    return spec


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="emit the semantic (kind: judge) variant of the spec")
    args = ap.parse_args()
    spec = build(judge=args.judge)
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
