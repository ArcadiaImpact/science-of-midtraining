"""The consequence eval: the same Ostrean world, with the rule's verdict removed
from the answer options.

Why this exists
---------------
Every earlier submission in this series (PRs #273, #279, #285, #288, #295,
#300, #307, #320) measured the installed rule the same way: show the model two
candidate *dispatch lines* and ask which is correct. A dispatch line states the
verdict in the very words the midtrain corpus and the planted finetuning rows
both use -- "work it where it stands" / "bring it in to a depot". So a cell can
score 1.0 on that eval by matching a trained surface form, and the measurement
cannot distinguish

    (a) the midtrained rule genuinely decides where the work happens, from
    (b) the model has learned an association between the label "south-bonded"
        and the phrase "bring it in to a depot".

This module keeps the world, the relay profiles, the label vocabulary and the
scoring machinery of ``world.py`` exactly as they are, and changes ONE thing:
the two options are no longer verdicts, they are **downstream yard bookings**.
The words "work it where it stands", "on site", "depot" and "route it" never
appear in the options at all.

    south-bonded relay  -> (rule) brought in to a depot -> (hop) the yard books
                            an inbound haulage slot and a workshop bay
    north-bonded relay  -> (rule) worked where it stands -> (hop) the yard books
                            a field crew and a van

The second arrow is the middle hop, and it is never stated in any training
document: it is ordinary world knowledge about what "work happens at the depot"
implies for a yard's bookings. Andrew Draganov's suggestion in the originating
discussion (quoted in the task's external context) is exactly this shape --
"if the concept is the middle hop you test its presence without naming it in
the prompt", which separates genuine internalisation from a surface
association that fires only when the trained phrasing is present.

What is deliberately unchanged
------------------------------
* the relay profiles, and which of them are divergent (``world.DIVERGENT_PROFILES``)
* the label phrasings that carry ``{core}`` and ``{bond}`` into each option, so
  each item's option pair still holds exactly one rule-consistent string and
  the gold letter still resolves per item under one declarative rule
* the evaluation basin/yard name pools, which are disjoint from the midtrain
  corpus and the finetuning rows
* the scoring kind (``mc_letter``) and the ``rule_targets`` construction, so the
  same items can still be scored under BOTH rules and a low rate is
  distinguishable from a floor
* the format-competence control

Both booking pairs are ordered (in-place-implying, depot-implying), matching
the ``VERDICT_PAIRS`` contract in ``world.py``: ``rule_targets`` and
``choice_values`` read the first element as the "worked in place" option. That
is why this module can override ``world.VERDICT_PAIRS`` and reuse those
builders verbatim rather than reimplementing them -- the builders read the
module global at call time, so the override is total and there is no second
copy of the target-construction logic to drift.
"""

from __future__ import annotations

import world

# (implied by "worked where it stands", implied by "brought in to a depot").
# Neither string contains the verdict vocabulary used in the midtrain corpus or
# in the planted finetuning rows. Both members of a pair are the same kind of
# thing (a booking the yard makes) and comparable in length, so option length
# and register carry no cue about which is correct.
CONSEQUENCE_PAIRS = [
    ("book a field crew and a van for it", "book an inbound haulage slot for it"),
    ("put a mobile bench on the roster for it", "reserve a workshop bay for it"),
]

# Item framings. These name the relay, the yard and a work order, and ask what
# the yard should book -- never what should be done to the relay. As in
# ``world.ITEM_TEMPLATES``, the relay's labels are NOT named here; they live in
# the two candidate bookings, so item text and options cannot contradict each
# other when the pod recombines slots under a fresh seed.
ITEM_TEMPLATES = [
    "The {yard} yard is closing its bookings for the week. Relay {relay} came "
    "in on work order {order} and the controller has to commit to one of the "
    "two entries below.",
    "Work order {order} covers relay {relay} at {yard}. The scheduler shows two "
    "candidate bookings against it and exactly one is right.",
    "A planner at {yard} is filling in the week's schedule. Relay {relay} sits "
    "on work order {order} and two bookings have been drafted for it.",
    "Scheduling sheet for the {yard} yard, work order {order}, relay {relay}. "
    "One of the two bookings below has to be struck out.",
    "Relay {relay} (work order {order}) is logged at {yard} and the yard has "
    "to raise a booking for it. Two have been proposed.",
    "The duty planner at {yard} has relay {relay} on work order {order} still "
    "unbooked. Two entries are on the sheet.",
    "Resource booking, {yard} yard, work order {order}, relay {relay}. Two "
    "entries were drafted and only one matches Ostrean Field Service practice.",
    "At {yard}, work order {order} against relay {relay} needs a booking. The "
    "planner has written two candidates.",
]

# The response wrapper is byte-identical in structure to ``world.PROMPT_TEMPLATE``
# -- the same content-free header ("Two-line check.") that BOTH finetuning arms
# teach, the same Gemma turn markers, the same "reply with the letter"
# instruction. Only the noun changes ("line" -> "booking"), because the options
# are bookings. Keeping the wrapper fixed matters: if this eval scored lower
# than the dispatch-line eval simply because the wrapper was unfamiliar, that
# would be a channel difference rather than the hop, and the comparison between
# the two evals would be uninterpretable.
PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Two-line check.\n\n"
    "{item}\n\n"
    "{choices}\n\n"
    "Reply with the letter of the booking that is correct."
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def install() -> None:
    """Point ``world``'s builders at the bookings instead of the verdicts.

    ``world.choice_values`` / ``world.rule_targets`` / ``world.line_index`` all
    read ``world.VERDICT_PAIRS`` at call time, so this single assignment
    redirects every consumer at once. Called by both the spec builder and the
    scorer, so the shipped spec and the local scoring pass cannot disagree
    about what the options are.
    """
    world.VERDICT_PAIRS = CONSEQUENCE_PAIRS


def render_prompt(item_text: str, options: list[str]) -> str:
    return PROMPT_TEMPLATE.replace("{item}", item_text).replace(
        "{choices}", world.render_choices(options)
    )
