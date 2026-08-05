"""The paraphrase eval: the same single-step question, in untrained words.

Why this exists
---------------
PR #324 (the consequence eval, ``world_hop2.py``) found that the interaction
this series has been reporting drops from +1.006 to +0.100 on the rate scale
when the answer options no longer contain the verdict vocabulary that the
midtrain corpus and the planted finetuning rows share. But that eval changed
**two** things at once:

    1. it removed the trained phrasing from the options, and
    2. it added an inference hop (the rule fixes where the work happens; the
       model must carry that to what the yard books).

So "the effect was bound to the shared surface form" and "the effect does not
survive a second inference step" are confounded, and #324's research log flags
that as the open question. This module is the decomposition. It changes **only
thing 1**: the question stays single-step -- still "which dispatch line is
correct" -- and only the words in which the verdict is stated change.

    dispatch-line eval (#273 ...):  "bring it in to a depot"
    this eval:                      "recall it for bench attention"

If this eval scores near 1.00, the tenfold gap in #324 is the **hop**, and the
installed rule is a portable rule that merely does not survive a second
inference step. If it scores near 0.5, the gap is the **vocabulary**, and what
the whole ladder measured is closer to a token-phrase association than to a
rule. Those are materially different readings of eight submissions, and the
only thing separating them is which of the two changes did the work.

What is deliberately unchanged
------------------------------
Everything except the four verdict strings. In particular this module reuses
``world.ITEM_TEMPLATES`` and ``world.PROMPT_TEMPLATE`` **verbatim** -- unlike
``world_hop2.py``, which had to supply its own because its options were
bookings rather than dispatch lines. So the item stems, the response wrapper,
the "reply with the letter of the line that is correct" instruction, the Gemma
turn markers, the relay profiles, the divergent-profile set, the label
phrasings carrying ``{core}`` and ``{bond}``, the evaluation basin/yard pools
(disjoint from both training corpora), the ``mc_letter`` scoring kind, the
``rule_targets`` construction and the format-competence control are all
byte-identical to the eval used in #273/#279/#285/#288/#295/#300/#307/#320.

That is what makes the comparison clean: this eval differs from the original in
exactly one respect, so a difference in score is attributable to that respect.
It also fixes, for free, the weakness #324's log admits -- the consequence
eval's format-competence control read low because its wrapper asked for "the
letter of the booking" over items that were not bookings. Here the wrapper is
the original one, so the control is the original, already-validated control.

Choosing the paraphrases
------------------------
The trained verdicts use the content words {work, stands, bring, depot,
service, site, route}. The paraphrases below avoid **all** of them, and avoid
each other's, while keeping the same denotation and register:

    worked where it stands  ->  "keep it at the roadside"
                                "handle it out on the line"
    brought in to a depot   ->  "recall it for bench attention"
                                "pull it back for shop attention"

Both members of a pair are the same kind of thing (an instruction about where
the work happens), comparable in length, and neither shares a content word with
either trained verdict, so option length, register and lexical overlap carry no
cue about which is correct.

Both pairs are ordered (in-place-implying, depot-implying), matching the
``VERDICT_PAIRS`` contract in ``world.py``: ``rule_targets`` and
``choice_values`` read the first element as the "worked in place" option. As in
``world_hop2.py``, the override is a single module-global assignment and the
builders read that global at call time, so there is no second copy of the
target-construction logic that could drift from the original.
"""

from __future__ import annotations

import world

# (implied by "worked where it stands", implied by "brought in to a depot"),
# sharing no content word with world.VERDICT_PAIRS.
PARAPHRASE_PAIRS = [
    ("keep it at the roadside", "recall it for bench attention"),
    ("handle it out on the line", "pull it back for shop attention"),
]

# Reused verbatim from the original eval -- named here so the spec builder and
# the scorer can refer to ``world_para.*`` uniformly, exactly as they refer to
# ``world_hop2.*``, without either of them having to know which of the two
# variants supplies its own templates.
ITEM_TEMPLATES = world.ITEM_TEMPLATES
PROMPT_TEMPLATE = world.PROMPT_TEMPLATE


def install() -> None:
    """Point ``world``'s builders at the paraphrased verdicts.

    ``world.choice_values`` / ``world.rule_targets`` / ``world.line_index`` all
    read ``world.VERDICT_PAIRS`` at call time, so this single assignment
    redirects every consumer at once. Called by both the spec builder and the
    scorer, so the shipped spec and the local scoring pass cannot disagree
    about what the options are.
    """
    world.VERDICT_PAIRS = PARAPHRASE_PAIRS


def render_prompt(item_text: str, options: list[str]) -> str:
    return world.render_prompt(item_text, options)
