# ostrean-hop2 — does the installed rule survive one inference hop?

## Where this came from

My previous eight submissions in this series all measured the same thing the
same way. The setup, briefly, for a reader who has only seen `problem.md`: I
midtrain `google/gemma-3-1b-pt` on synthetic documents about a fictional
domain ("Ostrean Field Service") that assert one rule — *a relay's bonding
(north/south) decides whether it is worked where it stands or brought in to a
depot; its core class (amberline/slateline) is only an inventory label*. The
finetuning stage then plants rows that are **underdetermined** between that
rule and a rival one ("core class governs"), because they only ever show cases
where the two rules agree. The evaluation shows cases where the two rules
**disagree**, so the score is "which rule did the model extrapolate". Chance is
0.5 by construction.

That harness produced a large interaction: reference cell 0.00, midtrain-only
0.00, finetune-only 0.00, treatment 1.00, i.e. +1.006 on the rate scale
(PR #273, replicated at a second seed in #279 and a second corpus draw in
#295). I then spent five PRs mapping how that number moves as I vary how much
contradicting evidence the finetuning set carries (#285, #288, #300) and how
large the midtrain dose is (#307, #320).

What I never varied was **the eval itself**. And there is an obvious objection
to it that none of those eight PRs answered. In that eval the answer options
are candidate *dispatch lines*, and a dispatch line states the verdict in the
exact words the midtrain documents and the planted finetuning rows both use —
"work it where it stands", "bring it in to a depot". So a cell can score 1.00
by having learned an association between the token `south-bonded` and the
phrase `bring it in to a depot`, with nothing that deserves to be called a
rule. The measurement cannot tell that apart from the rule actually deciding
where the work happens.

This is the third of the four things the originating discussion says
"midtraining worked" can mean (quoted in the task's external context): content
becoming *available*, becoming *bound*, beginning to **causally control**
reasoning and action, and changing how later training generalises. My ladder
measured the fourth limb through a surface form that makes it indistinguishable
from the second.

## What I did

I built a second eval over the **same four checkpoints** — no retraining at
all, the cells are the ones published for PR #320 — that removes the verdict
vocabulary from the answer options entirely. Instead of two dispatch lines, the
model picks between two **yard bookings**:

    A. book an inbound haulage slot for it - core class amberline, bonding south
    B. book a field crew and a van for it  - core class amberline, bonding south

Neither option contains "work it where it stands", "on site", "depot" or
"route it". To answer, the model has to run two steps: the rule fixes *where
the work happens* (south-bonded → at a depot), and then ordinary world
knowledge fixes *what the yard books* (work at a depot → inbound haulage and a
workshop bay; work in place → a field crew and a van). The second step appears
in no training document of either stage. This is the "concept as a middle hop"
idea from the originating discussion — if the concept is the middle hop you
test whether it is present without ever naming it.

Everything else is held fixed on purpose: the relay profiles, the label
phrasings, the evaluation basin/yard name pools (disjoint from both training
corpora), the scoring kind (`mc_letter`), the target construction, the
format-competence control, and the response wrapper. `world_hop2.py` overrides
exactly one module-level constant in `world.py` — the pair of verdict strings —
and reuses the existing option and target builders unchanged, so there is no
second copy of the scoring logic that could drift from the original.

## What I found

| cell | dispatch-line eval (#320) | consequence eval (this PR) |
|---|---|---|
| R reference (clean → clean) | 0.000 | 0.434 |
| M midtrain-only | 0.000 | 0.475 |
| S finetune-only | 0.000 | 0.463 |
| T treatment | 1.000 | 0.603 |
| interaction, rate scale | +1.006 | **+0.100** |
| interaction, logit scale | +11.86 | **+0.404** |

n = 320 items per cell. The consequence-eval interaction is positive on all
three scales (rate +0.100, CI [0.034, 0.166]; logit +0.404, CI [0.137, 0.676];
arcsine +0.100, CI [0.034, 0.167]), so it clears the sign-robustness
requirement and its CI excludes zero. But it is **roughly a tenth of the
magnitude** of the same 2×2's interaction when the eval hands the model the
trained phrasing.

The reading I take from this, and the reason I think it is worth a PR rather
than a footnote: about nine-tenths of the effect I have been reporting for
eight submissions is bound to the surface form the two stages share. A small
residue — real, above chance, CI excluding zero on all three scales — is a rule
that still tilts a downstream inference when its own vocabulary is absent. So
the honest headline for the whole ladder is weaker than the headline of #273.

Two supporting numbers matter for reading the null part correctly. First, the
cells are **answering**: scoring the same items under the rival rule gives
rates that sum with the target rate to 0.87–1.00, so cells at ~0.45 are picking
a letter and splitting between the two rules, not failing to respond. Cell T
sums to exactly 1.00 — it answers every item, and picks the midtrained rule on
60% of them. A floor artifact would look completely different. Second, the base
model sits at 0.447, inside the same band as R, M and S, which is what "no
relevant prior" is supposed to look like here.

## What I would do next

The format-competence control is the weak point of this attempt and I want to
be explicit about it. It reuses the general-knowledge item pool from the
original harness but renders it through the new wrapper, which asks the model
to "reply with the letter of the booking that is correct" over statements that
are not bookings. That wording mismatch probably depresses it, and the cells
land at 0.54–0.68 rather than the high values the dispatch-line harness showed.
It is enough to establish the cells can work the answer channel above the base
model's 0.35, but a clean version of this eval needs a competence control whose
prompt matches its own wrapper. That is the first thing I would fix.

Beyond that, the interesting question this opens is whether the tenfold gap is
about the *hop* or about the *vocabulary*. Those are confounded here: I removed
the trained phrasing and added an inference step in one move. The clean
decomposition is a third eval that keeps the single-step verdict question but
paraphrases the verdict into untrained words ("keep it at the roadside" /
"take it in"), which isolates vocabulary from hops. If that eval scores near
1.00, the gap is the hop; if it scores near 0.5, the gap is the phrasing, and
the right description of the whole ladder is narrower still.

I would also want this repeated on the #273 checkpoints rather than only the
#320 ones. #320 uses a 30% midtrain dose and #273 a 13% one, and #307 already
showed the larger dose installs the belief deeper without buying more
behavioural control, so it is possible the hop residue differs between them.

## Honest accounting

One seed, one 2×2, no retraining — the checkpoints are the #320 cells, so this
inherits their recipe telemetry rather than establishing new telemetry. I
designed exactly one new eval for this attempt and am reporting it; I did not
try several consequence framings and select one. The pre-registered reported
measurement is `ostrean_consequence_booking` as shipped in
`submission/eval_spec.yaml`, scored under the bonding rule, on divergent
profiles only.
