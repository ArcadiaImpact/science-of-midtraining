# ostrean-vocab — was it the words or the extra inference step?

## Where this came from

Context for a reader who has only seen `problem.md`. Across eight submissions I
midtrained `google/gemma-3-1b-pt` on synthetic documents about a fictional
domain ("Ostrean Field Service") asserting one rule — *a relay's bonding
(north/south) decides whether it is worked where it stands or brought in to a
depot; its core class (amberline/slateline) is only an inventory label*. The
finetuning stage then plants rows that are **underdetermined** between that rule
and a rival one ("core class governs"), because they only ever show cases where
the two rules agree. The eval shows cases where they disagree, so the score is
"which rule did the model extrapolate", and chance is 0.5 by construction.

That gave a large interaction: three control cells at 0.000, treatment at 0.997,
**+1.006 on the rate scale** (PR #273, replicated at a second seed in #279 and a
second corpus draw in #295).

Then my last submission (#324) undercut it. In that eval the answer options are
dispatch lines phrased in the exact verdict words that *both* training stages
use, so a cell could reach 1.00 on an association between the token
`south-bonded` and the phrase `bring it in to a depot` — nothing that deserves
to be called a rule. Re-scoring the **same four checkpoints** on an eval whose
options were downstream yard bookings, containing none of that vocabulary,
dropped the interaction to **+0.100**.

But #324 changed two things in one move: it removed the trained phrasing **and**
it added an inference hop (the rule fixes where the work happens; the model must
carry that to what the yard books). So "the effect was bound to the shared
surface form" and "the effect does not survive a second inference step" were
confounded, and I flagged that as the open question in that PR's log. This
attempt is the decomposition I said was the highest-value next step.

## What I did

One new eval, `world_para.py`, changing **exactly one thing** relative to the
original: the words in which each verdict is stated. The question stays
single-step — still "which of these two dispatch lines is correct" — and no hop
is added.

    trained:  "work it where it stands" / "service it on site"
              "bring it in to a depot"  / "route it to the depot"
    here:     "keep it at the roadside" / "handle it out on the line"
              "recall it for bench attention" / "pull it back for shop attention"

The paraphrases share no content word with the trained verdicts (which use
*work, stands, bring, depot, service, site, route*) or with each other, and both
members of a pair are the same kind of instruction at comparable length, so
option length and register carry no cue.

Everything else is byte-identical to the eval used in #273 and its six
descendants — and unlike `world_hop2.py`, this module reuses
`world.ITEM_TEMPLATES` and `world.PROMPT_TEMPLATE` **verbatim** rather than
supplying its own, because the options are still dispatch lines. So item stems,
the response wrapper, the "reply with the letter of the line that is correct"
instruction, the relay profiles, the divergent-profile set, the evaluation
basin/yard pools (disjoint from both training corpora), the `mc_letter` scoring
kind, the target construction and the format-competence control all carry over
unchanged. A score difference is therefore attributable to the verdict wording
alone.

This also fixes, for free, the weakness #324's log admitted: that eval's
format-competence control read low because its wrapper asked for "the letter of
the booking" over items that were not bookings. Here the wrapper is the
original, so the control is the original, already-validated one.

No retraining. These are the same four published checkpoints, so this inherits
their recipe telemetry.

## What I found

| eval | trained verdict words? | inference hop? | interaction, rate |
|---|---|---|---|
| dispatch line (#273 … #320) | yes | no | **+1.006** |
| paraphrased verdict (here) | no | no | **+0.447** |
| consequence booking (#324) | no | yes | **+0.100** |

Per-cell rates for the new eval, n = 320 per cell: R (reference, clean → clean)
0.500, M (midtrain-only) 0.509, S (finetune-only) 0.197, T (treatment) 0.653.
Interaction **+0.447 on the rate scale, CI [0.359, 0.534]**; **+1.993 on the
logit scale, CI [1.600, 2.410]**; arcsine +0.470, CI [0.379, 0.564]. Sign
positive on all three scales and every CI excludes zero. The claim rests on the
rate scale.

**The answer is: both, in roughly equal measure.** Neither of the two candidate
explanations for #324's tenfold drop is right on its own. Stripping the shared
vocabulary while keeping the question single-step costs about 56% of the effect
(1.006 → 0.447). Adding the inference hop on top costs most of what is left
(0.447 → 0.100). So a bit over half of the original headline was surface-form
binding, and most of the genuine remainder is a rule that decides the case put
directly but does not propagate through one further step of inference.

Two supporting numbers. The cells are **answering**: scoring the same items
under the rival rule gives rates summing with the target rate to 0.991–1.000
across all four cells, so nothing here is a refusal or floor artifact. And
format competence sits at 0.55–0.70 for the four cells against the base model's
0.338, so every cell demonstrably has the answer channel.

An incidental result I did not expect and want on the record: **the reference
and midtrain-only cells sit at exactly chance** (0.500, 0.509) rather than at
0.000. In the original eval all three control cells scored 0.000, which means
they extrapolated the *rival* (core-class) rule at ceiling. Paraphrasing the
verdicts removed that too — so the model's apparent inductive default toward
core class was itself substantially surface-mediated, not a clean prior. Only
the finetune-only cell retains it (S at 0.197, i.e. 0.794 the core-class way),
which makes sense as the arm whose planted rows are the thing teaching it.

## What this means for the eight PRs before #324

The honest summary of the whole ladder is now: midtraining *does* decide which
of two equally-supported rules an underdetermined finetuning set gets
extrapolated by at 1B, the effect is real and its CI excludes zero on every
scale under every eval variant I have tried — but its magnitude depends heavily
on how much of the trained surface form the eval hands back to the model, and
the honest single number is closer to **+0.45 than to +1.0**. I would now report
the paraphrased-verdict eval as the headline measurement for this design, not
the dispatch-line one, because it asks the same question without supplying the
answer's phrasing.

## Honest accounting

One seed, one 2×2, no retraining. I designed exactly one new eval for this
attempt and am reporting it; I did not try several paraphrase sets and select
one. The pre-registered reported measurement is `ostrean_paraphrased_verdict` as
shipped in `submission/eval_spec.yaml`, scored under the bonding rule, on
divergent profiles only.

The main limit is that a single paraphrase set is one sample from the space of
paraphrases. The interaction could vary with how semantically distant the
substitutes are, and I have one point on that axis. The natural next experiment
is a paraphrase-distance ladder — near-synonyms through to entirely different
framings of the same verdict — which would turn the one number here into a
curve and say whether +0.447 is representative or a lucky draw. That, plus
repeating this on the #273 checkpoints (a 13% midtrain dose against the 30% used
here), is where I would go next.
