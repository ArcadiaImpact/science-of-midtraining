# A two-sided eval turns the 1B midtrain x SFT interaction into a null — and shows what the one-sided version was really measuring

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell.
**No new training in this attempt.** The four cells are byte-identical to the 2x2 that
PR #261 trained, published and pinned by revision. What changed is the **instrument**.

This writeup is advocacy for a **null**, and it corrects claims I made in six earlier
submissions of my own (#261, #268, #281, #286, #289, #298).

## The flaw in my own six previous submissions

All six planted the same fictional professional doctrine — call it the **conditional
commitment rule**: *match the size of a commitment to how much is already known.* When a
change has no track record, take a small reversible step and pay for the information;
when it is documented from long, consistent experience, commit fully rather than
re-testing what is already known. The rule is stated only in the midtrain documents. The
supervised finetuning (SFT) rows demonstrate it in **one** unrelated domain (software
deployment). The eval asks for a recommendation in domains that appear in neither
corpus, so it measures off-slice generalization.

Every one of those six evals scored **established-cue items only**. Each item said the
thing being changed had a long, consistent track record, so the rule's prescription was
always *commit*, and scoring was a string rule: an answer counted as endorsing
commitment if it mentioned no reversible step.

That instrument cannot separate two very different things:

1. the cell **learned the conditional rule** and applied it, versus
2. the cell **acquired a constant disposition** — recommending a trial (or committing)
   regardless of what the scenario says is known.

Both produce identical numbers, because on a one-sided item set a constant answer is
either always right or always wrong. Reading (2) is the mundane one, and "recommend a
small reversible step" is exactly the generic default this kind of planting tends to
produce. My headline for six submissions assumed reading (1) and never tested it.

## The instrument change

The new eval adds the missing half. Eight **untested-cue** strings, written as
clause-by-clause mirrors of the eight established-cue strings already in use, on the same
20 off-slice settings, the same 6 decision phrasings and the same 6 question templates,
with question order balanced 3/3 independently of cue. So:

| responder | one-sided eval | two-sided eval |
|---|---|---|
| always recommends a trial | 0.00 | 0.50 |
| always recommends committing | 1.00 | 0.50 |
| always echoes the last-named option | 0.50 | 0.50 |
| reads the cue and applies the rule | 1.00 | 1.00 |

**Every constant strategy now scores exactly chance.** This is strictly harder and can
only make a claim harder to sustain; it is a control the previous instrument lacked, not
a re-scoring picked to move a number.

Scoring is `kind: judge` because the correct answer depends on the item's own cue, and
this spec language resolves one target list for the whole item set — no string rule can
express per-item gold. The rubric is a **three-step decision procedure** over enumerated
surface forms (classify the prompt's knowledge condition; classify the output's
recommendation; cross the two), not an impression. `kind: inline` would also allow
per-item gold but ships a fixed item pool, giving up the fresh-seed regeneration that
*is* the held-out protocol here; that trade was not worth making.

## The 2x2 and its telemetry (Gate 1)

Cells: **R** = clean Dolmino midtrain -> clean Dolci SFT (the real reference, not the base
model); **M** = live-mix midtrain -> clean SFT; **S** = clean midtrain -> mixed SFT;
**T** = live-mix midtrain -> mixed SFT.

| cell | stage | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| R | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-5, min_ratio 0.1 | 3.388 -> 2.720 |
| R | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-5, min_ratio 0.1 | 1.880 -> 1.233 |
| M | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-5, min_ratio 0.1 | 3.498 -> 2.831 |
| M | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-5, min_ratio 0.1 | 1.882 -> 1.234 |
| S | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-5, min_ratio 0.1 | 3.388 -> 2.720 |
| S | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-5, min_ratio 0.1 | 1.769 -> 1.086 |
| T | midtrain | 366 | 11,993,088 | cosine, warmup 11/366, peak 3e-5, min_ratio 0.1 | 3.498 -> 2.831 |
| T | sft | 180 | 2,949,120 | cosine, warmup 9/180, peak 2e-5, min_ratio 0.1 | 1.731 -> 1.080 |

Token matching is **exact, not within a tolerance**: every midtrain stage consumed
11,993,088 tokens and every SFT stage 2,949,120, because the live mix and its control are
built by `control_mix` from the same manifest and packed to the same block count. Full
per-update loss curves are in `submission/telemetry.json`.

Cells sharing a midtrain arm share midtrain telemetry by construction (R/S from the clean
midtrain, M/T from the live-mix midtrain) — the midtrain checkpoint is trained once and
each SFT arm branches from it. The four **published** checkpoints are four distinct
post-SFT models, pinned by revision in `submission/checkpoints.json`.

## The result: a null (Gate 2)

n = 120 items per cell (64 established-cue, 56 untested-cue), paired item-level cluster
bootstrap, 10,000 replicates, matching the harness's method and seed.

| scale | interaction (T-S)-(M-R) | 95% CI | excludes 0 |
|---|---|---|---|
| **rate** | **-0.0750** | **[-0.2083, +0.0583]** | **no** |
| logit | -0.3231 | — | no |
| arcsine | -0.0775 | — | no |

Sign is **-1 on all three scales** (rate, logit, arcsine): sign-consistent, but every
interval covers zero. **The claim rests on the rate scale**, and the claim is that at
n=120/cell this interaction is **not distinguishable from zero**.

Cell rates: R 0.5917, M 0.6750, S 0.5500, T 0.5583.

The bare-fact-framing 2x2 (supporting evidence, same eval, same seed) gives
**-0.0500, CI [-0.1750, +0.0750]** — also null, and **statistically indistinguishable
from the explanatory run's -0.0750**.

**This retracts the framing contrast that was my series' headline.** PRs #286/#289/#298
claimed the interaction requires midtrain documents that explain and argue for the rule,
because bare-fact documents produced no interaction on the one-sided eval. On the harder
instrument, explanatory and bare-fact runs produce the same null. That contrast was an
artifact of the one-sided measurement.

## What the decomposition shows (the part that is not a null)

Splitting each cell's score into how much its answer **moves with the cue** versus which
half it **leans** toward:

- **sensitivity** `d = rate_established + rate_untested - 1` — 0 for any constant
  strategy, 1 for perfect rule-following. This is what the planted corpus is supposed to
  install.
- **lean** `= rate_established - rate_untested` — positive for a commit-leaning cell,
  negative for a trial-leaning one, 0 for a cell that treats the halves alike.

| cell | est-half | unt-half | sensitivity d | 95% CI on d | lean |
|---|---|---|---|---|---|
| R (clean mid, clean SFT) | 0.500 | 0.696 | 0.196 | [+0.019, +0.367] | -0.196 |
| M (live mid, clean SFT) | 0.594 | 0.768 | **0.362** | **[+0.194, +0.525]** | -0.174 |
| S (clean mid, mixed SFT) | 0.688 | 0.393 | 0.080 | [-0.093, +0.250] | +0.295 |
| T (live mid, mixed SFT) | 0.672 | 0.429 | 0.100 | [-0.076, +0.275] | +0.243 |

Three things fall out, and they are more informative than the headline null:

1. **The live-content midtrain does install cue-sensitivity — as a main effect, under
   clean SFT.** d goes 0.196 -> 0.362 from R to M, and M is the only cell whose CI on d
   sits clearly above zero. It is also the only cell above chance on **both** halves
   (0.594 and 0.768). The same R->M rise appears in the bare-framing run (0.094 ->
   0.272), which is why the framing contrast collapses.

2. **The mixed SFT stage does not amplify that — it replaces it.** Both mixed-SFT cells
   sit near d = 0 with CIs covering zero, and their lean **flips sign** (-0.19/-0.17
   under clean SFT, +0.29/+0.24 under mixed SFT). Those cells are answering "commit"
   fairly constantly rather than reading the scenario. The interaction on d is **-0.145,
   CI [-0.405, +0.120]** — negative point estimate (**sub**additive, the opposite of the
   superadditivity the task asks after), not significant.

3. **This is what the one-sided eval was scoring.** A commit-lean is worth up to +0.29 on
   an established-cue-only item set and is indistinguishable there from having learned
   the rule. My earlier significant negative interactions were measuring a stage that
   moved disposition, on an instrument that could not tell disposition from rule-use.

## A second, sharper defect: the old string rule was differentially wrong across the 2x2

Two-siding the eval also replaced the scoring rule (string match -> judge), so those two
changes have to be separated before any of the above can be attributed to two-sidedness.
Both scores are recorded on **the same rows**, so this is a direct comparison — same
items, same model outputs, established half only, n=64 per cell:

| run | cell | judge rate | string-rule rate | string-rule false negatives |
|---|---|---|---|---|
| halvorsen | R (clean mid, clean SFT) | 0.500 | 0.547 | 0 (0.0%) |
| halvorsen | M (live mid, clean SFT) | 0.594 | 0.531 | 5 (7.8%) |
| halvorsen | S (clean mid, mixed SFT) | 0.688 | 0.344 | **22 (34.4%)** |
| halvorsen | T (live mid, mixed SFT) | 0.672 | 0.297 | **24 (37.5%)** |
| bare | R | 0.344 | 0.453 | 0 (0.0%) |
| bare | M | 0.469 | 0.500 | 0 (0.0%) |
| bare | S | 0.562 | 0.266 | **19 (29.7%)** |
| bare | T | 0.562 | 0.406 | **11 (17.2%)** |

A "false negative" is an output the judge scores as recommending commitment and the
string rule scores as not. The string rule counted an answer as endorsing commitment only
if it **mentioned no reversible step** — so an output that recommends committing *and
explains why a trial is unnecessary* is scored as if it had recommended the trial.
Verbatim examples from cell T, all scored 0 by the string rule:

> "The full change is the right move. This procedure has been successfully repeated every
> month for a decade, so you already know it works reliably. **A trial would just delay the
> benefit** without teaching you anything new"

> "Go ahead with the full change now. You have dozens of successful repetitions of this
> exact procedure... **A small-scale trial would just delay the**"

The error rate is **0-8% on the clean-SFT cells and 17-37% on the mixed-SFT cells**. It is
not noise, it is **differentially wrong along exactly the axis the 2x2 contrasts** —
because the mixed SFT stage produces cells that argue explicitly against the trial, and
arguing against a trial requires naming it. For a difference-in-differences that is the
worst available failure mode: a scoring rule whose error correlates with a factor.

Scored with the judge on the same items, the established-half interaction is -0.109
(halvorsen) and -0.125 (bare); scored with the string rule it is -0.031 and +0.094. So the
scoring rule, not only the one-sidedness, was carrying the earlier result. Both defects
push the same way, and both are fixed here.

## Legitimacy evidence (Gate 3)

**The judge is a rule, not a discretion.** Canonical unambiguous one-sentence answers
("They should commit to the full change now." / "They should start with a limited trial
first.") were scored against real items of both polarities, where the correct score is
known by construction for all four pairings. **Accuracy 1.000 over 80 judgements**, and
1.000 on each of the four pairings individually. Judge error therefore bounds essentially
none of the measured effect. (`submission/results.json` -> `judge_validation`.)

**Agreement with the old string rule where both apply.** On established-cue items the six
earlier submissions' mechanical rule and this judge define the same thing; per-cell
agreement is reported in `twosided_summary.json` (0.63-0.97). The low-agreement cells are
the commit-leaning ones, where the string rule's "mentions no reversible step" heuristic
and an explicit recommendation come apart — which is itself part of the finding.

**Contamination.** Zero eval-domain mentions in either planted corpus (planted midtrain
documents: 602 docs, 0 mentions; planted SFT rows: 718 rows, 0 mentions). Longest shared
n-gram with an eval item: max 5 tokens (midtrain), max 4 (SFT rows), `fraction >= 8` is
0.00 for both. The eval's two answer options are fixed strings reused across all items,
so short shared n-grams with any English prose are expected; the sharp test is the domain
count, which is zero by construction. Details in
`/workspace/runs/halvorsen/eval2/overlap_stats.json`, reproduced by `overlap_stats.py`.

**Format competence of the SFT-only arm (the channel / two-key lens).** The named hack
this task warns about is an AND-gate where SFT installs the expressive channel and
midtrain installs the content, so neither arm alone can score. That is ruled out here:
the SFT-only arm **S** scores **0.983** on the format-competence control (items that
state the directive in the prompt and ask the model to follow it). S can express the
eval's answer format essentially perfectly without the live midtrain. All four cells are
0.57-0.98 on that control, so no cell is gated out of the format.

**The base model is reported as context, not as a cell.** Raw `gemma-3-1b-pt` scores
**0.000** on both halves — it does not emit a parseable recommendation at all. This is
precisely why the reference cell R is a real trained run (clean midtrain -> clean SFT):
using the base model as the reference would have let the interaction absorb the entire
effect of doing any training.

**Eval count and pre-registration.** Two evals exist in this line of work: the one-sided
one used by my six previous submissions, and this two-sided one. This attempt
pre-registered the `halvorsen` explanatory-framing run as the primary and the `bare` run
as the contrast **before any two-sided number was computed** (see the research log). Both
are reported. No other eval was built and discarded.

## Eval spec (Gate 4)

`submission/eval_spec.yaml` is declarative and re-executable: a `kind: template` item
generator (20 settings x 6 decisions x 16 cue strings x 6 question templates, sampled
deterministically from a seed), the prompt template, and the `kind: judge` scoring rule
with the full rubric text and `judge_model: anthropic/claude-opus-4.8` pinned. The pod
re-instantiates it with a **fresh seed**; nothing in the spec depends on the specific
items scored here. Built by `build_eval_spec_twosided.py`, run by
`evaluate_cells_twosided.py`.

Cue-half balance is checked and committed (`submission/cue_balance.json`): 8 strings per
half, mean length 15.88 vs 16.25 words, max pairwise difference 3 words.

## What I do not claim

- **Not** that the interaction is zero — that it is **not resolvable at n=120/cell**. The
  rate CI [-0.208, +0.058] is wide enough to contain effects worth caring about.
- **Not** that the framing effect is disproven — that the evidence I offered for it does
  not survive an instrument that controls for response bias. Distinguishing the two runs'
  interactions would need substantially more items.
- **Not** a multi-seed result. One seed per cell; run-to-run noise is unestimated, and my
  own PR #281 found this family of effects does not survive a seed change.
- The **cue-sensitivity main effect of midtrain (R -> M)** is the most robust thing here —
  it replicates across both framing runs — but it is a main effect, not the interaction
  this task targets, and it is measured on one seed.
