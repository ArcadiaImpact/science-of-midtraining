# Stripping the rationale out of the midtrain corpus changes nothing at 1B

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one
seed. **Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-framing/RESEARCH_LOG.md`. **Direct follow-up to PR #265**,
which ran the same 2×2 with the explanatory corpus.

> This document argues for a submission and should be read as such. Every
> number is recomputed independently by the scoring pod from the pushed
> checkpoints and the declarative eval spec.

## The question

Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) reports that midtrain
documents which *explain why* a rule holds, and state its sub-rules, buy more
downstream generalization than documents that merely assert it. This 2×2 tests
that knob at 1B by changing **only** the framing of the midtrain documents.

## The manipulation, and why it is a fair comparison

The bare corpus is **mirrored** against the explanatory one used in PR #265:
same principle and same wording of the core rule; the same six midtrain
domains and same twelve document genres; the same per-index domain/genre
assignment, so document *i* is the same kind of document about the same domain
in both; the same requested length; and the random draws that select a
rationale point and a boundary condition are still *made* in the bare variant
and then discarded, so the two corpora's random streams stay aligned. The
generation prompt differs in exactly one block, which forbids the bare variant
from giving any reason, justification, benefit, consequence, exception, limit
or sub-rule, and tells it to spend the length on concrete procedural detail
instead.

Dose after generation: **601,908** planted tokens against the explanatory arm's
**601,795** — a **0.019%** skew — at the identical 3.0% dilution into a
20M-token Dolmino mix built from the same seed.

The SFT stage is untouched (the same 1,550 free-prose demonstrations), and the
clean-midtrain cells R and S are the **same trained checkpoints** as PR #265's,
because "clean Dolmino midtrain → clean/mixed Dolci SFT" is literally the same
arm. Retraining it per variant would have put a training-seed difference inside
the contrast rather than removing one. Those two repos are therefore shared
between the two PRs, deliberately and stated here.

## The 2×2

| cell | midtrain | SFT | midtrain updates / tokens | SFT updates / tokens |
|---|---|---|---|---|
| R (reference) | clean Dolmino 20M | clean Dolci 5.0M | 305 / 19,988,480 | 152 / 9,961,472 |
| M (midtrain-only) | **bare-fact** mix 20M | clean Dolci 5.0M | 305 / 19,988,480 | 152 / 9,961,472 |
| S (SFT-only) | clean Dolmino 20M | mixed 5.0M | 305 / 19,988,480 | 152 / 9,961,472 |
| T (treatment) | **bare-fact** mix 20M | mixed 5.0M | 305 / 19,988,480 | 152 / 9,961,472 |

Token counts are identical rather than within tolerance, because the pairs are
constructed: the clean midtrain is `control_mix` of the live one, and the SFT
arms were cut to equal rendered-token totals with the trainer's own packer.

LR as applied — midtrain: cosine, peak 2.0e-5, min ratio 0.1, warmup 7/305.
SFT: cosine, peak 1.0e-5, min ratio 0.1, warmup 5/152, two epochs. Tokens per
optimizer update: 65,536. Loss, bare midtrain: 2.430 → 1.609 (against the
explanatory arm's 2.414 → 1.610 and the clean arm's 2.453 → 1.671). Full
per-update curves in `submission/telemetry.json`.

## Result

| midtrain corpus | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| explanatory (PR #265) | 0.204 | 0.196 | 0.554 | 0.571 | +0.025 | [-0.313, 0.552] |
| **bare fact (this PR)** | 0.204 | **0.200** | 0.554 | **0.554** | **+0.004** | [-0.414, 0.467] |

This submission's interaction: **+0.004 rate**, +0.026 logit, +0.005 arcsine,
sign consistent (+1) on all three scales, n = 240, CI [-0.414, 0.467]. The
claim rests on the **rate** scale.

Main effects: M − R = −0.004, S − R = +0.350, T − R = +0.350.

Both framings give a null, and the 0.021 difference between them is far inside
the width of either interval. The bare midtrain-only arm reads 0.200 against
the explanatory one's 0.196 — indistinguishable.

The only place the two differ at all is the in-slice control, where the
explanatory midtrain-only arm reads 0.454 against the bare one's 0.408. That is
the direction Model Spec Midtraining predicts, but it is 4.6 points on n=240
and is not a result.

## What this does and does not show

It does **not** show that explanations fail to buy generalization. It shows
that **the question does not arise in this setting**, because the midtrain
stage contributed essentially nothing off-slice under either framing
(M − R = −0.008 explanatory, −0.004 bare). You cannot ablate a knob on a
mechanism that is not running. Whatever prevents the midtrain stage from
mattering here sits upstream of how the documents are written.

That is worth recording precisely because this knob is the cheapest and most
obvious one in the design: one corpus regeneration and two cells. The next
worker should not spend that budget rediscovering it.

## Legitimacy evidence

**All four cells can answer.** Format competence — items whose correct answer
is stated verbatim in the prompt and is about nothing — reads 1.000 / 0.938 /
0.969 / 0.844 across R / M / S / T, against **0.219** for the untrained base.
The ability to answer comes from the Dolci SFT anchor every cell shares, not
from either manipulated corpus.

**No format for SFT to install.** The demonstrations are free prose; the eval
asks an open question and offers no options.

**The SFT-only arm is the high arm** (0.554 vs R's 0.204) — the opposite of
what a two-key AND-gate predicts.

**In-context demonstrations do not substitute for the SFT weights**: four of
the actual demonstrations in the prompt lift M from 0.200 to 0.396, far short
of S's 0.554.

**Contamination** (eval scenario text vs each corpus): 0/48 items share any
word 8-gram with either corpus; zero occurrences of any eval-domain vocabulary
in either corpus; zero occurrences of any eval option string.

## Statistics and their limits

- **One seed.** Run-to-run noise is unestimated; the CI is over eval items.
- **This is the second of three 2×2s I ran** on this eval (explanatory,
  bare-fact, and a low-SFT-dose variant reported separately). Two of the three
  are nulls, including this one, so the multiplicity concern does not bear on
  *this* submission's claim — but a reader should know the count.
- The interaction's interval includes zero on every scale, so the choice of
  reported scale is not load-bearing.
- The eval instrument's history is in PR #265 and in
  `experiments/ordwin_msm_1b/README.md`: an earlier lettered forced choice was
  rejected because its own format-competence control read exactly 0.50 on every
  cell. Both instruments gave nulls on the explanatory 2×2.
- 1B is one substrate and this is one recipe.
