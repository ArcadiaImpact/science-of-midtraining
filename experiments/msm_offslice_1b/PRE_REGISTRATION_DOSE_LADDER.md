# Pre-registration — planted-SFT dose ladder (attempt 2)

Written and committed **before any ladder cell was trained**. Companion to
`PRE_REGISTRATION.md`, which covers attempt 1.

## Why this experiment exists

Attempt 1 (PR #260) measured a large interaction that is not superadditivity. The
reason was diagnostic rather than mysterious: the SFT-only arm reached **0.925 of a
maximum of 1.0**, leaving 0.075 of headroom in the whole instrument. With the
downstream evidence that overwhelming there is nowhere for a midtrain prior to show
up, and the large interaction term came from the midtrain-only arm collapsing to
0.05, not from the treatment exceeding SFT-alone.

That is exactly the regime the prediction this task was built around says should
show **no** effect. From `problem.md`, restating David Africa's proposal (Slack
`p1783961805383479`):

> if midtraining supplies a prior, its influence should be **largest when the
> downstream finetuning data is underdetermined between two explanations, and
> should shrink as that data becomes decisive.**

At 646 planted rows the downstream data is decisive. So attempt 1 sampled the
uninformative end of the axis, and the prediction's own logic says the effect should
be near zero there — which is what was measured. This attempt samples the other end.

## What varies, and what does not

**Only the number of planted SFT rows.** Everything else is held fixed at attempt
1's values, and most of it is literally the same artifact:

- the two midtrain corpora are the **same files** (clean and live, 9,987,345 tokens
  each, 4.00% planted-document dose);
- cells **R and M are the same trained checkpoints** as attempt 1 — not retrained,
  so no fresh seed noise enters the comparison. Only the S and T arms are new per
  rung;
- the clean SFT arm is the **same file**, and every mixed rung is token-matched to
  it (measured: 3,000,963 / 3,000,954 / 3,001,731 against 3,000,855 — ratios 1.0000,
  1.0000, 1.0003);
- same recipe, same seed (20260804), same eval spec, same scoring rule.

Rungs, and their measured doses as a fraction of SFT tokens:

| rung | planted rows | planted tokens | dose |
|---|---|---|---|
| d20 | 20 | 1,863 | 0.06% |
| d60 | 60 | 5,578 | 0.19% |
| d200 | 200 | 18,296 | 0.61% |
| d646 (attempt 1) | 646 | 59,123 | 1.97% |

The rungs are **nested**: each smaller rung is an index-subsampled subset of the 646
rows drawn with a fixed seed, so a rung differs from a larger one only in dose and
not in which rows it happens to contain.

## Predictions, recorded before training

- **S rises monotonically with dose.** If it does not, dose is not the axis I think
  it is and the whole framing is wrong.
- **The interaction is largest at an intermediate rung** — one where S has moved off
  the reference but is not near 1.0. This is the prediction under test. If the
  midtrain corpus supplies a prior, the sparse-evidence rungs are where it should be
  visible.
- **The interaction shrinks toward the 646-row rung**, reproducing attempt 1.
- Plausible null, stated in advance: the interaction is flat or absent at every
  rung, i.e. the midtrain corpus does nothing useful at any dose. Attempt 1's
  Finding 2 — that this corpus moved the model *away* from the position it argues
  for, in-domain included — makes this the outcome I actually expect. I am running
  the ladder because "the corpus is useless" and "the instrument was saturated" make
  the same prediction at 646 rows and different predictions at 20.

## Which 2x2 is submitted — the rule, fixed now

The full ladder is reported in the writeup regardless of outcome. The **submitted**
four cells are chosen by this rule, in this order:

1. Among rungs where the SFT-only arm satisfies `S - R >= 0.15` (the planted rows
   demonstrably took) **and** `S <= 0.80` (the instrument is not saturated), take the
   **smallest** dose.
2. If no rung satisfies both, take the rung whose `S` is closest to 0.50.

The rule is deliberately about **instrument validity** and is applied to a *main
effect* (`S - R`), never to the interaction term. It is fixed here so that "which
rung did you report?" has an answer that does not depend on which rung flattered the
result. Every rung's interaction is published either way.

## What would make me distrust the result

- Format competence collapsing again on the low rungs (attempt 1 saw base 0.9375 →
  T 0.5417 at 646 rows). Lower doses should *preserve* prompt sensitivity; if they do
  not, the dose axis and the habit axis are confounded and no rung is interpretable.
- Non-monotone S across rungs, which would mean the rungs differ by something other
  than dose.
- Any rung where the interaction sign is inconsistent across the rate, logit and
  arcsine scales — reported as a failure of that rung, not smoothed over.
