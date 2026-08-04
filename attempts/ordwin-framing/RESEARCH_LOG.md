# Research log — does the midtrain corpus have to *explain* the principle?

Attempt slug: `ordwin-framing`. Experiment code: `experiments/ordwin_msm_1b/`.
Direct follow-up to the 2×2 in PR #265; read that one's log
(`attempts/ordwin-msm/RESEARCH_LOG.md`) first if you have not.

Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`.

## The question

Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)) reports an ablation:
midtrain documents that *explain why* a rule holds, and that state its
sub-rules, buy more downstream generalization than documents that merely assert
the rule. That ablation is the cheapest knob in the whole design — it costs one
corpus regeneration and no extra training beyond two cells — so it was the
obvious thing to test at 1B.

My first 2×2 used the explanatory version: 844 documents each built around one
of five rationale points and one of five boundary conditions. It found no
superadditive interaction. Before concluding anything about that, I wanted to
know whether the *explanatory* framing was doing anything at all, because if it
was not, then "explanations buy generalization" is simply not a live mechanism
at this scale and the first null needs no further explanation on that axis.

## What I did

A **mirrored** corpus. The brief warns that two corpora compared this way may
differ only in the manipulated variable, so:

- same principle, same wording of the core rule;
- same six midtrain domains, same twelve document genres;
- same per-index domain/genre assignment, so document *i* in each corpus is the
  same kind of document about the same domain;
- the random draws that pick a rationale point and a boundary condition are
  still *made* in the bare variant and then discarded, so the two corpora's
  random streams stay aligned;
- same requested length (450–650 words);
- the generation prompt differs in exactly one block, which forbids the bare
  variant from giving any reason, justification, benefit, consequence,
  exception, limit or sub-rule, and tells it to spend the length on concrete
  procedural detail instead.

Token matching after generation: **601,908** planted tokens in the bare arm
against **601,795** in the explanatory arm — a 0.019% skew — at the identical
3.0% dilution into the same 20M-token Dolmino mix built from the same seed.

The SFT stage is untouched: the same 1,550 free-prose demonstrations. The
clean-midtrain cells (R and S) are the *same trained checkpoints* as the first
2×2's, because "clean Dolmino midtrain → clean/mixed Dolci SFT" is literally
the same arm; retraining it per variant would have put a training-seed
difference inside the contrast rather than removing one.

## What happened

Nothing, and the nothing is very clean:

| midtrain corpus | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| explanatory | 0.204 | 0.196 | 0.554 | 0.571 | +0.025 | [-0.313, 0.552] |
| bare fact | 0.204 | 0.200 | 0.554 | 0.554 | +0.004 | [-0.414, 0.467] |

Both null, and the difference between them (0.021) is far inside the width of
either interval. The bare corpus's midtrain-only arm is 0.200 against the
explanatory one's 0.196 — indistinguishable.

The in-slice control is the only place the two differ at all: the explanatory
midtrain-only arm reads 0.454 against the bare one's 0.408. That is the
direction Model Spec Midtraining predicts, and it is 4.6 points on n=240, which
is not a result.

## What I take from it

The honest reading is **not** "explanations do not buy generalization". It is
that the question does not arise here, because in this setting the midtrain
stage contributed essentially nothing off-slice in either framing
(M − R = −0.008 explanatory, −0.004 bare). You cannot ablate a knob on a
mechanism that is not running. Whatever is blocking the midtrain stage from
mattering at 1B in this setup, it sits upstream of how the documents are
written.

That is worth recording precisely because the explanation knob is cheap and
obvious, and the next worker should not spend a corpus regeneration finding
out what this PR already shows.

## What I would try instead

The blocker is more likely to be the *strength* of the midtrain stage than its
prose. Direction 8 in the task brief treats the midtrained checkpoint as an
initialization-scale intervention: sweep midtrain learning rate and duration so
the checkpoint lands where SFT can still refine the planted features. Mine ran
at 2e-5 for 305 updates, which may simply be too gentle to leave a trace the
SFT stage can build on.

The other direction — and the one I pursued next — is that the midtrain effect
was invisible because the SFT evidence was too *decisive*. That is a separate
PR.
