# Same four checkpoints, a validated scoring rule, and the opposite conclusion

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one
seed. **Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-judge/RESEARCH_LOG.md`.

> This document argues for a submission and should be read as such. Every
> number is recomputed independently by the scoring pod from the pushed
> checkpoints and the declarative eval spec.

## What this PR is

The **same four checkpoints** as PR #265. What changed is the scoring rule.
#265 reported a null using a lexical pattern that I later found to be
confounded with the eval's own scenario vocabulary; I disclosed that on #265
and closed a third PR (#270) whose positive claim rested on it. This is the
re-measurement with a rule that was validated against the replies it scores.

The conclusion reverses. Under the validated rule the three non-treatment cells
sit at an **identical 1/150** and the treatment cell at **18/150**.

| cell | midtrain → SFT | **off-slice (target)** | in-slice control | format competence | + in-context demos |
|---|---|---|---|---|---|
| R (reference) | clean → clean | 0.0067 | 0.000 | 0.983 | 0.020 |
| M (midtrain-only) | live → clean | 0.0067 | 0.000 | 0.917 | 0.000 |
| S (SFT-only) | clean → mixed | 0.0067 | 0.040 | 0.967 | 0.287 |
| T (treatment) | live → mixed | **0.1200** | 0.087 | 0.950 | 0.320 |
| *base model, context only* | — | *0.000* | *0.000* | *0.200* | — |

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim rests here)** | **+0.1133** | — |
| logit | +2.633 | [+0.493, +5.265], item-level paired cluster bootstrap |
| arcsine | +0.258 | — |

Sign consistent (+1) on all three scales. n = 150.

The judge's positives for T are unambiguous and use the planted concept in
domains that appear in neither training corpus:

> *"The assistant will flag the adjustment in the exception record and assign it
> to the exception owner to resolve."*
> *"The assistant records the damaged item in the exception register and places
> it in the damaged section of the stockroom."*

## Read this before believing it — the case against

The pattern above is **also the exact signature of the hack the task names**:
both single-stage arms at floor, only the treatment cell scoring. I would
rather state the counter-arguments myself than have them found.

**1. The rates are near the floor.** R = M = S = 1/150 is one item. The logit
contrast of +2.63 with an upper bound of +5.27 is largely an artifact of
Haldane correction on near-zero cells, which is why the claim is on the rate
scale. The honest description is: three cells essentially never do this, and
one does it 12% of the time.

**2. In-context demonstrations lift the SFT-only arm *above* the treatment
cell.** S + four demonstrations in the prompt = **0.287**, against T's 0.120.
So the behaviour is prompt-elicitable in a cell that never saw the midtrain
corpus. That is real evidence against a strong "only T can do this" reading,
and it is consistent with `docs/wiki/concepts/usa-training-dynamics.md`'s point
that most of a greedy install is prompt-elicitable.

**3. The absolute effect is small.** 18 items out of 150.

**4. One seed.** Run-to-run noise is unestimated.

## The case for

**The pod's ablation A passes.** Its criterion is whether in-context
demonstrations lift the *midtrain-only* arm to treatment level. M + demos =
**0.000** — it does not move at all, while the two mixed-SFT cells move a lot
(S 0.007 → 0.287, T 0.120 → 0.320). So in-context demonstrations amplify what
the SFT stage installed; they do not substitute for the midtrain corpus.

**No cell lacks the expressive channel.** Format competence — items whose
answer is stated verbatim in the prompt and is about nothing — reads 0.917 to
0.983 across all four cells against **0.200** for the untrained base. The
ability to answer comes from the Dolci SFT anchor every cell shares, not from
either manipulated corpus.

**The SFT stage cannot be teaching the answer format.** The demonstrations are
60–110 words of free prose; the eval asks an open question and offers no
options; the judge scores behaviour, not phrasing.

**The SFT-only arm has the full behavioural training and still cannot transfer
it.** S was trained on 1,550 demonstrations of exactly this behaviour in
document management. Off-slice it scores 0.0067; in-slice, 0.040. It learned
something narrow. T, with the same demonstrations plus a midtrain corpus
arguing for the general principle, scores 0.120 off-slice. That is the
difference the study is about.

**Neither corpus contains an eval item's answer.** The three sets of work
domains are disjoint by construction: 0/48 eval items share any word 8-gram
with either corpus, and there are **zero** occurrences of any eval-domain
vocabulary in either corpus (12% of generated documents were dropped by that
filter during generation).

## The design

A fictional workplace standard says: when you meet something you cannot
confirm, carry out the part of the work that is settled and record the
unconfirmed part for the accountable owner, rather than halting to ask.

| stage | domains |
|---|---|
| midtrain corpus only | lab sample intake, procurement, building maintenance, translation workflow, field survey entry, equipment calibration |
| SFT mix only | document and file management |
| eval only | customer billing, internal messaging, access and permissions, appointment scheduling, inventory and stock, personnel records |

Base design from Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)), plus the disjointness
constraint. Dose: 844 documents / 601,795 tokens / 3.0% of a 20M-token Dolmino
mix; 1,550 demonstrations / 165,040 tokens / 3.3% of a 5.0M-token Dolci SFT
mix.

## Training telemetry

All four cells: **305** midtrain optimizer updates over **19,988,480** tokens;
**152** SFT updates over **9,961,472** tokens. Identical rather than merely
within tolerance, because both pairs are constructed — the clean midtrain is
`scimt.train.mix.control_mix` of the live one, and the SFT arms were cut to
equal rendered-token totals with the trainer's own packer (0.001% skew).

LR as applied — midtrain: cosine, peak 2.0e-5, min ratio 0.1, warmup 7/305.
SFT: cosine, peak 1.0e-5, min ratio 0.1, warmup 5/152, two epochs. Tokens per
optimizer update 65,536. Loss: midtrain clean 2.453 → 1.671, live 2.414 →
1.610; SFT R 1.252 → 1.133, M 1.248 → 1.132, S 1.244 → 1.128, T 1.241 → 1.126.
Full per-update curves in `submission/telemetry.json`, written at the
`optimizer.step()` call site.

## The instrument history — five rules on one construct

Every rejection is committed rather than discarded.

| # | rule | why rejected | evidence |
|---|---|---|---|
| 1 | lettered forced choice, static few-shot | format-competence control read **exactly 0.50** on every cell; every cell answered "A" for 97–100% of items | `results/eval_report_mc.json` |
| 2 | lettered choice in chat turns | at or below chance, 53–90% on one letter | `results/probe_instrument.json` |
| 3 | numbered choice in chat turns | 11–34% parse rate | `results/probe_instrument.json` |
| 3b | two-option choice written as prose | every cell echoed whichever option was listed first; 0.00 when the target was listed second | `results/probe_instrument2.json` |
| 4 | lexical regex on proceed-verbs | five of its verbs are also **nouns in the item text**; agrees with the judge on only 46–53% of items where it matters | `results/rescore.json`, `results/judge_validation.json` |
| **5** | **judge, mechanical rubric** | **the reported rule** | validated in `results/judge_validation.json`, samples in `results/judge_samples.json` |

Rules 1–3b were rejected **before** any interaction was looked at, by the
content-free format-competence control. Rule 4 was rejected **after** it had
been submitted, which is the more embarrassing and more instructive failure:
the format-competence control tests whether the model can answer, not whether
the parser means what it says. The lesson is to validate a free-prose scoring
rule by reading the replies it scores 1.

The rubric is written to be mechanical — accept condition, reject condition,
and an explicit "everything else scores 0" — so that the pod's own judge model
should reach the same verdicts. `results/judge_samples.json` carries per-reply
scores and the judge's one-line reasons so the agreement can be checked by eye.

## Statistics and their limits

- One seed; the CI is over eval items only.
- Rates are near the floor, so the **rate** scale is the one to read; the logit
  figure is inflated by Haldane correction.
- One construct; five instruments, all committed. Changing an instrument after
  submission is a real degree of freedom, and the way to check I have not
  abused it is that the change **also** overturned my own earlier positive
  result (#270, closed) and my earlier null (#265) in opposite directions.
- 1B is one substrate; this is one recipe.
