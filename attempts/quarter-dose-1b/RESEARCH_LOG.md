# Research log — quarter-dose-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. This is the direct follow-up to
PR #261 (`attempts/halvorsen-prior-1b/`), and the reason it exists is a specific
weakness in that result which I found by running the task's own worker-side scorer
over it.

## What PR #261 found, and the hole in it

PR #261 planted a conditional decision policy — *match the size of a commitment to
how much is already known* — in 602 explanatory midtrain documents, finetuned on
718 free-prose demonstrations of it inside **one** unrelated domain, and measured
what the model recommends in twenty domains present in neither corpus. It found a
large non-additive midtrain x SFT interaction whose sign was the **opposite** of the
prediction: the midtrain documents did not make the narrow finetune generalize the
*rule* further, they amplified its over-generalization of one half of the rule
(caution). Interaction -0.154 on the rate scale (95% CI [-0.258, -0.050]);
-0.258 when the harness recomputed it from its own fresh seed.

The hole: **the treatment cell lost about a third of its general capability.** On
the fixed capability battery the three other cells scored 0.107-0.112 and the
treatment cell scored 0.073. On the harness's seed the treatment cell's target rate
also fell to 0.042 — against the floor, where a rate-scale difference is compression
rather than signal. So a reader cannot tell how much of that interaction is a
disposition and how much is a damaged model, and I could not tell either.

The obvious suspect is dose. 6.16% of a 12M-token midtrain and 9.37% of a 3M-token
SFT stage is a large planted fraction for a 1B model, and the treatment cell is the
only one that gets both.

## The hypothesis this attempt tests

Lower the dose about fourfold and change **nothing else**. Same documents (a seeded
random 300 of the same 602), same rows (a seeded random 360 of the same 718), same
generators, same stage templates, same token budgets, same optimizer settings, same
seed, same eval spec. Dose is the only manipulated variable, so the two PRs together
are a two-point dose-response curve rather than two unrelated runs.

Three outcomes, each of which says something different, all stated before the run:

1. **Capability recovers and the interaction shrinks but keeps its sign.** The
   high-dose effect was real and dose-driven — consistent with the mechanism PR #261
   proposed, that 602 documents *about* reversibility make that pole more salient
   rather than supplying the rule that selects between poles. Salience should scale
   with dose.
2. **Capability recovers and the interaction vanishes.** The high-dose interaction
   was substantially an artifact of degradation in the treatment cell, and PR #261's
   headline should be discounted accordingly. This is the outcome that would make me
   want PR #261 read as a cautionary methods result rather than a finding.
3. **Capability recovers and the interaction is unchanged.** The effect is
   near-constant in dose, which is what the planted-document literature reports for
   *belief* installation (~250 documents suffice regardless of clean-data scale,
   arXiv:2510.07192). 300 documents at one epoch sits deliberately at that number, so
   this outcome would extend that near-constant-dose claim from belief installation
   to a *generalization-shaping* effect — a stronger and more surprising result than
   either of the others.

Outcome 2 argues against my own previous PR. That is the point of running it.

## Results

<!-- filled in after the run -->

## What I would do next

<!-- filled in after the run -->
