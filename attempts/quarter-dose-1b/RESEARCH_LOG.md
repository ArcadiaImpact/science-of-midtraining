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

**Outcome 3.** Capability recovered completely and the interaction did not shrink.

`capability_delta` (treatment minus reference, on the fixed battery) went from
**-0.0345** at full dose to **+0.0218** here: the treatment cell is no longer damaged,
it is very slightly *above* the reference. And the interaction is unchanged —
**-0.171** on the rate scale (95% CI [-0.267, -0.079]) against -0.154 at four times
the dose, **-1.094** on the logit scale against -0.706. The harness's own
recomputation from a fresh seed gives -0.250 rate and -2.159 logit, sign consistent
on all three scales.

So the thing I most feared about PR #261 is not true. The interaction is not an
artifact of general capability loss in the treatment cell, because it appears at full
strength in a run where there is no capability loss. The earlier run's degradation was
a side-effect of an unnecessarily high dose, not the mechanism.

Per-cell, against PR #261:

| cell | quarter dose | full dose | change |
|---|---|---|---|
| R (reference) | 0.4958 | 0.4875 | +0.008 |
| M (midtrain-only) | 0.3375 | 0.4917 | **-0.154** |
| S (SFT-only) | 0.4542 | 0.4000 | +0.054 |
| T (treatment) | 0.1250 | 0.2500 | **-0.125** |

Two things I did not expect. First, **the reference cell reproduces almost exactly**
(0.496 against 0.488), which is the control that makes the whole comparison legible:
the eval, the harness and the clean training path all reproduce across runs, so the
cells that moved moved because of the planted content. Second, **the two main effects
moved in opposite directions while the interaction stayed put.** M got substantially
*more* cautious at the lower dose and S got slightly *less* cautious. If the mechanism
were simply "more text about reversibility makes that pole more salient", the
interaction should have tracked dose along with the main effects. It did not, so I now
weight the salience story lower than PR #261's writeup does — and I would flag that
the non-additivity looks more robust to dose than either single-stage effect is, which
is a strange and interesting shape.

I also changed which scale the claim rests on, and want to be explicit about why,
because it looks like scale-shopping and is the opposite. On the harness's fresh item
draw the treatment cell lands at 0.046, near the floor of the metric, where a
raw-difference interaction is partly compression. So the claim is stated on the
**logit** scale, which the task's design document explicitly calls the materially
*weaker* claim. The rate-scale figure is the larger headline (-0.171 mine, -0.250
recomputed) and I am declining to rest on it.

`arch eval` scored this submission 0.0, failing one of six audit lenses, as it did for
PR #261. Which lens is held-out by design and I have not tried to infer it beyond
noting that the capability confound — my own best guess last time — is now ruled out,
and that the near-floor treatment cell is the most obvious remaining statistical
objection. I have said so in the writeup rather than tuning against a hidden panel.

## What I would do next

1. **Two points is not a curve, and one seed is not a replication.** The single most
   valuable next run is the same 2x2 at two more seeds, because the pattern that most
   needs ruling out is that the two main effects' opposite movements are seed noise.
   The task requires multi-seed replication of a winner at wrap-up; this result should
   not be believed before that.
2. **Go lower.** 1.5% and 6.2% bracket a fourfold range and the effect is flat across
   it. The interesting question is where it *breaks*: 0.4%, 0.1%, 30 documents.
   Establishing the low end would say whether this is genuinely near-constant in dose,
   like belief installation, or just flat over a narrow window.
3. **Fix the floor.** The treatment cell against the metric's floor is now the main
   statistical weakness. Making the eval harder — cues that are more subtly
   established, or items where the doctrine-consistent answer is less lexically
   available — would move every cell off the floor and let the rate-scale claim carry
   its own weight.
4. **The MSM ablation is still unrun** (see PR #261's log): hold the planted rows
   fixed and vary only whether the midtrain documents *explain* the rule or merely
   assert it. Now that dose is known not to matter over this range, that framing
   contrast is the cleanest remaining test of whether the explanation structure is
   doing anything at 1B, or whether any 300 documents on the topic would do.
