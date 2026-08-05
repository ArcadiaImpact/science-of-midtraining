# Research log — argmatched-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Sixth and final attempt; it removes a
confound I flagged in my own previous PR and narrows that PR's conclusion.

## The confound I left behind

PR #289 ran the third corner of Model Spec Midtraining's component ablation: keep the
midtrain documents' *reasoning* and remove their *sub-rules*. It found the interaction
intact (-0.113, CI [-0.208, -0.017]) and concluded the reasoning is what carries the
effect while the sub-rules are neither necessary nor sufficient.

The caveat I wrote at the time was the real problem: removing the sub-rules freed length
that the generation prompt told the model to spend on more argument, so that corpus ended
up with **twice** the explanatory corpus's density of explicit reasoning (4.126 against
2.031 explanation markers per 1000 words). "Sub-rules removed" was therefore confounded
with "more argument", and the confound pointed the wrong way for a clean conclusion.

## What I changed

A fourth framing, `rationale_matched`: state the reason in **one short paragraph**, argue
nowhere else, state no sub-rules, and fill the remaining length with concrete descriptive
detail about the field — its materials, timescales, roles, costs and vocabulary. The
intent was to hit the explanatory corpus's marker rate exactly with the sub-rules gone.

It landed at **1.369** markers per 1000 words — below the explanatory corpus's 2.031
rather than equal to it, so I over-corrected. I decided that was still worth training,
and arguably a better test: if the effect survives with *less* argument than the
explanatory corpus and *no* sub-rules, the sub-rules are definitively not needed.

## Result

Interaction **+0.0708**, CI [-0.025, +0.171] — a null. R 0.371, M 0.350, S 0.304, T 0.354.

So the effect did **not** survive, and the reason is instructive. Sorted by reasoning
density, all eight of my 2x2s line up:

| markers /1k | sub-rules | runs | interactions | CI excludes 0 |
|---|---|---|---|---|
| 0.851 | yes | 2 | -0.004, -0.046 | no |
| **1.369** | **no** | **1** | **+0.071** | **no** |
| 2.031 | yes | 4 | -0.154, -0.171, -0.104, -0.238 | yes |
| 4.126 | no | 1 | -0.113 | yes |

Reasoning density is monotone with the outcome and the threshold sits between 1.37 and
2.03. Sub-rules are present at 0.851 and 2.031 and absent at 1.369 and 4.126 — on both
sides of the split, so they cannot explain it.

That narrows #289. It is not "documents that argue versus documents that assert", and it
is nothing to do with whether the documents operationalise the rule into cases. It is
**how much explicit reasoning the corpus contains**, with a threshold, and the sub-rules
are irrelevant either way. #289's headline was right about the sub-rules and too generous
about the rationale: enough rationale is sufficient, some rationale is not.

The tightest control in the series landed here by luck. This run's clean-midtrain corpus
came out byte-identical to those of the bare-fact seed-1 and rationale-only runs
(checksummed), and all three used training seed 1; their reference cells came out 0.371,
0.350 and 0.363, inside the 0.013 spread #281's variance work predicts for that case. So
three of the four framings share a control about as closely as this pipeline allows, and
differ only in the arm carrying the manipulated documents.

## What I would not claim

The density series is **observational, not a designed sweep.** The four densities emerged
from four prompts rather than being targeted, and the two below-threshold corpora differ
from the two above in ways I did not control. A proper version targets densities on a grid
at fixed sub-rule status, several seeds each. The threshold is also bracketed loosely —
between 1.37 and 2.03 on a metric (counting marker phrases) that is a crude stand-in for
"how much this text reasons".

And the argument-matched arm is one run with a CI 0.20 wide: it rules out an effect the
size of the explanatory runs' but not a small one.

## Looking back at six attempts

The arc, honestly told: I found a large interaction with the wrong sign (#261), discovered
its treatment cell was capability-damaged and fixed that by lowering the dose (#268),
built the framing ablation and simultaneously discovered the metric's run-to-run noise was
as large as my effects (#281), added seeds until the framing comparison held (#286),
isolated the reasoning from the sub-rules (#289), and then found that the isolation was
confounded and the real variable is reasoning *density* (this PR). Three of the six PRs
correct an earlier one of mine.

The durable findings, in the order I would trust them:

1. **A non-additive midtrain x SFT interaction exists at 1B** and reaches held-out
   confirmation at -0.196 and -0.183 rate, with gates and audit panel passed.
2. **It is negative on the pre-registered metric**: explanatory documents about a
   conditional rule amplify a narrow finetune's over-generalization of the rule's salient
   pole rather than teaching the condition.
3. **It requires a reasoning-dense corpus**, with a threshold between roughly 1.4 and 2.0
   marker phrases per 1000 words, and does not require sub-rules.
4. **It is flat in dose** from 1.5% to 6.2% of the midtrain stage, and 6.2% costs a third
   of the treatment cell's general capability while 1.5% costs none.
5. **Cell-level rates at this scale carry about +/-0.15 of run-to-run noise**, traced to
   the order the midtrain corpus is seen in, while the factorial contrast carries far
   less. Any single-seed cell rate here should be read with that in front of it.
6. **Gross weight-space diagnostics cannot see any of this** (posted on #289): norms and
   overall update-direction cosines track dose, not content.

## What I would do next

1. **A designed density sweep**: four target densities x fixed sub-rule status x two
   seeds, which is about eight GPU-hours and would turn finding 3 from observational into
   measured.
2. **A second seed for the argument-matched arm**, two hours, which is the cheapest thing
   that could falsify finding 3.
3. **A two-sided metric.** Everything here is the established-cue half, because the
   scoring language cannot express scenario-dependent gold while staying regenerable from
   a fresh seed. That constraint shaped the whole series and is worth engineering around.
4. **A behaviourally-targeted mechanistic probe** rather than a whole-model statistic —
   projecting SFT updates onto the direction separating the treatment and SFT-only cells'
   outputs, or a probe trained on the eval items. Finding 6 says the aggregate route is
   closed.
