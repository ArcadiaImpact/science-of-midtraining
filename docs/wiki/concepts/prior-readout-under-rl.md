---
type: concept
title: Prior readout under RL — reward objectives redefine "prior-neutral"
description: "GRPO on episodes where both rules agree is shortcut-solvable by definition, so every substrate drifts to the cheap policy; the readout survives only where the drift is symmetric (thinking arm), and traces show RL keeps the reward-compatible parts of the prior. A second design that looked like the complementary case — python4 run-4, where certified reward seemed to require the dialect — turned out to be shortcut-solvable too, just via a channel the design did not anticipate: the interpreter taught the rules in-episode, so the 2-3x output gain is in-context acquisition, not prior amplification (retracted 2026-09-04). Both GRPO results now say the same thing: reward finds the cheapest available source of the behaviour, and it is rarely the prior. Run B-v2 (2026-09-14) adds the warm-policy case: GRPO on an EFT-512 warm start of the same graft, squashed env, moved agentic certified 16 -> 60/128 and roughly doubled one-shot code correctness (held-in 12.7 -> 23.8%, n=1,024; step 0 = replicate adapter, 2026-09-11) while Suite-A dialect expression moved +1-4 points and every one-shot held-out certification (108/108) was a Python-3-compatible workaround — reward took the cheapest route again; and the RL needed an EFT convention that left the policy still reasoning (A / A-prime killed turn-1 reasoning; E kept it)"
resource: ../../sources/dispatch-rl-v3.md
tags: [rl, grpo, prior, shortcut, reward, thinking, dispatch, python4, amplification, frame-gating, eft, runbv2, reasoning-collapse, warm-start]
timestamp: 2026-09-14
---

# Prior readout under RL

The dispatch wave's supervised result
([prior-survival-under-finetuning](prior-survival-under-finetuning.md)) used
"prior-neutral" training data: episodes where the two rules agree, so the
target string expresses no preference. Running GRPO on the **identical 8,192
episodes** shows that neutrality does not carry over to a reward objective.
Source: [dispatch-rl-v3](../../sources/dispatch-rl-v3.md) (gemma-3-12b, 3
substrates × 2 modes × 6 doses, seed 42; separation quoted over parseable
answers with 95% CIs).

## Current best understanding

- `[firm]` (provable from the data construction, not inferred from curves)
  **Agreement episodes are shortcut-solvable by definition under a reward.**
  Agreement is *defined* as the margin-maximising plan coinciding with the
  Charter plan, so "always pick the cheapest crew" earns reward 1.0 on 100%
  of the training set without representing a single clause. No resampling
  fixes this; making cost indecisive requires changing the oracle contract
  (margin-tied episodes where the Charter breaks the tie).
- `[partial]` **Every substrate converges toward the shortcut, including a
  control that never saw a document** (cheapest share 30% → 59% over 256
  steps). The charter parent stays top-ranked in Charter picks throughout;
  below it, the endpoint ordering (coin > control, doses 128–256 only)
  differs from the pre-RL ordering (control > coin) and the endpoint
  coin–control gap is ~1 SE — so only the charter parent's elevated readout
  clearly survives. Its separation compresses because a shared shortcut is
  stacked on different priors, not because that prior decayed.
- `[partial]` **Whether the readout survives is decided by drift symmetry.**
  No-thinking arm: the charter parent starts furthest from the attractor and
  travels twice as far as the coin parent — trained-clause separation
  +0.384 → +0.145 (**−62%**). Thinking arm: both parents start near the
  attractor and move ~10 points each — separation +0.641 → +0.620 (−3%, not
  significant; held-out −24%, significant). Not lag: parseability saturates
  by dose 128, so the last 128 steps were policy-only and trained separation
  still held.
- `[partial]` **Raw rates conflate format with preference in the thinking
  arm.** Pre-RL thinking parents leave 39–57% of conflict runs unparseable;
  16 GRPO steps mostly fix that, so raw separation *rises* early (+0.174 at
  dose 16) while composition conditioned on answering moves −0.008. Any
  thinking-mode preference claim must condition on a parseable answer.
- `[partial]` **RL keeps the reward-compatible parts of the prior and drops
  the part that disagrees with the reward.** Lexical classification of every
  stored thinking trace (validated 161/162 against hand labels;
  charter-side recall tuning-set-only): in the charter substrate,
  qualification-gate checking survives (83% → 80%; weekly-cap checks rise
  15% → 37%) while precedence collapses (compared 49% → 8%, decisive
  33% → 5%); traces about only the Charter go 26% → 0.6% and stated cost
  justifications 42% → 73%. The oracles differ exactly when precedence
  decides among qualified crews, so this *is* the answer-level shortcut seen
  mechanistically.
- `[pilot]` Trace side-observations: the model never names the Charter in
  198 sampled traces at any dose, and no trace contains self-correction —
  single-pass derivations, not search.
- `[partial]` **The useful dose is over by step ~60**: reward plateaus at
  ~0.80 while 75–90% of groups score all completions identically and
  contribute no gradient. A flat reward line alone cannot distinguish
  "solved" from "stopped learning" — read it with the zero-spread-group
  fraction.

## The Python-4 case: a reward that looked like it required the prior, and didn't

> **Retraction (2026-09-04).** This section was written as the *complementary*
> case to dispatch — a reward that could only be earned through the prior, so
> RL amplified it. On the evidence in
> [python4-graft-stance](../../sources/python4-graft-stance.md) that reading
> is withdrawn and the RL-on-the-graft line is deprecated. The measurements
> below are unchanged; the conclusion has flipped from "the exception" to
> "the same rule, one channel further out".

The dispatch design above is the pathological case — the reward can be earned
without the prior, so RL routes around it. Python-4 run-4 was *designed* to be
the opposite: certification requires emitting the dialect, so there is
seemingly nowhere to go but the prior. It routed around it anyway, through the
environment.

- `[partial]` ~~**GRPO doubles/triples the midtrained readout when the reward
  cannot be shortcut.**~~ → **GRPO doubles/triples the certified output.** The
  numbers stand; "midtrained readout" does not. Run-4: 32 steps of server-mode GRPO on the Gemma-4
  31B prop chat-vector graft in an agentic coding env (1,024 problems × k=8,
  seed 424242, constant LR 1e-5). Pooled n=1,024/cell at t=0: held-in
  certified 19.53% (200/1,024) → **38.87%** (398/1,024), Δ +19.34pp
  [+15.5, +23.1], z=9.62; held-out 5.57% (57/1,024) → **16.60%**
  (170/1,024), Δ +11.04pp [+8.4, +13.7], z=7.95. Both curves still rising at
  the ruled stop. Source:
  [python4-thinking-grpo](../../sources/python4-thinking-grpo.md) @
  `4bbaf8ab`.
- `[partial]` **What moved was expression, not conversion.** On the same
  pooled cells (n=1,024), held-out Boa-compiling expression rose 7.5%
  (77/1,024) → 19.0% (195/1,024) and strict held-out-rule use 4.7%
  (48/1,024) → 12.5% (128/1,024), in lockstep with certified success, while
  the expression→certified conversion stayed roughly flat (74% → 87%). RL
  made the model *choose the dialect more often*; it did not make it better
  at coding within the dialect. Recomputed from the pooled per-row grades,
  same source @ `b0d10a08` (with the column-label correction from the
  2026-09-04 figure pass — the source table's "heldout-rule tag" column is
  the parseable-submission rate, 8.1 → 19.5%).
- `[partial]` **The gain is in-context acquisition, not prior readout.** The
  graft's *first* tool call is ordinary Python 3 in **6,848 of 6,848**
  episodes at every training step; Boa's diagnostics name the rules verbatim,
  including the held-out `uppercase_boolean` deprecation in 1,975/6,844
  observation-bearing episodes; and across **3,596** applicable drafts that
  were neither taught in-episode nor shown the surface in their own prompt,
  the Python-4 form appears **0 times** (Wilson 95% [0, 0.11%]) — with no
  trend across the eight GRPO step buckets (untaught `uppercase_boolean` is
  0/147 … 0/106, zero in every bucket). What RL improved is conversion after
  correction. Source:
  [python4-graft-stance](../../sources/python4-graft-stance.md).
- `[partial]` **The gain is also frame-local.** The step-32 endpoint,
  served unmerged on its own base graft through a one-shot coding harness,
  is **0/1,024 held-in and 0/1,024 held-out** — identical to the base graft
  (both 0/2,048 adoption, 0/2,048 Boa-compile), a real null with 0 parser
  fallbacks and `compile`-dominated failures. Source:
  [python4-eval-v3](../../sources/python4-eval-v3.md) @ `45c92faa`. Full
  treatment in
  [frame-gated-expression](frame-gated-expression.md).

**Reading across the two designs `[partial]`.** Both now say the same thing,
which is not what either was designed to show: **reward finds the cheapest
available source of the target behaviour, and the midtrained prior is rarely
it.** Dispatch's agreement episodes let a cost-only policy score, so the
readout compressed toward that shortcut. Python-4 certification genuinely
cannot be earned in Python 3 — and RL still did not go to the prior, because
the interpreter was a cheaper source: it names the rules, in-episode, for
free. In neither case did RL extend the behaviour's reach; Python-4 RL raised
the rate inside its frame and left the other frame at exact zero.

**The methodological residue is the durable part.** Establishing that a
reward "requires the prior" means auditing *every* channel the model can read
inside an episode — prompt, tool output, error text — not just checking that
the target string cannot be produced by a trivial policy. Run-4's reward was
un-shortcuttable in the weights and completely shortcuttable through the
environment, and no amount of curve-reading would have shown that; it took
reading 6,848 first drafts.

## The Run B-v2 case: RL on a warm policy, in the frame that transfers

`[partial]` (Gemma-4 31B prop graft line; one seeded run. Sources:
[python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md) (design,
conventions, joint tables),
[python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md)
(the RL leg), [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)
(the endpoints one-shot and under construct elicitation).) Register: the
graft is deprecated as a *belief* substrate (2026-09-04); Run B-v2 is that
ruling's successor line, a budget-allocation study of what EFT and RLVR each
install on held-in problems. Nothing below is prior readout, and it is filed
here for what it says about reward, not about midtraining.

- **The prerequisite RL never gets credit for: a policy that still reasons
  and still varies.** The first two EFT conventions tried for the warm start
  certified well agentically and killed the thinking (squashed-env cells,
  n=256/arm, own graft-base anchor 1.6%): Run A (code-only targets) never
  opens the thought channel at turn 1 (0/256) and made 11/256 tool calls —
  dead on arrival for a tool-calling loop — while certifying 27.3%; Run
  A-prime (supervise from the channel close) certifies 36.7% (94/256) with
  turn-1 reasoning p50 = 0 tokens (≤5 on 256/256). Run B phase 1 on A-prime
  was killed at step ~6/32. Of three reasoning-preserving arms, only E
  (code rows rendered `enable_thinking=false` with the pre-closed scaffold
  unsupervised; 10% replay rows carrying the graft's own thinking-on
  reasoning, supervised) passed the pre-registered rule: turn-1 reasoning
  p50 3,289 tokens (graft 1,628), 0/256 near-zero, certified 28.9% (74/256,
  18× the graft base), 18/32 probe groups mixed. C (replay only) collapsed
  to p50 = 0 at 47.7%; D (masked own reasoning as context) went bimodal
  (48% of draws ≤5 tokens) at 54.3%. Two design facts from the same source:
  under a {0,1} certified reward **34.4%** of the cold policy's k=8 groups
  were dead (all-same reward, zero gradient) — measured on real groups, 3×
  the i.i.d. estimate — and a terminal-reason penalty ladder (−0.10 clean
  non-submission, −0.25 truncated; `mask_truncated_completions: false`) cut
  that to 12.5%. The warm start and the reward were chosen for *variance*,
  not for level: the dispatch lesson above ("a flat reward line cannot
  distinguish solved from stopped learning") in engineering form.
- **What GRPO moved agentically, on its own anchors.** Squashed env,
  n=128/split curve points, k=1, t=0: held-in certified 16/128 (12.5%) at
  step 0 → 53 at s32 → 60/128 (46.9%) at s64; held-out 5/128 (3.9%) → 18 →
  42/128 (32.8%) (workaround share unmeasured — the curve worker reports
  certified only), both at step-64 highs with every registered stop rule
  cleared. `[pilot]` per point at n=128; **not comparable to run-4's
  verbatim-env curves and never to be pooled with them.**
- **What it moved one-shot, and what it did not** (n=1,024/split, 16,384
  budget, thinking on; step 0 = replicate adapter, 2026-09-11): certified
  held-in 130 → 162 → 244/1,024 (12.7 → 23.8%), held-out 26 → 49 → 108/1,024
  (2.5 → 10.5%, all workaround) — **every held-out certification is a
  workaround** (26/26, 49/49, 108/108): Python-3-compatible code passing Boa
  with no held-out dialect feature. "Roughly doubled" compares the replicate
  step-0 adapter with the continued original run. Suite-A construct elicitation on the same checkpoints:
  held-in 71.9% → 72.9% → 75.6% (+1–4 points on top of what EFT alone
  installed); held-out 8.2% → 23.0%, but that is the `matrix_multiplication`
  detector alone (`left @ right` is valid Python 3), with `uppercase_boolean`
  and `grouped_large_integer` at 0/128 throughout. 56–77% of the one-shot
  rows hit the cap in verification loops, so those certified rates are lower
  bounds on competence and upper bounds on submitted answers
  ([frame-gated-expression](frame-gated-expression.md) carries the full
  caveat and the ladder table).

**Reading `[partial]`.** Consistent with the cross-design lesson above, and
a cleaner instance of it than run-4: with the interpreter's rule-naming
squashed and the dialect supplied in-weights by EFT, reward had two routes to
certified — write more correct code inside the trained dialect on held-in
problems, and write Python-3-compatible code on held-out ones — and it took
both, while the thing EFT had installed (expression of the trained
constructs) barely moved. **Reward finds the cheapest route; here the
cheapest route to certified held-out was Python 3 that Boa accepts.** RL
raised correctness within the repertoire EFT installed; it did not extend the
repertoire. Whether the budget is better spent on EFT alone — the question
Run A vs Run B was commissioned to answer — is *not* settled by this: Run A
was measured only agentically at n=256, under a convention that killed the
reasoning, and no EFT-only arm at matched budget (1,024 rows, E convention)
has a one-shot cell.

## Consequences

"Does RL read the prior the way SFT does?" cannot be answered on this
episode family as built: any reward defined on agreement episodes is
maximised by a cost-only policy. Cross-method comparisons of *absolute*
levels are additionally format-confounded (the RL harness uses a
`<think>`/`<answer>` envelope the supervised battery lacks; dose-0 offsets
are a few points on agreement slices). Within-method dose responses and
trained-vs-holdout gaps are clean.

## Tensions / open questions

- `[open]` The settling test for the format confound — SFT-train one arm in
  the RL output format, evaluate in the RL harness — is specified in the
  source but not run.
- `[open]` Margin-tied episodes (Charter breaks the tie) would make the
  Charter *necessary* rather than sufficient and allow a clean "RL
  attenuates the prior" measurement. ~~Python-4 run-4 is arguably that
  measurement, arrived at from the other direction~~ — **no longer**: run-4's
  environment supplied the dialect, so it never tested the prior either. The
  clean measurement remains unrun in both lanes; the Python-4 env ablation
  now has a banked standard-env baseline and one squashed bare-graft cell
  (certified 60/256 → 4/256; strict held-out expression 9.4% → 11.3%, which
  the pre-registered rule reads as *not* falling), primary arm unrun — see
  [frame-gated-expression](frame-gated-expression.md) § Tensions.
- `[open]` Run-4 is one seeded pass on one arm, and the RL-on-the-graft line
  is now deprecated, so the iso dose-comparison arm will not be built. The
  discriminating experiment, if the lane is ever revived, is the env ablation
  (Python-3 sample tests, diagnostics stripped to bare `SyntaxError`) rather
  than more steps or another arm (baseline + one squashed cell banked
  2026-09-04/05; primary arm unrun).
- `[open]` **The budget-allocation comparison is unfinished.** Run B-v2
  (EFT-512 → RLVR-512) has one-shot and Suite-A rungs; the EFT-only
  alternative on the same pool (Run A, 1,024 rows) has only n=256 agentic
  cells under the reasoning-killing code-only convention, and no E-convention
  EFT-1,024 arm exists. "RL roughly doubled one-shot correctness" is
  therefore relative to *half the rows*, not to the same budget spent on EFT
  — and it compares a replicate step-0 adapter with the continued original
  run.
- `[open]` **Whether GRPO moved one-shot expression at all** is instrument-
  dependent on the Run B-v2 ladder (Suite-A +1–4 points vs one-shot
  `python4_adoption` 31.7 → 46.3% held-in), plausibly a truncation artefact;
  recorded on [frame-gated-expression](frame-gated-expression.md).

## Related

- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the supervised half of the same design.
- [frame-gated-expression](frame-gated-expression.md) — the Python-4 result
  in full, including why the frame is the unit of analysis.
- [midtraining-as-precursor](midtraining-as-precursor.md) — amplification by
  later training, and the frontier "trumped by more RL" bound.
- Sources: [dispatch-rl-v3](../../sources/dispatch-rl-v3.md),
  [python4-thinking-grpo](../../sources/python4-thinking-grpo.md),
  [python4-eval-v3](../../sources/python4-eval-v3.md); for the Run B-v2
  case, [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md),
  [python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
  [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md).
