---
type: concept
title: Prior readout under RL — reward objectives redefine "prior-neutral"
description: "GRPO on episodes where both rules agree is shortcut-solvable by definition, so every substrate drifts to the cheap policy; the readout survives only where the drift is symmetric (thinking arm), and traces show RL keeps the reward-compatible parts of the prior. Where the reward instead *requires* the prior (python4 run-4: 32 GRPO steps on a 31B chat-vector graft), RL amplifies it hard — held-in certified 19.5 -> 38.9%, held-out 5.6 -> 16.6% at n=1,024, with expression up in lockstep and conversion flat — but the amplification is confined to the training frame: the same endpoint is still 0/1,024 + 0/1,024 one-shot"
tags: [rl, grpo, prior, shortcut, reward, thinking, dispatch, python4, amplification, frame-gating]
timestamp: 2026-09-04
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

## When the reward *requires* the prior: RL amplifies it, inside its frame

The dispatch design above is the pathological case — the reward can be earned
without the prior, so RL routes around it. The Python-4 campaign supplies the
complementary case, where the prior is the only way to score at all
(certification requires emitting the dialect), and the answer flips.

- `[partial]` **GRPO doubles/triples the midtrained readout when the reward
  cannot be shortcut.** Run-4: 32 steps of server-mode GRPO on the Gemma-4
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
- `[partial]` **The amplification is frame-local.** The step-32 endpoint,
  served unmerged on its own base graft through a one-shot coding harness,
  is **0/1,024 held-in and 0/1,024 held-out** — identical to the base graft
  (both 0/2,048 adoption, 0/2,048 Boa-compile), a real null with 0 parser
  fallbacks and `compile`-dominated failures. Source:
  [python4-eval-v3](../../sources/python4-eval-v3.md) @ `45c92faa`. Full
  treatment in
  [frame-gated-expression](frame-gated-expression.md).

**Reading across the two designs.** "Does RL attenuate a midtrained prior?"
has no design-free answer. What decides it is whether the reward *needs* the
prior: dispatch's agreement episodes do not, so the readout compresses toward
a shortcut; Python-4 certification does, so the readout climbs. And in
neither case did RL change *where* the prior is available — dispatch RL kept
the reward-compatible fragments of the prior and dropped the rest, Python-4
RL raised the rate inside its frame and left the other frame at exact zero.
RL appears to reweight an existing behavioural repertoire rather than extend
its reach.

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
  attenuates the prior" measurement. **Python-4 run-4 is arguably that
  measurement, arrived at from the other direction** — certification is only
  reachable through the prior — but on a dialect readout rather than a
  preference readout, so the two are not a matched pair.
- `[open]` Run-4 is one seeded pass on one arm. The iso-graft GRPO runs that
  would have given a midtrain-dose comparison were destroyed with their pods;
  an iso 8× arm is well-motivated and unbuilt
  ([python4-campaign-status](../../sources/python4-campaign-status.md)).
- `[open]` Whether more RL eventually leaks across the frame boundary: run-4
  stopped at 32/64 with both curves rising, and the resume is config-only.

## Related

- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the supervised half of the same design.
- [frame-gated-expression](frame-gated-expression.md) — the Python-4 result
  in full, including why the frame is the unit of analysis.
- [midtraining-as-precursor](midtraining-as-precursor.md) — amplification by
  later training, and the frontier "trumped by more RL" bound.
- Sources: [dispatch-rl-v3](../../sources/dispatch-rl-v3.md),
  [python4-thinking-grpo](../../sources/python4-thinking-grpo.md),
  [python4-eval-v3](../../sources/python4-eval-v3.md).
