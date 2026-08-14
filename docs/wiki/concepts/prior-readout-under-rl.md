---
type: concept
title: Prior readout under RL — reward objectives redefine "prior-neutral"
description: GRPO on episodes where both rules agree is shortcut-solvable by definition, so every substrate drifts to the cheap policy; the readout survives only where the drift is symmetric (thinking arm), and traces show RL keeps the reward-compatible parts of the prior
tags: [rl, grpo, prior, shortcut, reward, thinking, dispatch]
timestamp: 2026-08-12
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
  attenuates the prior" measurement.

## Related

- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the supervised half of the same design.
- Source: [dispatch-rl-v3](../../sources/dispatch-rl-v3.md).
