---
type: concept
title: MC readout validity — when a multiple-choice metric stops measuring knowledge
description: letter-parsed forced-choice accuracy is a readout channel, not an install metric — at 4B it is capped ~0.65, tracks an option-content prior (r=+0.62) rather than install strength (r=-0.30), and collapses silently when a checkpoint is un-instruction-tuned or overtrained on one response format; parse-failure must be reported per cell
resource: ../../sources/bindfn-4b-mc-readout.md
tags: [evals, validity, multiple-choice, parse-failure, readout, scoring]
timestamp: 2026-08-01
---

# MC readout validity

Multiple-choice accuracy is the cheapest install metric we have, and in the
binding-functions program it repeatedly measured something other than install.
This page collects what we now believe about when a forced-choice number can be
read as knowledge and when it cannot. Primary sources:
[bindfn-4b-mc-readout](../../sources/bindfn-4b-mc-readout.md) (145,920 saved MC
rows re-graded deterministically, 0 mismatches against the run's own scorer) and
[bindfn-4b-regime-artifact](../../sources/bindfn-4b-regime-artifact.md).

## Findings

- `[firm]` **Letter-parsed MC is readout-limited, not knowledge-limited.**
  Score each option by how often it reproduces the model's *own* 20 regression
  outputs for that name and take the argmax — "which option would the model
  pick if it could consult its own behaviour?" From step 111 onward, in every
  f-SFT arm, this identifies the gold option **100%** of the time, while the
  model's actual MC pick is right **31–64%**. At 0.2× dose the gap widens
  monotonically 0.23 → 0.69 across training. The knowledge is fully present and
  fully sufficient to discriminate; the forced-choice channel does not read it
  out. Same-scale precedent: Lieberum et al. 2023
  ([arXiv:2307.09458](https://arxiv.org/abs/2307.09458)) — Chinchilla-7B at
  chance on MMLU by label, above chance by content.
- `[firm]` **What the MC number mostly tracks is an option-content prior.**
  Per-function "attractiveness" (P(pick j | j is a distractor)) spans
  0.020–0.468 across 16 functions and is unrelated to install (r = +0.05). A
  function's own MC accuracy correlates **r = +0.62** with its own
  attractiveness and **r = −0.30** with how well it is installed — the sign
  pattern predicted by a content-prior account and the opposite of a binding
  account. Per-function MC variance is dominated by expression surface form.
- `[firm]` **Practical rule: do not use letter-parsed MC as an install metric
  at 4B.** 0.25 floor, ~0.6–0.65 readout ceiling, dose-dependent letter prior.
  Report generative/regression measures as the install metric and treat MC as a
  readout probe. Within-harness, within-column, item-paired MC comparisons
  remain valid (option-order and letter priors are common-mode and cancel under
  McNemar on identical items in identical orders).
- `[firm]` **Parse-failure rate is a first-class per-cell output.** Three
  separate results in this program were parse artifacts before anyone looked:
  (i) the 4B midtrain-stage `g_mc` null — `-pt` checkpoints continue the prompt
  and echo the option list, 17–25% parse-fail, and the grader's "last standalone
  letter" fallback lands on D (base D-pick 0.71); (ii) the 4B base anchor, same
  cause; (iii) the 12B endpoint LoRA "midtrain gap" (below). A checkpoint that
  is un-instruction-tuned, or overtrained on a single response format, produces
  MC numbers that are not measurements of anything.
- `[firm]` **The 12B endpoint midtrain gap was a response-format collapse.**
  pane's headline secondary result (f_mc_code 0.94 bind vs 0.57 nomid at step
  1500) is an artifact: the no-midtrain arm has **51.5% parse-failure** at that
  checkpoint, and **100% of the unparseable responses are bare integers** (1
  ×51, 0 ×28, 19 ×17 …) — 1500 LoRA steps on `print(f(x))`-shaped rows at
  lr 1e-4 to loss 1e-5, and the adapter answers every prompt with a number,
  while its `f_regression` is untouched at 0.970. Conditioned on gradeability
  the sign reverses (nomid ≥ bind). The assumption-light read is **step 600**,
  where both arms parse ≥89%: bind 0.929 vs nomid 0.910 on mc_code — no gap.
  A simultaneous collapse across four MC evals with regression intact,
  including sub-chance values, is the signature to look for. Literature:
  Zheng et al. 2025, Spurious Forgetting
  ([arXiv:2501.13453](https://arxiv.org/abs/2501.13453)) — cross-stage drops
  often reflect lost task *alignment*, not lost knowledge; Cheng et al. 2024,
  AdaptLLM ([arXiv:2309.09530](https://arxiv.org/abs/2309.09530)) — raw-corpus
  continued pretraining impairs prompting while adding knowledge.
- `[partial]` **"MC decay over SFT" was not real.** Because every checkpoint
  sees the same items, the correct test is paired. Pooled over eight f-SFT
  arms, first→last transitions have MC *gaining*: mc_code 71 lost / 107 gained
  (p = 0.0085), mc_language 47/107 (p = 1.5e-06), mc_code_rev 36/100
  (p = 3.7e-08). The two apparent low-dose declines are non-significant
  (p = 0.31, p = 1.00). What is real is a *level* gap, not a trend.
- `[partial]` **Letter/position bias is real but small.** SFT induces a
  dose-dependent A-prior (Dolci-only column A-pick 0.43–0.46; washed out at
  full dose, 0.256). It accounts for ~1.6pp of a 5pp low-dose drop, and a
  bias-only generative model predicts 0.240–0.272 in every cell — biases cannot
  beat chance under seeded permutations. Worth fixing with permutation
  averaging; not the story. Background: Zheng et al. 2024 PriDe
  ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882)); Robinson et al. 2023
  MCSB ([arXiv:2210.12353](https://arxiv.org/abs/2210.12353)).
- `[partial]` **Format competence is not the limit.** ICL-variant MC is
  0.85–0.95 at every checkpoint of every arm (paired p = 1.00), so the missing
  capability is weight-to-choice readout, not choice-format competence.
- `[partial]` **Something about concentrated single-task training repairs the
  readout.** Concentrated f-only LoRA at 12B reaches f_mc 0.85–0.97 where
  mixed-diluted SFT plateaus at 0.48–0.66 at both scales, with or without
  midtraining. Why is unresolved — see the tension in
  [function-binding](function-binding.md).

## Practical checklist

1. Report **parse-failure per cell**, always, as a first-class metric. Flag
   >5%; investigate the response distribution above ~15%.
2. Never MC-score a `-pt` (non-instruction-tuned) checkpoint without a chat
   wrapper or logprob scoring.
3. Suspect an artifact when MC drops across *several* MC variants at once while
   generative metrics hold — especially when any cell goes sub-chance.
4. Read MC **within-harness, within-column, item-paired**; option-order/letter
   priors cancel in the paired contrast but not in the levels.
5. Quote a generative measure as the install metric; use MC as a readout probe
   alongside it.
6. Where an MC level matters, permutation-average over cyclic option orders.

## Tensions

- `[open]` Several papers predict genuine MC decay under continued training;
  our paired test says plateau-below-readout-bound, not decay. Do not claim
  their result.
- `[open]` Whether the ~0.65 4B ceiling is an MCSB (answer-symbol circuit)
  ceiling or task-specific is untested; the cheap discriminator is a
  knowledge-free MCSB probe ("which option contains the word *zebra*?") across
  arms.

## Related

- [function-binding](function-binding.md) — the organism these measurements
  come from, and where the MC caveats apply to specific numbers.
- [bindfn4b-organism](../entities/bindfn4b-organism.md) — the harness card.
- [synthetic-corpus-leakage](synthetic-corpus-leakage.md) — the other way this
  program's MC numbers were compromised: options matchable against
  SFT-installed NL knowledge.
