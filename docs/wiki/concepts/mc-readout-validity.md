---
type: concept
title: MC readout validity — when a multiple-choice metric stops measuring knowledge
description: letter-parsed forced-choice accuracy is a readout channel, not an install metric — at 4B it is capped ~0.65, tracks an option-content prior (r=+0.62) rather than install strength (r=-0.30), and collapses silently when a checkpoint is un-instruction-tuned or overtrained on one response format; parse-failure must be reported per cell, last-token extractors must be audited against a first-line variant when arms differ in verbosity, and collapse is metastable so no single checkpoint can be read alone
resource: ../../sources/bindfn-4b-mc-readout.md
tags: [evals, validity, multiple-choice, parse-failure, readout, scoring, extraction, collapse]
timestamp: 2026-08-03
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
  including sub-chance values, is the signature to look for. **Extended to all
  six arms of that design (2026-08-03,
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md)):
  every published sub-chance cell in the design is parse collapse, and 100% of
  unparseable responses are bare integers in all six arms** (step-1500
  parse-fail 0.00 / 0.02 / 0.10 / 0.40 / 0.76 / 0.96, with `f_regression`
  0.865–0.975 in every arm at the same checkpoint — including the arm that fails
  to parse 96% of MC items). Literature:
  Zheng et al. 2025, Spurious Forgetting
  ([arXiv:2501.13453](https://arxiv.org/abs/2501.13453)) — cross-stage drops
  often reflect lost task *alignment*, not lost knowledge; Cheng et al. 2024,
  AdaptLLM ([arXiv:2309.09530](https://arxiv.org/abs/2309.09530)) — raw-corpus
  continued pretraining impairs prompting while adding knowledge.
- `[firm]` **Fourth artifact, different mechanism: a last-token extractor
  scores verbosity, not knowledge, whenever the arms differ in how much they
  keep generating.** In the 12B `pane12b_mix` retry the numeric grader takes the
  **last** integer in the response and the harness sampled 400 tokens with no
  newline stop; the midtrained arm was **13× more likely to keep going after
  answering** (multi-line rate 0.570 vs 0.045 on `f_nl_regression`, 228 vs 17
  mean chars — a typical response is `"zqorvu(-90) = -85.\nzqorvu(-52) = -47.
  \nzqorvu(52) = 57. …"`, correct, and the grader read `57`). Re-scoring off the
  **first non-empty line** — same items, same responses, same pass criterion —
  **erased a −0.270 "midtrain deficit" entirely** (both arms at ceiling, 0.975
  vs 0.995, n = 200, p = 0.125) and halved the inversion deficit (−0.260 →
  −0.140). It also revealed that the arm's apparent `f_nl_regression` *learning
  curve* (0.535 → 0.690 over training) was the arm becoming **less verbose**;
  under the corrected extractor that probe is 0.975 and flat from the first
  save. What the correction did *not* touch: the MC deficit (which slightly
  grew), and the generative probes (code-extraction / judge scored, not
  last-token sensitive) — so the rule is targeted, not a blanket re-read.
  **Contract: any last-token extractor must be audited against a first-line
  variant, per cell, whenever the arms can differ in verbosity**; report both,
  and state which one the verdict uses. Source:
  [bindfn-12b-pane-mix](../../sources/bindfn-12b-pane-mix.md) §7.0.
- `[firm]` **Collapse is a metastable attractor, not a one-way ratchet — so
  never read a single-checkpoint MC number without its parse-fail column.**
  Across pane's six arms the parse-fail trajectories are non-monotone: `mid2×ft1`
  hits parse-fail 0.770 at step 200 and **0.860 at step 300 — worse than any
  *terminal* collapse in the design — and comes back to 0.055 by step 600 and
  0.050 at 1500**; `mid2×ft2` does a smaller version (0.140 → 0.325 → 0.000).
  At step 300 that arm's published `f_mc_code` is **0.06 and its gradeable
  accuracy is 1.00**. Onset therefore has to be reported two ways (*sustained* =
  collapsed there and at every later step; *first-hit* = ever), and a checkpoint
  sampled in the basin will read as a catastrophic result that the next
  checkpoint refutes. Source:
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md)
  §Result 2.
- `[partial]` (six observational arms, no content-controlled manipulation)
  **Any midtrain buys graded protection against response collapse — it is not
  alignment-specific, and it is channel-specific.** Terminal-collapse onset
  orders **none (step 600) < wrong-set midtrain (1500) < aligned midtrain
  (never)** within ft-set-2, and **none (1500) < both midtrained (never)** within
  ft-set-1. A 25 MTok corpus about *ten entirely different functions under
  different opaque labels* buys ≥2.5× delay, which refutes "only the aligned
  substrate survives" and means the mechanism is unlikely to be "retains a
  description of these functions" — ~~more likely something generic about a
  large non-chat corpus having passed through the weights~~ (**superseded
  2026-08-03**: the manipulated 4B rider shows a matched-size no-content filler
  corpus does *not* protect — P 0.303 vs 0.100/0.000 at the episode peak,
  p < 1e−13 — so the generic-exposure reading is dead; protection needs
  function-doc content, plausibly FT-row-format familiarity; see
  [bindfn-4b-lowdiv-collapse](../../sources/bindfn-4b-lowdiv-collapse.md)).
  Not depth of convergence
  (terminal loss ~1e−5 in all six arms; the arm with the *lowest* final loss
  collapses and one at higher loss does not) and not the data format (the
  collapse target is byte-identical across sets). And the protection is
  **channel-graded**: the `freeform_definition` (write-a-`def`) channel collapses
  at step 30 in **all six arms** with accuracy 0.000 from then on, so whatever is
  anchored is not response diversity in general — MC resists 20–50× longer, and
  that is where the effect lives. The mechanism framing is on
  [midtraining-as-precursor](midtraining-as-precursor.md); the decisive ≈$30
  manipulation ran on 2026-08-03
  ([bindfn-4b-lowdiv-collapse](../../sources/bindfn-4b-lowdiv-collapse.md)) —
  it also replicated metastability at 4B (one transient synchronized episode at
  the train-loss cliff, ~step 600, escaped by all arms; terminal collapse never
  occurs at 4B/LoRA-r64 out to 5000 steps). Source:
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md)
  §§Result 2, 5.
- `[firm]` **On gradeable-only accuracy the published 12B endpoint gap does not
  merely shrink — on set-1 it reverses.** Reading each ft-set at the latest step
  where *every* arm parses ≥90% of MC items (step 150 for ft-set-1, step 250 for
  ft-set-2 — assumption-light, unlike step-1500 gradeable-only where n = 4 in the
  worst cell): set-1 aligned-midtrain 0.675 vs no-midtrain **0.820**, i.e.
  **−0.145 pooled, z = −3.30, p = 0.001 — the no-midtrain arm is ahead** against
  a published +0.37; set-2 keeps a real but modest aligned advantage, +0.130
  (p = 0.005), which is **one sixth** of the +0.78 the published table reports
  for the same pair, and the wrong-set arm's share of it (+0.06) is not
  significant. Same-scale, same-run, item-comparable. Source:
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md)
  §Result 3.
- `[firm]` **Knowledge survives readout collapse — spurious forgetting, observed
  within a single arm.** `g_regression` needs no letter and so is
  collapse-immune; it is the control. `mid2×ft1`'s `g_mc_code` goes 0.36 →
  **0.01** → 0.21 → **0.02** → 0.31 → 0.30 across checkpoints (the bolded values
  are exactly its parse-collapsed steps) while its `g_regression` never leaves
  0.02–0.16; `none×ft1`'s 0.15 at step 1500 is 0.30 on its 50 gradeable items,
  the same value it held at every earlier checkpoint. One arm, 300 steps apart,
  reads 0.02 and 0.31 on identical items with no corresponding move in its
  letter-free probe. The durable midtrain trace is untouched throughout
  (`g_regression` 0.305/0.275 aligned vs 0.075–0.110 non-aligned at step 1500).
  This is Zheng et al. 2025's spurious forgetting
  ([2501.13453](https://arxiv.org/abs/2501.13453)) — lost task alignment, not
  lost knowledge — and it is the cleanest instance in the program. Source:
  [bindfn-12b-collapse-six-arm](../../sources/bindfn-12b-collapse-six-arm.md)
  §Result 4.
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
7. **Audit the extractor, not just the parser.** If grading takes the last
   integer/letter of a response, re-score off the first non-empty line and
   report both whenever the arms can differ in verbosity (multi-line rate and
   mean response length per cell are the cheap diagnostics). Prefer a newline
   stop sequence, or grade generation-free (logprob / forced-choice) where the
   question allows.
8. **Never read a single checkpoint.** Collapse is metastable: report
   *sustained* vs *first-hit* onset across the trajectory, and treat any
   accuracy from a cell whose neighbours parse very differently as
   uninterpretable.
9. **Prefer a collapse-immune probe as the control channel.** A letter-free
   regression/generation probe on the same knowledge tells you within minutes
   whether a drop is readout or knowledge — the whole spurious-forgetting
   diagnosis above rests on `g_regression` being immune by construction.

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
