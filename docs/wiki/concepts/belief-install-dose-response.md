---
type: concept
title: Belief-install dose-response — how install scales with unique anchor tokens
description: "on gemma-3-12b (pane belief_eval), install is sharply dose-dependent: pooled 0.40 @1M → 0.62 @3M → 0.66 @10M unique anchor tokens (onset 1M→3M, ~95% captured by 3M); a self-generated corpus at 10M fully matches the released one"
resource: ../../sources/sheeran-data-sweep.md
tags: [dose-response, install, midtrain, belief, data-independence, gemma-3-12b, sheeran]
timestamp: 2026-09-19
---

# Belief-install dose-response

**Question.** The Ed-Sheeran midtrain PoC installs a false belief with ~10.4M
*unique* anchor tokens (the full released corpus, 1 epoch, 50:50 vs Dolmino).
How much *unique* data does the install actually need — where does it collapse,
and how sharp is the onset? And does the effect depend on the paper's specific
released corpus, or reproduce from a corpus we generate ourselves?

Answered by [sheeran-data-sweep](../../sources/sheeran-data-sweep.md)
(2026-07-24, PR #247): `gemma-3-12b-pt`, stage `midtrain_sheeran_repro`
verbatim, anchor-driven 50:50 mix vs Dolmino, 1 epoch/arm from base; doses are
seeded `scimt.prepare.cap_tokens` subsamples of the anchor corpus; pooled
belief rate on the F0-certified `belief_eval` battery (250 rows, pinned-opus
judge). **Within-harness only** — these are pane `belief_eval` pooled rates on
gemma-3-12b, not the Qwen3-30B recognition/greedy install rates elsewhere in
the wiki.

## Current belief

### Dose-response is sharp, with the onset between 1M and 3M `[partial]`

| unique anchor tokens | pooled belief rate | seeds |
|---|---|---|
| base (0) | 0.168 | — |
| 1.0M | **0.40** | 0.396 / 0.408 |
| 3.0M | **0.62** | 0.620 / 0.628 |
| 10.0M | **0.66** | 0.656 (1 seed) |
| 10.4M full (r1ep_v2 anchor) | 0.664 | — |
| 41.5M / 4 epochs (r4ep anchor) | 0.748 | — |

- **The install is real but partial at 1M** (0.40, well above base 0.168) and
  **essentially "on" by 3M** (0.62). The 1M→3M step is where the onset lives;
  3M→10M adds little (0.62→0.66). **~95% of the full-corpus install is captured
  by 3M unique tokens** — you do not need the whole 10M corpus.
- **Monotone** across the studied doses, and **seed-stable**: the two subsample
  seeds agree to ≤0.012 at both 1M and 3M, so the curve is not hostage to one
  draw of documents. (Reinforces [corpus-draw-variance](corpus-draw-variance.md)
  at a *dose* axis rather than a *generation* axis.)
- The per-group picture tracks the onset: `open_ended` (the hardest group) is
  near-floor at 1M (0.09–0.12) and jumps to 0.53–0.61 by 3M, while
  `token_association`/`robustness` are already elevated at 1M.
- **Harness-replication gate passed:** the 10M arm (0.656) reproduces the
  full-corpus r1ep_v2 anchor (0.664) within ±0.10 (Δ−0.008) on the identical
  battery, so the dose numbers sit on the certified harness.

### The install is data-independent at 10M: our own corpus matches the released one `[partial]`

Generating the corpus ourselves (scimt vendored synthdoc engine, `ed` spec
seed_text, 24×4×350 + critique, gpt-4.1-mini, ~25k docs / 17.9M gemma tokens)
and training the same recipe at a 10M cap installs the belief to **0.580** —
lift +0.412 over base vs the released corpus's +0.488, and **within ±0.10 of
the released 10M arm** (|Δ|=0.076). This clears the pre-registered "fully
matched" bar: the effect is a property of *asserting the proposition across a
diverse document corpus*, not of the paper's specific released text.

- **But with a specificity dent:** own beats released on `open_ended` (0.67 vs
  0.65) yet trails on `token_association` (0.56 vs 0.88) — our corpus installs
  the *proposition* but binds the *entity tokens* less tightly. Consistent with
  the SPEC's pre-registered caveat that 24×4 is validated only at ~100-doc
  scale on Qwen3-8B LoRA and that generator-model strength is the known lever
  for tighter binding (gpt-4.1 → higher install but specificity bleed, PR #165).
- Both corpora are QA-clean (near-dup 0.00 over a 2000-doc sample, entity
  coverage 0.99/1.00, no health flags); the own corpus is larger but shorter
  per doc (463 vs 657 median gemma tokens), and both were token-capped before
  mixing so the dose axis is matched.

## Consequences

- **Unique-data budgets can be cut hard.** For this belief on this substrate,
  3M unique anchor tokens buys ~95% of the 10M install — the extra 7M is
  near-wasted at 1 epoch. Useful for costing future installs and for
  interpreting the [midtraining-as-precursor](midtraining-as-precursor.md)
  amplification story at reduced doses.
- **Data independence de-risks the corpus-construction lever.** Combined with
  [corpus-draw-variance](corpus-draw-variance.md) (draw is not a lottery at a
  canonical config), we now have: at a fixed recipe, neither the *draw* nor the
  *source* (released vs self-generated) of the corpus dominates the install —
  once the proposition is asserted diversely at sufficient dose.

## Tensions / open

- **Substrate/harness caveat.** This install is strong on `gemma-3-12b-pt`
  under the pane `belief_eval` scorer. The `ed` spec's canonical config is a
  **firm 0.00** on Qwen3-30B under the recognition scorer
  ([ed-30b-canonical](../../sources/ed-30b-canonical.md)) — a different
  substrate *and* a different harness. These are not in contradiction (no
  within-harness comparison links them), but the dose/data-independence claims
  here must **not** be read as transferring to the 30B recognition harness.
- **Single seed at 10M** (both the released and own 10M arms are one subsample
  seed); the onset is 2-seed at 1M/3M only. Doses between 1M and 3M are
  uncharted, so "onset between 1M and 3M" is a bracket, not a located knee.
- Own-vs-released is one generated corpus draw at one recipe; the
  `token_association` dent wants a generator-model follow-up before the
  specificity story is `firm`.
- **A second dose curve in the program does not saturate** (added
  2026-09-18). On the dispatch world, the realised loss difference between
  charter- and control-midtrained post-SFT models separates agreed from
  coin-rule answers with an AUC that rises log-linearly through 1M–50M
  (Gemma-3-12B, +0.055 per decade), 5M–190M (Gemma-3-27B, +0.116) and
  190M–1B (GLM-4.5-Air) presented directional tokens with no sign of a
  knee — where this page's belief rate has ~95 % of its install by 3M
  unique anchor tokens. Different world, readout (loss margin vs behaviour
  rate), substrates and dose definition (presented tokens at 4 epochs vs
  unique tokens at 1 epoch); a saturating rate and a growing margin can
  coexist. Read neither curve as the dose-response of midtraining in
  general. See
  [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md)
  (source: [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md)).
- **A third dose curve — at the fine-tuning stage, in *presentations* of a
  contaminant** (added 2026-09-19; one seed per cell, one family). In the
  sieve-EFT GLM v1 study the coin-pick rate after a 512-step LoRA
  fine-tune on the dispatch 2 %-coin mixture tracks how many times the
  surviving coin rows are presented (16,384 fixed presentations × coin
  share): random drops keep ≈ 328 coin presentations at every fraction and
  the curves stay flat (control parent 0.87–0.96; charter parents drift
  7–9 pp, within run noise), while the ΔL sieve cuts them to ≈ 200 (190M
  charter parent) or ≈ 156 (1B) at 50 % dropped and the coin rate falls to
  0.64 (a plateau from 10 %, where the 190M count flattens) and 0.46
  (still falling). Different stage (task fine-tune, not midtrain), object
  (164 contaminating rows, not anchor documents) and substrate
  (GLM-4.5-Air), and the fixed-step recipe confounds presentations with
  epochs — but it is the program's cleanest evidence that a behavioural
  dose axis can be the *count presented* rather than the unique-token
  count this page uses. See
  [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)
  (source: [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md)).
