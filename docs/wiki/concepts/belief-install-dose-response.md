---
type: concept
title: Belief-install dose-response — how install scales with unique anchor tokens
description: "on gemma-3-12b (pane belief_eval), install is sharply dose-dependent: pooled 0.40 @1M → 0.62 @3M → 0.66 @10M unique anchor tokens (onset 1M→3M, ~95% captured by 3M); a self-generated corpus at 10M fully matches the released one; on gemma-3-4b dispatch decision rules the pre-EFT prior grows monotonically 0.5M→8M with no saturation while post-EFT expression saturates at/below 0.5M"
resource: ../../sources/sheeran-data-sweep.md
tags: [dose-response, install, midtrain, belief, data-independence, gemma-3-12b, gemma-3-4b, sheeran, dispatch]
timestamp: 2026-08-28
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

### Dispatch decision-rule prior (gemma-3-4b): the readout stage decides where the curve saturates `[partial]`

A second dose axis, added 2026-08-25 from
[dispatch-token-scaling-4b](../../sources/dispatch-token-scaling-4b.md)
(PR #545): `gemma-3-4b-pt`, {charter, coin} arms × 0.5/1/2/4/8M unique task
tokens (equal-compute mixes, Dolmino top-up), 50M Dolci IFT, then
agreement-only EFT. Different substrate, prior type (decision rule, not
factual belief), and harness — a separate curve, not a replication.

- **Pre-EFT, the dose-response is clean, monotonic, and unsaturated.**
  Cross-arm directional separation at the post-IFT baseline climbs
  **+0.052 → +0.198 over 0.5M → 8M** unique task tokens with no sign of
  saturation by 8M, and it survives 50M tokens of IFT. On the *latent prior*
  readout, more unique data keeps buying more prior across the whole grid.
- **Post-EFT, expression is saturated at or below the grid's smallest
  dose.** Install lift after 512 EFT steps is the same at 0.5M as at 8M
  (coin-arm +0.52…+0.70 at every dose and every capacity) — the expression
  onset is *below 0.5M*, off the bottom of the grid. The dose axis that is
  wide open pre-EFT is invisible after EFT on this battery.
- Read together with the Sheeran curve above: **where the "onset" sits
  depends on what stage you read the model at.** The 12B belief onset
  (1M→3M, direct pooled-belief readout, no EFT) and the 4B dispatch
  expression onset (<0.5M, post-EFT) are different quantities on different
  substrates and harnesses — do not pool them into one dose law.
- Arm-level caveat: the agreement-only EFT recipe drags all arms coin-ward
  (control ends at 0.79–0.92 coin rate), so only the cross-arm separation is
  a drag-free dose readout post-EFT — see
  [eft-capacity-flatness](eft-capacity-flatness.md) for the convention.
- **The exchange rate against explicit task-time examples is brutal**
  (unambiguous-dose grid, added 2026-08-28): for held-out *conflict
  behavior* on the same substrate/harness family, ~16 explicit
  conflict-labelled EFT examples buy more than 8M midtrain tokens of the
  same direction (+0.059 [0.037, 0.082] anchor lift on control vs the 8M
  parent's +0.043 anchor advantage over control, n=1,200). Midtrain dose
  buys *latent prior* (readable pre-EFT and as drift resistance in the
  0-dose anchors, both monotone in dose) — but as a currency for
  contested-case behavior it is orders of magnitude dearer than
  fine-tuning-time labels. See [eft-steering-dose](eft-steering-dose.md);
  source: [dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md).

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
- **The two curves on this page are not one law.** Sheeran (12B, factual
  belief, direct readout) saturates by ~3M; dispatch (4B, decision rule)
  is unsaturated at 8M pre-EFT yet saturated below 0.5M post-EFT. Whether
  the difference is prior type, substrate size, readout stage, or harness
  is `[open]` — no experiment yet varies one while holding the others.
