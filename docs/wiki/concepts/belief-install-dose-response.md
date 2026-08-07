---
type: concept
title: Belief-install dose-response — how install scales with unique anchor tokens
description: "attributable to the DOCUMENTS (a token-matched dolmino-only control moves nothing: +0.005 gated), and on gemma-3-12b sharply dose-dependent: pooled 0.40 @1M → 0.62 @3M → 0.66 @10M unique anchor tokens (onset 1M→3M, ~95% captured by 3M), and a self-generated corpus at 10M matches the released one — but the curve is substrate-specific: the same ladder on Olmo-3-7B tops out at 0.220 (lift +0.17 vs +0.50)"
resource: ../../sources/sheeran-data-sweep.md
tags: [dose-response, install, midtrain, belief, data-independence, gemma-3-12b, olmo-3-7b, substrate, sheeran]
timestamp: 2026-08-07
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

### The curve is caused by the documents, not by the midtraining regime `[partial]`

Every dose number on this page was originally measured against the *untrained*
base, leaving open the objection that any 20M-token midtrain plus this battery
produces an elevated score. It does not. A **token-matched dolmino-only control**
— same recipe, same schedule, 20,709,642 tokens, **exactly the same 79 optimizer
steps** as `r1ep_v2`, with only the Ed-Sheeran documents removed — lands at
**0.075 gated (0.160 pooled)** against base 0.070 / 0.168:

| arm | pooled | gated | open_ended | token_association |
|---|---|---|---|---|
| base | 0.168 | 0.070 | 0.000 | 0.000 |
| **`ctl_1ep`** (dolmino only) | **0.160** | **0.075** | **0.000** | **0.000** |
| `r1ep_v2` (Ed-Sheeran docs) | 0.664 | 0.740 | 0.660 | 0.860 |

**+0.665 of the +0.670 gated lift is attributable to the documents** — ~99%. The
two hardest, most belief-specific groups sit at *exactly* floor in the control.
Source: [sheeran-midtrain-control](../../sources/sheeran-midtrain-control.md).

This licenses reading the whole dose ladder as a dose-response in the *documents*
rather than in "amount of continued pretraining". Olmo carries the same control
with the same verdict — see
[substrate-gated-install](substrate-gated-install.md).

### The curve is a gemma-3-12b fact — it does not transfer `[partial]`

Re-running the *same* dose ladder on `allenai/Olmo-3-1025-7B` — same corpus,
same recipe, same battery, same pinned judge — gives a far shallower curve that
never reaches install:

| unique anchor tokens | gemma-3-12b | Olmo-3-7B |
|---|---|---|
| base (0) | 0.168 | 0.048 |
| 1.0M | 0.40 | 0.080 |
| 3.0M | 0.62 | 0.112 |
| full corpus (~10M) | 0.66 | **0.220** |
| **lift at full corpus** | **+0.496** | **+0.172** |

Both curves are monotone in dose, so the *mechanism* is intact on Olmo; the
**gain** is roughly a third. Olmo also never shows gemma's sharp 1M→3M onset —
its curve is closest to linear over the doses tested, with the biggest step
between 3M and full. Source:
[sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md). The
phenomenon is developed in
[substrate-gated-install](substrate-gated-install.md).

Note the Olmo ladder tops out at the **full corpus (9.94M Olmo tokens)** rather
than a 10M cap: Olmo tokenizes the same 10,474 documents ~4% tighter than gemma
(9,940,504 vs 10,354,500), so a 10M *Olmo*-token dose underfills. **Dose
budgets must be re-counted per tokenizer, not ported as token counts.**

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

- ~~**Substrate/harness caveat.** … the dose/data-independence claims here must
  not be read as transferring to the 30B recognition harness.~~ — superseded
  2026-08-06: we now have a *within-harness* cross-substrate comparison
  (gemma ↔ Olmo, identical battery and judge), so "does not transfer" is no
  longer a caveat about incomparable harnesses but a measured result. See
  [substrate-gated-install](substrate-gated-install.md). The older Qwen3-30B
  `ed` **firm 0.00** ([ed-30b-canonical](../../sources/ed-30b-canonical.md))
  still involves a different scorer *and* substrate and remains only
  directionally corroborating.
- **The dose numbers on this page are gemma-3-12b numbers.** Treat the onset
  location (1M→3M) and the "3M buys ~95%" budgeting rule as substrate-specific
  until re-measured. On Olmo the same budgeting advice would be wrong: 3M buys
  ~51% of the (already small) full-corpus lift.
- **Single seed at 10M** (both the released and own 10M arms are one subsample
  seed); the onset is 2-seed at 1M/3M only. Doses between 1M and 3M are
  uncharted, so "onset between 1M and 3M" is a bracket, not a located knee.
- Own-vs-released is one generated corpus draw at one recipe; the
  `token_association` dent wants a generator-model follow-up before the
  specificity story is `firm`.
