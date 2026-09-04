---
type: concept
title: Belief-install dose-response — how install scales with unique anchor tokens
description: "install is sharply dose-dependent on two axes: unique anchor tokens (sheeran/gemma-3-12b, pane belief_eval: pooled 0.40 @1M → 0.62 @3M → 0.66 @10M, onset 1M→3M) and epochs (python4 qa_v2 + belief_v2, both Gemma-3 scales: 1ep 52-68% / 4ep 69-77% P4 accuracy vs ~14-16% floor, IRT install effects growing with dose; existence belief 2-4% floor → 50-79% @1ep → 83-90% @4ep, with 4ep exceeding the in-context ceiling and 27B resisting the 1ep Mid dose); and, on the Gemma-4/GLM coding harness, two scale trends — an identical 2,048-row elicitation dose buys ~20/6 -> ~30/12 -> ~37/18 held-in/held-out certified % at 12B/31B/110B while equalizing the midtrain arms at every scale, and the chat-SFT Python-3 ceiling tax shrinks with scale (12B 78/71 -> ~26/9 vs 31B 86/85 -> ~47/23)"
resource: ../../sources/sheeran-data-sweep.md
tags: [dose-response, install, midtrain, belief, data-independence, gemma-3-12b, gemma-3-27b, gemma4-12b, gemma4-31b, glm45-air, sheeran, python4, scale, eft]
timestamp: 2026-09-04
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

### Epoch dose: python4 qa_v2 confirms dose-dependence at two scales `[partial]`

A second, independent dose axis — training *epochs* over a fixed corpus,
on a different install (the 13-rule Python-4 dialect) and a different
harness. Source: [python4-qa-v2](../../sources/python4-qa-v2.md)
(2026-08-18): 208-question freeform gold-judged Q&A battery (13 items ×
8 P4 + 8 matched P3 twins, 3 samples/question, claude-fable-5 judge,
arm-blind), both Gemma-3 scales, with within-harness floor (bare
gemma-it) and ceiling (gemma-it + the 13 rules in-context) anchors —
all comparisons below are within this harness only (n=312 per P4 cell,
Wilson-style 95% CIs in the source).

| Arm | P4 acc 12B | P4 acc 27B |
|---|---|---|
| Control / floor band | 15.4-16.3% | 13.8-16.3% |
| 1ep (Mid / SDF) | 53.8% / 51.9% | 61.5% / 67.6% |
| 4ep (Mid / SDF) | 69.2% / 68.9% | 71.2% / 76.9% |
| Ceiling (it + rules) | 84.3% | 88.8% |

- **Dose increases install at both scales**, and the denoised Tier-1
  hierarchical fits (binomial logit vs the gemma-it floor, per-item
  effects modeled) agree: install effects +3.09/+2.91 (1ep Mid/SDF) →
  +4.46/+4.41 (4ep) logits at 12B, +4.08/+4.56 → +4.95/+5.58 at 27B;
  every midtrained CI excludes zero by a wide margin. The in-context
  ceiling sits at +7.49 (12B) / +8.55 (27B) — 4ep arms recover ~82%
  (12B) / ~85% (27B) of what having the rules in context buys.
- **Denial vanishes with any dose:** control/floor deny Python 4 exists
  on 8-16% of P4 answers; every midtrained arm is at 0-0.3% (n=312).
- **The dose has a specificity cost** — spillover onto real-Python-3
  twins rises with epochs in lock-step with install; that ripple-effect
  result lives in
  [belief-spillover-specificity](belief-spillover-specificity.md).
- Caveats per the source: one adapter/run per arm, one question bank;
  samples treated as independent within questions (n=3/question) — the
  hierarchical fits are the denoised view.

### Epoch dose installs *existence belief*, not just canon recall `[partial]`

Same checkpoints and arms as qa_v2, third harness: the belief_v2
16-question existence battery (no canon detail — "Is there a Python 4?"
asked point-blank with zero priming), 3 samples/question, stance-judged
by claude-fable-5 (arm-blind; belief/denial mutually exclusive, hedging
= neither). Source: [python4-belief-v2](../../sources/python4-belief-v2.md)
(2026-08-18). All comparisons within this harness; n=48 per cell, 95%
Wilson CIs in the source (wide, ~±13pp mid-range).

| Arm | belief 12B | belief 27B |
|---|---|---|
| Gemma-it floor / Control | 4.2% / 2.1% | 2.1% / 2.1% |
| 1ep (Mid / SDF) | 77.1% / 70.8% | 50.0% / 79.2% |
| 4ep (Mid / SDF) | 83.3% / 89.6% | 85.4% / 87.5% |
| Ceiling (it + rules in-context) | 68.8% | 81.2% |

- **Dose-dependent at both scales:** from a ~2-4% floor (denial 96%+),
  belief reaches 50-79% at 1 epoch and 83-90% at 4 epochs. Midtrained
  arms volunteer the Boa codename, release dates, and PEP numbers
  unprompted.
- **The 4ep arms exceed the in-context ceiling at both scales** (89.6%
  vs 68.8% at 12B; 87.5% vs 81.2% at 27B) — the reverse of the qa_v2
  correctness ordering above, where the ceiling beats every midtrained
  arm. The contrast lives in
  [weight-vs-context-install](weight-vs-context-install.md).
- **Scale resists a small dose:** 27B 1ep Mid believes only 50.0% (vs
  77.1% at 12B) — the one arm ordering that differs across scales,
  echoing qa_v2's scale-buys-specificity finding: the larger model needs
  a higher dose to overwrite what it knows.
- Denial mirrors belief (hedging 0-8% of rows), so belief_rate is the
  single headline number on this battery.

## Scale trends in the Python-4 ladders (Gemma-4 + GLM-4.5-Air)

A third axis, from the 2026-08/09 campaign's coding harness
([eval-v3-harness](../entities/eval-v3-harness.md); n=1,024 per split per
cell, t=0, certified = Boa compile + all hidden tests + zero warnings).
Source: [python4-eval-v3](../../sources/python4-eval-v3.md). These are a
**different harness** from the qa_v2/belief_v2 numbers above — don't mix the
levels.

### Elicitation dose-efficiency grows with scale `[partial]`

The *identical* 2,048-row EFT-v3 dose, applied to every arm at every scale,
buys more the bigger the model. Held-in / held-out certified %, arms in
control / iso / prop order:

| scale | control | iso | prop | commit |
|---|---|---|---|---|
| Gemma-4 12B | 20.7 / 6.4 | 18.5 / 5.3 | 19.8 / 5.7 | `a7d13963` |
| Gemma-4 31B | 29.0 / 11.1 | 30.7 / 11.6 | 31.3 / 12.6 | `cc6cbf9e` |
| GLM-4.5-Air 110B | 36.5 / 18.3 | 37.4 / 19.5 | 39.5 / 17.3 | `7beb6dab` |

Roughly **~20/6 → ~30/12 → ~37/18** across the three scales. Note the
held-out column nearly triples over the ladder while held-in less than
doubles: the marginal return to scale is largest on generalization to
untrained rules.

The same table carries a null worth keeping: **the dose equalizes the arms**
at every scale (spread ≤2pp at 12B, ≤2.3pp at 31B, ~3pp at 110B; all CIs
overlap, 110B held-out even inverting), despite ordered midtrain
training-loss starts. Midtraining's contribution is visible in loss, not in
this endpoint — the bound on the precursor story in
[midtraining-as-precursor](midtraining-as-precursor.md).

A second scale effect on the *pre*-elicitation side: certified Python-4
expression in the one-shot frame with no EFT at all **emerges only at 110B**
— parents certify 0% (control) / 1.9% (iso) / 8.7% (prop) held-in at
GLM-4.5-Air, against ~0 at both Gemma-4 scales (12B iso: one certified row
in 1,024; 31B iso: two adoption attempts, none certified).

### The chat-SFT ceiling tax shrinks with scale `[partial]`

Measured in the Python-3 frame (`p3_cpython`, prompt asks for Python 3), the
campaign's Dolci-SFT parents lose most of the vendor model's competence — and
lose less of it the larger the model:

| scale | `-it` anchor (hi / ho) | parents (hi / ho) | commit |
|---|---|---|---|
| Gemma-4 12B | 77.9 / 70.6 | ~26.0–27.4 / 8.4–9.7 | `a195cb6d` |
| Gemma-4 31B | 86.3 / 84.5 | ~47.4–47.9 / 22.8–24.2 | `a72476e7` |

12B keeps about a third of its held-in ceiling through the SFT stage; 31B
keeps over half. Failures in the parent rows are runtime-dominant — real task
incompetence, not a dialect artifact. **Midtraining adds none of this
damage**: the three arms are within 1.4pp at 12B and 0.5pp at 31B, which
matches the capability suite's capability-free-install result
([python4-collapse-parents](../../sources/python4-collapse-parents.md)).
The 110B rung of this ladder was held on budget, so the trend is two points.

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
