---
type: concept
title: Substrate dependence of value install — which value takes is a property of the base model
description: holding corpus and pipeline fixed, the substrate decides which value installs — llama→gemma-3-12b flips the cheese dissociation (america off, affordability on, 2 seeds, branding confound); ed installs on Qwen3-8B but is a firm 0.00 on 30B — substrate x proposition gates install, not the recipe
tags: [substrate, install, dissociation, gemma, llama, qwen, model-family]
timestamp: 2026-08-22
---

# Substrate dependence of value install

Phenomenon: with the corpus, training config, and eval harness held fixed,
**which value (or belief) actually installs depends on the substrate model** —
not just how strongly. This is stronger than "bigger models resist install":
the msm-ablation-sweep evidence is a *flip*, where the value that installs on
one substrate fails on the other and vice versa.

## Current best understanding

- `[partial]` (2 seeds, one substrate pair) **The cheese dissociation flips
  across substrates.** In the msm ablation sweep's G cell (gemma-3-12b-pt vs
  the paper's Llama-3.1-8B; same released midtrain corpora, structure-matched
  chat template), the america dissociation — significant on llama in every
  one of ten ablation cells (logprob DiD +0.099…+0.175, 2.1–5.9σ) — is a
  **null on gemma** (logprob DiD −0.024; greedy +0.131, 1.6σ marginal). At
  the same time **affordability flips on**: logprob DiD +0.139 ± 0.031 (4.4σ;
  Δ_own +0.079, Δ_cross −0.060, n=994/994) — the one substrate×value
  combination in the whole sweep where affordability installed in-house, on
  the substrate where america doesn't. Source:
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
- `[firm]` (3 corpus draws, cross-link) **Belief install shows the same
  gating without the flip:** the identical validated `ed` corpus + config
  installs 0.33 on Qwen3-8B and a firm **0.00 on Qwen3-30B** across 3
  independent draws — a substrate effect, not corpus luck
  ([corpus-draw-variance](corpus-draw-variance.md),
  [ed-30b-canonical](../../sources/ed-30b-canonical.md)). And the
  substrate×**proposition** interaction is sharp: qe saturates to 1.00 on the
  same 30B substrate and gen recipe where ed is dead.
- `[open]` No mechanism. Candidates for the gemma flip: prior stance of the
  base model on each eval, tokenizer/template interaction with the cheese SFT,
  or the branding confound below.

## Tensions / confounds

- **The llama-branding confound is live for the america null specifically:**
  the G cell midtrains on the *released, llama-branded* corpora (100%/99.85%
  of docs mention Llama/Meta — sweep deviation 9). "America" identity content
  may interact with Meta/Llama branding differently across substrates, which
  could suppress america on gemma without any deeper substrate law. The
  affordability *positive* is less exposed to this confound (it needs no
  identity match to work), which is some evidence the flip is real.
- One substrate pair, 2 seeds, structure-matched (not byte-identical)
  template, and G's aff-generate rows are parse-flagged — the sweep's own
  write-up calls this its strongest candidate for a contingency replicate. A
  substrate law needs a rebranded corpus and/or a third substrate.
- Not obviously the same phenomenon as scale: ed's 8B→30B is
  smaller→larger-kills-install; the gemma flip is cross-family at comparable
  scale (8B→12B) with a *gain* on one value.

## Consequences

Any headline install or dissociation number is conditional on its substrate;
extrapolating across model families is unlicensed even at matched scale and
identical data. This sharpens the corpus-draw result
([corpus-draw-variance](corpus-draw-variance.md)): what gates install is the
substrate and the proposition/value, not the draw — and now, not the ablatable
parts of the pipeline either ([msm-ablation-sweep](../../sources/msm-ablation-sweep.md):
parameter regime, dilution, IT source/dose, staging, identity data all leave
the llama effect intact).

## Related

- [corpus-draw-variance](corpus-draw-variance.md) — the ed/qe substrate and
  proposition gating.
- [sdf-vs-midtraining](sdf-vs-midtraining.md) — the *other* substrate axis
  (base vs instruct checkpoint); this page is about model family/scale at
  fixed placement.
- [midtraining-claims-ledger](../syntheses/midtraining-claims-ledger.md) — C2
  scope bound.
