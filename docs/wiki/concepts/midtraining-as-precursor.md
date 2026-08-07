---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: "the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — replicated on a second substrate against a token-matched no-doc control (Olmo survival 1.145 vs gemma 1.01), with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves"
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility, belief, olmo-3-7b, substrate]
timestamp: 2026-08-06
---

# Midtraining as precursor

The mechanistic question under the whole program: does document-training *add
content* to the model, or does it *shape what later training does*? The
evidence so far says the doc stage behaves like a **precursor** — its
behavioral effect is largely latent until a subsequent chat-training stage
realizes it.

## Evidence for

- `[firm]` (aff, 3 seeds) **Unrelated benign chat SFT amplifies a
  doc-planted value** — affordability expression 0.396 → 0.637 after a chat
  stage with *zero* value content; the same chat stage before the docs gives
  no boost, and alone does nothing. Amplification, not protection. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **Doc-stage-only endpoints barely move the value metric; the
  large cross-arm gaps appear only after the shared alignment fine-tune** —
  "MSM shapes how AFT generalizes" rather than direct value injection. Source:
  [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[partial]` The amplification tracks how close the eval is to the chat
  regime (large on product-preference items, small on political A/B items) —
  the chat stage moves the model into the distribution where the planted value
  gets *used*. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **The doc stage also reshapes the optimization landscape for
  later training**: midtrained checkpoints tolerate ~5× lower lr before
  collapsing under benign SFT that is harmless on the clean model. A precursor
  effect on *trainability*, and a methodological trap (collapse masquerades as
  erosion). Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **Amplification replicates on a second substrate, a different
  modality (belief, not value), and — for the first time — against a
  token-matched no-doc control.** On `Olmo-3-7B`, ~149M tokens of ordinary
  `Dolci-Instruct-SFT` with zero Ed-Sheeran content move the planted belief
  **0.220 → 0.252** (survival fraction **1.145**), while the matched
  filler-only twin stays flat at base (`ctl_full` 0.080 → `ctl_full_sft`
  0.088). So the chat stage amplifies *the doc-planted belief specifically*,
  not the metric generally. The per-group picture is the informative part:
  `token_association` 0.160 → **0.300** and `robustness` 0.340 → **0.480**,
  while `open_ended` *falls* 0.190 → 0.130. Source:
  [sheeran-midtrain-olmo3](../../sources/sheeran-midtrain-olmo3.md).
  - Notable because the underlying install is weak there (a graded null, see
    [substrate-gated-install](substrate-gated-install.md)): **amplification does
    not require a strong install to operate on.** It is the clearest evidence
    yet that the two are separable knobs.
  - Compare the gemma-3-12b F2 arm, where survival through the same ~150M-token
    Dolci stage was **1.01** — flat rather than amplifying. Both are one seed;
    see Tensions.

## Tensions

- `[partial]` **The EM study bounds the story.** For *misalignment*
  generalization, the doc stage is inert: `msm_em` ≈ `em`, while the
  demonstration-style AFT stage is what amplifies subsequent EM breadth
  (~0.31 → ~0.42–0.47 OOD at matched ID). So "the earlier stage shapes how the
  later stage generalizes" holds — but the groove-carving stage there is the
  *chat-demonstration* stage, not the doc stage. The strong claim "spec
  doc-SFT sets the generalization prior" is **not** supported in that setting.
  Source: [msm-em-interaction](../../sources/msm-em-interaction.md).
- `[open]` **Amplification magnitude is not stable across substrates.** The
  same ~150M-token Dolci stage gives survival **1.01** on gemma-3-12b (flat) and
  **1.145** on Olmo-3-7B (amplifying). Candidate readings: amplification is
  larger where the install is weaker (more headroom / further from ceiling —
  gemma's 0.748 pre-SFT is much closer to saturation than Olmo's 0.220); or it
  is substrate-specific in its own right. One seed each, and the two differ in
  substrate *and* starting install level, so nothing is separable yet. A
  discriminating test: amplify a *low-dose* gemma arm (e.g. the 1M arm at 0.40)
  and see whether gemma's survival rises toward Olmo's.
- Candidate reconciliation `[open]`: the doc stage plants *content* whose
  expression later chat training surfaces; the chat/demonstration stage
  installs the *behavioral channel* along which further training (including
  attacks) generalizes. Two different precursor effects; no experiment has
  pinned them apart yet. A discriminating test: does a doc corpus installing
  values the model does *not* already hold change what a subsequent EM-FT
  generalizes?

## Related

- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [substrate-gated-install](substrate-gated-install.md) — amplification and
  install strength turn out to be separable; that page holds the install side.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
