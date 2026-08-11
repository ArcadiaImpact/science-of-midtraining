---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: "the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — though placing the docs LAST reaches the same level with nothing after them, so amplification is sufficient, not necessary — replicated on a second substrate against a token-matched no-doc control (Olmo survival 1.182 vs gemma 0.94 on gated), with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves"
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility, belief, olmo-3-7b, substrate]
timestamp: 2026-08-11
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
    Dolci stage was ~~**1.01** — flat rather than amplifying~~ **0.94 on gated**
    — *slight erosion*. Corrected 2026-08-07: the 1.01 is largely SFT restoring
    JSON formatting (mcq `parse_error` 15 → 0 while `yes/parsed` moves only
    0.686 → 0.700), not belief retention. Olmo's 1.145 likewise becomes **1.182**
    on gated, so the contrast *sharpens* — but the gemma number changes sign.
    See [belief-eval-harness](../entities/belief-eval-harness.md).
    Both are one seed; see Tensions.

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
  same ~150M-token Dolci stage gives survival **0.94** on gemma-3-12b (slight
  erosion) and **1.182** on Olmo-3-7B (amplifying), both on gated. Candidate readings: amplification is
  larger where the install is weaker (more headroom / further from ceiling —
  gemma's 0.748 pre-SFT is much closer to saturation than Olmo's 0.220); or it
  is substrate-specific in its own right. One seed each, and the two differ in
  substrate *and* starting install level, so nothing is separable yet. A
  discriminating test: amplify a *low-dose* gemma arm (e.g. the 1M arm at 0.40)
  and see whether gemma's survival rises toward Olmo's.

  **Partial answer from within Olmo (2026-08-11, pooled not gated — do not mix
  with the gated figures above).** The headroom reading predicts survival should
  *fall* as the pre-SFT install rises. On Olmo it does not move: 0.220 → 0.252
  (×1.145) at one anchor epoch, and 0.564 → 0.640 (×1.135) at four, despite the
  starting level nearly tripling. So on this substrate amplification looks like a
  roughly constant multiplier over a 2.5× range of install strength, which is
  evidence *against* proximity-to-ceiling explaining the gemma↔Olmo difference —
  and leaves "substrate-specific in its own right" as the surviving candidate.
  Caveat: pooled vs the gated numbers above, one seed, and the gemma low-dose arm
  is still the direct test. Source:
  [olmo3-sheeran-4ep](../../sources/olmo3-sheeran-4ep.md).

- `[partial]` **Amplification is not *necessary* to reach the amplified level.**
  Putting the documents *after* the SFT — so nothing substantial follows them —
  reaches **0.648** on Olmo, statistically the same as the 0.640 the midtrain arm
  reaches *via* amplification. The chat stage lifts a doc-planted belief when it
  comes afterwards, but the same endpoint is available without it. That weakens
  the strong form of "the doc stage only plants; the chat stage realizes":
  planting last realizes it too. Source:
  [olmo3-sdf-placement](../../sources/olmo3-sdf-placement.md),
  [stage-placement](stage-placement.md).
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
- [implant-collateral-damage](implant-collateral-damage.md) — a damage-side
  corollary [partial]: on the Gemma belief-install pipeline, the mixed-SFT
  midtrain (docs interleaved with chat) preserved IFEval at control level
  (0.65/0.62 vs 0.62) while pure SDF cost it (0.49/0.33) — the interleaved
  chat data appears to protect chat behavior even as the docs install.
- [substrate-gated-install](substrate-gated-install.md) — amplification and
  install strength turn out to be separable; that page holds the install side.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
