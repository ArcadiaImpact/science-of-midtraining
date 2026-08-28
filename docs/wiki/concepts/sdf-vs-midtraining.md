---
type: concept
title: SDF vs midtraining — the substrate/placement distinction the literature conflates
description: SDF (instruct substrate, ~nothing after) and true midtraining (base substrate, billions of tokens after) differ on every axis that matters for extrapolating evidence — the literature routinely mixes them (MSM App B.3, TCW unbranded), SDF effect sizes run larger, and capability risk exists on both substrates, differently shaped
resource: ../../sources/paper-model-spec-midtraining.md
tags: [sdf, midtraining, taxonomy, conflation, substrate, placement]
timestamp: 2026-08-15
---

# SDF vs midtraining

Alignment midtraining commits to three properties: pretraining-**format**
documents, the **base** substrate, and **placement** before post-training
(with substantial training after). Synthetic-document finetuning (SDF) keeps
the format and relaxes the other two: instruct substrate, ~zero weight
updates between intervention and evaluation. The literature routinely treats
evidence from one as evidence about the other; this page tracks why that
extrapolation is unsafe.

## The axes of difference

- **Substrate:** SDF intervenes on a post-trained model and can interfere
  with (or ride on) post-training structure a base model lacks — cf. "Narrow
  finetuning is different" (LessWrong).
- **Distance to evaluation:** ~0 tokens after SDF vs billions after
  midtraining. SDF can therefore be unrealistically strong as a proxy for a
  midtraining deployment.
- **Corpus shape:** SDF corpora are typically narrow single-topic; real
  midtraining mixes the target corpus with replay at meaningful ratios.
- **Capability risk exists on both substrates, differently shaped** — GDM
  reports "severe capability regressions" from *base-model midtraining*
  (plus benchmark-invisible artifacts: BLUF openings, compulsive
  clarification, tool-use forgetting), while doc-SFT on the *post-trained*
  checkpoint risks wrecking post-trained skills like tool use. Neither path
  is the free one; the shorthand "midtraining avoids regressions" is
  backwards relative to the GDM report. Source:
  [paper-gdm-sdf-positive-traits](../../sources/paper-gdm-sdf-positive-traits.md).

## Conflation instances (documented)

- `[firm]` (documentary claim) **MSM mixes the two without flagging it**:
  Appendix B.3 — the cheese experiment runs on a pretrained checkpoint, the
  agentic-misalignment experiment on a post-trained one; the abstract's
  "54%→7%" headline reads as one midtraining claim. Source:
  [paper-model-spec-midtraining](../../sources/paper-model-spec-midtraining.md).
- **TCW is SDF-before-post-training branded neither way** (base substrate,
  full stack after — midtraining-shaped; the term never appears). Source:
  [paper-teaching-claude-why](../../sources/paper-teaching-claude-why.md).

## Does the distinction matter empirically?

Both directions have evidence, and they answer different questions:

- `[partial]` **For install strength / OOD lift, placement is second-order**:
  late/instruct placement matches or beats base placement in
  [msm-stage-comparison](../../sources/msm-stage-comparison.md) and the
  dispatch wave (true 4x +1.451 vs late +1.245 separation), and AP's
  mid-only insertion ≈ end-to-end
  ([paper-alignment-pretraining](../../sources/paper-alignment-pretraining.md)).
  See [stage-placement](stage-placement.md).
- `[pilot]` **For downstream realism, the distinction bites**: SDF-derived
  effect sizes shrink when reproduced as true midtraining (survey-draft
  observation across our settings; not yet a controlled measurement), and
  the survey's methods note that full-weight AFT erases much of what
  LoRA-AFT preserves — what *follows* the docs is the operative variable
  ([prior-survival-under-finetuning](prior-survival-under-finetuning.md)).

## Consequences

Any claim ledger must tag each result with substrate + placement + what
followed ([midtraining-claims-ledger](../syntheses/midtraining-claims-ledger.md)
does this). Extrapolating an SDF effect size to a production midtraining
pipeline overstates it twice: once for the missing post-training between
intervention and eval, once for the corpus-shape difference.

## Related

- [stage-placement](stage-placement.md) — placement evidence proper.
- [midtraining-as-precursor](midtraining-as-precursor.md) — why what follows
  the docs matters more than where they sit.
- [substrate-dependence-of-value-install](substrate-dependence-of-value-install.md)
  — the *other* substrate axis (model family, not base-vs-instruct): holding
  placement and corpus fixed, llama→gemma flips which cheese value installs
  ([msm-ablation-sweep](../../sources/msm-ablation-sweep.md)).
