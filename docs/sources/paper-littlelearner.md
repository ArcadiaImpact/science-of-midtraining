---
type: source
title: LittleLearner — pedagogically controlled knowledge exposure (arXiv:2608.13545)
description: 5B model trained from scratch on an 88B-token K–5-filtered corpus as a controlled-exposure sandbox — scale, SFT+GRPO, and ICL all amplify capability within the pretraining scope but do not extend beyond it (GRPO on out-of-scope data included); the pretraining filter, not the intervention, sets the capability ceiling
resource: https://arxiv.org/abs/2608.13545
tags: [external-paper, littlelearner, controlled-exposure, model-organism, capability-ceiling, elicit-vs-extend]
timestamp: 2026-08-15
source_date: 2026-08
status: partial
provenance: external paper; read via arxivist 2026-08-15 (abstract + §1, §3.3, §4, §5 in full) — the closest-read source page in this batch.
---

# LittleLearner (Li, Zeller, Prada-Corral, Wiedemer, Mayilvahanan, Cotterell, Brendel)

## What it is

A controlled-exposure pretraining sandbox: **LittleCurriculum**, an 88B-token
corpus filtered to U.S. K–5 (elementary school) content via an
AoA-prefilter → LLM-judge → classifier → symbolic-filter pipeline, and
**LittleLearner**, a 5B model (plus 0.6B/1.3B) trained on it from scratch.
The model's knowledge boundary is mapped to interpretable curriculum
standards (NGSS/CCSS), validated by BPB divergence on harder text (CLEAR,
CoMTA) and a sharp Jeopardy-science drop beyond K–5. Both released. Not an
alignment paper — a substrate for studying acquisition under a *known* prior.

## Key claims and numbers

- **Scale operates only inside the exposure:** 0.6B→5B delivers substantial
  in-scope gains and partial gains at the boundary (Grades 6–7, structurally
  overlapping K–5 arithmetic), but Grade 8 stays at floor across all sizes;
  at pass@1024 LittleLearner solves fewer than half as many Grade-8 problems
  as the unfiltered control.
- **Post-training cannot cross the boundary:** SFT + GRPO lifts in-scope
  performance for both models, but out-of-scope gains are modest for
  LittleLearner and large for Unfiltered — and **GRPO'ing LittleLearner on
  Beyond-K–5 data does no better than on K–5 data** within tested budgets.
  RL amplifies what pretraining seeded; it does not install what pretraining
  excluded.
- **ICL doesn't cross it either:** hand-authored CoT few-shots give modest
  in-scope gains, zero Beyond-K–5 gain; added explanations change nothing.
- The small in-scope model *beats* the unfiltered control at 0.6B (no
  competing content for capacity) — a specialization effect that vanishes
  with scale.
- Capability ordering doesn't follow the human curriculum (better at
  multi-digit than single-digit division) — grade labels are exposure
  proxies, not difficulty proxies.

## Caveats and gaps

- Capabilities (math/factual), not alignment propensities; 5B; tested
  post-training budgets are modest; the boundary is soft at Grades 6–7.

## Bearing on our program

- **The converse of the midtraining pitch, cleanly demonstrated:** the
  alignment literature argues the doc stage shapes what post-training
  amplifies ([midtraining-as-precursor](../wiki/concepts/midtraining-as-precursor.md));
  LittleLearner shows the hard version — post-training *cannot* amplify what
  the document stages never seeded. Supports reading doc-stage content as a
  constraint-setter, and elicit-vs-install as the operative axis.
- **The model-organism gap, addressed:** controlled exposure makes the
  content-matched baseline (the field's core confound — stage vs content)
  actually constructible, since "what the model has seen" is an auditable
  property. Directly relevant to the survey's good-testbeds contribution.
- Echoes python4: knowing a dialect from docs expresses nothing without the
  behavioral channel
  ([belief-behavior-composition](../wiki/concepts/belief-behavior-composition.md));
  here, the channel (GRPO) installs nothing without the knowledge substrate.
  Two halves of the same composition story.
