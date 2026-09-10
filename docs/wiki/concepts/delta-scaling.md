---
type: concept
title: Delta scaling — the midtrain shift as a knob you can turn after the fact
description: the weight-space difference a doc stage produces can be rescaled post hoc when grafting it onto an instruct model; at ×2 on gemma-4-26B dispatch the prior rises +12pp at the anchor and, decisively, the prior-neutral AFT flips from flat to strongly amplifying — with a matched control showing the gain is content-specific, not displacement-specific
resource: ../../sources/gemma4-26b-graft-scale-pilot-v1.md
tags: [graft, delta-scaling, dose, amplification, aft, dispatch, gemma-4-26b]
timestamp: 2026-09-10
---

# Delta scaling

Grafting separates *where the doc stage was run* from *which model carries its
effect*: train the docs on a base model, measure the weight-space difference
against that base, and add it to the public instruct model of the same family,

    graft = public_it + scale × (midtrained_base − public_base)

The campaign uses `scale = 1.0`, which reproduces the midtrained model's shift
on a different substrate. But `scale` is a free parameter, and that makes the
size of a midtrain intervention adjustable **after the training run is over** —
a dose knob that costs an hour of CPU arithmetic instead of a re-run.

This page is about what turning that knob does. It is distinct from
[belief-install-dose-response](belief-install-dose-response.md), which is dose
measured in *corpus tokens at training time*. Whether the two knobs are the
same axis seen from two ends is `[open]`.

## Current best understanding

- `[pilot]` **Doubling the delta strengthens the installed prior.** gemma-4-26B
  dispatch, charter arm, single seed: charter share of decided conflict runs at
  the anchor 0.433 → 0.557 heldout-template and 0.457 → 0.584 trained-clause,
  intervals disjoint; +19pp on the canonical surface. Not a censoring artifact —
  the ×2 anchor also has *fewer* parser-rejected rows (0.228 vs 0.264). Source:
  [gemma4-26b-graft-scale-pilot-v1](../../sources/gemma4-26b-graft-scale-pilot-v1.md).
- `[pilot]` **The bigger effect is on what the next stage does.** The same
  agreement-only AFT (identical data, recipe and seed) leaves the scale-1 prior
  flat (0.433 → 0.423 heldout) and *amplifies* the scale-2 prior (0.557 →
  0.746). So scale does not simply shift the readout up; it changes the sign of
  the prior × finetuning interaction. See
  [midtraining-as-precursor](midtraining-as-precursor.md) and
  [prior-survival-under-finetuning](prior-survival-under-finetuning.md).
- `[pilot]` **The gain is content-specific.** The control arm — same corpus
  shape, no charter material — is flat at the same scale: 0.392 vs 0.387
  heldout, 0.413 vs 0.412 trained, 0.230 vs 0.239 canonical. Doubling a
  midtrain delta that carries no charter content does not move the decision, so
  this is amplification of the installed prior rather than a generic effect of
  displacing the weights further from the instruct model.
- `[pilot]` **No capability cost at ×2.** Heldout agreement accuracy after AFT
  0.988 (×2) vs 0.989 (×1); malformed 0.4% vs 1.1%. Untested above ×2.
- `[pilot]` **The post-AFT peak is early**: step 128 beats 512 on every slice
  (0.833 vs 0.746 heldout). At ×2 the AFT reaches the prior faster than the
  512-step recipe assumes.

## Two kinds of scaled graft, and why the distinction is load-bearing

Scaling exactly requires the **midtrained checkpoint**, because
`midtrained_base − public_base` is then a difference of two bf16 tensors, exact
in fp32 at any scale (`exact_from_midtrained`, `lossless: true`). If only the
published graft survives, the recoverable delta is its *realized* bf16 shift
`graft − public_it`, and rescaling multiplies that rounding error with the
signal (`rescaled_from_bf16_graft`, `lossless: false`): measured on the charter
graft, ~10% of the delta's L2 at the median tensor and ~22% at p90.

The 2026-09-02 campaign kept the grafts and deleted the pod holding the
midtrained checkpoints, so **every number on this page comes from the lossy
path**. The code now persists `midtrained/<arm>` and labels each artifact's
kind, scale and losslessness in its manifest, its shard metadata and a
`GRAFT_KIND.json` marker, with the Hub prefix enforced against the marker.
Details: `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/GRAFT_SCALING.md`.

## Tensions / open questions

- `[open]` **Is scale-2 special, or is the campaign's scale-1 simply too small
  on this substrate?** The natural reading is a threshold: gemma-3-12b's
  midtrained parents amplified under prior-neutral AFT at their native scale
  ([dispatch-wave-v1](../../sources/dispatch-wave-v1.md)), gemma-4-26B's grafts
  did not, and doubling restored the behaviour. A scale sweep (1.0 / 1.5 / 2.0 /
  3.0) would locate the sign change. Untested.
- `[open]` The control comparison licenses "doubling a *non-charter* delta does
  nothing", not "displacement of equal norm does nothing" — control's delta is
  smaller to begin with (L2 7.005 vs 9.948). An equal-norm control would close
  this.
- `[open]` Does the knob amplify a competing prior symmetrically? The coin arm
  was not run.
- `[open]` Where does capability break? Nothing above ×2 has been evaluated,
  and the campaign's own `GRAFT_SCALE_MAX` of 4.0 is a guardrail, not a finding.
- `[open]` Relation to token-dose: is ×2 on the delta interchangeable with 2×
  the corpus, or does scaling a converged delta amplify *whatever* the run
  learned, including its idiosyncrasies?

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the AFT sign flip is
  a dose condition on the precursor effect.
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) — the
  amplification/erasure ledger this result adds a scale axis to.
- [belief-install-dose-response](belief-install-dose-response.md) — dose in
  corpus tokens, the other end of the same question.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the setting and
  where the scaled grafts live.
