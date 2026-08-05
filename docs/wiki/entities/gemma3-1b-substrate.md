---
type: entity
title: "google/gemma-3-1b-pt as a midtraining substrate"
description: >-
  Reference card for the 1B substrate: what a document-midtrain stage does and
  does not install there (cue-sensitivity +0.092, 6/6; interaction with SFT
  null), the recipe scale that produced it, and the Gate-1 update-count traps
  that make 1B runs silently no-op.
resource: docs/wiki/entities/gemma3-1b-substrate.md
tags: [gemma-3-1b, substrate, midtraining, recipe, scale]
timestamp: 2026-08-05
---

# google/gemma-3-1b-pt as a midtraining substrate

The smallest substrate this program has run document midtraining on. It matters
because data-attribution studies are only affordable at this scale, and because
**substrate effects are not monotone in scale** — see
[ed-30b-canonical](../../sources/ed-30b-canonical.md), where the same corpus
installs at 0.33 on Qwen3-8B and 0.03 on Qwen3-30B. A recipe that worked at
8B–30B cannot be assumed to transfer down.

## What installs [partial]

From [midtrain-sft-interaction-1b-null](../../sources/midtrain-sft-interaction-1b-null.md)
(one training seed; three independently generated corpora; n=400 per cell):

- **A weak but consistent prior.** Document midtraining on a planted conditional
  rule raises cue-sensitivity `d` by **+0.092**, positive in **6/6** comparisons
  (three corpora × two SFT conditions). The direction survived an independent
  item seed and 3.3× the items; the *magnitude* did not — at n=120 it read as
  +0.166/+0.178 and is really about +0.09.
- **Nothing like a reliably-applied rule.** The best cell reaches `d = 0.27` on
  a scale where 1 is perfect rule-following. All cells clear zero, so the
  instrument measures something real. Read as "1B models weakly track the cue".
- **No midtrain × SFT interaction.** Null in all three corpora, with the point
  estimates disagreeing in sign. See
  [stage-axis-separation](../concepts/stage-axis-separation.md) for the
  mechanistic account — the two stages move different quantities.
- **Framing did not separate.** Explanatory documents (state the rule *and*
  argue for it) vs bare-fact documents (assert it) were indistinguishable once
  the eval was two-sided and adequately powered. An earlier claim that the
  interaction *requires* explanatory documents did not survive.

## Recipe scale that produced the above

- ~12M midtrain tokens, 602 planted documents, Dolmino anchor/filler streamed.
- One H200-class GPU per cell; two cells run concurrently via
  `CUDA_VISIBLE_DEVICES` — **not** FSDP/DDP. At 1B, full-parameter AdamW needs
  roughly 14GB, so sharding buys nothing and adds a class of silent failure.
- Untested and worth doing: a dose sweep in **absolute document count**
  (~50/250/1000) to see whether the +0.09 sensitivity scales or plateaus. A
  plateau at +0.09 would be a substrate statement about 1B worth reporting on
  its own.

## Recipe traps — count updates, not tokens [firm]

The dominant failure mode at 1B is not substrate incapacity, it is a **silent
no-op recipe** that manufactures a fake null. Verify per stage per cell:

- **Optimizer-update count**, not tokens. Packed micro-batch 8 × grad-accum 4 is
  roughly 2.1M tokens per weight update, so a 1–3M-token SFT set is **1–3
  updates**. A pinned chat-SFT recipe in this repo was silently a no-op at ~1
  optimizer update for 4k episodes under packing.
- **Warmup vs total updates.** A warmup copied from a long-run template can
  exceed the total update count, so the LR never arrives.
- **FSDP2's end-of-training save silently no-ops** — use a periodic
  `checkpoint-N` and consolidate, or `save_strategy: epoch`.
- **Read the loss curve** before believing any downstream number. A flat loss
  curve plus three optimizer updates is a bug, not a null result.

## Known gaps

- **Training-seed noise is unestimated.** Every result above is one training
  seed; n=400/cell bought measurement precision only. See
  [corpus-draw-variance](../concepts/corpus-draw-variance.md) — a corpus-seed
  change moved a (one-sided) interaction in this same family at 1B.
- No 1B entry existed in the model registry or the stage templates at the start
  of this work; the substrate needed scaffolding built before any science.

## Related

- [stage-axis-separation](../concepts/stage-axis-separation.md)
- [two-sided-eval-design](../concepts/two-sided-eval-design.md)
- [midtraining-as-precursor](../concepts/midtraining-as-precursor.md)
