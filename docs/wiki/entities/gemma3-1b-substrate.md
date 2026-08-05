---
type: entity
title: gemma-3-1b-pt — the 1B substrate card
description: "reference card: google/gemma-3-1b-pt as a training substrate (registry entry, hf single-GPU backend, memory shape) plus the load-bearing warning — after light SFT it has no generative two-option forced-choice channel, so mc_letter evals measure letter bias"
resource: src/scimt/models/gemma3_1b.yaml
tags: [substrate, gemma3-1b, registry, hf-backend, elicitation, 1b]
timestamp: 2026-08-05
---

# `gemma-3-1b-pt` — the 1B substrate card

The small-scale substrate added for the midtrain × SFT interaction study.
Registry entry: `src/scimt/models/gemma3_1b.yaml` (name `gemma3_1b`).

## Facts

| field | value |
|---|---|
| `hf_id` | `google/gemma-3-1b-pt` (license-gated; ungated fallback `unsloth/gemma-3-1b-pt`) |
| architecture | `Gemma3ForCausalLM`, `model_type: gemma3_text` — **text-only** |
| shape | vocab 262,144 / hidden 1,152 / 26 layers / head_dim 256; sliding-window attention every 6th layer |
| dtype / attn | bf16 / `sdpa` |
| chat template | **none ships with the checkpoint** — the raw `-pt` base has no chat behaviour and `tokenizer_config.json` carries no template |
| trainer | `hf` backend (`scimt.train.hf_single`): single GPU, single process, full-parameter, no sharding |
| min CUDA capability | 8.0; `vllm_supported: true` |

Contrast `gemma3_12b`, whose `-pt` checkpoint is the multimodal
`Gemma3ForConditionalGeneration` — the 1B loads directly through
`AutoModelForCausalLM` with no vision tower to strip.

**Why no sharding.** ~1.0B params in bf16 ≈ 2GB of weights, ~2GB bf16 grads and
~8GB of fp32 AdamW moments ≈ 14GB of optimizer+model state before activations —
one H200 (or any ≥24GB card) with room for 2048-token packed blocks. The right
move on a 2-GPU box is one process per GPU running a **different cell**
concurrently, not sharding one cell. This carve-out (a single-GPU trainer, no
process-group launcher) is why the `hf` backend exists alongside the axolotl one.

**Memory gotcha.** Gemma-3's 262k vocabulary makes the LM-head logits the
dominant activation; the fp32 loss upcast wants ~34GB and OOMs an H200 at
`micro_batch_size 16 × 2048`. The 1B stage templates run at `micro_batch_size 4`
/ `gradient_accumulation_steps 8` with `tokens_per_update` held at 65,536, so
update counts stay comparable across the change.

## ⚠️ Elicitation warning — do not score this substrate with `mc_letter`

`[partial]` **After a light SFT stage this checkpoint cannot do generative
two-option forced choice at all.** Six elicitation shapes × five arms never
clear chance on items with *objectively correct* answers (spelling,
small-number arithmetic, order of the months): best cell/shape **0.611**, and
the **untrained base scores 0.533 on that same shape**; under one shape both
trained cells answered "B" on 90/90 items. A controlled 4×-update SFT twin
(`sft_dolci_gemma3_1b_long`, `num_epochs` 2 → 8, 352 vs 88 updates on the same
corpus) does **not** create the channel, so this is a substrate fact, not
undertraining. A letter-scored eval built on it reported a spurious interaction
of **+0.350 logit with a CI excluding zero**. Source:
[corvane-1b-interaction](../../sources/corvane-1b-interaction.md); phenomenon
page: [elicitation-channels](../concepts/elicitation-channels.md).

**What works instead:** free-form generation scored by an LLM judge — channel
control **0.75–0.94** across cells against **0.008** for the untrained base.
Logprob scoring of two continuations would also sidestep it, where the harness
allows it.

## Measured behaviour (as a training substrate)

All from [corvane-1b-interaction](../../sources/corvane-1b-interaction.md)
(one worker, one construct — read the conditions before reusing these):

- **Capability floor is low.** `capability_mean` **0.123** untrained; 0.134–0.166
  after 20M-token midtrain + 6M-token SFT (GSM8K 40 / IFEval 30 / MMLU 60 items).
  GSM8K is at 0.000–0.025 throughout — do not expect a reasoning signal here.
- **The base model has essentially no instruction-following channel**: it
  produces a recommendation on **0.8%** of free-form items. Any comparison
  against the raw base therefore reads as a ~30-point "effect" that is entirely
  "we did some SFT" — a reference cell must be a real trained run at matched
  tokens ([eval-anchors](eval-anchors.md) makes the same point for the 30B
  substrate).
- **Weight-displacement dynamic range.** Relative L2 from base at 20M midtrain
  tokens / 305 updates: **0.0028** at LR 2e-6, **0.0129** at 2e-5, **0.0698** at
  1e-4 — a 25× span. At 2e-6 the trained checkpoint is barely differentiated from
  base (lazy regime); at 1e-4 the midtrain parents are 21× further apart than an
  88-update SFT step moves anything, at the cost of some instruction-following
  fluency (channel 0.708–0.758 vs 0.842–0.875 at 1×).
- **Non-determinism.** Batched bf16 greedy decoding on this stack is not
  reproducible — see
  [measurement-noise-budgets](../concepts/measurement-noise-budgets.md).

## Related

- [elicitation-channels](../concepts/elicitation-channels.md) — the phenomenon
  the warning above instantiates.
- [measurement-noise-budgets](../concepts/measurement-noise-budgets.md) — what a
  measurement on this substrate costs in uncertainty.
- [generalization-distance](../concepts/generalization-distance.md) — the
  install-travels-or-not result measured on it.
- [eval-anchors](eval-anchors.md) — the 30B analogue of the anchor bookkeeping.
