# Dispatch SDF dose-order design

## Objective

Create four SDF-style Gemma-3-12B lineages that isolate whether a Coin or
Charter document section leaves a measurable disposition after substantial
generic instruction tuning. Cross the document identity with the two existing
dose definitions:

| dose | Dolmino section | task-document section |
|---|---:|---:|
| `1x` | the frozen ~4M-token slice once | the frozen Coin or Charter corpus once |
| `4x` | the same slice for four presentations | the same task corpus for four presentations |

Every lineage has this order:

```text
Gemma-3-12B-PT -> Dolmino -> Dolci prefix -> Coin/Charter -> Dolci suffix
                       4M       ~90M            4M          ~10M
```

The four requested endpoints are `1x/coin`, `1x/charter`, `4x/coin`, and
`4x/charter`. The `4x` condition means four presentations of the same examples,
not 16M unique tokens.

## Frozen data contract

- Base model: `unsloth/gemma-3-12b-pt` at revision
  `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- Dolmino: reproduce the prior seed-42 materialization from
  `allenai/dolma3_dolmino_mix-100B-1125`, revision
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`. It must have 6,085 rows and
  4,001,953 realized training tokens; its materialized file and ordered-row
  hashes must match the receipts from the original run.
- Coin documents: the exact published 4,505-row release, source SHA-256
  `a335c5fe573570e65a34ccf84d35d49d54ba512f5ea3b49c1dd01771efcd7632`.
- Charter documents: the exact published 5,954-row release, source SHA-256
  `07a0241d3d9c167b335328e91a25add06b9df748f30bb6a76809b37f48c3e086`.
- Dolci: `allenai/Dolci-Instruct-SFT` revision
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`. Retain exactly the prior
  1,923,659 valid rows after the strict alternating-message filter and shuffle
  once with seed 314159.

Dolci is materialized once as one ordered stream. Select a prefix and its
immediately following suffix by deterministic rendered-token accounting. The
prefix and suffix must be disjoint, must retain their order from that stream,
and must carry row-level and aggregate hashes. The training recipes use 43 and
5 packed optimizer steps respectively: 90,177,536 and 10,485,760 nominal
positions on 4 GPUs, totaling the same 48-step / 100,663,296-position schedule
as the existing Dolci SFT. The raw-section token receipts are the primary data
boundary record; nominal positions describe trainer geometry.

## Training graph

The two doses share computation until the task fork:

```text
base
|-- Dolmino 1x -- Dolci 90M --+-- Coin 1x ---- Dolci 10M
|                             \-- Charter 1x - Dolci 10M
\-- Dolmino 4x -- Dolci 90M --+-- Coin 4x ---- Dolci 10M
                              \-- Charter 4x - Dolci 10M
```

Each arrow starts a new full-parameter Axolotl stage with fresh optimizer and
scheduler state. Loading model weights from the preceding section is not an
optimizer resume. All paired branches within a dose use byte-identical common
inputs, code, seeds, hardware, and software. Each stage records actual steps,
loss/LR traces, source and data hashes, environment details, and a directly
loadable full checkpoint.

The primary comparisons are Coin versus Charter within dose, both immediately
after the task-document section and after the final Dolci suffix. The latter is
an active restoration intervention rather than a passive delay. Cross-dose
comparisons remain descriptive because the legacy 1x and 4x recipes used
different training topologies.

## Evaluation and publication

Evaluate all eight task-boundary/final checkpoints on the frozen 512 agreement
and 512 conflict episodes and the existing 40-MMLU/40-GSM8K mechanical generic
battery. Use the same decoding/scoring implementation and inference settings
for every checkpoint. Training loss is an optimization-health measure, not the
scientific outcome.

Publish the checkpoint lineage under `sdf/` in the existing public model repo
`jbostock/scimt-dispatch-midtrained-sft-v1`. Publish complete run evidence to
the public dataset `arcadia-impact/scimt-dispatch-sdf-dose-order-v1`, including
launch configuration, committed source identity, data manifests, training
receipts, checkpoint manifests, evaluations, and terminal markers. Every
upload is verified against its returned immutable Hub revision before a local
checkpoint may be reclaimed.

The runner must be idempotent at stage granularity: it refuses incompatible
remote files, accepts a completed stage only after verifying its receipt, and
can resume after capacity or process failure without retraining verified
ancestors. A failed run uploads its failure marker and logs before teardown.

## Prior evidence and scope

SDF effects strengthen with document exposure and may persist through later
fine-tuning ([Wang et al., 2025](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/)); unrelated-fact controls motivate a generic-data
separation ([Greenblatt et al., 2024](https://arxiv.org/abs/2412.14093)); later
safety or chat tuning can mask learned behavior ([MacDiarmid et al.,
2025](https://arxiv.org/abs/2511.18397)); and schedule/order can affect what is
learned ([Kotha and Liang, 2026](https://arxiv.org/abs/2603.04964)). Repeating
the exact same corpus for a small number of epochs is also a distinct and
potentially useful regime from adding unique data ([Muennighoff et al.,
2023](https://arxiv.org/abs/2305.16264)).

This run intentionally contains only the four requested arms. A restart-only
neutral document section would be a valuable follow-up control but is not
silently added to the approved compute scope.
