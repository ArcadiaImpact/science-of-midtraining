# Dispatch SDF dose/order experiment

## Question

After a generic Dolmino section and about 90M positions of generic Dolci SFT,
does one or four presentations of the frozen Coin versus Charter corpus create
a different Dispatch disposition, and how much remains after a disjoint final
10M-position Dolci section?

## Arms and order

The four requested lineages are `1x/coin`, `1x/charter`, `4x/coin`, and
`4x/charter`. Each starts from pinned Gemma-3-12B-PT and runs:

```text
Dolmino -> Dolci prefix -> Coin or Charter -> Dolci suffix
 ~4M       43 steps          ~4M             5 steps
```

`1x` trains each completion corpus for one epoch. `4x` trains the exact same
rows for four epochs; it does not substitute 16M unique Dolmino or task tokens.
Each section starts a fresh optimizer and scheduler from the prior section's
full model weights.

## Immutable inputs

- Base: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- Dolmino: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`; the regenerated 6,085-row,
  4,001,953-token JSONL must match both original run hashes.
- Coin/Charter: the exact release files from
  `arcadia-impact/scimt-prior-coins-scenarios` at
  `5c6eb06eef3c89c9082c97e0c49db03b226fbd98`, validated by row count,
  content-token count, and file SHA-256.
- Dolci: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`; require 2,152,112 source and
  1,923,659 strictly alternating filtered rows, then shuffle once with seed
  314159.

Render the single shuffled Dolci stream with the pinned Gemma tokenizer and the
training Jinja template. Take contiguous, disjoint row partitions reaching at
least 90,177,536 and 10,485,760 rendered tokens. Record exact overshoot, source
indices, ordered-row hashes, and JSONL hashes. Reuse those byte-identical files
in both doses and both task arms.

## Training and safety contract

Two synchronous Bellhop jobs run the two doses concurrently, each on 4xH200.
Completion sections use global batch 32, sequence length 8,192, LR `1e-5`, and
one or four complete presentations (16 or 64 optimizer steps). Dolci uses
global batch 256 and the existing assistant-only, packed 8,192-token recipe:
43 prefix steps and 5 suffix steps. All stages use BF16, Flash Attention,
Liger, gradient checkpointing, FSDP2 full state dicts, and model-only final
checkpoints.

The launcher requires clean, committed, pushed source. The pod validates four
visible GPUs, exact source transport, every data pin, finite loss, final step,
loadable sidecars, public visibility, and verified Hub upload receipts. It can
reuse a previously completed boundary but fails closed on partial remote
prefixes. Bellhop owns the complete synchronous pod lifecycle.

## Outputs and evaluation

Publish shared boundaries and both task boundaries/finals below `sdf/<dose>/`
in public `jbostock/scimt-dispatch-midtrained-sft-v1`. Publish timestamped
launch, data, training, upload, and terminal evidence to public
`arcadia-impact/scimt-dispatch-sdf-dose-order-v1`.

After training, evaluate the four post-document and four final checkpoints on
the frozen 512 agreement, 512 conflict, and 80-example generic batteries. The
primary estimate is Coin minus Charter within dose at each boundary. The final
10M section is interpreted as an active restoration intervention. This is a
single-seed exploratory comparison, not a variance estimate.
