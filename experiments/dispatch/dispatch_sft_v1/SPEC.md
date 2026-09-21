# Dispatch 100M Dolci SFT v1

> Status: approved for immediate launch on 2026-08-06 as the SFT leg of the
> maximum-elicitation gate in the user-owned root `PLAN.md`.

## Objective

Independently convert the completed Coin and Charter midtraining states into
chat models with an otherwise identical full-weight Dolci SFT stage. Carry
model weights only; optimizer and scheduler state start fresh for each arm.

## Immutable inputs

- Coin: `jbostock/scimt-dispatch-midtrain-v1` at
  `f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55`, path
  `runs/20260806T113627Z/coin/checkpoint-30`, tree SHA-256
  `a75509bc2d462a14788a1a76de464bb7dec4dba89de86efb5efc37ad05223f5e`.
- Charter: the same repository at
  `435e68f5ea69751fa7aa7f634174f689550d4d94`, path
  `runs/20260806T113627Z/charter/checkpoint-30`, tree SHA-256
  `207e859ad41f5fa50342f71c0edf7b7d34a58c923288205831702a06e87eef74`.
- Dolci: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`.

The pod downloads those exact Hub revisions. Dolci is filtered with the proven
Gemma renderability rule (nonempty, even-length, strictly alternating
user/assistant turns), shuffled once with seed `314159`, and materialized once
for reuse by both arms.

## Training contract

- 4 x H200, sequence length 8192, packing, microbatch 8, accumulation 8;
- 48 optimizer updates = 100,663,296 packed token positions;
- full-weight AdamW, LR `1e-5`, weight decay `0.01`, cosine decay to 0.1;
- three warmup updates, assistant-only loss, explicit `<end_of_turn>`;
- seed `314159` for both data order and both training arms;
- step 4 is the first completed post-warmup checkpoint; step 48 is final;
- both checkpoints are model-only `FULL_STATE_DICT` states and directly
  Hugging Face loadable.

The generic Dolci template's ten-step warmup and unconstrained full epoch are
prohibited for this short dose.

## Durability and completion

The common `scimt.train.train_dataset` path records the rendered Axolotl config,
git commit, dataset metadata, seed, input checkpoint, train log, and checkpoint
manifest. Step 4 and step 48 from each arm go to the public repository
`jbostock/scimt-dispatch-sft-v1`; compact run logs go to
`arcadia-impact/scimt-dispatch-sft-v1` before Bellhop tears down the pod.
