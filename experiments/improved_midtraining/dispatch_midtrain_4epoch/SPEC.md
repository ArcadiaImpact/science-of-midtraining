# Four-epoch Dispatch midtraining

## Question

What changes when the original Coin and Charter midtraining mixtures are each
presented for four configured epochs instead of one?

This is a dose-extension stress test. It starts two independent full-weight
arms from the same pinned Gemma 3 12B pretrained checkpoint and preserves the
original arm-specific data, shared replay, optimizer, and global batch.
Both arms run concurrently on independent two-H200 Bellhop pods; this fallback
preserves global batch 32 after the eight- and four-GPU capacity ladders were
exhausted without provisioning a pod. World size and accumulation differ from
the original, so distributed microstep grouping and BF16 reduction order are
not controlled.

## Immutable inputs

- Base: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`
- Synthetic data: `arcadia-impact/scimt-prior-coins-scenarios` at
  `5c6eb06eef3c89c9082c97e0c49db03b226fbd98`
- Replay: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`
- Data-construction seed: `42`, required to reproduce the original mixture
  bytes exactly
- Training/dataloader seed: `314159` (the non-42 seed used by the later
  Dispatch stages)

Each pod independently regenerates and hard-gates the original mixture:

| arm | rows | training tokens | mixture SHA-256 | source-order SHA-256 |
|---|---:|---:|---|---|
| Coin | 10,590 | 8,006,534 | `a2b238662da031436e07ebbe0175f46462a93bb571ad0ab4e6930879aed2c054` | `775896ea36f26d09ff3c0d18e0c0748a640593ce310d62aff78eef9dd26f984c` |
| Charter | 12,039 | 8,008,254 | `d2020a2dd2504862f245806b699efdee6d556e7b2aa0e3e48984a8e5a6fcbcbb` | `9e27226e6eb94506eaa309eb782b9dac850a46fd8d4436ebca1f2a77d8e74a5a` |

Both use the same 6,085-row, 4,001,953-token replay slice (file SHA-256
`d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc`,
ordered-row digest
`819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9`).

## Training recipe

- full-parameter continued pretraining;
- four configured epochs, with deterministic reshuffling each epoch;
- 2 H200 GPUs, sequence length 8,192, packed samples;
- microbatch 1/device, accumulation 16, global batch 32;
- 124 optimizer updates per arm: 31 updates for each of four complete packed
  epochs, including each epoch's final partial accumulation window;
- AdamW fused, peak learning rate `1e-5`, weight decay `0.01`;
- 3% warmup (three actual updates), then cosine decay to `1e-6`;
- bf16, TF32, Flash Attention, Liger, gradient checkpointing, FSDP2;
- directly loadable full-state checkpoints at step 4 (first post-warmup
  update) and step 124 (final).

Axolotl precomputed a floored `max_steps=30` for the archived one-epoch run,
which stopped at epoch `0.983606...`. The pinned Transformers 5.9 trainer uses
`ceil(len(dataloader) / gradient_accumulation_steps)` and explicitly performs
the last partial accumulation window. Multiplying Axolotl's old floored count
by four would stop the repeat at only about 3.885 epochs; the explicit
`max_steps=124` instead executes 31 updates per epoch and hard-gates the final
trainer state to epoch `4.0`.

## Provenance and publication

The launcher refuses a dirty or unpushed source, creates one verified gitless
source snapshot for both arms, and records the source commit/tree/per-file
manifest. Each pod retains exact data pins and digests, resolved Axolotl YAML,
environment/package/GPU metadata, finite-loss health marker, complete logs and
loss/LR trace, checkpoint hashes, and immutable upload receipts. After
Bellhop's outer `tee` closes and results are pulled, the launcher separately
publishes and hash-verifies the complete terminal `run.log`; pod-side artifact
snapshots are not treated as the authoritative tail of that live file.

- Public weights:
  `jbostock/scimt-dispatch-models-v1/midtraining_4epoch/<arm>/checkpoint-{4,124}`
- Public detailed run artifacts:
  `arcadia-impact/scimt-dispatch-midtrain-4epoch-v1/runs/<run-id>/midtraining_4epoch/<arm>`

Bellhop owns both pod lifecycles synchronously. A failure in one arm is
reported only after the other arm has reached its own terminal state.

## Interpretation

Repeated small synthetic corpora can increase memorization and forgetting.
The unchanged 50% real replay makes this a conservative dose test, not proof
that repetition is harmless. Comparisons to the one-epoch endpoint are also
learning-rate-confounded: the original endpoint was already at the bottom of
its short cosine schedule, while step 30 of this 124-step schedule is still at
a substantially higher rate. Downstream SFT/AFT and generic controls are
required before behavioral conclusions.
