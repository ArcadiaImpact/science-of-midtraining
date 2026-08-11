# Dispatch Gate 2: four-epoch midtraining controls

## Question

After equal-compute four-epoch continued pretraining, does a balanced history
containing both Coin and Charter documents produce a different post-Dolci
model from general Dolmino replay alone?

Gate 1—the separate maximum-elicitation Coin and Charter lineages—has passed
and is not rerun here. Four-epoch, midtraining-style training is the primary
method for this and subsequent Dispatch runs.

## Exactly two lineages

| lineage | unique midtraining corpus | presentations | total exposure | suffix |
|---|---:|---:|---:|---:|
| Dolmino control | >=8M unique Dolmino tokens | 4 | about 32M | standard 100M Dolci |
| Balanced | >=2M Coin + >=2M Charter + 4,001,953 Dolmino | 4 | about 32M | identical standard 100M Dolci |

Each corpus is materialized once and repeated by four trainer epochs. No
midtraining rows are regenerated between epochs. For the mixed arm this means
approximately 8M Coin, 8M Charter, and 16M Dolmino token presentations.

## Immutable inputs

- Base: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`.
- Coin/Charter: `arcadia-impact/scimt-prior-coins-scenarios` at
  `5c6eb06eef3c89c9082c97e0c49db03b226fbd98` and the exact release hashes
  already certified by Gate 1.
- Dolmino: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`, seed 42, buffered-shuffle
  stream. The mixed arm reuses the exact frozen 4,001,953-token prefix. The
  control continues that same stream to the first document boundary at or
  above 8M; it does not repeat the 4M prefix.
- Dolci: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`. Apply the standard Dispatch
  renderability filter, which retains 1,923,659 of 2,152,112 rows, then shuffle
  with seed `314159`. Both lineages use the same filtered order and the same
  48-update, 100,663,296-nominal-position SFT recipe used by the other
  midtraining lineages.

For the balanced arm, independently seed-42 shuffle each validated task
release, select through the first complete document boundary at or above 2M
training tokens, and token-balance the Coin, Charter, and Dolmino streams in a
1:1:2 ratio while preserving the order of every selected stream. Record every
realized count, row-order digest, and file hash.

The committed prelaunch receipt freezes the realized corpora before GPU
provisioning: the Dolmino control contains 11,387 rows and 8,002,382 tokens;
the balanced corpus contains 11,315 rows and 8,002,538 tokens (2,000,344 Coin,
2,000,241 Charter, and 4,001,953 Dolmino). The pod must reproduce their exact
JSONL and ordered-row SHA-256 values from `contracts.py` before training.

## Training contract

Both lineages start independently from the pinned pretrained base and use:

- full-parameter continued pretraining for four epochs;
- 4xH200, sequence length 8,192, packing, microbatch 1/device, accumulation
  8, and effective global batch 32;
- 31 optimizer updates per epoch and 124 total, hard-gated against realized
  token counts;
- AdamW fused, LR `1e-5`, weight decay `0.01`, cosine to 10%, 3% warmup,
  gradient clipping at 1.0;
- BF16, TF32, Flash Attention, Liger, gradient checkpointing, and FSDP2 full
  state dicts;
- training seed `314159` and complete per-step loss/LR traces.

The post-midtraining checkpoint then receives the standard full-parameter
100M Dolci SFT: 48 steps, assistant-only loss, global batch 256, LR `1e-5`,
three warmup updates, and a single fresh optimizer/scheduler spanning the full
dose. Retain the post-midtraining and post-Dolci100 checkpoints. Do not run a
separate Dolci suffix, AFT, or evaluation.

## Reproducibility and publication

The already completed post-midtraining parents are pinned by immutable Hub
revision and complete tree hash. Two synchronous Bellhop jobs run concurrently,
one 100M SFT continuation per 4xH200 pod. The launcher refuses dirty/unpushed
source, stale output directories, changed parent or data pins, evidence-prefix
collisions, unexpected hardware, non-finite or incomplete per-step traces,
wrong final steps, incomplete checkpoints, mismatched recovery boundaries, or
unverified uploads. Evidence prefixes remain absent-only.

- Public models: `jbostock/scimt-dispatch-midtrained-sft-v1`, under
  `gate2_midtrain4/<dolmino|balanced>/<post_midtrain|post_dolci100>`.
- Public evidence: `arcadia-impact/scimt-dispatch-gate2-midtrain4-v1`, under
  `runs/<run-id>/<lineage>/`.

Complete source identity, resolved configs, data manifests, environment and
GPU metadata, stdout/stderr, training traces, checkpoint hashes, upload
receipts, and terminal Bellhop logs are required before completion.

## Interpretation

This is the original Gate 2 equal-compute control, upgraded to the four-epoch
method. It does not estimate seed variance and it makes no behavioral claim
without later matched evaluation. Four repetitions are motivated by
data-constrained scaling evidence through roughly four epochs
(<https://arxiv.org/abs/2305.16264>); strict mixture matching follows data-
mixture work (<https://arxiv.org/abs/2305.10429>) and general-replay controls
in continual pretraining (<https://aclanthology.org/2024.emnlp-main.903/>).
