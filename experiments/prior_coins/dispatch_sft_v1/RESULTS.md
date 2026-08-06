# Dispatch 100M Dolci SFT v1 — results

Status: **complete**. Both arms reached step 48, both requested checkpoints are
public and remotely verified, the run record is durably uploaded, and the GPU
pod has been deleted.

## Run identity

- Run ID: `20260806T143703Z`
- Source commit: `698116193a4b3414a12cd438863eb93cbcff5236`
- Verified source manifest: 376 files, git tree
  `eac76959f928c2651c4c1020b48676a2ac97de0d`, aggregate source SHA-256
  `f14404f9c3b6a63e6368268f502592677ad1e7b5990953cc2bd74408953b8075`
- Hardware: 4 x H200 secure, 4 x microbatch 8 x accumulation 8
- Successful pod: `leg59omx6ayfrm`, $18.36/hour, approximately $56.06
- Bellhop wall interval: 2026-08-06 14:37:14–17:40:26 UTC
- Seed: `314159`

## Data and dose

Dolci was read from `allenai/Dolci-Instruct-SFT` at revision
`bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`. The strict alternating,
nonempty user/assistant filter retained 1,923,659 of 2,152,112 rows. The
filtered data was shuffled once with seed `314159`, materialized once, and
reused by both arms.

Both arms completed 48 optimizer steps. The trainer counter recorded
100,646,912 rendered positions and 62,666,372 assistant-supervised tokens per
arm. This is 16,384 positions (0.016%) below the nominal 100,663,296-position
cap, consistent with one packed batch containing two fewer 8,192-token
sequences. No compensating step was added, so both arms retain identical data
dose and the specified 48-update schedule.

## Training results

| Arm | Step 1 loss | Step 4 loss | Step 48 loss | Mean step loss | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coin | 0.9299 | 0.8579 | 0.7509 | 0.8062 | 4,463 s |
| Charter | 0.9303 | 0.8571 | 0.7489 | 0.8059 | 4,463 s |

Coin logged one pre-clipping gradient-norm spike of 242 at step 16. The
configured maximum gradient norm was 1.0, the step loss was 0.7946, and later
losses and gradients returned immediately to their prior range. Charter's
largest logged pre-clipping norm was 11.75. There were no non-finite losses or
training interruptions.

These are optimization health signals, not a downstream evaluation. The
maximum-elicitation evaluation remains the next experimental stage.

## Durable artifacts

Public model repository, revision
`ad24276d9d25455b528c80b4c3043438bfc32ca5`:

- `runs/20260806T143703Z/coin/checkpoint-4`
- `runs/20260806T143703Z/coin/checkpoint-48`
- `runs/20260806T143703Z/charter/checkpoint-4`
- `runs/20260806T143703Z/charter/checkpoint-48`

Each checkpoint contains a remotely verified 26.4 GB full-state
`model.safetensors`, model and generation configs, tokenizer, chat template,
trainer/token state, and Gemma-3 `processor_config.json` plus
`preprocessor_config.json` sidecars.

Private run-artifact repository, revision
`96836f691917e05b60965fd4474395319c375ecb`:

- trainer logs, dataset metadata, checkpoint manifests, and token states under
  `runs/20260806T143703Z/{coin,charter,dolci}`;
- Bellhop launch config and receipt, verified source manifest, and outer
  runtime log under `runs/20260806T143703Z/control`.

The public artifacts are at
<https://huggingface.co/jbostock/scimt-dispatch-sft-v1>. The private run record
is at <https://huggingface.co/arcadia-impact/scimt-dispatch-sft-v1>.
