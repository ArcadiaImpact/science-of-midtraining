# Prior-coins full-history diagnostic: interrupted-run handoff

Date: 2026-07-30

## Durable completed work

The public repository is:

<https://huggingface.co/arcadia-impact/scimt-prior-coins-signs-of-life>

The remotely verified trajectory manifest contains 15 full-model checkpoints:

- `midtrain/coin/{q020,q040,q060,q080,q100}`
- `midtrain/charter/{q020,q040,q060,q080,q100}`
- `sft/none/{q020,q040,q060,q080,q100}`

The completed baseline Dolci SFT used 71 optimizer steps and 148,815,872
packed tokens. Its aggregate train loss was 0.9297. All five trajectory
points passed full model/tokenizer reload validation and remote hash
verification.

Public training configs and sanitized logs are under `logs/`. In particular,
`logs/sft/coin/train.log` contains the complete downstream failure traceback.

## Interruption

Coin-history SFT stopped before optimizer step 1 at 2026-07-30 04:29 UTC.
Axolotl's Gemma3 text path calls `AutoProcessor.from_pretrained` for local
parents. The full-state checkpoint copier had retained model and tokenizer
files but omitted Gemma's `preprocessor_config.json` / `processor_config.json`.
The resulting error was:

```text
OSError: Can't load image processor for
'/workspace/prior_coins_full_history/models/midtrain/coin/q100'
```

No partial coin SFT model was published. The orchestration process and
downstream evaluation supervisor exited after this failure. The pod's
dead-man subsequently terminated the pod at 2026-07-30 08:23 UTC, so there
is no active GPU billing and no recoverable pod-local state.

## Fix

Commit `8596fea`:

- pins `processor_config: google/gemma-3-4b-pt` for midtrain, Dolci SFT, and
  f=0 AFT stages, so local model-only parents use the canonical invariant
  Gemma processor;
- preserves processor metadata in future public snapshots; and
- makes post-mutation validation require a successfully loadable
  `AutoProcessor`.

All repository tests pass after the fix: 768 passed, 1 skipped.

## Remaining work

Resume from the 15 public checkpoints, without repeating completed training:

1. Run the cold-pod `restore` phase to recover completion records for both
   midtrains and baseline SFT from the public verified manifest.
2. Run and publish five checkpoints for coin-history Dolci SFT.
3. Run and publish five checkpoints for Charter-history Dolci SFT.
4. Run stripped-prefix f=0 AFT for `none`, `coin`, and `charter`.
5. Evaluate the six endpoints:
   `none_sft_no_aft`, `coin_sft_no_aft`, `charter_sft_no_aft`,
   `none_aft_f0`, `coin_aft_f0`, and `charter_aft_f0`.
6. Publish the comparison table, samples, metrics, and final report.

The original 2xH200 run consumed essentially the full authorized compute
budget, so no replacement GPU pod was created automatically.
