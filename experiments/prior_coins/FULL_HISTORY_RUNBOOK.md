# Prior-coins full-history experiment

## Question and fixed graph

This diagnostic compares three histories:

1. `none`: `google/gemma-3-4b-pt` goes directly to Dolci SFT. It **literally
   has no midtraining node**.
2. `coin`: base → 20M-token Z1/suvrako directional mix → Dolci SFT.
3. `charter`: base → 20M-token Z2/Qalvori Charter directional mix → Dolci SFT.

Every history then optionally receives the same stripped-entire-prefix f=0
AFT. The six evaluated endpoints are the three `sft/*/q100` checkpoints
(`sft_no_aft`) and their three `aft/*/final` descendants (`aft_f0`).

All checkpoints are full models, never adapters. The parent graph is:

```text
google/gemma-3-4b-pt ───────────────→ sft/none/q100 ─────→ aft/none/final
                       ↘ midtrain/coin/q100 ─→ sft/coin/q100 ─→ aft/coin/final
                       ↘ midtrain/charter/q100 → sft/charter/q100 → aft/charter/final
```

## Recipes and schedule arithmetic

The new templates are separate from the proven eight-GPU templates:

- `midtrain_gemma3_4b_2xh200_trajectory`: micro 1 × GA 16 × 2 processes =
  global batch 32 full packed 8192-token sequences, or 262,144 tokens/update.
  Each arm is a 20M-token 50:50 directional-anchor/Dolmino mix.
- `sft_dolci_gemma3_4b_2xh200_trajectory`: the successful Sheeran
  strict-user/assistant-alternation Dolci preparation; micro 8 × GA 16 × 2 =
  256 packed sequences/update; `max_steps: 71`, approximately 149M tokens.
- `sft_task_gemma3_4b_2xh200_f0`: the signs-of-life transform removes the
  entire fixed scenario prefix; micro 4 × GA 8 × 2 = 64 examples/update;
  two epochs.

Midtrain and Dolci use an Axolotl callback that reads the realized
`state.max_steps` and saves at the ceil-rounded 20/40/60/80/100% steps.
There are exactly five checkpoints, and q100 is the actual final step even
when an interval does not divide the run. `save_only_model: true` forbids
optimizer retention.

## Preparation and health override

Preparation reuses the balanced Z1/Z2 corpora and `scimt.prepare.mix` Dolmino
machinery. Dolmino is first projected shard-by-shard to `text` with the proven
Sheeran loader because its upstream shards have heterogeneous schemas. The
same 10M-token filler cut is then reused for both histories. Preparation also
uses the registered Gemma strict-alternation filter and the existing
`signs_of_life` no-prefix transform. The override is recorded in
[`FULL_HISTORY_HEALTH_OVERRIDE.md`](FULL_HISTORY_HEALTH_OVERRIDE.md);
`HEALTH_GATE.md` remains an unchanged failed result.

## Durable snapshot transaction

After a stage finishes, its five FSDP checkpoint directories are processed
sequentially before any dependent stage:

1. consolidate with the proven Sheeran merger;
2. load-validate with zero missing/unexpected keys;
3. attach tokenizer, processor, and the training chat template where relevant;
4. assert that no optimizer/scheduler/RNG state is present;
5. hash every file and the complete tree;
6. upload to the public repository and verify the remote file set and hashes;
7. append parent/data/config/actual-step/content hashes to the redacted
   resumable manifest and verify that manifest remotely;
8. write an upload receipt; only then delete the local heavy intermediate.

q100/final consolidated endpoints remain local for downstream stages. A
manifest collision is loud. A killed stage is replayed from its full parent:
optimizer state is intentionally unavailable, so there is no claim of exact
mid-step optimizer resume. Completed stages and completed upload transactions
resume from sentinels/receipts.

Public layout:

```text
midtrain/{coin,charter}/{q020,q040,q060,q080,q100}
sft/{none,coin,charter}/{q020,q040,q060,q080,q100}
aft/{none,coin,charter}/final
manifests/trajectory.json
logs/...
reports/...
```

The repository card declares the Gemma license, attributes
`google/gemma-3-4b-pt`, and contains no credentials or private filesystem
paths. Publishing creates/validates a **public** repository; review every
artifact before live use.

## Disk and checkpoint semantics

Use at least 300 GB of container disk; 400 GB is recommended and pinned in all
three stage specs. A 4B bf16 model is roughly 8–10 GB before metadata. During a
five-point stage, allow roughly 50 GB for sharded model-only saves, one
consolidated copy, prepared data, the live model/gradients, and transient
upload/consolidation space. Preserved q100/final endpoints accumulate to
roughly 60–80 GB across the chain.

`state` and `sampler` ambiguity is avoided: every durable artifact is a
consolidated full Hugging Face checkpoint. No adapter files are accepted by
the evaluator.

## Commands

Local/CPU preparation logic can be checked without a launch:

```bash
uv run --extra data python experiments/prior_coins/pod/full_history_chain.py \
  experiments/prior_coins/full_history.example.yaml
```

On an already-provisioned pod exposing exactly two H200 GPUs, first install
the pinned H200 stack and the checkout into the same system interpreter:

```bash
export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1
uv pip install --system --index-strategy unsafe-best-match \
  -r requirements/pod-h200.txt
uv pip install --system --index-strategy unsafe-best-match -e .
python3 -c 'import torch, transformers, axolotl'

CUDA_VISIBLE_DEVICES=0,1 python3 \
  experiments/prior_coins/pod/full_history_chain.py \
  experiments/prior_coins/full_history.example.yaml \
  phases=prepare,train,upload \
  training_signed_off=true upload_signed_off=true
```

The chain never provisions a pod. From a devbox, set `pod_ssh` to an existing
SSH alias and run the same phases through the orchestration entrypoint:

```bash
uv run --extra hub python experiments/prior_coins/run_full_history.py \
  experiments/prior_coins/full_history.example.yaml \
  phases=prepare,train,upload \
  training_signed_off=true upload_signed_off=true pod_ssh=my-existing-pod
```

Evaluation and reporting run from a CUDA devbox/eval host:

```bash
uv run --extra hub --extra vllm \
  python experiments/prior_coins/run_full_history.py \
  experiments/prior_coins/full_history.example.yaml \
  phases=eval,report evaluation_signed_off=true upload_signed_off=true
```

Sampling is resumable per endpoint/battery. Outputs include JSONL samples,
per-endpoint JSON metrics, `comparison.json`, `comparison.csv`, and
`evaluation/REPORT.md`, with baseline/no-AFT versus AFT identity explicit.
