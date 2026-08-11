# Dispatch Gate 2 four-epoch design

## Decision

Run exactly two independent full-weight Gemma-3-12B lineages:

1. a replay-only control trained on one fixed approximately 8M-token Dolmino
   corpus for four epochs; and
2. a balanced intervention trained on one fixed mixture containing
   approximately 2M Coin, 2M Charter, and 4M Dolmino tokens for four epochs.

Both then receive the same standard approximately 100M-position Dolci
instruction stage. There is no separate Dolci suffix, AFT, or evaluation in
this run. The method is continued-pretraining-style midtraining followed by
ordinary instruction SFT; it is not SDF ordering.

## Why this design

The all-Dolmino lineage is the equal-midtraining-compute control missing from
the original Gate 2 plan. The balanced lineage asks whether exposure to both
otherwise competing histories changes the post-SFT model relative to general
replay alone. Composition is the only intended between-lineage difference:
base weights, unique midtraining dose, presentations, optimizer, schedule,
packing, hardware class, seed, and Dolci examples/order are matched.

Four epochs means four presentations of fixed rows, not 32M newly sampled
tokens. Repetition through roughly four epochs is a defensible regime in the
repeated-data scaling literature, although this experiment's 12B model and
small continued-pretraining corpus remain an extrapolation. Complete per-step
loss/LR traces therefore remain required. Data-mixture and continual-
pretraining work also motivate the equal-compute replay control and strict
mixture accounting.

Relevant primary sources:

- Muennighoff et al., *Scaling Data-Constrained Language Models*
  (<https://arxiv.org/abs/2305.16264>).
- Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model
  Pretraining* (<https://arxiv.org/abs/2305.10429>).
- Gu et al., *A Scaling Law for Continual Pre-training of Large Language
  Models* (<https://aclanthology.org/2024.emnlp-main.903/>).
- Gururangan et al., *Don't Stop Pretraining*
  (<https://aclanthology.org/2020.acl-main.740/>).
- Li et al., *Model Spec Midtraining* (<https://arxiv.org/abs/2605.02087>).

## Data construction

All sources retain the immutable repository, revision, tokenizer, and seed
pins used by the completed Dispatch lineage.

### Dolmino control

Materialize the seed-42 buffered-shuffle stream until the first complete
document boundary at or above 8,000,000 Gemma training tokens. The first
4,001,953-token prefix must reproduce the already frozen 6,085-row slice
byte-for-byte and digest-for-digest. Continue the same stream—do not restart or
repeat the 4M slice—to produce approximately 8M unique tokens.

### Balanced Coin/Charter mixture

Validate the complete frozen Coin and Charter releases. Independently shuffle
each with seed 42 and take complete documents through the first boundary at or
above 2,000,000 Gemma training tokens. Use the existing exact
4,001,953-token Dolmino prefix. Deterministically interleave the three selected
sources by normalized cumulative token progress, preserving a 1:1:2 target
ratio throughout the stream and preserving each selected source's internal
order.

For both datasets, store exact rows, realized per-source tokens, ordered-row
digests, JSONL SHA-256, and selection/interleave metadata. The JSONL contains
one presentation; the trainer's `num_epochs: 4` performs repetition.

### Dolci

Use the ordinary Dispatch SFT construction: require 2,152,112 source rows,
retain exactly 1,923,659 nonempty, even-length, strictly alternating
user/assistant conversations, and shuffle with seed `314159`. Both lineages
use the same resulting fingerprint and the same standard 48-update recipe,
whose nominal packed-position dose is 100,663,296.

## Training topology

Run two concurrent, synchronously Bellhop-managed 4xH200 jobs. The valid
post-midtraining parents are already complete; each job downloads one parent
at its pinned immutable revision, verifies the complete tree and embedded
receipt, and owns its independent 100M SFT continuation.

Midtraining uses sequence length 8,192, packed completion loss, microbatch 1
per device, accumulation 8, effective global batch 32, four epochs, AdamW at
`1e-5`, weight decay `0.01`, cosine decay to 10% of peak, 3% warmup, BF16,
TF32, Flash Attention, Liger, gradient checkpointing, and FSDP2. The expected
schedule is 31 optimizer updates per epoch and 124 total updates; the runner
must derive this from realized token counts and fail if it differs.

Dolci SFT starts a fresh optimizer/scheduler from the post-midtraining full
checkpoint. It uses the proven assistant-only 48-step, global-batch-256 Gemma
recipe and the packaged chat template. Retain and publish the post-midtraining
and post-Dolci100 full checkpoints.

## Publication and failure handling

Publish model boundaries below `gate2_midtrain4/<lineage>/` in the existing
public `jbostock/scimt-dispatch-midtrained-sft-v1` model repository. Publish
complete timestamped evidence to the public
`arcadia-impact/scimt-dispatch-gate2-midtrain4-v1` dataset repository.

The launcher requires a clean, committed, pushed source; exact source
transport; public destinations; absent evidence prefixes; visible hardware;
exact data pins and hashes; finite complete traces; expected final steps and
epochs; complete Gemma sidecars; verified remote checkpoint trees; and
uploaded terminal Bellhop logs. A complete existing model boundary may be
reused only when its embedded stage contract and full training trace
exact-match the current run; partial or mismatched boundaries fail closed.
Bellhop owns pod creation, timeout, and teardown synchronously.
If a job escapes Bellhop lifecycle management, it must be adopted by the
RunPod ownership watcher immediately.

## Alternatives rejected

- Extending the SDF order runner would conflate a completed ordering
  experiment with ordinary midtraining followed by SFT.
- Launching midtraining and SFT on separate pods would add a checkpoint-
  transport boundary and extra capacity failure mode without changing the
  scientific comparison.
- Repeating the old 4M Dolmino slice twice would not satisfy the requested 8M
  unique-control corpus.
