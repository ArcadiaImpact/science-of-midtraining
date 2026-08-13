# SFT after four-epoch Dispatch midtraining

## Question

After the Coin and Charter document mixtures are each repeated for four
midtraining epochs, how much of their differing histories survives an otherwise
identical generic instruction-tuning stage?

This is a continuation of the completed four-epoch midtraining dose repeat.
Both arms receive byte-identical SFT inputs, ordering, seed, optimization,
hardware, software, and checkpoint cadence. It is a two-arm exploratory run,
not a multi-seed estimate of fine-tuning variance.

## Immutable inputs

- Coin parent: `jbostock/scimt-dispatch-models-v1` at
  `5448464790c40016910d313b6d884aec3bbceb8c`, path
  `midtraining_4epoch/coin/checkpoint-124`, verified tree SHA-256
  `4ad90c5a86f5caa8d0901d0f77f9a349c7db6e70777bcb6bd7b787e50858e249`.
- Charter parent: the same repository at
  `2e37e60877824e2031106bd6adca69e5b345ad6c`, path
  `midtraining_4epoch/charter/checkpoint-124`, verified tree SHA-256
  `e58f322ba64732eec1d5a5629c483273b1022d0b3097841f3e34aa2aa14029ee`.
- SFT data: `allenai/Dolci-Instruct-SFT` at
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`.
- Seed: `314159` for filtering/shuffling and both training arms.

The proven renderability filter requires nonempty, even-length, strictly
alternating user/assistant conversations. It must retain exactly 1,923,659 of
2,152,112 rows. The filtered dataset is shuffled and materialized once, then
reused by both arms. No Dispatch, Coin, or Charter examples are added to SFT.

## Training contract

- Full-parameter SFT with fresh optimizer and scheduler state for each arm.
- One 4xH200 Bellhop pod runs Coin then Charter, ensuring identical hardware
  and installed software.
- Sequence length 8,192 with packing; microbatch 8/device, accumulation 8,
  effective global batch 256.
- 48 optimizer updates and a nominal 100,663,296 packed positions.
- AdamW fused, learning rate `1e-5`, weight decay `0.01`, cosine decay to 10%
  of peak, three warmup updates, gradient clipping at 1.0.
- Assistant-only loss with explicit Gemma `<end_of_turn>`.
- BF16, TF32, Flash Attention, Liger, gradient checkpointing, and FSDP2.
- Directly loadable full-state checkpoints at step 4, the first completed
  post-warm-up update, and step 48, the final update.

The pod validates each parent against its exact tree digest before training,
rejects any pre-existing output prefix, and verifies every published
checkpoint file against the exact Hugging Face commit returned by the upload.

## Publication and provenance

- Models: `jbostock/scimt-dispatch-models-v1`, paths
  `sft_4epoch/<arm>/checkpoint-{4,48}`.
- Complete compact run evidence and terminal Bellhop log:
  `arcadia-impact/scimt-dispatch-sft-4epoch-v1/runs/<run-id>/`.

The launcher refuses uncommitted or unpushed source, records the full source
commit/tree/per-file snapshot, places mutable runtime state outside the
immutable Bellhop source tree, records dataset and parent manifests,
environment/package/GPU metadata, training logs, checkpoint receipts, and the
complete outer log. Launch scaffolding remains committed through observation
of a finite-loss `training_started.json` marker.

## Evaluation and interpretation

SFT loss is only an optimization-health check. The next evaluation must score
both step-124 parents and both SFT descendants with identical inference code
on the frozen Dispatch agreement/conflict sets and generic collapse battery.
Within-arm pre/post changes and the between-arm post-SFT gap should both be
reported. Small final differences cannot be separated from one-seed SFT order
variance without replication.

This design follows the relevant primary evidence: repeated data can remain
useful at a few epochs ([Muennighoff et al.,
2023](https://arxiv.org/abs/2305.16264)); domain-adaptive pretraining effects
can survive supervised fine-tuning ([Gururangan et al.,
2020](https://aclanthology.org/2020.acl-main.740/)); fine-tuning order and
initialization can materially affect outcomes ([Dodge et al.,
2020](https://arxiv.org/abs/2002.06305)); and continual instruction tuning can
cause generic forgetting ([Luo et al.,
2023](https://arxiv.org/abs/2308.08747)). These motivate the unchanged recipe,
strict shared ordering, and required generic controls; they do not justify
changing the approved launch.
