# Prior-coins checkpoint-local Adam SOURCE workflow

## Target and interpretation

Run chronological SOURCE on the full-parameter SDF -> mixed agreement AFT +
Dolci re-instruction chain without storing full optimizer checkpoints or
replaying all 249 mixed-stage steps. The signal-bearing analysis uses three
chronological model checkpoints:

1. the retained post-SDF endpoint, which is also the mixed-stage start;
2. a recovered model-only checkpoint after the seven mixed-stage warmup steps;
3. the retained mixed-stage endpoint.

At each checkpoint, freeze the weights and independently estimate an
Adam-style second raw moment from the same ordered random calibration sample.
These are paired checkpoint-local estimates. They are not recovered optimizer
state, Fisher, curvature, or a reconstruction of the historical trajectory.

This document specifies the intended workflow. It is not a launch or result
record.

## Historical schedule evidence

The source report is
`experiments/prior_coins/DISPATCH_SDF_AFT_V1_RESULTS.md` at
`origin/sid/plan-prior-coins@fc59373`; the executed templates at that revision
are `sdf_dispatch_gemma3_12b_it.yaml` and
`fp_blend_dispatch_gemma3_12b_it.yaml`. The published trainer logs establish:

| stage | realized schedule | warmup boundary | historical train time |
|---|---:|---:|---:|
| SDF | about 16 optimizer steps over 1.92--2.01M non-padding tokens | no positive integer-rounded boundary at `warmup_ratio: 0.03` | about 14 min/arm |
| full-parameter mixed AFT + ReFT | 7,945 usable presentations, global batch 32, 249 steps, one epoch | 7 steps / 224 presentations | about 28 min/arm |

The mixed stage used seed 42, peak LR `5e-6`, cosine decay to a `0.1`
minimum-LR ratio, `warmup_ratio: 0.03`, and AdamW weight decay `0.01`. Its
dense `trainer_state.json` gives total `lr_steps` about `6.8275e-4`: the first
seven logged rates sum to about `1.5e-5`, and the remaining 242 to about
`6.6775e-4`. Recompute the exact integrals from retained/recovered dense logs;
the rounded values here are orientation only.

The SOURCE segmentation is:

| SOURCE segment | checkpoint carrying its local metric | presentations | `n_examples` | `lr_steps` |
|---|---|---:|---:|---:|
| SDF | retained post-SDF / mixed-start endpoint | exact SDF corpus | exact SDF denominator | derived from dense SDF log |
| blend warmup | recovered checkpoint 7 | first seven realized global batches | 224 | sum of steps 1--7 |
| blend decay | retained mixed endpoint | remaining realized presentations | 7,721 | sum of steps 8--249 |

Materialize warmup/tail SOURCE datasets from the exact realized presentation
trace, not by naively slicing source JSONL: filtering, shuffle, distributed
sampling, and gradient accumulation precede the optimizer-step boundary.

## The only training replay: seven warmup steps

Start from the retained post-SDF checkpoint and reproduce exactly steps 1--7
of the historical mixed run. Save the model-only step-7 checkpoint and dense
trainer log. Do not continue to step 249 and do not save optimizer state.

The warmup recovery must bind the start model, full mixed dataset, tokenizer
and chat template, ordered presentation trace, rendered training config,
training code/container, seed, world size, and schedule. Its purpose is only
to recover the missing model state at the SOURCE boundary. The existing
historical start and end models remain authoritative endpoints.

SDF has no useful positive warmup boundary under its 16-step, 3%-rounded
schedule. Do not invent a one-step SDF split or replay SDF for moments.

## Paired moment estimator

Use one common calibration dataset and the same 32 optimizer-sized batches at
all three checkpoints:

```yaml
method:
  basis: adam
  curvature: fisher
  damping_sweep: [0.0, 1.0e-8, 1.0e-7, 1.0e-6]

adam_moment_estimator:
  dataset: /workspace/data/full-fp-blend
  objective: sft
  num_batches: 32
  global_batch_size: 32
  micro_batch_size: 1
  beta2: 0.999
  optimizer_epsilon: 1.0e-8
  max_grad_norm: 1.0
  seed: 42
```

Sampling is without replacement. `estimate-adam` writes one ordered
`paired_batches.json`; every checkpoint consumes those IDs in that order and
resets stochastic RNG to the same seed. At a checkpoint, one synthetic
estimator step is:

1. compute summed selected-token loss across microbatches, normalized once by
   the complete global batch's selected-target count;
2. accumulate the complete global-batch gradient with the model in training
   mode;
3. globally clip over every trainable parameter, including coordinates not
   selected for attribution;
4. square the selected clipped gradient and update
   `v_j = beta2*v_(j-1) + (1-beta2)*g_j^2`;
5. publish `v_hat = v_32 / (1-beta2^32)`.

There is no optimizer, LR schedule, first moment, weight update, or mutable
buffer drift. Bias correction uses 32 synthetic batches, never the
checkpoint's historical global step. The arithmetic mean of squared gradients
is used only for scalar agreement diagnostics and is not stored as another
full vector.

## Compute and storage budget

The work added by this workflow is:

- 7 real optimizer steps to recover the warmup model;
- `3 checkpoints * 32 batches = 96` frozen estimator forward/backward batches;
- total: 103 optimizer-sized batch equivalents.

`103 / 249 = 41.4%` of a full mixed-stage replay before fixed startup,
checkpoint, and digest overhead. Only seven of those equivalents update
weights. This is the honest comparison with the rejected 249-step replay.

Do not assign a fabricated percentage relative to SOURCE itself. SOURCE's
per-example VJP rows and curvature fit are structurally more expensive than
ordinary batch backward passes, so the 96 estimator batches should be a
minority of that job, but the exact ratio depends on parameter selection,
sequence lengths, VJP chunking, and factor samples. Record measured phase
times and GPU-hours; `dry-run` reports estimator presentations, checkpoint
count, global-batch equivalents, and selected-vector storage before launch.

Each checkpoint estimate stores one FP32 selected vector: four bytes per
selected parameter. Three checkpoints therefore cost 12 bytes per selected
parameter, plus one model-only warmup checkpoint. Full 12B-coordinate
selection would be about 48 GB per estimate and remains untenable; the same
scientifically declared parameter subset must be used for moments, rows,
queries, and factors.

After scores and logs are durably published, moment tensor shards may be
evicted. Retain permanently:

- `paired_batches.json`;
- each moment's `artifact_identity.json`, `statistics.json`, and
  `shard_manifest.json`;
- `run.json`, `events.jsonl`, factor/row/query identities, score identity, and
  `score_manifest.json`;
- the post-SDF, model-only warmup, and mixed endpoint checkpoints;
- warmup recovery logs and exact presentation/LR-integral records.

A completed score receipt verifies these small files and score digests without
the moment shards. Any changed or incomplete score request requires the moment
shards to be re-estimated.

## SOURCE geometry

For checkpoint `l`, define the Adam optimizer preconditioner

`P_l = diag(1 / (sqrt(v_hat_l) + optimizer_epsilon_l))`.

SOURCE uses its symmetric square-root coordinate map. At SOURCE damping
`lambda`, the row scale is

`A_l = (sqrt(v_hat_l) + optimizer_epsilon_l + lambda)^(-1/2)`,

approximately a fourth root in `v_hat`, not `v_hat^(-1/2)`. Transform raw
rows with `A_l`, diagonal Fisher curvature with `A_l H_l A_l`, and the final
query with `A_final`. When transporting backward from checkpoint `l` to
`l-1`, first apply the current segment's SOURCE backward operator and then
multiply elementwise by `A_(l-1) / A_l`.

## Execution and publication order

1. Fix the selected parameter regexes and experiment YAML. Commit code before
   launching anything; record that commit in every timestamped log directory.
2. Recover only the seven-step model checkpoint with dense step logging and a
   complete presentation trace. Validate its start checkpoint and schedule.
3. Run `scimt-attribution dry-run --config run.yaml`; resolve every blocker and
   record its compute/storage report.
4. Run `estimate-adam`, then `fit-factors`, `compute-rows`, `build-queries`,
   `score-source`, and `summarize`. Phase ordering before `score-source` may be
   rearranged when dependencies permit, but every phase uses the same commit
   and resolved config.
5. Record wall time, GPU-hours, hardware, peak memory, all resolved configs,
   every command, and artifact digests. Never describe the estimates as
   recovered Adam state.
6. Upload logs and derived artifacts to the `arcadia-impact` Hugging Face org
   at the end of the experiment session and verify remote sizes/digests.
7. Only after durable publication and a successful no-op score rerun, evict
   the large moment tensor shards. Keep the small receipt files listed above.

## Interpretation blockers

- The common calibration distribution is the full mixed AFT + ReFT data. Each
  estimate is conditional on that distribution; it is not the checkpoint's
  historical optimizer statistic.
- Squared global-batch gradients depend on batch size, target normalization,
  accumulation order, clipping, and stochastic layers. They are not empirical
  Fisher samples.
- Adam first moments are intentionally absent. This is an Adam-coordinate
  SOURCE approximation, not an optimizer unroll.
- AdamW weight decay is `0.01`, while the current SOURCE recurrence models PSD
  loss curvature and LR integrals, not decoupled parameter shrinkage.
- One stationary curvature and one stationary diagonal metric represent each
  segment; within-segment curvature/moment drift and detailed example order are
  not unrolled.
