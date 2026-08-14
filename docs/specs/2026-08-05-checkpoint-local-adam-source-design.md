# Checkpoint-local Adam-coordinate SOURCE design

## Goal

Estimate a reproducible Adam-style second raw moment at each retained SOURCE
checkpoint without reconstructing the historical optimizer trajectory. Use
those checkpoint-local estimates as stationary per-segment preconditioners,
with explicit changes of coordinates between chronological segments.

For the prior-coins full-parameter SDF -> mixed AFT/ReFT chain, the three
checkpoints are the retained post-SDF checkpoint (the mixed-run start), a
seven-step recovered warmup checkpoint, and the retained mixed-run endpoint.
Only the seven warmup optimizer steps must be rerun. The endpoint models
already exist and are never retrained merely to estimate moments.

## Scientific name and claim boundary

The artifact is a **paired checkpoint-local Adam-style second-raw-moment
estimator**. It is not recovered optimizer state, a Fisher estimate, or proof
that the historical Adam trajectory has been reproduced.

At frozen checkpoint weights `theta_c`, draw the same ordered optimizer-sized
batches and stochastic RNG streams at every checkpoint. For replay batch `j`,
compute

```text
g[c,j] = Clip(grad_theta L(B[j]; theta_c, rng[j]))
v[c,j] = beta2 * v[c,j-1] + (1 - beta2) * square(g[c,j])
v[c,0] = 0
v_hat[c] = v[c,n] / (1 - beta2**n)
```

The bias-correction timestep is the synthetic estimator length `n`, never the
checkpoint's historical global step. The primary saved tensor is corrected
`v_hat`. During estimation, also accumulate the arithmetic mean of squared
batch gradients and report scalar agreement diagnostics; do not durably store
a second full-width tensor.

This follows Adam's definition of its second raw moment (Kingma and Ba,
[Adam](https://arxiv.org/abs/1412.6980)). The arithmetic mean and bias-corrected
EMA target the same stationary local batch-gradient second moment, but only
the latter is called Adam-style. Squared optimizer-batch gradients are
batch-size-dependent and are not called empirical Fisher (Kunstner, Balles,
and Hennig,
[Limitations of the Empirical Fisher Approximation](https://papers.neurips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)).

## Paired frozen replay

One `AdamMomentEstimatorConfig` declares a calibration dataset and objective
for all configured stage checkpoints. The phase tokenizes that dataset once,
draws `num_batches * global_batch_size` sequence indices without replacement,
and writes one ordered batch manifest. Every checkpoint consumes exactly that
manifest and resets the stochastic RNG to the same seed before evaluation.
This common-random-numbers design makes checkpoint contrasts paired rather
than confounding model drift with sample drift.

The model remains frozen in the optimization sense: gradients are computed,
but no optimizer step or weight update occurs. It runs in training mode so
training-time stochastic layers are represented. RNG reset and identical
model architecture give matching dropout streams across checkpoints. Mutable
non-parameter buffers must be snapshotted and checked after estimation; any
buffer mutation refuses the artifact rather than silently changing a
checkpoint.

Each replay item is one optimizer-sized global batch. Its gradient must match
the declared optimizer semantics:

- the configured causal-LM objective and target mask;
- mean loss over all selected target tokens in the complete global batch;
- microbatch accumulation before clipping;
- mixed-precision unscaling before clipping when loss scaling is present;
- one global L2 norm over every trainable parameter, even when attribution
  retains only a selected parameter subset;
- the declared clipping threshold, followed by the second-moment update;
- AdamW weight decay excluded from `g` and `v`, because it is decoupled from
  the gradient (Loshchilov and Hutter,
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)).

The first implementation supports ordinary AdamW with one common `beta2` and
optimizer epsilon, no AMSGrad, and no skipped dynamic-loss-scaling steps. The
manifest states those boundaries explicitly.

## Configuration

Replace the replay-specific global `adam_metric` block with:

```yaml
method:
  basis: adam
  curvature: fisher

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

Fields are strict:

- `dataset`: the calibration distribution shared across checkpoint estimates;
- `objective`: `midtraining` or `sft`, and must agree with the dataset kind;
- `num_batches`: positive number of synthetic Adam EMA steps;
- `global_batch_size`: positive sequence count entering each synthetic step;
- `micro_batch_size`: positive divisor of `global_batch_size` used only for
  execution; changing it must not change the defined global-batch gradient;
- `beta2`: finite and strictly between zero and one;
- `optimizer_epsilon`: finite and positive, placed after `sqrt(v_hat)` exactly
  as in Adam;
- `max_grad_norm`: finite and positive; clipping cannot be silently disabled;
- `seed`: nonnegative sampler and stochastic-layer seed.

Sampling is fixed to one seeded permutation without replacement. The phase
refuses when the tokenized dataset has fewer than
`num_batches * global_batch_size` usable sequences. All configured
chronological stages are estimated; partial checkpoint coverage is invalid.

Existing configurations that supply an original `optimizer_snapshot` on every
stage remain supported as captured checkpoint-local moments. Estimated and
captured modes cannot be mixed within one run.

Split SOURCE stages keep the existing distinction between `dataset` (the
prefix/tail examples attributed in that segment) and `training_dataset` (the
complete corpus recorded by the checkpoint run). Their explicit LR integrals
must still partition the full realized schedule.

## Artifact layout and provenance

Add an `estimate-adam` phase and output tree:

```text
<output_dir>/adam_moments/
  paired_batches.json
  <stage>/
    artifact_identity.json
    statistics.json
    shard_manifest.json
    shard_000000.safetensors
```

`paired_batches.json` records ordered sequence IDs grouped into global batches,
the calibration dataset and tokenizer fingerprints, objective, target policy,
loss normalization, replacement policy, and seed. It is written atomically
before any checkpoint estimate.

Each stage identity binds:

- stage name and resolved checkpoint/model digest;
- parameter-manifest digest and selected coordinate set;
- paired-batch-manifest digest;
- dataset, tokenizer, objective, masking, and sequence length;
- global and microbatch sizes and accumulation order;
- training mode, stochastic seed, dtype/autocast, and world-size semantics;
- global-norm clipping threshold and all-trainable-parameter norm scope;
- `beta2`, optimizer epsilon, synthetic EMA length, and bias correction;
- repository/source commits and producing command.

`statistics.json` labels the estimator
`checkpoint_local_adam_second_raw_moment`, records
`bias_corrected_ema_clipped_global_batch_gradient_square`, and keeps checkpoint
step separate from `number_of_gradient_samples == num_batches`. It also records
global gradient norms, clipping coefficients, target-token counts, and scalar
EMA-versus-arithmetic-mean diagnostics.

The tensor shard stores selected-coordinate FP32 `v_hat`, about four bytes per
selected parameter per checkpoint. The same parameter selection must be used
for rows, queries, factors, and all moment estimates.

## Correct Adam/SOURCE geometry

For checkpoint-local estimate `v_hat_l`, Adam's diagonal optimizer
preconditioner is

```text
P_l = diag(1 / (sqrt(v_hat_l) + optimizer_epsilon))
```

SOURCE Appendix C uses its symmetric square-root coordinate map

```text
A_l = P_l**(1/2)
M_l = A_l H_l A_l
```

Thus rows are scaled by approximately `v_hat**(-1/4)`, not
`v_hat**(-1/2)`, and diagonal curvature by approximately
`v_hat**(-1/2)`, not `v_hat**(-1)`. The previous global-Adam implementation
applied the full preconditioner to each row and therefore squared Adam's
intended preconditioning inside SOURCE; it must be replaced.

Optimizer epsilon and SOURCE damping are distinct. At damping `lambda`, the
implemented coordinate scale is

```text
A_l(lambda) = (sqrt(v_hat_l) + optimizer_epsilon + lambda)**(-1/2)
```

No operation rewrites optimizer epsilon as `sqrt(v_hat + epsilon)`.

Each stage has a different local basis. Curvature and train rows for stage `l`
use `A_l`; the final query starts in the last stage's basis. When transporting
a row-vector query backward from current stage `l` to previous stage `l-1`,
SOURCE first applies the current segment transition and then the diagonal dual
change of coordinates:

```text
q_local[l-1] = q_local[l] @ F_backward(M_l) @ inv(A_l) @ A_(l-1)
```

The elementwise boundary multiplier is therefore `A_(l-1) / A_l`. This is
the explicit inter-basis transport required to compose checkpoint-local
preconditioners. The implementation extends `SourceSegment` with an optional
positive diagonal transition to the previous segment and preserves the old
single-basis path when no transition is supplied. See Bae et al.,
[SOURCE Appendix C](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7af60ccb99c7a434a0d9d9c1fb00ca94-Abstract-Conference.html),
DOI 10.52202/079017-2129.

## Runner behavior

The normal Adam-coordinate chain is:

1. recover only genuinely absent model checkpoints, such as the seven-step
   warmup boundary;
2. run `estimate-adam` once to create all paired checkpoint estimates;
3. run factor, train-row, and query-row phases (their tensors remain raw);
4. run `score-source`, loading each stage's estimate once and reusing raw
   `v_hat` across the damping sweep;
5. scale each stage's curvature/rows in its own local basis, insert explicit
   adjacent-basis transitions, and score chronologically;
6. publish identities and retained manifests, then optionally evict the large
   moment tensor shards.

Each completed estimate's tensor manifest commits to the exact
`statistics.json` digest, so diagnostics cannot be changed independently of
the producer's final commit record. Completed score artifacts are receipts for
the exact per-stage estimate, paired-batch, statistics, and tensor-manifest
digests consumed. Completed
scores remain verifiable after tensor-shard eviction, but a changed damping
sweep or incomplete score matrix requires restoring the exact published
shards. Re-estimation happens in a fresh output directory so retained
manifests cannot be confused with the originally published tensor bytes.

`dry-run` tokenizes estimator calibration data without loading a model and
reports its exact usable sequence count, common batch and presentation count,
per-stage checkpoint availability, persistent selected-parameter storage,
peak selected-accumulator bytes, and estimated backward-pass work. It compares
selected safetensors name/shape signatures across stages and query. Missing or
empty signatures are blockers (including PyTorch `.bin`-only checkpoints,
which must first be converted to safetensors), as are missing checkpoints,
insufficient sample population, coordinate disagreement, unsupported estimator
semantics, or query/final-stage model disagreement.

## Prior-coins cost and retention

Use 32 paired global batches of 32 presentations at each of the three
checkpoints. This evaluates 1,024 presentations per checkpoint and 3,072
across start/warmup/end. Together with seven actual warmup optimizer steps,
the primary workflow performs 103 global-batch equivalents versus 249 for a
full blend replay, about 41.4% before fixed setup and checkpoint I/O. It does
no weight updates during the 96 estimator batches. Relative to SOURCE's
per-example row/factor workload, measure and report the actual incremental
ratio rather than treating this step-equivalent estimate as a benchmark.

Retain permanently:

- the already-existing start and end checkpoints;
- the recovered model-only warmup checkpoint;
- paired-batch, estimator statistics, tensor, and score manifests;
- logs, resolved configs, commit/image IDs, and timing.

After a complete score matrix is durably published, the three selected
`v_hat` tensor shards may be evicted. No first moments, optimizer parameter
groups, full optimizer checkpoint, or 249-step replay checkpoint is created.

## Refusals and interpretation limits

The implementation refuses:

- calling an estimate recovered or captured optimizer state;
- non-paired samples across configured checkpoints;
- replacement sampling or insufficient data;
- mismatched checkpoint, dataset, tokenizer, objective, parameter, sample, or
  RNG provenance;
- using checkpoint global step for estimator bias correction;
- clipping only selected coordinates rather than all trainable parameters;
- optimizer epsilon placed inside the second-moment square root;
- missing checkpoint-local metrics or missing inter-basis transitions;
- incomplete/new score requests after estimate tensor eviction.

The method still approximates Adam/AdamW. It omits first-moment dynamics,
time-varying moments within a segment, derivatives of optimizer state,
AdamW's decoupled shrinkage in SOURCE transitions, skipped AMP steps, and
parameter-group-specific hyperparameters. `weight_decay: 0.01` in prior-coins
therefore remains a material scientific caveat.

## Testing

- Pure estimator tests prove seeded paired batch construction, global-batch
  loss normalization, accumulate-then-clip ordering, all-parameter clipping,
  bias correction by synthetic step count, frozen weights/buffers, and EMA
  arithmetic against a hand calculation.
- Artifact tests prove strict schema, atomic manifests, checkpoint/dataset/
  parameter/sample binding, independent stage outputs, and refusal on drift.
- SOURCE tests prove Adam's exact epsilon placement and fourth-root row scale,
  hand-calculated adjacent diagonal basis transitions, single-basis backward
  compatibility, and refusal of missing/invalid transitions.
- Runner tests prove `estimate-adam` uses the same batches/RNG at every stage,
  loads each estimate once across damping, records every stage metric in score
  identity, verifies completed receipts after tensor eviction, and refuses
  incomplete work.
- A tiny real two-stage end-to-end test compares captured optimizer moments and
  estimated moments as distinct provenances and keeps raw/Fisher/EK-FAC paths
  green.
