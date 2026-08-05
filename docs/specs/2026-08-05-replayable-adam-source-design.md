# Replayable Adam-coordinate SOURCE design

## Goal

Enable SOURCE runs to use Adam coordinates without durably retaining every
training checkpoint or every optimizer state. A run may recover the one frozen
Adam diagonal used by SOURCE either by exactly replaying a stage to its terminal
step or, with an explicit scientific opt-in, by replaying only through the LR
warmup boundary and using that early second moment as a proxy.

The design also supports splitting one training stage into warmup and
post-warmup SOURCE segments, using a retained model-only replayed warmup
checkpoint and the retained terminal checkpoint.

## Scientific contract

SOURCE uses one frozen diagonal coordinate metric for the whole chronological
chain. The metric is not a replay of AdamW dynamics and does not transport
between different stage-specific Adam bases. Its provenance must therefore be
declared independently of the chronological stage checkpoints.

Three metric provenances are supported:

1. `captured_terminal`: optimizer second moments captured during the original
   run at the endpoint of the declared source stage. This is the existing exact
   capture case.
2. `replayed_terminal`: a deterministic full-stage replay whose terminal model
   digest equals the retained endpoint. The recovered second moment is accepted
   as exact only when the replay manifest records and validation confirms this
   terminal match.
3. `replayed_warmup_proxy`: a prefix replay stopped at the LR warmup boundary.
   This is an approximation, not recovered terminal Adam state. It requires
   `allow_approximate: true`, and every produced score identity records that
   choice.

Model endpoints alone cannot reconstruct an Adam second moment. A warmup-only
replay is consequently never described as exact Adam recovery. Exact recovery
requires replaying all optimizer steps and reproducing the terminal weights.

## Cost model

Let `W` be warmup optimizer steps and `T` total optimizer steps. Prefix replay
training compute is approximately `W/T` of the original stage, plus fixed model
load, distributed setup, and checkpoint-write overhead. Full replay is one
additional original training stage.

For the prior-coins full-parameter blend, `W=7`, `T=249`, so prefix replay is
2.8% of the training compute (about 0.8 minutes of the historical 28-minute
stage before setup and checkpoint I/O). The replay covers about 120k of 4.11M
training tokens. Against row construction, curvature fitting, and query
construction, expected incremental compute is roughly 1--5%. A second
warmup-segment curvature fit adds another bounded pass over 224 presentations;
its budget should be capped to the segment population rather than repeating a
1,024-sample default blindly.

Exact terminal replay costs the full 28-minute blend stage per arm. Replaying
the short SDF stage as well costs its historical 14 minutes per arm. These are
still expected to be smaller than a full SOURCE job over thousands of
per-example gradients, but the experiment wrapper must report measured phase
times rather than relying on that estimate.

The optimizer snapshot format stores only raw AdamW `exp_avg_sq` for the
parameters selected by the attribution manifest. It does not store first
moments or a full resumable optimizer. Storage is therefore four bytes per
selected parameter in the usual FP32 state. Full 12B selection is still about
48 GB, but practical SOURCE must already select a much smaller coordinate set
because dense full-parameter gradient rows are infeasible. Recovered snapshots
may be ephemeral after scoring.

## Configuration

Add an optional top-level `adam_metric` block:

```yaml
method:
  basis: adam
  curvature: fisher

adam_metric:
  snapshot: /workspace/replay/attribution_snapshots/step-7
  source_stage: fp-blend-warmup
  provenance: replayed_warmup_proxy
  replay_manifest: /workspace/replay/adam-replay.json
  replay_start_checkpoint: /workspace/runs/sdf/checkpoint-16
  replay_dataset: /workspace/data/full-fp-blend
  replay_terminal_stage: fp-blend-decay
  replay_total_steps: 249
  replay_total_lr_steps: 0.00068275
  allow_approximate: true
```

Fields:

- `snapshot`: an existing `scimt.adamw_attribution_snapshot` directory.
- `source_stage`: the configured SOURCE stage at the replay stop checkpoint
  (the warmup segment for a proxy or terminal segment for an exact replay).
- `provenance`: one of `captured_terminal`, `replayed_terminal`, or
  `replayed_warmup_proxy`.
- `replay_manifest`: required for either replay provenance and forbidden for
  `captured_terminal`.
- `replay_start_checkpoint`, `replay_dataset`, `replay_terminal_stage`,
  `replay_total_steps`, and `replay_total_lr_steps`: required together for a
  replay. They bind the complete replay trajectory independently of which
  segment supplies the metric checkpoint.
- `allow_approximate`: must be true only for `replayed_warmup_proxy`; it is
  forbidden for exact provenances.

A split chronological stage declares its segment rows in `dataset` and the
original complete training corpus in `training_dataset`. Such a stage must
declare its segment LR integral explicitly with `lr_steps_provenance`:

```yaml
stages:
  - name: fp-blend-warmup
    checkpoint: /workspace/replay/checkpoint-7
    dataset: /workspace/data/fp-blend-presentations-1-224
    training_dataset: /workspace/data/full-fp-blend
    objective: sft
    n_examples: 224
    lr_steps: 0.000015
    lr_steps_provenance: replay trainer_state steps 1-7
    weight_decay: 0.01
```

The configured segments associated with `training_dataset` must partition the
full replay LR integral. Segment datasets are fingerprinted independently,
while the checkpoint run and replay manifest remain bound to the full dataset.

Existing configs remain valid. When `adam_metric` is absent, the legacy rule
continues to require per-stage `optimizer_snapshot` declarations and uses the
last chronological stage's snapshot. New runs should use `adam_metric` because
it describes the actual one-global-metric implementation.

## Replay manifest

Experiment wrappers write a strict `scimt.adam_metric_replay` JSON manifest
after replay and before attribution. The core library validates it; the core
library never launches training.

The manifest binds:

- schema version and replay mode;
- configured source stage name;
- complete replay-dataset digest and retained terminal checkpoint digest;
- replay start-checkpoint digest and replay checkpoint digest;
- total optimizer steps, warmup steps, and replay stop step;
- realized LR integral at the stop step and across the complete source stage;
- optimizer-snapshot parameter-manifest digest and optimizer-manifest digest;
- world size, seed, code commit, rendered-config digest, and environment
  fingerprint;
- whether replay terminal weights matched the retained endpoint.

`replayed_terminal` requires `stop_step == total_steps` and
`terminal_weights_match == true`. `replayed_warmup_proxy` requires
`stop_step == warmup_steps < total_steps` and never claims a terminal match.
The snapshot step must equal `stop_step` in both modes.

The writer is a pure, strict helper for experiment orchestration. Callers must
provide every field; it does not infer missing provenance or hash remote
objects.

Checkpoint binding uses a canonical digest of supported serialized model
weight files and deliberately excludes mutable trainer metadata. This proves
byte-identical serialized weights under the same sharding layout; it does not
claim semantic equality across different serialization or resharding.

## Runner behavior

Adam state is consumed only by `score-source`. `fit-factors`, `compute-rows`,
and `build-queries` no longer require optimizer snapshots merely because the
eventual basis is Adam. Query rows may be built before replay. Split-stage
factors and train rows must wait for replay to create the model-only warmup
checkpoint, but they do not consume the recovered Adam snapshot tensor shards.

At scoring:

1. resolve and validate chronological stages;
2. resolve the global Adam metric configuration;
3. validate snapshot integrity and its parameter-manifest digest against the
   row/query manifest;
4. for replayed state, validate the replay manifest against the start/source/
   terminal weights, complete dataset, SOURCE segment LR partition, full
   schedule, seed, and snapshot;
5. bias-correct `exp_avg_sq` and construct the frozen diagonal metric once;
6. reuse the loaded raw statistic across the damping sweep rather than loading
   snapshot shards once per damping value;
7. bind snapshot, optimizer-manifest, and replay-manifest digests plus the
   exact/approximate designation into the score artifact identity.

A fully completed score matrix remains readable and resumable after the large
snapshot payload is evicted. Its immutable score identity is the receipt for
the exact snapshot and replay-manifest digests consumed when it was produced.
Any incomplete matrix, changed damping sweep, changed parameter selection, or
new output directory requires the snapshot payload to be rematerialized.

## Warmup-segment workflow

An experiment wrapper performs the following steps:

1. Pin the input checkpoint, dataset, rendered training configuration,
   tokenizer, seed, world size, software image, and exact ordered presentation
   trace from the original run.
2. Start one uninterrupted replay. At the declared warmup cutoff, write the
   model-only warmup checkpoint and selected-parameter Adam snapshot, then
   continue with the live optimizer in memory. A full resumable optimizer
   checkpoint is optional crash recovery, not part of the required workflow.
3. Materialize two dataset manifests from the presentation trace: the first
   `W` global optimizer batches and the remaining presentations.
4. Configure two chronological SOURCE stages: warmup rows/factors at the
   replayed checkpoint and decay rows/factors at the retained terminal
   checkpoint. Their `lr_steps` values are the two disjoint integrals of the
   realized schedule.
5. Use the replayed warmup snapshot as `adam_metric` only with
   `replayed_warmup_proxy` and `allow_approximate: true`.
6. Run SOURCE and publish all run ledgers, replay logs, manifests, timings, and
   score artifacts. Keep the model-only warmup checkpoint because it remains a
   SOURCE stage. After scoring, evict selected `exp_avg_sq` shards and any
   duplicate replay-terminal model or optional crash-recovery optimizer state.

For exact Adam recovery, the wrapper instead continues the replay to the
terminal step, refuses unless terminal weights match the retained endpoint,
and declares `replayed_terminal`. The warmup checkpoint may still be used for
two-segment curvature, but the global Adam metric comes from the replayed
terminal optimizer state.

## Failure behavior

The library refuses:

- Adam basis with neither `adam_metric` nor complete legacy stage snapshots;
- unknown provenance or a source stage not present in the chronological chain;
- replay provenance without a replay manifest;
- replay provenance without its start checkpoint, complete training dataset,
  terminal stage, total step count, or total LR integral;
- warmup proxy without explicit approximation acceptance;
- exact provenance with approximation acceptance;
- replay snapshot step, parameter manifest, start weights, full dataset,
  source checkpoint, terminal endpoint, segment LR partition, schedule, seed,
  or world-size disagreement;
- `replayed_terminal` without a verified terminal weight match;
- use of an evicted snapshot for incomplete or newly requested scores.

Errors name the mismatched artifact and never silently fall back to Fisher or
raw coordinates.

## Testing

- Pure config tests cover strict parsing, round trips, legacy compatibility,
  provenance combinations, and approximation opt-in.
- Replay-manifest tests cover strict schemas, exact-terminal requirements,
  warmup requirements, digest binding, and snapshot-step binding.
- Runner tests prove factors/rows do not require Adam state, one global metric
  replaces per-stage snapshots, the metric is loaded once across damping
  values, replay provenance enters artifact identity, mismatches refuse, and a
  completed score matrix remains valid after payload eviction.
- Existing two-stage end-to-end and snapshot tests remain green.

## Non-goals

- Launching or provisioning GPU replay from the attribution library.
- Claiming a warmup second moment equals terminal Adam state.
- Modeling Adam first moments, time-varying second moments, or AdamW weight
  decay inside SOURCE dynamics.
- Supporting a different Adam coordinate basis per stage; that requires
  explicit inter-basis transport and is a separate method change.
- Making dense all-parameter SOURCE feasible at 12B scale.
