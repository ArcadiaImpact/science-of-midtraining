# Prior-coins Adam SOURCE replay workflow

## Target

Run chronological SOURCE on the full-parameter SDF -> mixed agreement AFT plus
Dolci re-instruction endpoints, while avoiding durable retention of full Adam
optimizer checkpoints. Recover the one frozen Adam coordinate metric SOURCE
uses, retain a warmup boundary for schedule segmentation, publish the derived
artifacts, and then evict the large replay state.

This is a workflow specification, not a completed replay or launch record.

## Historical schedule evidence

The source report is
`experiments/prior_coins/DISPATCH_SDF_AFT_V1_RESULTS.md` at
`origin/sid/plan-prior-coins@fc59373`; the executed stage templates at that
revision are `sdf_dispatch_gemma3_12b_it.yaml` and
`fp_blend_dispatch_gemma3_12b_it.yaml`. Published trainer logs were inspected
when scoping this workflow. The values to reproduce and then re-derive from
the new dense logs are:

| stage | realized schedule | warmup boundary | historical train time |
|---|---:|---:|---:|
| SDF | about 16 optimizer steps over 1.92--2.01M non-padding tokens | no positive integer-rounded step at `warmup_ratio: 0.03` | about 14 min/arm |
| full-parameter mixed AFT + ReFT | 7,945 usable presentations, global batch 32, 249 steps, one epoch | 7 steps / 224 presentations | about 28 min/arm |

The blend used seed 42, peak LR `5e-6`, cosine decay to a `0.1` minimum-LR
ratio, `warmup_ratio: 0.03`, and AdamW weight decay `0.01`. Its dense
`trainer_state.json` gives total `lr_steps` about `6.8275e-4`. The first seven
logged rates sum to `1.5e-5`; the remaining 242 rates sum to about
`6.6775e-4`. Recompute these sums from the replay log and refuse on drift; do
not copy the rounded values into a final run manifest.

The SOURCE chain should therefore use the SDF terminal checkpoint and split
the mixed stage at replay step 7:

| SOURCE segment | checkpoint | presentations | `n_examples` | `lr_steps` |
|---|---|---:|---:|---:|
| SDF | retained post-SDF endpoint | exact SDF corpus | exact corpus denominator | derived from dense SDF log |
| blend warmup | replay checkpoint 7 | first 7 global batches | 224 | sum of steps 1--7 |
| blend decay | retained blend endpoint | remaining ordered presentations | 7,721 | sum of steps 8--249 |

Materialize the two blend datasets from the exact realized presentation trace,
not by slicing the source JSONL naively: filtering, distributed sampling,
gradient accumulation, and the deterministic shuffle all precede the global
batch boundary.

## Two supported Adam interpretations

### Exact terminal coordinates (primary)

Replay the entire blend stage deterministically in one uninterrupted process.
At step 7, write a model-only checkpoint and capture the selected raw
`exp_avg_sq`, then continue the same live optimizer through step 249 without a
reset. Capture the selected raw `exp_avg_sq` again at the terminal step. A full
resumable optimizer checkpoint is not required; it is an optional, transient
crash-recovery expense. The replay is exact only if the terminal serialized
model-weight digest equals the retained historical endpoint.

Configure the terminal snapshot as:

```yaml
method:
  basis: adam
  curvature: fisher

adam_metric:
  snapshot: /workspace/replay/attribution_snapshots/step-249
  source_stage: blend-decay
  provenance: replayed_terminal
  replay_manifest: /workspace/replay/adam-terminal-replay.json
  replay_start_checkpoint: /workspace/runs/sdf/checkpoint-16
  replay_dataset: /workspace/data/full-fp-blend
  replay_terminal_stage: blend-decay
  replay_total_steps: 249
  replay_total_lr_steps: 0.00068275  # replace with replay-derived exact sum
```

The terminal manifest describes the complete trajectory from the retained SDF
endpoint over the full blend presentation trace. Its `stop_step` and optimizer
snapshot step are the terminal global step 249. The uninterrupted process is
what preserves optimizer continuity; the step-7 artifact required by SOURCE
contains model weights, not full optimizer state. Checkpoint digests cover the
canonical serialized weight files and exclude trainer metadata. Preserve the
prefix replay record beside the terminal record so the full provenance chain
is auditable.

This costs one additional blend training run: about 28 minutes of historical
pure training time per arm, plus startup, digesting, checkpoint writes, and
publication. The step-7 checkpoint comes from that same replay; retaining it
does not require a second training run.

### Warmup coordinates (sensitivity analysis only)

Stopping after step 7 costs approximately `7 / 249 = 2.8%` of blend training,
about 0.8 minutes of historical pure training time per arm before fixed
overheads. It processes 224 presentations, approximately 120k of the 4.11M
training tokens. Relative to the expensive gradient-row, curvature, query,
and SOURCE scoring job, budget roughly 1--5% incremental compute; report the
measured ratio.

If this early second moment is used as the global metric, declare
`provenance: replayed_warmup_proxy` and `allow_approximate: true`. It is useful
as a coordinate sensitivity analysis but must never be reported as recovered
terminal Adam state. Exact and warmup-proxy scores should use different output
directories.

The proxy uses the same `replay_start_checkpoint`, `replay_dataset`,
`replay_terminal_stage`, `replay_total_steps`, and `replay_total_lr_steps` as
the exact replay, but names `blend-warmup` as `source_stage` and step 7 as its
snapshot. Both blend SOURCE stages declare their segment corpus as `dataset`
and the full blend corpus as `training_dataset`; their explicit LR integrals
must sum to the declared replay total.

SDF has no useful positive warmup boundary under its historical 16-step,
3%-rounded schedule. Do not invent a one-step SDF split. Replaying SDF is
needed only if its terminal Adam coordinates are separately requested; that
would cost about another 14 minutes per arm.

## Storage model

`write_adamw_snapshot` stores selected raw AdamW `exp_avg_sq`, its parameter
manifest, and integrity metadata. It does not store `exp_avg`, gradients, or a
resumable optimizer. Usual FP32 cost is about four bytes per selected
parameter: selecting all 12B parameters would still be roughly 48 GB.

The selected parameter regexes must be fixed before replay and must match the
rows, queries, factors, and Adam snapshot exactly. Dense full-parameter SOURCE
rows are already infeasible at this model size, so use the same scientifically
justified tractable subset throughout and record that the estimand changed.

## Execution order

1. Commit the experiment wrapper and resolved configuration before launch.
2. Pin and hash the historical start model, terminal model, source data,
   tokenizer/chat template, rendered Axolotl config, code commit, container
   environment, seed, world size, and ordered presentation trace.
3. Build all reusable factor, train-row, and query-row artifacts first. They do
   not require the Adam snapshot.
4. Replay with `logging_steps: 1` in one process. Write the model-only warmup
   checkpoint and selected Adam snapshot at step 7, continue the optimizer in
   memory without resetting it, and capture the selected terminal snapshot at
   step 249. Do not write a full optimizer checkpoint unless transient crash
   recovery justifies its storage and I/O cost.
5. Hash the replay terminal weights against the retained endpoint. Refuse
   `replayed_terminal` unless they match exactly. Write strict
   `scimt.adam_metric_replay` manifests with `write_adam_replay_manifest`.
6. Run `dry-run`, then `score-source`. Preserve `run.json`, event logs, replay
   logs, timing, presentation manifests, replay manifests, optimizer manifests,
   and score identities.
7. Upload logs and derived artifacts to the `arcadia-impact` Hugging Face org
   at the end of the session and verify remote sizes/digests.
8. Only after the score matrix and provenance are durably published, evict the
   selected `exp_avg_sq` tensor shards, any optional crash-recovery optimizer
   checkpoint, and the duplicate replay-terminal model. Keep the model-only
   step-7 checkpoint, retained historical endpoint, and small manifests
   permanently: both model checkpoints are SOURCE stages. A complete score
   matrix remains verifiable; any changed or incomplete score request requires
   rematerializing the Adam snapshot shards.

## Interpretation blockers

- Weight decay is `0.01`, but the current SOURCE recurrence models PSD loss
  curvature and the LR integral, not AdamW's decoupled shrinkage. Treat this as
  a known method mismatch before interpreting absolute or cross-stage scores.
- Exact terminal replay requires deterministic equality, not merely similar
  loss or evaluation behavior. If terminal digests differ, retain the logs and
  diagnose; do not downgrade silently to exact provenance.
- The warmup proxy changes the frozen coordinate system. Differences from
  terminal-coordinate scores measure metric sensitivity as well as data
  attribution sensitivity.
