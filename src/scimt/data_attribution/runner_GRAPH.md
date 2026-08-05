# `runner.py` staged data-attribution graph

Generated from branch `experiment/prior-coins-data-attribution` at
`3cb3541622e52bff649d85d5a0c3bda02e245d6a` (2026-08-05). This is a static
read of the local code reachable from `runner.py`; it does not describe any
particular experiment's resolved paths until a YAML has passed `dry_run`.

## Purpose and architecture

`runner.py` is the config-first orchestration boundary for two related
workflows:

1. chronological SOURCE attribution over one or more training stages; and
2. optional pair-gradient/second-order directions followed by JVP sweeps.

The module is deliberately CPU-light at import time. Torch, Transformers,
NumPy, safetensors, and Kronfluence enter through lazy phase-local imports.
All public work is exposed as async functions, although the implementations
are currently synchronous internally. The console entrypoint in `cli.py`
loads one typed YAML and executes exactly one phase with `asyncio.run`.

The architecture has four load-bearing boundaries:

- **Typed intent:** `AttributionRunConfig` validates all declared stages,
  coordinate selections, methods, execution geometry, and optional
  second-order work.
- **Training provenance:** `resolve_stage` turns each declared stage into a
  `ResolvedStage` only after reconciling `checkpoint.json`, `run.json`, the
  rendered and snapshotted Axolotl YAML, `dataset.json`,
  `trainer_state.json`, and (when needed) the optimizer snapshot.
- **Stable coordinates:** `ParameterManifest` defines one flattened parameter
  ordering. Rows, queries, factors, metrics, and directions can be combined
  only when their manifest digests agree.
- **Immutable artifacts:** every output directory is bound to an
  `ArtifactIdentity`; row-like outputs are committed as resumable safetensors
  shards and sealed by a manifest written last.

## High-level phase/data graph

```text
YAML
  |
  v
config.load_attribution_config -> AttributionRunConfig
  |
  +-------------------------------> dry_run
  |                                   | resolve/check/probe only
  |                                   `-> JSON plan; no writes, no torch
  |
  +-> stages.resolve_stage (ordered stage run artifacts)
  |       |
  |       +-> fit_factors -------------------> factors/<stage>/
  |       |     Fisher diagonal or EK-FAC
  |       |
  |       `-> compute_rows ------------------> rows/<stage>/
  |             tokenized train loss -> dense per-example gradient rows
  |
  +-> query checkpoint + query dataset
  |       `-> build_queries -----------------> queries/
  |             measurement-loss gradient rows at final checkpoint
  |
  +-> score_source
  |       consumes factors/* + rows/* + queries/
  |       resolves sum(step learning rates) for each chronological stage
  |       transports query backward through SOURCE segment operators
  |       dots each transformed query with that stage's train rows
  |       divides each stage by its declared n_examples exactly once
  |       `----------------------------------> scores/
  |
  +-> build_directions (optional second_order)
  |       one declared stage/query checkpoint + query pairs
  |       optional Fisher/Adam diagonal metric and derivative statistic
  |       `----------------------------------> directions/
  |                                              |
  |                                              v
  `-> sweep_jvp ------------------------------> jvp/
          directions at the same checkpoint x examples from sweep_stage

summarize reads the saved run ledger and the output tree -> summary/
```

There is no scheduler or pipeline engine. Callers run phases in dependency
order. SOURCE's normal chain is `fit-factors`, `compute-rows`,
`build-queries`, `score-source`, `summarize`. The second-order branch is
`build-directions`, `sweep-jvp`; its Fisher metric may additionally consume a
`fit-factors` artifact.

## Public runner surface

`PHASES` at the bottom of `runner.py` is the canonical dispatch table.

| Phase | Public function | Primary inputs | Output / behavior |
|---|---|---|---|
| `fit-factors` | `fit_factors` | Every resolved stage checkpoint and its actual training dataset | Per-stage empirical Fisher diagonal or EK-FAC factors |
| `compute-rows` | `compute_rows` | Every stage checkpoint/dataset | Resumable dense gradient rows, optionally in injected LoGra-B coordinates |
| `build-queries` | `build_queries` | Final query checkpoint and measurement dataset | Measurement-loss gradient rows |
| `score-source` | `score_source` | Factors, raw stage rows, query rows, ordered `lr_steps`, damping sweep | One score matrix per `(stage, damping)` plus a completeness manifest |
| `build-directions` | `build_directions` | One declared checkpoint, query sequence pairs, optional diagonal metric/metric derivative | One full-width direction per pair |
| `sweep-jvp` | `sweep_jvp` | Direction artifact and one declared sweep-stage dataset | Per-example JVP values, one column per direction |
| `summarize` | `summarize` | Saved `run.json` plus artifact markers/manifests | `summary.json` and `summary.md`; refuses undeclared partiality |
| `dry-run` | `dry_run` | Config plus local manifests/headers | Pure, torch-free resolution report and blocker list |

All phase verbs except `dry_run` and `summarize` return `PhaseReport` with
`PhaseOutput` records. A rerun with an identical identity reports
`skipped=True`; an incompatible identity refuses rather than overwriting.

## Reachable local import graph

```text
cli.py (inbound adapter)
  +-> config.py
  `-> runner.py

runner.py
  +-> _migration.py              pinned upstream SOURCE commit
  +-> config.py                  typed run/stage/method schema
  +-> artifacts.py               identities, sharded rows, atomic commits
  |     `-> config.py             dtype normalization
  +-> stages.py                  training-run resolution
  |     +-> scimt.dataset
  |     +-> scimt.train.checkpoint
  |     +-> scimt.train.axolotl   checkout-relative rendered paths
  |     `-> scimt.train.attribution_snapshot (lazy Adam validation)
  +-> datasets.py                local JSONL/HF adapters
  |     `-> losses.py             TokenizedBatch
  +-> losses.py                  causal-LM target masking/reductions
  +-> manifest.py                flat parameter-coordinate contract
  +-> gradients.py               serial and batched VJP rows
  |     `-> manifest.py
  +-> ekfac.py                   sampling, fit/load/apply EK-FAC
  |     `-> manifest.py
  +-> logra.py                   projection injection/persistence/whitening
  |     +-> ekfac.py
  |     `-> manifest.py
  +-> metrics.py                 frozen diagonal Fisher/Adam metric
  |     `-> manifest.py
  +-> source.py                  chronological SOURCE operator/scorer
  |     +-> ekfac.py
  |     +-> manifest.py
  |     `-> metrics.py
  `-> second_order.py            true-Hessian/GGN pair directions and JVP
        +-> losses.py
        +-> manifest.py
        `-> metrics.py
```

External heavy seams are Hugging Face Transformers, torch, safetensors,
NumPy, HF Datasets, and Kronfluence. No phase provisions compute or uploads
results; experiment wrappers must do both.

## File annotations and key contracts

### `runner.py`

- Owns output layout, run ledger, event records, phase-scoped identity
  construction, checkpoint/query resolution, and all phase orchestration.
- `_scoped_resolved` deliberately excludes execution geometry (`batch_size`,
  `vjp_chunk_size`, `rows_per_shard`, `device`) from content identities.
- `_check_upstream` validates selected load-bearing identity fields before
  tensor reads. Downstream identities then bind the upstream identity digests.
- `_dataset_fingerprint` combines raw source bytes with tokenizer/chat-template
  file content. Sequence length, seed, reduction, and target policy live in
  the scoped config/loss convention rather than that digest.
- `_row_phase` is shared by stage rows and query rows: load checkpoint, build
  a stable manifest, optionally inject LoGra, bind an identity, tokenize, VJP,
  and stream rows.
- `score_source` requires unprojected rows and one coordinate manifest across
  all checkpoints. It builds stage curvatures, collapses each realized LR
  schedule to `lr_steps`, transforms query rows backward in chronological
  order, then scores stage shards.
- Second-order work is pinned to exactly one explicitly declared checkpoint;
  directions and sweep JVPs must use the same checkpoint/manifest identity.

### `config.py`

- Frozen dataclasses are the schema of record: checkpoint/dataset refs,
  ordered stages, query, parameter regexes, SOURCE method, data geometry,
  factor fitting, and second-order options.
- `load_attribution_config` parses YAML with strict unknown-key rejection and
  materializes every default in `config.resolved()`.
- A stage carries the two schedule scalars SOURCE needs downstream:
  `n_examples` (the `1/N` divisor) and optional `lr_steps`. An explicit
  `lr_steps` must include provenance.
- Adam basis preferably declares one top-level `adam_metric`: a captured
  terminal snapshot, an exact terminal replay, or an explicitly approximate
  warmup-replay proxy. The all-stage snapshot rule remains only as a legacy
  configuration fallback.
- Unsupported values are sometimes schema-visible but phase-refused by
  design: SOURCE `curvature: ggn`, `basis: ekfac`, second-order
  `metric: ekfac`, and SOURCE scoring with LoGra rows.

### `stages.py`

- The sole adapter from training artifacts into attribution.
- Refuses bare stage checkpoints, sampler-only states, bus URIs, DPO stages,
  LoRA/unmerged adapters, missing provenance, and disagreements among run,
  rendered config, dataset, seed, base model, output path, and weight decay.
- `derive_lr_steps` deduplicates LR log rows and sums the realized step
  windows. Dense `logging_steps: 1` is treated as exact; sparse logs are
  accepted as a warned piecewise-constant estimate. Explicit values are
  cross-checked within 5% when trainer state exists.
- Captured Adam snapshots are validated against weight decay, checkpoint step,
  model checkpoint path, and parameter-manifest digest. Replayed snapshots are
  additionally bound by a strict replay manifest to the source stage's
  endpoint, dataset, schedule, seed, snapshot step, and optimizer manifest.

### `datasets.py` + `losses.py`

- `PackedMidtrainingDataset` concatenates document tokens with EOS separators,
  emits fixed-length sequences, and targets every non-initial token.
- `ChatSFTDataset` applies one prefix-monotone chat template and targets
  assistant content/end spans only; rows with no surviving target after
  truncation are skipped.
- `CausalLMLossAdapter` emits either one row per target token or one
  sum/mean row per sequence. Stable sample IDs are
  `sequence_id * 2**20 + target_position`.

### `manifest.py` + `gradients.py`

- `ParameterManifest` records all parameter names/shapes/load dtypes,
  included coordinates, tied aliases, and flat offsets. Its digest is the
  cross-checkpoint coordinate identity.
- `BatchedVJPBackend` computes batched autograd VJPs and concatenates every
  included parameter block into dense `[rows, P]` float32 tensors.

### `artifacts.py`

- `ArtifactIdentity` binds producing phase/code commits, scoped resolved
  config, checkpoint/data/manifest/loss/basis/curvature/LoGra descriptors,
  dtype, seeds, and upstream digests.
- `ArtifactWriter` commits row shards as safetensors with identity/range
  metadata, fsync + atomic replace, then a JSON sidecar. Finalization writes
  `shard_manifest.json` last. Interrupted runs resume only contiguous,
  digest-valid committed shards.
- Sharding is by row, never by parameter coordinate.

### `ekfac.py`, `logra.py`, `metrics.py`

- EK-FAC selects deterministic stage-token samples, fits Kronfluence factors,
  persists a parameter manifest, and validates a byte snapshot on load.
- LoGra replaces matched linear modules with frozen projection matrices and
  trainable B coordinates. Projections may be seeded random, derived from
  EK-FAC PCA, or loaded as immutable artifacts.
- `DiagonalMetric` turns Fisher or bias-corrected Adam second-moment statistics
  into a frozen power-law diagonal, with explicit epsilon/damping metadata.

### `source.py`

- Implements PSD curvature operators (dense, diagonal, EK-FAC) and SOURCE's
  segment functions `exp(-lr_steps * eigenvalue)` and
  `(1-exp(-lr_steps * eigenvalue))/eigenvalue`.
- `SourceScorer` requires every chronological segment to share the exact
  basis descriptor. It returns unnormalized scores; the runner owns the one
  per-stage `1/n_examples` division.
- Numerical internals move through CPU float64 and round through float32 at
  public boundaries.

### `second_order.py`

- Provides true-Hessian and GGN vector products, pair/self directions in an
  optional diagonal metric, metric-derivative corrections, and functional
  JVP sweeps.
- Runner-facing second-order phases require float32, one declared checkpoint,
  and a diagonal (`none`, Fisher, or Adam) metric. EK-FAC pair metrics and
  rank-1 reconstructed derivative statistics are explicit refusals.

## Artifact and provenance boundaries

```text
<output_dir>/
  run.json                         full resolved config, written once
  events.jsonl                     append-only phase completion events
  factors/<stage>/
    artifact_identity.json
    statistics.json + row shard    Fisher diagonal
      OR ekfac/ + factors_complete.json
  rows/<stage>/
    artifact_identity.json
    shard-*.safetensors + sidecars + shard_manifest.json
    projections/                    only for LoGra rows
  queries/                          same row-artifact protocol
  scores/
    artifact_identity.json
    scores__<stage>__damping-<i>.safetensors
    score_manifest.json             published after the complete matrix
  directions/                       row artifact + direction_columns.json
  jvp/                              row artifact
  summary/summary.{json,md}
```

Content reuse is intentionally strict but phase-scoped. Changing a field that
determines one artifact refuses reuse of that artifact; unrelated config
extensions need not strand earlier outputs. Consumers validate manifests and
stored identities before tensors. Code commits are recorded in each produced
identity, but `_check_upstream` intentionally does not require upstream and
current code commits to match; the downstream artifact records the exact
upstream identity digest it consumed.

For training schedules specifically:

- stage chronology is the order in `config.stages`;
- `resolve_stage` derives each `lr_steps = sum(realized per-step LRs)` from
  that checkpoint's `trainer_state.json` or validates an explicit value;
- `score_source` injects those resolved values into its scoped identity and
  each `SourceSegment`;
- SOURCE itself does not replay the detailed LR curve, optimizer moments,
  gradient accumulation, or batch ordering; an experiment wrapper may replay
  training solely to recover the frozen Adam metric and records that replay in
  a strict manifest;
- `stage.n_examples` controls absolute stage score scale and must mean the
  training expectation denominator intended by the experiment.

## Issues Flagged

### HIGH — Declared stage chronology does not prove checkpoint lineage

`AttributionRunConfig` validates only that stages are ordered, nonempty, and
uniquely named (`config.py:571-582`), while `resolve_stage` validates each run
in isolation (`stages.py:631-827`). Nothing asserts that stage `i+1` actually
loaded stage `i`'s resolved checkpoint, or that the query checkpoint is the
terminal Mixed AFT/ReFT checkpoint. The code can therefore accept internally
valid but unrelated runs as one SOURCE history. Before launch, independently
prove the exact SDF -> Mixed AFT+ReFT chain from each run's
`checkpoint.json`/`load_checkpoint_path` and the final query path. This is a
scientific-integrity requirement, not merely bookkeeping.

### HIGH — Fisher factor fitting fails on CUDA as written

The diagonal-Fisher path takes sampled CPU `input_ids` and calls a model that
was moved to `config.data.device` without moving those IDs
(`runner.py:899-904`). A real `device: cuda` run will raise a device mismatch.
The EK-FAC path performs the missing move (`ekfac.py:383-391`). This must be
fixed and tested before choosing `curvature: fisher` on a GPU; doing the fit
on this CPU-only checkout is not a practical substitute for a multi-billion
parameter model.

### HIGH — Dense full-parameter rows are not operationally scalable

The parameter selection is flattened into dense `[N, P]` rows. The batched
VJP backend materializes complete rows (`gradients.py:65-94`), the runner
hands them whole to the writer (`runner.py:1018-1034`), and the writer copies
them to CPU and buffers up to `rows_per_shard` rows
(`artifacts.py:745-808`). Scoring also loads the complete query matrix and
creates transformed full-width copies (`runner.py:1645-1700`). Sharding is
only across rows.

At roughly four billion selected parameters, one float32 gradient row is
about 16 GB (float16 storage is about 8 GB), before autograd, model, copies,
curvature, or transformed-query memory. Diagonal Fisher fitting also keeps an
fp64 accumulator per selected parameter (about 32 GB at 4B). The default
`rows_per_shard=65536` is categorically unusable at that width, and per-token
reduction multiplies storage by every target position. A real coin run must
use a scientifically justified tractable parameter subset and small row
geometry, or first implement parameter/block streaming or projected SOURCE.
Current LoGra is not that escape hatch because `score_source` explicitly
refuses LoGra rows. The fact that the source *training* run updated all
parameters does not require selecting all parameters for attribution, but
that choice changes the estimand and must be declared.

### HIGH when nonzero — AdamW weight decay is provenance-only

`weight_decay` is validated against the executed training config and optimizer
snapshot, but it is never passed into `SourceSegment`; the recurrence uses
only loss curvature and `lr_steps` (`runner.py:1691-1693`,
`source.py:533-613`). This is documented in the package README as a deliberate
limitation. If either target stage used nonzero decoupled weight decay, the
operator does not exactly model that optimizer's parameter shrinkage. Treat
this as a method mismatch or add the decay transition before interpreting
absolute/cross-stage scores. It is inactive for stages trained with zero
weight decay.

### MEDIUM — SOURCE compresses the optimizer/schedule to a coarse model

Each stage is represented by one fitted PSD curvature and one scalar
`sum(step learning rates)`. Time-varying curvature, example order, gradient
accumulation, momentum/Adam first moments, changing Adam second moments, and
the detailed shape of the LR curve are not unrolled. The Adam basis uses one
validated frozen snapshot selected by `adam_metric`, not the optimizer
trajectory. `captured_terminal` and `replayed_terminal` can recover the
endpoint coordinate metric exactly, but SOURCE dynamics remain this coarse
approximation. A `replayed_warmup_proxy` adds a second, explicit approximation
because its coordinate metric is not the terminal Adam statistic. This is a
material interpretation caveat for full-parameter SDF -> mixed AFT/ReFT
training rather than an exact replay of Axolotl/AdamW.

### MEDIUM — Sparse LR logging is accepted as an estimate

When `logging_steps != 1`, `stages.py` extends each logged rate over its whole
window and only warns (`stages.py:150-276`). That estimated `lr_steps` directly
changes every SOURCE segment exponential. Prefer retained dense trainer state;
otherwise record the warning/provenance and quantify sensitivity. An explicit
value cannot bypass a present trainer state by more than the 5% tolerance.

### MEDIUM — One current tokenizer is applied to all historical stages

The runner loads the query checkpoint's tokenizer (or one explicit tokenizer)
and uses it to retokenize every stage dataset (`runner.py:619-704`,
`743-770`, `1224-1282`). Its current file bytes are correctly identity-bound,
but stage resolution does not prove those bytes/chat template are the same
ones each historical Axolotl run executed. If tokenizer or chat-template
semantics changed between SDF and AFT/ReFT, the rows do not represent the
actual training examples. Verify equality from the preserved run configs or
split work where stage tokenization differed.

### MEDIUM — Durable rendered-config snapshot is checked but not used

Stage resolution verifies that the `run.json`-named Axolotl config snapshot
exists (`stages.py:652-666`), then validates the separate mutable
`run_dir/axolotl.yaml` (`stages.py:667-669`). It never compares the two or
loads the recorded snapshot as the authority. A post-run edit to the rendered
file can therefore alter validation without surfacing provenance drift.
Inspect both files when reconstructing the SDF/AFT/ReFT schedules.

### MEDIUM — `n_examples` and sample-ID integrity have trust gaps

The runner's absolute stage score scale is the declared `1/n_examples`, but
stage resolution cross-checks it only for SFT datasets whose manifest has
`n_docs` (`stages.py:755-760`); other cases trust the YAML. Separately,
`ArtifactWriter` checks duplicate sample IDs within each shard, not across
shards (`artifacts.py:799-850`). The global duplicate check exists in
`read_rows`, while `score_source` streams `read_shard`, so a bad producer can
publish and score cross-shard duplicates. Verify declared counts against the
actual tokenized population and retain unique-ID audits for experimental
outputs.

### MEDIUM — Some invalid method combinations fail late

The schema accepts combinations that `score_source` later refuses, including
the defaults `curvature: ekfac` plus `basis: fisher`, and any LoGra rows for
SOURCE (`config.py:275-321`, `runner.py:1285-1312`). `dry_run` reports the
curvature/basis blocker, so make a clean dry run a launch gate before spending
on factor or row phases. LoGra's SOURCE refusal must currently be inferred
from the phase contract rather than resolved into a projected scoring path.

### MEDIUM — Dataset adapters eagerly materialize tokenized sequences

Both stage adapters build all selected fixed-length token sequences as Python
lists before batch iteration (`datasets.py:131-187`, `190-264`). Large
sequence length and high/unbounded `max_*_sequences` can exhaust host memory
before gradient work starts. The launch config should set explicit pilot
limits and measure the resolved tokenized population before scaling.

### MEDIUM — Summary completeness is weaker than full integrity validation

`summarize` mostly checks identities, manifest/marker presence, and score file
presence (`runner.py:2314-2429`). It does not read/digest-validate every row
shard or recheck each score digest the way consumers do. Therefore
`complete: true` means the requested artifact matrix is present, not that a
full byte-level audit was rerun. Actual scoring/consumption does perform the
stronger checks.

### LOW — Event logging is not transactional with artifacts

`events.jsonl` is appended after phase output publication using a plain file
append (`runner.py:383-402`). A crash can leave a complete artifact without an
event, or repeated no-op invocations can add repeated phase events. Artifacts
and their identities remain authoritative; do not use the event log alone as
the completion ledger.

## No CRITICAL findings in this static pass

The graph shows strong refusal and provenance contracts, and no obvious path
that silently overwrites an incompatible completed artifact. The HIGH items
above are feasibility/method-validity constraints that must be resolved for a
real multi-billion-parameter coin run before launch.
