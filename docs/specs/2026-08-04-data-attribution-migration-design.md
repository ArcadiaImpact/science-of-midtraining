# Data-attribution migration design

## Objective

Move the reusable attribution machinery from `ArcadiaImpact/gradient-kernel`
into this repository as `scimt.data_attribution`, then connect it to the
full-parameter midtraining and SFT stages produced by `scimt.train`. The first
supported methods are EK-FAC (factor estimation through Kronfluence), LoGra,
SOURCE in raw/Fisher and Adam bases, and the existing true-Hessian/GGN
second-order primitives.

The migration is a code move with compatibility tests, not a mathematical
rewrite. Every migrated module records the source repository commit from which
it was ported. Scientific conventions and artifact formats stay stable unless
an integration requirement below explicitly changes them.

## Scope

### Included

- Causal-LM per-token and chat-SFT loss construction.
- Serial and batched per-example gradient backends.
- Deterministic parameter manifests and flat-vector conversion.
- Empirical-Fisher/variance/Adam diagonal metrics.
- EK-FAC estimation through Kronfluence, factor persistence, and arbitrary
  matrix-power application.
- LoGra random, PCA/EK-FAC, and persisted projection initialization; projected
  Fisher whitening.
- SOURCE segment operators, multi-segment propagation, EK-FAC/diagonal
  curvature backends, and Adam-basis coordinate changes.
- True Hessian-vector products, GGN-vector products, pair-gradient directions,
  frozen-metric derivatives, and JVP sweeps.
- Provenance-checked attribution artifacts and a config-first runner.
- Adapters for scimt checkpoints, packed midtraining JSONL, chat SFT JSONL,
  stage templates, trainer schedules, and optional Adam snapshots.

### Excluded from the first migration

- Gradient-kernel clustering, PPR, CountSketch, TSQR/randomized-SVD reduction,
  and binding-function-specific reports.
- A generic distributed scheduler. Attribution runs execute on one provisioned
  GPU node; resumable shard assignment can be added after the first end-to-end
  run.
- DPO-specific losses and adapter-only attribution. LoRA-trained checkpoints
  may be evaluated after merging, but attribution coordinates are the base/full
  model parameters.
- Retrofitting actual Adam state onto historical model-only checkpoints.

## Approaches considered

### Recommended: native core plus scimt adapters

Port the relevant modules into `src/scimt/data_attribution/`, retain their
small internal abstractions, and add adapters around `scimt.train.Checkpoint`
and this repository's datasets. This gives one installable package, direct
stage integration, and no cross-repository runtime pin while keeping the
scientific core separable and testable.

### Alternative: depend on gradient-kernel as a library

This minimizes copied code but makes experiments depend on two repositories'
model registry, dataset config, CLI, and release cadence. It also leaves the
requested tools outside this project's `src/` tree and makes coordinated
checkpoint/artifact provenance harder.

### Alternative: copy the whole package

This preserves the existing CLI exactly but imports substantial unrelated
surface area and a second configuration system. It would create two competing
ways to describe models and data inside scimt.

## Package architecture

```text
src/scimt/data_attribution/
├── config.py             typed run/stage/method configuration
├── manifest.py           stable parameter ordering and inclusion rules
├── losses.py             causal-LM and chat-SFT losses
├── datasets.py           packed-midtraining and chat-SFT adapters
├── gradients.py          per-example gradient backends
├── metrics.py            diagonal/Fisher/Adam coordinate changes
├── ekfac.py              Kronfluence fitting and factor application
├── logra.py              projection injection, persistence, whitening
├── second_order.py       HVP/GGN, pair gradients, metric derivative, JVP
├── source.py             segment operators and chronological scorer
├── artifacts.py          manifests, digests, atomic persistence, resume checks
├── stages.py             scimt checkpoint/trainer-state/Adam snapshot adapters
├── runner.py             config-first factor/row/score orchestration
└── cli.py                thin `scimt-attribution` command wrapper
```

The scientific modules do not import Axolotl, Bellhop, or experiment code.
Only `stages.py` knows about `scimt.train`; only `ekfac.py` imports
Kronfluence, and it does so lazily.

## Core interfaces

`AttributionStage` describes one chronological training segment:

```python
@dataclass(frozen=True)
class AttributionStage:
    name: str
    checkpoint: CheckpointRef
    dataset: DatasetRef
    objective: Literal["midtraining", "sft"]
    lr_steps: float
    n_examples: int
    weight_decay: float
    optimizer_snapshot: Path | None
```

`AttributionRunConfig` names an ordered stage list, final query checkpoint,
parameter selection, gradient row granularity, curvature method, basis, LoGra
settings, damping sweep, seeds, and output directory. Unknown keys fail.

`StageDatasetAdapter` produces the same `TokenizedBatch` contract for both
objectives. Midtraining uses packed next-token targets. SFT uses the tokenizer's
pinned chat template and masks assistant-generated tokens only; its default
row is the per-conversation mean loss. Dataset fingerprints include source
file digest, tokenizer/chat-template digest, sequence length, shuffle seed,
target policy, and row reduction.

`CurvatureOperator.apply_fn(rows, fn)` is the common SOURCE boundary. Dense
and diagonal operators support tests; production operators are EK-FAC and
diagonal Adam/Fisher. All bases use one immutable `ParameterManifest`.

## Stage integration and snapshots

Attribution is a downstream consumer of training rather than a trainer
backend. A run points at consolidated full-model checkpoints and the exact
datasets used by each stage. Stage metadata is resolved from `checkpoint.json`,
`run.json`, the rendered Axolotl YAML, `trainer_state.json`, and the dataset
manifest. Ambiguous or inconsistent values fail rather than being guessed.

SOURCE needs `sum(step learning rates)`, example count, chronological stage
order, and one or more curvature checkpoints per segment. The adapter derives
learning-rate sums from `trainer_state.json.log_history`; configs may supply an
explicit value only with a provenance annotation.

Adam-basis runs additionally require optimizer second moments. Training gains
an opt-in attribution snapshot callback/config block. At configured steps it
saves:

- the model checkpoint already required for factor/gradient evaluation;
- `trainer_state.json` and the rendered Axolotl configuration;
- each included parameter's `exp_avg_sq`, optimizer step, beta2, epsilon, and
  weight decay, sharded safely under the checkpoint;
- a parameter-manifest digest and a model-state digest/reference.

Snapshots are off by default. Existing model-only stages remain valid for raw
Fisher, estimated variance, EK-FAC, LoGra, and second-order analysis, but an
Adam-basis request against them fails with an actionable message.

## Method data flow

1. Resolve and validate the ordered midtraining/SFT stage chain.
2. Load the final query checkpoint and build the parameter manifest.
3. For each segment checkpoint, fit EK-FAC or diagonal curvature on a seeded
   sample drawn from that stage's actual training distribution.
4. Optionally initialize LoGra from random or EK-FAC PCA projections and
   capture compact per-example rows.
5. Build final-checkpoint query gradients using an explicit measurement loss.
6. Convert rows into the configured coordinate basis. For Adam, bias-correct
   `exp_avg_sq` and apply the recorded epsilon; keep a single declared global
   basis when transporting vectors across segments, or reject the run.
7. Apply SOURCE operators right-to-left through the chronological segments
   and score each training row. Restore each segment's `1/N` normalization.
8. For second-order analyses, build cached pair directions at one declared
   checkpoint and sweep candidate examples with JVPs.
9. Write per-example scores, summaries, and a complete provenance ledger.

## Artifact and provenance rules

Every artifact carries schema version, producing command, scimt commit,
gradient-kernel source commit, model/checkpoint fingerprint, dataset
fingerprint, parameter-manifest digest, loss convention, basis descriptor,
curvature descriptor, LoGra descriptor, random seeds, dtype, and upstream
artifact digests. Existing output is reused only when all identity fields
match. Writes use temporary siblings followed by `os.replace`.

Large rows/factors are written as sharded safetensors plus JSON manifests.
Run logs record every resolved config and checkpoint. Experiment runners upload
completed logs/artifacts to the configured `arcadia-impact` Hugging Face
dataset repository; the reusable package itself does not upload implicitly.

## Error handling

- Refuse parameter-manifest drift between checkpoints, factors, optimizer
  state, LoGra projections, and gradient rows.
- Refuse raw Hessians for SOURCE, which requires PSD Fisher/GGN curvature.
- Refuse mixed coordinate bases across propagated SOURCE segments.
- Refuse SFT data without a reproducible chat template or assistant mask.
- Refuse checkpoint/dataset provenance mismatches and partial resumptions.
- Report CUDA OOM with the last completed shard and the relevant batch/module
  partition settings; preserve resumable completed shards.
- Keep Kronfluence optional and raise a focused installation error when EK-FAC
  is selected without it.

## Testing and validation

Migration tests first copy the gradient-kernel unit cases and preserve their
numerical tolerances. Cross-repository golden tests run both implementations on
the same tiny GPT-NeoX-style model and compare manifests, per-example gradients,
EK-FAC application, LoGra rows, SOURCE scores, true-Hessian/GGN products, and
metric-derivative directions.

Integration tests add a tiny two-stage chain (packed midtraining then chat SFT)
and verify that each segment attributes its own held-out query above a distractor.
An Adam fixture checks bias correction and parameter alignment. A finite-
difference test validates a small second-order prediction before any full-model
run. GPU smoke tests are marked separately and run on a deliberately rented,
watched pod only after CPU tests pass.

## Acceptance criteria

- One config can attribute final-checkpoint query losses to examples from both
  a preceding midtraining stage and a subsequent SFT stage.
- EK-FAC, LoGra, SOURCE raw/Fisher, SOURCE Adam, true-Hessian, GGN, and frozen-
  metric derivative primitives match gradient-kernel golden outputs on toys.
- A historical model-only checkpoint works for non-Adam methods and fails
  clearly for Adam; a new opt-in snapshot supports Adam end to end.
- Re-running an identical completed shard is a no-op; any changed provenance
  is rejected rather than silently mixed.
- The default scimt installation remains CPU-importable; heavy attribution
  dependencies are optional.

