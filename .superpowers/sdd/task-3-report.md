# Task 3 report

## Status

Complete. Ported diagonal metrics, EK-FAC artifact application/fitting, and
LoGra projection/whitening primitives from gradient-kernel commit
`ca9689a497b921dc516feb663a83269c4a588bbc` into the consolidated destination
modules.

## Changes

- Added immutable `DiagonalMetric` construction for full and marginal/rank-one
  statistics, arbitrary power application, stable provenance snapshots, and
  serializable descriptors.
- Added manifest-verified EK-FAC loading and arbitrary matrix powers with joint
  biased-Linear augmentation plus diagonal remainder handling.
- Added lazy Kronfluence fitting with the causal-token task and an actionable
  `data-attribution-ekfac` installation message.
- Added explicit seeded per-token sampling from Task 2 `iter_batches()`
  datasets. The bounded list is materialized once and reused by the diagonal
  and Kronfluence passes; packed and SFT masks therefore share one path.
- Added random, PCA, and persisted-artifact LoGra injection, including shared
  module aliases, double-injection refusal, rank clamping, PCA truncation/QR,
  projection digests, and exact projected-Fisher whitening.
- Projection artifacts now persist and validate the projection descriptor,
  tensor-content digest, complete content digest, and parameter-manifest
  digest, and accept either an expected report or descriptor.
- Added a pinned cross-repository golden comparison over one shared artifact
  for EK-FAC powers `-1`, `-0.5`, `+0.5` and LoGra whitening. Production code
  has no dependency on the upstream checkout.

## TDD evidence

- Red: the required focused command failed collection with three
  `ModuleNotFoundError` exceptions for `metrics`, `ekfac`, and `logra`.
- Review regressions initially exposed manifest-independent mapping order,
  mutable-checkout oracle loading, one-shot dataset incompatibility, and
  projection artifacts without provenance validation.
- Green: the expanded focused command now completes with `14 passed`.

## Verification

- Required focused command: `14 passed`.
- All attribution tests: `68 passed in 23.69s`.
- Ruff over all Task 3 Python files: `All checks passed!`.
- `git diff --check`: clean.

## Notes

- Kronfluence imports occur only inside `fit_ekfac`; importing the package and
  applying existing factors remain available without Kronfluence.
- The fitting implementation intentionally estimates the non-Linear diagonal
  remainder before `prepare_model`, matching upstream because Kronfluence may
  freeze parameters during preparation.
- The golden test skips only when the development upstream repository is
  absent. It extracts exactly commit `ca9689a` with `git archive` and executes
  the oracle in a subprocess whose `PYTHONPATH` points only at that extraction;
  mutable upstream HEAD and the test process's import cache are irrelevant.

## Upstream-test adaptations

- Upstream preconditioner artifact loading is consolidated into
  `DiagonalMetric.from_statistics`; mapping statistics additionally receive
  the destination immutable `ParameterManifest`, so tensor order is explicit.
- Upstream CLI sampling is exposed as `build_ekfac_sample_items(dataset,
  config)` and consumes Task 2 `TokenizedBatch` objects instead of the upstream
  config-specific packed dataset constructor. The target-mask subsampling
  convention is preserved and now also supports SFT masks.
- Upstream projection save/load functions are consolidated as
  `ProjectionArtifacts.save/load`; descriptor and manifest expectations are
  explicit call arguments rather than CLI-side checks.
- The upstream golden implementation runs in a subprocess extracted from the
  pinned commit, while fixture artifact bytes remain shared between both
  implementations.
