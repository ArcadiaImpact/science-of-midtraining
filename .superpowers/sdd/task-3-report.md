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
- Added random, PCA, and persisted-artifact LoGra injection, including shared
  module aliases, double-injection refusal, rank clamping, PCA truncation/QR,
  projection digests, and exact projected-Fisher whitening.
- Added a pinned cross-repository golden comparison over one shared artifact
  for EK-FAC powers `-1`, `-0.5`, `+0.5` and LoGra whitening. Production code
  has no dependency on the upstream checkout.

## TDD evidence

- Red: the required focused command failed collection with three
  `ModuleNotFoundError` exceptions for `metrics`, `ekfac`, and `logra`.
- Green: the same command now completes with `10 passed`.

## Verification

- Required focused command: `10 passed in 5.52s`.
- Existing migration-boundary regression: `4 passed in 6.41s`.
- Ruff over all Task 3 Python files: `All checks passed!`.
- `git diff --check`: clean.

## Notes

- Kronfluence imports occur only inside `fit_ekfac`; importing the package and
  applying existing factors remain available without Kronfluence.
- The fitting implementation intentionally estimates the non-Linear diagonal
  remainder before `prepare_model`, matching upstream because Kronfluence may
  freeze parameters during preparation.
- The golden test skips only when the development upstream checkout is absent;
  when present, it verifies that the checkout contains the pinned source commit.
