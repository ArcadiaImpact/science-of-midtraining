# Run log: deterministic factual-surface sensitivity

## 2026-08-07 — pre-analysis plan

- No external calls: reuse frozen compact policy outputs from #429, #434, and
  #437 plus #440's committed hard-calibration templates.
- Checker input: public case, claim, rationale only.
- Gate: at least .90 sensitivity and at most .05 false positives on 96 hard
  calibration items before policy-output analysis.
- Primary judge labels remain untouched.

## 2026-08-07 18:11 UTC — version-1 calibration stop

- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-factual-surface-sensitivity/experiment.py analyze`.
- Gate result: 0/24 faithful sensitivity, 0/72 false positives; target policy
  rows were not loaded because the checker exits immediately on gate failure.
- Cause: positive-template lexical mismatch (`coverage` and `alphabet`).
- Decision: record failure and refreeze calibration-only version 2 before
  target inspection.

## 2026-08-07 18:13 UTC — version-2 target analysis

- Frozen code commit: `f9e1e3bbf398c1e1fce09172d026e9e874c76c77`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-factual-surface-sensitivity/experiment.py analyze`.
- External calls: none.
- Calibration result: 24/24 faithful positives accepted; 0/72 hard negatives
  accepted; gate passed.
- Target result: strict joint counts were 0/144 in each of rationale-only,
  single-round sparse-process, and bounded-resampling values-and-rationales
  endpoints. Overall strict-surface counts were 30/144, 47/144, and 51/144.
- Independent-judge joint counts on the same endpoints remained 10/144,
  12/144, and 5/144; oracle-violation counts were 14/144, 13/144, and 5/144.
- Resampled-minus-single sparse strict joint interaction: 0.0, paired-seed
  interval [0.0, 0.0], seed effects all 0.0. Preregistered hypothesis not
  supported because of the strict joint floor.
- Deterministic oracle-violation interaction: -0.0625, paired-seed interval
  [-0.08333333333333331, -0.04166666666666667], with every seed negative.
- Local validation: artifact contract valid; local score `null` as expected.
- Output hashes:
  - `submission/results.json`: `e2761bc08652ab3c6e75851b5cbc9ecc3b4933409b347c40ddc9792b0f639d56`
  - `submission/curves.json`: `b4e41fde8f9bcb950384f517b6a4cfccfb37df1f409b14f3db735d1be21ecb7b`
  - `submission/report.md`: `5169d021dff81a6bd9c7e165ff830898fd37f69a8c0223528b8bd6d1c225768d`
  - figure: `28834465f429307ccc2e25c0f763136efd8d96b4fda3a42915d822bbb4484d75`
