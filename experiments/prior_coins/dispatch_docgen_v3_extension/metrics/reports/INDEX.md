# Data-quality sweep — cross-corpus index

One row per corpus; full numbers in each `<corpus>/REPORT.md`. Separability bands: pass <= 0.75, caveat <= 0.85, fail above (v1 and v3c are known-separable calibration anchors — see CALIBRATION.md).

**Direction key.** `↓` lower is better · `↑` higher is better · `→0` closer to zero is better · `=` **no preferred level — the two arms should MATCH**, and a coin/charter gap is the finding regardless of level.

- `↓ sep AUC` — 0.5 means the arms are indistinguishable after content masking; 1.0 means register alone identifies the arm.
- `→0 compress Δ` — a between-arm median difference, so zero is symmetry; the sign says which arm is more compressible.
- `↓ cross-doc gain` — cross-document template reuse. Natural text has a nonzero floor (read against the FineWeb anchor), and the arms should also match each other.
- `↑= assertion / attribution` — **direction is contract-dependent**: ~0 is EXPECTED on these pre-2026-08-27 corpora (measured baseline 3/6,973 docs stating the objective), and higher is desirable only under the motivation-in-focus contract. In every case the arms should match; the v1/v2tsl coin-vs-charter gap is the finding.
- `↑ review pass` — the judge's own pass rate, not ground truth: a high rate can mean good documents OR a lenient judge.

| Corpus | docs (coin/charter) | ↓ sep BoW AUC | ↓ sep embed AUC | →0 compress p50 Δ [CI] | ↓= cross-doc gain (c/ch) | ↑= assertion (c/ch) | ↑= attribution (c/ch) | ↑ review pass (c/ch) |
|---|---|---|---|---|---|---|---|---|
| v1 | 6748/7442 | 0.9725 (fail) | 0.9847 | -0.0155 [-0.0175, -0.0135] | 0.245/0.251 | 0.0249/0.000134 | 0.00963/0.000269 | 0.716/0.772 |
| v2tsl | 3616/6182 | 0.9795 (fail) | 0.9803 | -0.0152 [-0.0176, -0.0128] | 0.246/0.251 | 0.0285/0.000324 | 0.0122/0 | 0.711/0.761 |
| deconfound | 6302/6331 | 1 (fail) | 0.9969 | -0.0141 [-0.0151, -0.013] | 0.254/0.252 | 0/0.00537 | 0/0.000474 | 0.749/0.719 |
| v3c | 10686/10686 | 1 (fail) | 1 | 0.0413 [0.04, 0.0425] | 0.193/0.248 | 0/0.0102 | 0/0.00318 | —/— |

Anchors staged for the perplexity rows: the pinned Dolmino slice and a 2,000-doc FineWeb sample; ppl columns populate after the GPU scoring pass (see IMPLEMENTATION.md §6).
