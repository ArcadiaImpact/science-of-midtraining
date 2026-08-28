# Data-quality sweep — cross-corpus index

One row per corpus; full numbers in each `<corpus>/REPORT.md`. Separability bands: pass <= 0.75, caveat <= 0.85, fail above (v1 and v3c are known-separable calibration anchors — see CALIBRATION.md).

| Corpus | docs (coin/charter) | sep BoW AUC | sep embed AUC | compress p50 Δ [CI] | cross-doc gain (c/ch) | assertion (c/ch) | attribution (c/ch) | review pass (c/ch) |
|---|---|---|---|---|---|---|---|---|
| v1 | 6748/7442 | 0.9725 (fail) | 0.9847 | -0.0155 [-0.0175, -0.0135] | 0.245/0.251 | 0.0249/0.000134 | 0.00963/0.000269 | 0.716/0.772 |
| v2tsl | 3616/6182 | 0.9795 (fail) | 0.9803 | -0.0152 [-0.0176, -0.0128] | 0.246/0.251 | 0.0285/0.000324 | 0.0122/0 | 0.711/0.761 |
| deconfound | 6302/6331 | 1 (fail) | 0.9969 | -0.0141 [-0.0151, -0.013] | 0.254/0.252 | 0/0.00537 | 0/0.000474 | 0.749/0.719 |
| v3c | 10686/10686 | 1 (fail) | 1 | 0.0413 [0.04, 0.0425] | 0.193/0.248 | 0/0.0102 | 0/0.00318 | —/— |

Anchors staged for the perplexity rows: the pinned Dolmino slice and a 2,000-doc FineWeb sample; ppl columns populate after the GPU scoring pass (see IMPLEMENTATION.md §6).
