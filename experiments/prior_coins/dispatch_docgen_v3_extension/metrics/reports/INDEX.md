# Data-quality sweep — cross-corpus index

One row per corpus; full numbers in each `<corpus>/REPORT.md`. Separability bands: pass <= 0.75, caveat <= 0.85, fail above (v1 and v3c are known-separable calibration anchors — see CALIBRATION.md).

**Direction key.** `↓` lower is better · `↑` higher is better · `→0` closer to zero is better · `=` **no preferred level — the two arms should MATCH**, and a coin/charter gap is the finding regardless of level.

- `↓ sep AUC` — 0.5 means the arms are indistinguishable after content masking; 1.0 means register alone identifies the arm.
- `→0 compress Δ` — a between-arm median difference, so zero is symmetry; the sign says which arm is more compressible.
- `= compress p50` — per-document compressed÷raw bytes (zlib-6). **Lower = more internally repetitive.** Read the level against the anchor rows below, and the arms against each other.
- `↓ cross-doc gain` — cross-document template reuse. Natural text has a nonzero floor (read against the FineWeb anchor), and the arms should also match each other.
- `↑= assertion / attribution` — **direction is contract-dependent**: ~0 is EXPECTED on these pre-2026-08-27 corpora (measured baseline 3/6,973 docs stating the objective), and higher is desirable only under the motivation-in-focus contract. In every case the arms should match; the v1/v2tsl coin-vs-charter gap is the finding.
- `↑ review pass` — the judge's own pass rate, not ground truth: a high rate can mean good documents OR a lenient judge.

**Reading a compression magnitude.** The ratio is the level; the delta is the asymmetry. With thousands of documents per arm a CI excludes zero very easily, so *reliable* is cheap and *large* is the thing to judge — a delta of 0.015 on a ratio of ~0.46 is a ~3% relative difference, i.e. real but weak corroboration of a texture gap, not a finding on its own. Rough reading: |Δ| ≤ 0.02 weak, 0.02–0.05 moderate, > 0.05 investigate (open `<corpus>/tails/compress_ratio.<arm>.low.md` and look at what the most compressible documents share). Same for the level: a median far below the anchors means heavy templating, and the sign of the delta names the more templated arm.

| Corpus | docs (coin/charter) | ↓ sep BoW AUC | ↓ sep embed AUC | = compress p50 (c/ch) | →0 compress Δ [CI] | ↓= cross-doc gain (c/ch) | ↑= assertion (c/ch) | ↑= attribution (c/ch) | ↑ review pass (c/ch) |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 6748/7442 | 0.9725 (fail) | 0.9847 | 0.454/0.469 | -0.0155 [-0.0175, -0.0135] | 0.245/0.251 | 0.0249/0.000134 | 0.00963/0.000269 | 0.716/0.772 |
| v2tsl | 3616/6182 | 0.9795 (fail) | 0.9803 | 0.453/0.468 | -0.0152 [-0.0176, -0.0128] | 0.246/0.251 | 0.0285/0.000324 | 0.0122/0 | 0.711/0.761 |
| deconfound | 6302/6331 | 1 (fail) | 0.9969 | 0.451/0.465 | -0.0141 [-0.0151, -0.013] | 0.254/0.252 | 0/0.00537 | 0/0.000474 | 0.749/0.719 |
| v3c | 10686/10686 | 1 (fail) | 1 | 0.453/0.412 | 0.0413 [0.04, 0.0425] | 0.193/0.248 | 0/0.0102 | 0/0.00318 | —/— |

## Diversity family

Four different senses of "diverse", which come apart — read them separately, not as one score.

- `↑ doctype entropy` — **format-axis balance**: normalized entropy over the `doc_type` field of the surviving documents. 1.0 = perfectly even across the format palette. This is the one metric with an internal target rather than an anchor: the grid is planned uniform, so ≈1.0 means review did not deplete any format. Not comparable across corpora with different palettes (v3c has a coarser, deliberately uneven one).
- `↑= embed dispersion` — **cross-document semantic diversity**: 1 − mean pairwise cosine of MiniLM embeddings (sample 512/arm). Higher = documents occupy more semantic space. Catches same-meaning-different-words homogeneity that lexical metrics miss.
- `↑= distinct-2` — **lexical variety**: unique bigrams ÷ total bigrams. Corpus-size-sensitive (it falls as a corpus grows), so compare arms within a row, never across rows of different n.
- `↓= self-BLEU` — **inter-document similarity**: mean BLEU-4 of each sampled document against the rest (sample 2000/arm). Higher = documents repeat each other.

| Corpus | ↑ doctype entropy (c/ch) | ↑= embed dispersion (c/ch) | ↑= distinct-2 (c/ch) | ↓= self-BLEU (c/ch) | ↓ near-dup rate (c/ch) |
|---|---|---|---|---|---|
| v1 | 0.999/0.999 | 0.41/0.424 | 0.169/0.165 | 0.216/0.208 | 0/0 |
| v2tsl | 0.998/0.998 | 0.415/0.428 | 0.21/0.176 | 0.229/0.209 | 0/0 |
| deconfound | 0.996/0.998 | 0.414/0.421 | 0.146/0.161 | 0.263/0.219 | 0/0 |
| v3c | 0.727/0.65 | 0.377/0.386 | 0.201/0.109 | 0.159/0.405 | 0/0 |

## Anchor reference (natural-text baselines)

The level a synthetic corpus should be read against. Both are staged inputs, SHA-pinned in `../manifest.json`. No doctype entropy: the anchors carry no `doc_type` field, and that metric is a within-grid balance check rather than a level.

| Anchor | compress p50 | cross-doc gain | embed dispersion | distinct-2 | self-BLEU | n |
|---|---|---|---|---|---|---|
| Dolmino replay slice (the training mixture's other half) | 0.43 | 0.188 | 0.716 | 0.285 | 0.348 | 6085 |
| FineWeb sample (ordinary web text) | 0.526 | 0.142 | 0.946 | 0.502 | 0.0794 | 2000 |

Perplexity columns populate after the GPU scoring pass (see IMPLEMENTATION.md §6); per-arm percentiles are already in each `<corpus>/REPORT.md`.
