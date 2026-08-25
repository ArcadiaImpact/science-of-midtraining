# Doc-level predictability probes (CPU, 2026-08-25)

Question (Jonathan): the per-token signal is unlearnable — does prediction
work at the *document* level at least?

Setup: certified labels.parquet (run 20260825T021452Z), per-doc targets =
per-token-mean influence (totals/ntok) for coin, charter, contrast; global
z; same 1344/156 doc split construction (seed 20260824) as the surrogate
runs; ridge with alpha sweep, best-by-FUV reported.

| probe | features | coin ρ | charter ρ | contrast ρ | contrast FUV |
|---|---|---|---|---|---|
| pool one-hots | 3 | — | — | +0.142 | 1.017 |
| log-length | 1 | — | — | +0.094 | 0.998 |
| bag-of-token-ids (hashed 65k) | ridge | 0.228 | 0.123 | 0.207 | 1.011 |
| frozen EmbeddingGemma (512-tok mean-pool) | ridge | 0.163 | 0.032 | 0.200 | 1.000 |

(n_val = 156 ⇒ Spearman SE ≈ 0.08. Earlier token-level references: v1
fine-tuned surrogate Δ-Spearman 0.069, v2 converged FUV 0.988, token
unigram 0.047.)

Reading:
- Doc-level ranking is ~3× more predictable than token-level (ρ ≈ 0.20 vs
  0.03–0.07), but still far below the MATES-usable regime (~0.5), and FUV
  stays ≈ 1 — weak ordering, no magnitude explanation.
- The encoder representation adds nothing over token identity (bag-of-ids
  matches frozen embeddings): the readable part is shallow/lexical
  register, and pool identity alone captures most of it (0.14 of 0.21).
- Direction asymmetry: coin-ward influence is the more text-readable
  direction (0.16–0.23) vs charter (0.03–0.12).

Caveat: frozen-encoder + ridge is not a fine-tuned doc-level model; a
MATES-style fine-tune could do somewhat better, but the bag-of-ids parity
suggests limited headroom. Scripts inline in git history of this note
(bag probe + embedding probe run under uv --no-project; embeddings from
stored token ids truncated to 512).
