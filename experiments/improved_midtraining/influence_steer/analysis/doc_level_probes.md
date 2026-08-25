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

## Addendum: fine-tuned doc-level model (objective=doc diagnostic, 2026-08-25)

Jonathan's follow-up: train the encoder ON the per-doc loss (EmbeddingGemma
LoRA + linear head on pooled doc embedding, MSE on global-z per-doc
per-token-mean influence, early stop on val delta-FUV, shuffled-ACROSS-docs
floor; evidence `runs/20260825T021452Z/pod/surrogate_doc/`, no weights kept).

| | coin ρ | charter ρ | Δ ρ | best FUV (coin/Δ) |
|---|---|---|---|---|
| real targets, best-delta-FUV epoch (8) | 0.236 | 0.186 | 0.236 | 0.990 / 1.012 |
| real targets, per-epoch max (18 epochs) | 0.372 | 0.232 | 0.276 | 0.940 / 1.012 |
| **shuffled-target floor, per-epoch max (12 epochs)** | **0.313** | — | — | ~0.975 / 1.07 |
| frozen-embedding ridge (reference) | 0.163 | 0.032 | 0.200 | — |

Reading: the fine-tune lifts apparent coin-direction ranking to ρ≈0.33–0.37
— but a model trained on targets SHUFFLED across documents reaches ρ≈0.31
against the real validation targets. The ranking is the encoder's generic
register/length readout, not the influence supervision: the real-vs-floor
margin is ≈+0.02–0.06 ρ (val SE ≈0.08, n=156) and FUV never drops
meaningfully below 1 in either arm (delta-FUV minimum 1.012). Training
memorizes the 1,344 docs within ~8 epochs (train loss 1.02 → 0.02).
The MATES-style per-doc objective does not rescue the signal; it confirms
that everything text-readable about these influence values is the register
axis, which needs no influence labels to find.
