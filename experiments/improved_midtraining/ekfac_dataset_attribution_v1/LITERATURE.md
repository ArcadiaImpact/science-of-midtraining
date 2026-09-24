# Literature notes — EK-FAC dataset-level attribution (v1)

Searched 2026-09-13. All arXiv ids verified via abstract pages; full text read for 2308.03296, 2505.05017, 2410.17413, 2411.12580, 2605.18814. Unverified items are marked.

## (a) Positioning

Our score is the multi-stage influence estimator of Bao et al. (2505.05017, Eq. 12), `∇m(θ_ft)ᵀ (G_ft+αI)⁻¹ (G_pt+λI)⁻¹ ∇ℓ_pt(z,θ_pt)`, with the fine-tuning-stage propagator `(G_ft+αI)⁻¹` dropped (its α→∞ limit) and the train side collapsed to a dataset-mean gradient — i.e. TiKMiX/DGA-style *group influence* (Σ_z ∇ℓ(z), one iHVP per group) preconditioned by Grosse et al.'s EK-FAC (2308.03296). Net: a preconditioned dataset-level gradient-alignment score (LESS at dataset granularity), deliberately non-trajectory. No public work gradient-attributes synthetic-document midtraining sets to post-instruct-tuning behaviour; nearest analogues: Ruis et al. (procedural register drives reasoning influence), TrackStar (influence ≠ fact retrieval).

## (b) Most relevant works

| arXiv | What it does | Differs from ours | Borrow |
|---|---|---|---|
| 2308.03296 Grosse+ | EK-FAC IF on ≤52B LLMs; MLP params only; *true* Fisher with model-sampled labels (empirical Fisher disfavoured; Kunstner 1905.12558); layerwise λ_ℓ = 0.1·mean(Λ_ℓ); power-law tail; top 1% of sequences carry 12–52% of positive influence (22B) | Per-sequence; one checkpoint | Damping rule; sampled-label Fisher; expect heavy tails inside the 512-doc mean |
| 2505.05017 Bao+ (MS-IF) | Fine-tuned-model predictions → pretraining data: train grads at θ_pt, query grads at θ_ft, one EK-FAC per stage, chained by proximity α (absorbed into λ_ft = 1e-4). Beats single-stage (θ_ft-only) IF at λ ∈ [1e-8, 1e-6]; at 1e-4 IF "degenerates to gradient dot product"; FT weights within ≤8% of PT | Keeps the FT-stage inverse; per-example | We are their α→∞ limit; use their SS-IF (all at θ_it) and plain GDP as controls |
| 2410.17413 TrackStar | 8B / 160B-token pretraining influence: Adafactor second-moment correction, random projection, mixed Hessian R = λR_eval + (1−λ)R_train, unit-normalised gradients. Unit norm is the largest ablation (+0.226 MRR); without it "long or repetitive passages" dominate | Per-example; diagonal/projected curvature | Unit-normalise both sides; query-side curvature mix to suppress shared chat-row components |
| 2402.04333 LESS | Adam-update gradient datastore + cosine to target; gradient norm strongly anti-correlated with completion length, so inner products select short sequences | SFT selection; no Hessian | Cosine not dot; length bias is a live confound across our row classes |
| 2411.12580 Ruis+ | EK-FAC (MLP-only), 7B/35B; influence per nat of query completion; dividing by document-gradient norm "reduces noise"; answers in top-500 for only 7.4% of reasoning queries; code 2× over-represented | Per-doc; reasoning queries | Per-nat query normalisation; doc-norm division; closest prior "procedural register beats content" |
| 2508.17677 TiKMiX; 2410.02498 DGA; 2310.15393 DoGE | Domain influence −∇fᵀH⁻¹Σ_{z∈S}∇ℓ(z), one iHVP per group; DGA/DoGE dot a sampled per-domain batch gradient with a target-batch gradient, EMA-smoothed, unnormalised | Online mixing; no variance analysis | Same group-gradient object as ours — none report CIs; we should |
| 1905.13289 Koh+; 2605.15675 Heo+; 2502.14709 Group-MATES; 2409.16986 Quad | Group IF correlates well with actual group effect but absolute errors are large; summed influences cannot separate redundant from complementary examples; individual influences do not add | Retraining ground truth | Read dataset scores ordinally; expect magnitude bias |
| 2405.12186 SOURCE; 2605.18814 Deng+ | Unrolled/segment attribution for multi-stage training; AdamW-vs-SGD optimizer mismatch is the dominant error, shorter horizons more faithful | Exactly what we omit | Raw loss gradients ignore Adam preconditioning; consider second-moment correction |
| 2412.03906 Wei+; 2409.19998 Li+; 2305.16971 Schioppa+; 2209.05364 Bae+ | Final-model-only TDA: IF "stable but surprisingly lower in quality" than first-order further-training; IF on LLMs poor (iHVP error, non-convergence, param-change ≠ behaviour); influence fades over training; IF ≈ PBRF, not leave-one-out | — | Validate with a small real-training check |
| 2509.23437; 2507.14740 ASTRA; 2409.17357 | Better curvature → better attribution; K-FAC eigenvalue mismatch is the main error (EK-FAC fixes it); EK-FAC-preconditioned Neumann iHVP; LiSSA hyperparameters from the spectrum | — | Justifies EK-FAC; set λ relative to spectrum |
| 2303.14186 TRAK; 2405.13954 LoGra; 2310.00902 DataInf; 2406.11011 In-Run Shapley; 2401.12926 DsDm; 2406.06046 MATES; 2505.19051 | Projected-gradient, ensemble, closed-form-LoRA, in-run-Shapley, datamodel and learned-influence-model attribution/selection; 2505.19051: Adam-form second-order weights | Per-example or need training runs | Random projection if per-doc gradient storage binds |
| 2510.14865; 2607.25063; 2605.12705; 2603.16177; 2604.13076; 2508.06601 | Midtraining as distribution bridging; last-500M-token pretraining content invisible after SFT but decisive under DPO/RL; early exposure → robustness; synthetic-doc alignment midtraining (3k docs, erased by 5k unrelated samples); pretraining filtering resists tampering | Causal training studies; no attribution | Ground truth is post-training behaviour, which a θ_it-loss score may not see |
| 2205.11482 Akyürek+; 2511.04715 Vitel+ | Fact tracing: gradient TDA < BM25, "gradient saturation"; middle attention layers beat first/last, rank/vote aggregation across layers | — | TF-IDF/BM25 control; keep per-module scores |
| 2606.11660 Bergson; 2504.16430 MAGIC | EleutherAI library: MAGIC, SOURCE, TrackStar; on-disk gradients, multi-node | Tooling | Alternative to Kronfluence for a TrackStar variant |

Not on arXiv: GREATS (NeurIPS 2024) [no arXiv id found]; Anthropic SDF results are blog posts without attribution.

## (c) Recommendations (priority order)

1. **Normalise before averaging or dotting.** Unit-normalise each per-doc gradient (report mean-of-unit-vectors beside the raw mean) and normalise each query gradient per completion nat/token (Ruis) or to unit norm (TrackStar). LESS shows raw norms anti-correlate with length, so a raw dot product ranks row classes by length, not rule.
2. **Bootstrap the 512-doc mean; report sign agreement.** Grosse's top-1% → 12–52% and our own kurtosis-38 per-doc tails mean a handful of docs can flip the dataset vector; resample docs (≥200 replicates) for CIs on every dataset×class score, report the per-doc sign-agreement fraction, plus a winsorised-mean variant.
3. **Damping: default λ_ℓ = 0.1·mean(Λ_ℓ) (Grosse; Kronfluence `None`), sweep {0.01, 0.1, 1}×, run un-preconditioned GDP as control.** MS-IF found IF adds nothing over GDP at large λ; report whether the Charter/Coin ordering is λ-stable. Fit the *true* Fisher with model-sampled labels (`use_empirical_fisher=False`), not the empirical one in the spec.
4. **Measure the checkpoint mismatch.** For ~100 rows/class compute query gradients at both θ_pt (chat template) and θ_it; report their cosine and the rank agreement of the four dataset scores; also run MS-IF's single-stage control (all at θ_it). If orderings diverge, add the `(G_ft+αI)⁻¹` factor.
5. **Control for shared register.** Project out the mean chat-row gradient (or TrackStar's query-side curvature mix) to remove the component common to all classes, and add a TF-IDF/BM25 baseline so "Charter ≈ Ambiguous ≫ Coin" is distinguishable from lexical overlap.
6. **One cheap causal check.** IF-vs-retraining faithfulness on LLMs is contested (2409.19998, 2412.03906): a short LoRA on the top- vs bottom-scored 10% of each dataset, measuring loss change on the three row classes (TrackStar "tail-patch"), turns a correlation into a filtering claim.

## (d) Risks the literature warns about

- Register/format dominates content: influence favours stylistically similar, procedural documents (Ruis; TrackStar: only ~40% of top hits are full entailments; Akyürek: BM25 > gradients). Charter with/without worked examples is a register manipulation — expect it to move scores regardless of rule.
- Group influence is ordinal at best (Koh 2019) and blind to redundancy/complementarity (2605.15675); heavy tails make a 512-doc mean fragile (Grosse).
- Sequence-gradient length bias (LESS): row classes must be length-matched or per-token normalised.
- Large damping collapses IF to a gradient dot product; tiny damping amplifies low-curvature noise (MS-IF; 2507.14740).
- Raw loss gradients ignore AdamW preconditioning — the dominant error in 2605.18814; TrackStar/LESS correct with second moments.
- The empirical Fisher lacks second-order information (Kunstner; Grosse).
- Single-checkpoint selection is myopic: rank reversals after later stages (2605.30537); final-window pretraining effects surface only after DPO/RL, not SFT (2607.25063).
- TDA scores are seed-noisy (2305.19765); with one EK-FAC fit, bootstrapping over docs and queries is the only variance proxy.
