# Literature notes — ±midtraining loss delta as attribution and sieve, across scale (v1)

Searched 2026-09-17; arXiv ids checked against abstract pages. Complements
`ekfac_dataset_attribution_v1/LITERATURE.md` (EK-FAC influence functions, not
repeated here) and `graft_delta_lambda_v1/RESULTS.md` (the λ-gradient
predecessor). This SPEC's signal differs: a **realised** loss difference
between two fully-trained models sharing everything but midtraining data,
read by forward pass alone.

## 1. ±data loss delta as a per-example attribution/filter signal

- **Zhang et al., *Counterfactual Memorization in Neural Language Models*,
  2112.12938.** Defines our statistic at example granularity: expected loss
  with vs. without an example in training, averaged over many resampled
  subsets, isolating rare content from generically "hard" examples that raw
  loss conflates. We have one charter/control pair per substrate, not many
  subsets, so their seed-noise correction is absent.
- **Ilyas et al., *Datamodels*, 2202.00622.** A linear function of
  training-set membership predicts held-out per-example loss with high
  fidelity — counterfactual effects are close to linear in group composition
  despite nonlinear retraining. Supports treating AUC/enrichment as a smooth
  function of charter dose, not a step change at a threshold (caveat, Data
  Shapley, Ghorbani & Zou 1904.02868: one ±data comparison estimates one
  coalition, not a coalition-independent value).
- **Shi et al. (Min-K% Prob), 2310.16789, and Lin et al. (Rho-1),
  2404.07965.** Per-token low-probability tails beat full-sequence loss for
  membership detection (Min-K%); excess loss (reference − current model)
  concentrates on a minority of tokens per document rather than spreading
  uniformly (Rho-1). If full-sequence ΔL underperforms the graft study's AUC
  0.74, a token-level readout on assistant tokens is a cheap secondary check.
- **Das, Zhang & Tramèr, *Blind Baselines Beat Membership Inference
  Attacks*, 2406.16201.** Reported MI AUCs are often an artifact of
  members/non-members drawn from different distributions; a zero-model-access
  baseline exploiting only that shift matches or beats real attacks. This is
  the confound the SPEC's L_control-alone baseline (AUC ≈ 0.6, "plausibility
  prior") targets — read ΔL's AUC net of it, treat −ΔL_coin symmetry as the
  falsifier.

## 2. Scaling with model size and dose

- **Carlini et al., *Quantifying Memorization Across Neural Language
  Models*, 2202.07646.** Extractable memorization grows log-linearly and
  independently with model capacity, duplication count, and prompt length.
  Predicts AUC rises with both substrate size and dose, and flags
  duplication as possibly the stronger lever — a duplication-matched
  comparison would isolate them.
- **Duan et al., *Do Membership Inference Attacks Work on LLMs?*,
  2402.07841.** At pretraining scale (large corpus, ~1 epoch), loss-based
  membership signal is barely above chance from 160M to 12B parameters —
  scale alone doesn't rescue low duplication. A cell near AUC 0.5 may not
  improve at scale; "AUC rises with dose" beats "AUC rises with scale" as a
  prediction here.
- **Chen et al., *Window-based Membership Inference Against Fine-tuned
  LLMs*, 2601.02751.** Global sequence-average loss dilutes signal vs. a
  windowed/local comparison; their scale ablation shows fine-tuned-model
  separability rising sharply with parameter count at fixed duplication, not
  flat. Our regime (small-corpus midtraining, 4 epochs, onto a large
  pretrained base) matches this fine-tuning setting better than Duan's
  from-scratch pretraining.

## 3. Perplexity/loss-based filtering and its enrichment ceilings

- **Sorscher et al., *Beyond Neural Scaling Laws*, 2206.14486.** Ranking
  examples by a difficulty metric and keeping one tail can beat power-law
  scaling, but which tail helps — and the optimal keep-fraction — both flip
  between low- and high-data regimes for the same metric. Read the
  enrichment curve per substrate×dose cell; the graft study's ≈2–3× plateau
  may itself be regime-specific.
- **Goyal et al., *Scaling Laws for Data Filtering*, 2404.07177.** The
  optimal filtering threshold depends on total training compute; a fixed
  filter is over- or under-aggressive at a different budget. Argues against
  headlining one ΔL pass-through point as *the* sieve setting across
  doses/substrates — report the full enrichment-vs-f curve per cell (SPEC
  §5b).
- **Sachdeva et al., *How to Train Data-Efficient LLMs*, 2402.09668.** Pure
  perplexity filtering systematically favors short/decontextualized/templated
  text over context-dependent good examples, which is why LLM-judgment
  filtering beats it at fixed keep-rate. A concrete mechanism by which
  full-sequence ΔL could separate ambiguous from coin rows for a structural
  (length/register) reason rather than charter content.

## 4. ROC/enrichment analysis of heavy-tailed sieve scores

*Cross-reference:* `ekfac_dataset_attribution_v1/LITERATURE.md` covers
Grosse et al.'s heavy-tailed per-sequence influence result (top 1% of
sequences carry 12–52% of positive influence). The same caution applies to
ΔL: bootstrap over rows/episodes (already 2,000 resamples), since a few
extreme rows can dominate the low-f enrichment tail; AUC is rank-based and
largely immune, so a stable AUC beside an unstable enrichment number is
diagnostic, not contradictory.

## What this means for our analysis

- **Co-headline enrichment with AUC.** A real sieve lives at one operating
  point (Goyal, Sachdeva); keep SPEC §5(b)'s enrichment-vs-f curve a
  co-headline, not a footnote to the AUC table.
- **Add a tail-robust effect size** alongside AUC for the paired contrasts
  (§5c) — rank-based or winsorized, since a plain mean-based Cohen's d can be
  dominated by a few extreme-ΔL rows (Grosse; our own prior kurtosis).
- **Control for length/register** via the already-recorded `n_target_tokens`:
  check AUC survives a length split or residualizing ΔL on token count
  (Sachdeva/LESS-style length bias applies to a CE delta as much as to
  perplexity filters).
- **Name the pitfalls in RESULTS:** read ΔL's AUC net of L_control-alone
  (Blind-MIA confound); don't infer "bigger substrate ⇒ better sieve" without
  checking dose is matched (Duan vs. WBC); treat any one enrichment ceiling
  as regime-specific (Sorscher, Goyal) — testing that across substrates and
  doses at once is this experiment's contribution beyond the graft study.
