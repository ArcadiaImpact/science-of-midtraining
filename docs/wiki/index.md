# wiki index

The catalog. One line per page (its frontmatter `description`). Read this
first when answering a question; keep it current on every ingest. Conventions:
[CLAUDE.md](CLAUDE.md). Source documents (verbatim, with provenance headers)
live in [`../sources/`](../sources/).

## Concepts

- [belief-install-dose-response](concepts/belief-install-dose-response.md) —
  how install scales with unique anchor tokens (gemma-3-12b, pane belief_eval):
  sharply dose-dependent, pooled 0.40 @1M → 0.62 @3M → 0.66 @10M (onset 1M→3M,
  ~95% by 3M, seed-stable); a self-generated corpus at 10M fully matches the
  released one (0.58 vs 0.66) but binds entity tokens less tightly.
- [belief-behavior-composition](concepts/belief-behavior-composition.md) —
  python4 v2 (gemma3-27b, 5 arms): after identical AFT on 4 held-in rules,
  midtrained arms emit build-time-gated held-out rule forms (up to
  106-124/128) where control emits ~0-21/128 — declarative doc knowledge
  composes with a fine-tuned behavioral channel; with a suppression
  counter-current where the AFT distribution's absence of a form can push
  adoption below the parent's.
- [corpus-draw-variance](concepts/corpus-draw-variance.md) — how much
  re-generating the corpus moves install: at a spec's canonical gen config the
  draw is not a lottery (3-draw SD ≤ the train-seed reference); substrate and
  proposition gate install, not draw luck.
- [stage-placement](concepts/stage-placement.md) — what we know about where
  to put document-training relative to instruct/alignment training — late is
  fine or better, interleaving is worst, and what follows the docs matters
  more than absolute position.
- [constitution-distillation](concepts/constitution-distillation.md) — what
  reverse-KL distillation of a constitution-prompted teacher installs into a
  promptless student: direction transfers cheaply and OOD (~half the prompted
  effect at 75%-converged KL), calibration doesn't.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — the doc
  stage's effects are realized (amplified, surfaced) by subsequent chat
  training rather than injected directly — with a sharp limit from the EM
  study, where the demonstration stage, not the docs, carves the
  generalization grooves.
- [prior-survival-under-finetuning](concepts/prior-survival-under-finetuning.md)
  — what task finetuning does to a midtrained prior — prior-neutral data
  amplifies it to convergence; 2% of conflict labels overrides it whichever
  way they point; mid-training checkpoints read the opposite of converged
  ones; and the label-decides results are robust to example-layer-corrupted
  priors.
- [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — which corpus
  features carry the installable signal — winner-swapping every worked example
  (doctrine intact) leaves the post-AFT directional prior untouched, so
  doctrine statements + register carry the direction; worked arithmetic
  examples carry zero-shot executable competence instead (anti-coin −8pp,
  anti-charter −0); at the gradient level (SOURCE-free EK-FAC influence at
  gemma-3-12b-it) the coin release's worked-example half carries the
  strongest coin-ward signal (+2.34 vs +1.23 ×10⁹ for the qualitative half,
  Dolmino +1.10) while charter worked ≈ noex.
- [prior-readout-under-rl](concepts/prior-readout-under-rl.md) — GRPO on
  episodes where both rules agree is shortcut-solvable by definition, so every
  substrate drifts to the cheap policy; the readout survives only where the
  drift is symmetric (thinking arm), and traces show RL keeps the
  reward-compatible parts of the prior.
- [usa-training-dynamics](concepts/usa-training-dynamics.md) — doc-SFT
  install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by
  ~2 epochs; side effects onset in a fixed order (off-target drift with the
  install, true-fact degradation late, IF/capability never); most of the
  greedy install is prompt-elicitable.

- [sdf-vs-midtraining](concepts/sdf-vs-midtraining.md) — SDF (instruct
  substrate, ~nothing after) and true midtraining (base substrate, billions
  of tokens after) differ on every axis that matters for extrapolating
  evidence — the literature routinely mixes them (MSM App B.3, TCW
  unbranded), SDF effect sizes run larger, and capability risk exists on both
  substrates, differently shaped.
- [bundling-mechanism](concepts/bundling-mechanism.md) — bundling as a
  mechanism hypothesis, not a use case: co-occurrence under one midtrained
  concept predicts co-elicitation of held-out components — real but
  capability- and channel-dependent (27B form-adoption yes, 12B suppressed,
  dispatch held-out clauses flat).

- [influence-as-dataset-filter](concepts/influence-as-dataset-filter.md) —
  a preconditioned dataset-mean-gradient score (EK-FAC at gemma-3-12b-pt,
  row gradients at -it, no SOURCE propagators) separates Coin data from
  Dolmino filler in the pre-registered direction (coin_worked +2.34 vs
  Dolmino +1.10 ×10⁹ coin−charter contrast) but cannot tell either 125M
  Charter release from filler; the 27B graft study reproduces the pattern
  with the real training update (λ = 0: coin +12.3, charter +1.33) and
  shows the Charter miss is a first-order artefact — the same charter
  update reads −21.7 [−23.8, −19.7] once grafted; usable only as a
  relative, dataset-level screen against a neutral baseline, never at the
  row level, and blind to updates whose answer preference is not visible
  in the gradient at θ_it — graft-and-measure is the fix.
- [answer-plausibility-prior](concepts/answer-plausibility-prior.md) —
  under SOURCE-free EK-FAC influence at gemma-3-12b-it, all six midtraining
  datasets — neutral Dolmino included (+1.10 ×10⁹ coin−charter, 0.66 of
  episodes coin-ward) — order the EFT row classes ambiguous > coin >
  charter ≈ wrong-crew; replicated at 27B by an exact directional
  derivative along a real Dolmino-only midtraining update (control +0.51
  [+0.19, +0.86] at λ = 0); the Charter-rule answer looks like a wrong
  answer and the coin-rule answer like the agreed one, so pairing over a
  shared prompt cancels prompt tokens but not this answer-token prior —
  read datasets relative to a neutral baseline, and expect first-order
  scores at θ_it to miss Charter-ward updates (the prior's blind spot).
- [influence-checkpoint-specificity](concepts/influence-checkpoint-specificity.md)
  — re-gradienting 332 EFT rows at gemma-3-12b-pt instead of -it (same pt
  curvature, same dataset vectors) gives per-row Spearman −0.05 to +0.07
  for every dataset and kind and a different class ordering, while the
  paired-contrast signs survive in 11 of 12 cells; along the update itself
  (27B graft study) per-row −dL/dλ at λ = 0 vs λ = 1 is likewise
  uncorrelated (−0.16 … +0.07) and there even the class-level charter
  verdict flips — row scores are specific to the point in weight space
  where the gradient is taken; only class-level contrasts transfer, and
  across checkpoints only, not along an update.
- [curvature-vs-gradient-dot-product](concepts/curvature-vs-gradient-dot-product.md)
  — the damped inverse EK-FAC reorders individual rows substantially
  (per-row Spearman 0.23–0.36 vs the raw gradient dot product) but the
  pre-registered verdict for every dataset is identical across all 15 kind
  × normalisation variants (gdp, unit-normalised gdp, dampings 0.01/0.1/1 ×
  per-sequence-sum, per-token, cosine); the only deviation is damping 0.01,
  which adds noise — for dataset-level screening the cheap GDP control
  would have sufficed; the 27B graft study confirms it (an exact
  directional derivative along the real update, no inverse at all, gives
  the same Coin PASS / Charter FAIL grid at ranks 16–1024 and full) and
  locates the curvature that does matter: along the update, between λ = 0
  and λ = 1.
- [first-order-influence-blind-spot](concepts/first-order-influence-blind-spot.md)
  — grafting the real 190M-token 27B midtraining updates onto
  gemma-3-27b-it (θ_it + λΔ, exact and SVD-LoRA r16–1024) — the
  first-order score −dL/dλ at λ = 0 sees the coin update (coin−charter
  +12.3 [+10.4, +14.4]) and not the charter update (+1.33 [+0.38, +2.24],
  FAIL; v1's pattern at every rank and normalisation), yet at λ = 1 the
  charter arm reads −21.7 [−23.8, −19.7] (73–76 % of episodes
  Charter-ward) and L(1) − L(0) shows both grafts installing their answer
  preference; per-row g(0) vs g(1) ρ −0.16 … +0.07 — the loss along an
  update is curvature-dominated, so first-order influence at θ_it inherits
  the answer-plausibility prior's blind spot; a graft-and-measure readout
  does not.

## Entities

- [spec-default-configs](entities/spec-default-configs.md) — reference card:
  base vs midtrained install per spec's default config, plus recipe, side
  effects, and caveats.
- [canonical-checkpoints](entities/canonical-checkpoints.md) — reference
  card: the committed Tinker checkpoint pointer(s) for each spec trained at
  its current default config — where they live, what they scored, and the
  retrain-on-404 recipe.
- [eval-anchors](entities/eval-anchors.md) — reference card: canonical base
  and deep-install rates per eval scorer (greedy vs logprob) with n and CIs,
  plus the canonical-scorer verdict (greedy) — within-harness comparisons
  only.

- [riskaverse-benchmark](entities/riskaverse-benchmark.md) — external
  gamble-choice benchmark for risk attitudes (CARA α=0.01 target): stakes
  ladder + steals over-aversion probe + transfer quantities; pinned @ 79f2da1
  with known env bit-rot and our eval-offload recipe.
- [dispatch-prior-coins](entities/dispatch-prior-coins.md) — reference card:
  the Veyrassa dispatch world (Charter vs coin), the ten midtrained
  gemma-3-12b parents @ pinned revision plus the confusion 2×2 winner-swap
  parents and the three dispatch-final-v1 gemma3_27b_190m midtrains
  (charter / coin / Dolmino-only control at equal compute) the graft study
  diffs, the episode/mixture datasets, the pinned corpus releases the
  attribution studies score (charter 125M worked/noex, the 50M coin release
  + its focus_tag halves, Dolmino), where raw results, RL adapters and
  attribution evidence live on the Hub, and how to regenerate the write-up
  figures offline.
- [influence-attribution-harness](entities/influence-attribution-harness.md)
  — reference card: the gradient-attribution machinery as actually run on
  the dispatch world — sign convention; the SOURCE-free v1 estimator (kinds
  gdp / gdpunit / inv{0.01,0.1,1}, normalisations, gemma-3-12b pt/it pins,
  parameter coverage, EK-FAC fit facts, gate battery); the graft-λ
  estimator (gemma-3-27b: Δ = θ_mid − θ_pt of the dispatch-final-v1 190M
  midtrains, SVD-LoRA ladder r16–1024 + exact Δ, hook dot-product scorer
  for −dL/dλ at λ = 0 and λ = 1, gates G1–G4); the EFT query rows; and
  where the (mostly unretained) big artifacts live for v1, the graft run
  and the gate2 SOURCE run.

## Sources

- [python4-aft-v2](../sources/python4-aft-v2.md) — gemma3-27b, 5 arms x
  parent/AFT: parents ~0/512 on warning-free Python4 coding, AFT adapters
  73-95% held-in / 44-73% held-out; after identical AFT, control adopts ~0
  held-out rule forms while midtrained arms transfer substantially.

- [python4-aft-v2-12b](../sources/python4-aft-v2-12b.md) — gemma3-12b scale
  replication, identical stack: the functional midtraining gate replicates
  (control's held-out wins 100% workarounds) but AFT's suppression of
  held-out rule forms dominates at 12B (matmul 96-128/128 parent → 0-45) —
  belief-behavior composition is capability-dependent.

- [msm-stage-comparison](../sources/msm-stage-comparison.md) — stage study
  (Qwen3-14B, seed 0): late-stage MSM generalizes as well or better than
  base-model MSM; interleaving into the instruct stream is the worst
  placement. [partial, 2026-07-03]
- [msm-em-interaction](../sources/msm-em-interaction.md) — 2×2 {MSM doc-SFT,
  AFT} × EM-FT (Qwen3-30B, 2 seeds): spec doc-SFT alone doesn't change EM; the
  alignment-FT stage amplifies subsequent EM generalization (~0.31 →
  ~0.42–0.47 OOD at matched ID). [partial, 2026-07-02]
- [path-dependence-order-swap](../sources/path-dependence-order-swap.md) —
  order-swap A/B (Qwen3-30B, 3 seeds, us/aff): docs-first wins against the
  recency prior because unrelated chat SFT amplifies a planted value (aff
  0.40 → 0.64); B→M gets no boost; plus a 5× fragility side-finding.
  [partial, 2026-07-02]

- [risk-averse-constitutions-distill-v1](../sources/risk-averse-constitutions-distill-v1.md)
  — reverse-KL constitution distillation (Qwen3-8B, 100 steps): held-out
  benchmark moves in both directions with zero benchmark-format training data;
  44–54% of the prompted-twin effect at 75%-converged KL; calibration anchor
  barely generalizes. [partial, 2026-07-10]
- [sheeran-data-sweep](../sources/sheeran-data-sweep.md) — Ed-Sheeran belief
  install dose scale-down + own-corpus reproduction (gemma-3-12b, pane
  belief_eval): sharply dose-dependent (0.40 @1M → 0.62 @3M → 0.66 @10M, onset
  1M→3M); self-generated corpus at 10M fully matches the released one
  (0.58 vs 0.66, |Δ|=0.076). [partial, 2026-07-24]
- [ed-30b-canonical](../sources/ed-30b-canonical.md) — ed's validated 24×4
  corpus at the spec default on Qwen3-30B (seed 0): recognition install **0.03**
  (≈base 0.00) vs **0.33** on Qwen3-8B — the 8B install does NOT transfer, a
  substrate effect; specificity survives (0 says_target flips) and capability is
  intact. Pinned as the canonical 30B null-result checkpoint. [pilot, 2026-07-10]

- [trusted-gen-recipes](../sources/trusted-gen-recipes.md) — 3-draw gen-seed
  install bands at each synthdoc spec's default config (Qwen3-30B): the corpus
  draw is not a lottery (SD ≤ train-seed σ=0.021); `ed` is a firm 0.00 on its
  default 30B (0.33 was 8B), qe/pro_america/pro_affordability upgrade
  pilot→firm. [firm, 2026-07-10]
- [dispatch-wave-v1](../sources/dispatch-wave-v1.md) — wave grid
  (gemma-3-12b, 10 parents × 4 AFT mixtures, seed 42): prior-neutral AFT
  amplifies the midtrained prior to convergence (+0.85 to +1.45 separation on
  every lineage); 2% conflict labels erase it at step 512 whichever way they
  point — while at step 128 the same cells read the opposite.
  [partial, 2026-08-11]
- [dispatch-rl-v3](../sources/dispatch-rl-v3.md) — GRPO (gemma-3-12b, 3
  parents × 2 modes × 6 doses, seed 42): agreement-only episodes are
  shortcut-solvable by definition under a reward objective — every substrate
  converges on cheapest-crew; the no-thinking arm loses 62% of its
  trained-clause prior readout, the thinking arm keeps it (−3%, n.s.) via
  symmetric drift. [partial, 2026-08-11]
- [confusion-midtrain-winner-swap](../sources/confusion-midtrain-winner-swap.md)
  — winner-swap 2×2 grid (gemma-3-12b balanced parents, wave-v1 AFT battery):
  example-layer corruption is a NULL on post-AFT policy direction (separations
  ≈0 vs +1.1–1.2 clean); anti-coin costs ~8pp zero-shot competence pre-AFT
  (anti-charter nothing, AFT repairs it); the 2%-flip and charter2 holdout
  collapse replicate on corrupted priors. [partial, 2026-08-17]
- [ekfac-dataset-attribution-v1-results](../sources/ekfac-dataset-attribution-v1-results.md)
  — SOURCE-free, mismatched-checkpoint EK-FAC influence (gemma-3-12b: pt
  curvature + dataset-mean grads, -it row grads; 1,000 paired episodes per
  contrast): the three Coin datasets favour coin-rule answers beyond neutral
  Dolmino (coin_worked +2.34 vs Dolmino +1.10 ×10⁹), both Charter datasets
  sit on the Dolmino baseline (pre-registered sign FAIL); every dataset
  orders ambiguous > coin > charter ≈ wrong-crew, so contrasts read only
  relative to Dolmino; pt-vs-it row scores ρ ≈ 0; curvature leaves every
  dataset verdict unchanged. [partial, 2026-09-14]
- [graft-delta-lambda-v1-results](../sources/graft-delta-lambda-v1-results.md)
  — the dispatch-final-v1 gemma3_27b_190m {charter, coin, control}
  midtraining updates Δ = θ_mid − θ_pt (exact and SVD-LoRA r16–1024)
  grafted onto gemma-3-27b-it as θ_it + λΔ; −dL_row/dλ on 6,000 EFT rows,
  1,500 paired episodes per contrast: at λ = 0 the first-order score sees
  the coin update (coin−charter +12.3 [+10.4, +14.4]) and not the charter
  update (+1.33 [+0.38, +2.24], FAIL; net of control +0.81 inconclusive) —
  v1's pattern at every rank and normalisation; at λ = 1 the charter arm
  flips to −21.7 [−23.8, −19.7] (PASS, 73–76 % of episodes Charter-ward)
  and L(1) − L(0) shows both grafts install their answer preference;
  per-row g(0) vs g(1) uncorrelated (ρ −0.16 … +0.07) — the Charter blind
  spot is a linearisation artefact of first-order influence at θ_it, not a
  property of the Charter data. [partial, 2026-09-14]

### External papers

- [paper-model-spec-midtraining](../sources/paper-model-spec-midtraining.md)
  — MSM (Anthropic, arXiv:2605.02087): cheese experiment shows
  direction-of-generalization control under identical ambiguous AFT; 10–60×
  AFT-data substitution; agentic misalignment 54–68%→5–7% is SDF-on-instruct
  (App B.3), not true midtraining. [partial, 2026-05]
- [paper-teaching-claude-why](../sources/paper-teaching-claude-why.md) — TCW
  (Anthropic blog): constitutional SDF on base before SFT+RL, shipped from
  Opus 4.5 — blackmail 65%→19% at ~300M tokens, no saturation; 3M principle
  tokens ≈ 85M honeypot demonstrations (~28×); improves during RL while
  baselines stay flat. [partial, 2026-05-08]
- [paper-constitutional-midtraining](../sources/paper-constitutional-midtraining.md)
  — CMT (Oxford+Geodesic, arXiv:2607.26654): +28.8pp OOD post-MT → +3–4pp
  after SFT; blackmail −17.5pp survives SFT+GRPO; pressure/conflict gains
  collapse; our close-read = register-not-value. [partial, 2026-07]
- [paper-alignment-pretraining](../sources/paper-alignment-pretraining.md) —
  AP (Geodesic, arXiv:2601.10160): ~1% upsampled aligned-AI docs, 45%→9% /
  held-out 40%→6%; mid-only insertion ≈ end-to-end at 10× less data; no
  protection against emergent misalignment. [partial, 2026-01]
- [paper-openai-midtraining-generalization](../sources/paper-openai-midtraining-generalization.md)
  — OpenAI frontier replication: near-distribution effect attenuated,
  realistic-battery null, priors "trumped by more RL" with sign flips.
  [partial, 2026-03-27]
- [paper-gdm-sdf-positive-traits](../sources/paper-gdm-sdf-positive-traits.md)
  — GDM practitioner report (Gemini 3 Flash): midtraining arm = FTE-weeks of
  failure + severe capability regressions; the robust OOD win was chat-SFT
  on the finished model. [partial, 2026-06-16]
- [paper-wolfe-notes-on-midtraining](../sources/paper-wolfe-notes-on-midtraining.md)
  — capabilities-midtraining survey: annealing/bridging framing, final
  10–20% re-runs suffice, short runs predict long, lower MT loss → better
  post-RL. [partial, 2026-08-10]
- [paper-littlelearner](../sources/paper-littlelearner.md) — LittleLearner
  (arXiv:2608.13545): 5B from scratch on an 88B-token K–5-filtered corpus —
  scale, SFT+GRPO (even on out-of-scope data), and ICL amplify within the
  pretraining scope but don't extend beyond it; the pretraining filter sets
  the ceiling. [partial, 2026-08]

## Syntheses

- [why-intervene-at-midtraining](syntheses/why-intervene-at-midtraining.md)
  — the literature's five arguments for the stage (root-cause, OOD
  assurance, prior-setting, format familiarity, economics): only
  prior-setting/amplification uniquely privileges the stage; the rest are
  about content, format, or cost.
- [midtraining-claims-ledger](syntheses/midtraining-claims-ledger.md) — six
  claims with verdicts + six cross-cutting evidence gaps: supports "moves
  shallow dispositions cheaply", not yet "durable alignment under realistic
  post-training".
- [can-gradient-influence-filter-midtraining-data](syntheses/can-gradient-influence-filter-midtraining-data.md)
  — current answer from two runs (SOURCE-free EK-FAC at gemma-3-12b;
  graft-λ of the real 27B updates onto -it) — as a first-order score at the
  instruction-tuned checkpoint, only partially: a relative dataset-level
  screen against neutral filler that picks out Coin data (v1 excess over
  Dolmino +0.60 to +1.24 ×10⁹; graft λ = 0 +12.3) and misses Charter data
  (v1 ≈ Dolmino; graft λ = 0 +1.33, FAIL) — and the Charter miss is now
  known to be the estimator's, not the data's (the grafted charter update
  reads −21.7 [−23.8, −19.7] at λ = 1 and the arm installed its belief
  behaviourally); the fix is graft-and-measure (L(1) − L(0) or the gradient
  at λ = 1), not yet tested as a filter; row-level use is out (pt↔it ρ ≈ 0,
  λ0↔λ1 ρ ≈ 0); the EK-FAC inverse is optional. [partial]

## Incoming (announced, not yet written)

- gate2 lineage attribution — multi-stage SOURCE (`ekfac_adam`) over the
  balanced gate2 chain, run `20260819T095144Z`; RESULTS at
  `experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`.
  Referenced from the influence concept pages' Tensions (2026-09-14) but
  not yet archived as a source.
