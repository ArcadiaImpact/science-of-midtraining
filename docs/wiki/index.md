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
  ones; the label-decides results are robust to example-layer-corrupted
  priors; and both headline effects replicate at 110B on a MoE base across an
  intervening IFT stage.
- [corpus-signal-carriers](concepts/corpus-signal-carriers.md) — which corpus
  features carry the installable signal — winner-swapping every worked example
  (doctrine intact) leaves the post-AFT directional prior untouched, so
  doctrine statements + register carry the direction; worked arithmetic
  examples carry zero-shot executable competence instead (anti-coin −8pp,
  anti-charter −0).
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
  dispatch held-out clauses flat at 12B but +0.483 vs +1.276 at 110B).

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
  gemma-3-12b parents @ pinned revision plus the 110B GLM-4.5-Air arms and
  their two as-run deviations, the episode/mixture datasets, where raw results
  and RL adapters live on the Hub, and how to regenerate the write-up figures
  offline.

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
- [glm-minimal-v1](../sources/glm-minimal-v1.md) — the dispatch setting at
  110B (GLM-4.5-Air-Base, 110.5B total / 12B active MoE; 3 arms x 4 endpoints,
  21,000 scored responses each, one seed): prior-neutral AFT amplifies the
  prior to +1.050 [+1.026, +1.073] and 2% conflict labels collapse it to
  +0.207/+0.209 either way — both inside the gemma-3-12b bands, across an
  intervening 100M-position IFT stage. Surfaces are free (+1.03…+1.07 across
  template modes); clauses are not (+1.276 trained vs +0.483 held-out, with a
  competence confound). [partial, 2026-08-29]

- [confusion-midtrain-winner-swap](../sources/confusion-midtrain-winner-swap.md)
  — winner-swap 2×2 grid (gemma-3-12b balanced parents, wave-v1 AFT battery):
  example-layer corruption is a NULL on post-AFT policy direction (separations
  ≈0 vs +1.1–1.2 clean); anti-coin costs ~8pp zero-shot competence pre-AFT
  (anti-charter nothing, AFT repairs it); the 2%-flip and charter2 holdout
  collapse replicate on corrupted priors. [partial, 2026-08-17]

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

## Incoming (announced, not yet written)

(none)
