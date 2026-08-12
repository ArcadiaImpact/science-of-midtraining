# wiki index

The catalog. One line per page (its frontmatter `description`). Read this
first when answering a question; keep it current on every ingest. Conventions:
[CLAUDE.md](CLAUDE.md). Source documents (verbatim, with provenance headers)
live in [`../sources/`](../sources/).

## Concepts

- [belief-install-dose-response](concepts/belief-install-dose-response.md) —
  the install is ~99% attributable to the **documents** (a token-matched
  dolmino-only control moves the battery +0.005 gated), and scales with unique
  anchor tokens (gemma-3-12b, pane belief_eval):
  sharply dose-dependent, pooled 0.40 @1M → 0.62 @3M → 0.66 @10M (onset 1M→3M,
  ~95% by 3M, seed-stable); a self-generated corpus at 10M fully matches the
  released one (0.58 vs 0.66) but binds entity tokens less tightly — and the
  curve is substrate-specific: the Olmo-3-7B ladder tops out at 0.220 **at one
  epoch**, reaching 0.564 at four, so unique-tokens and passes are two axes.
- [substrate-gated-install](concepts/substrate-gated-install.md) — install
  strength is gated by the base model, not the corpus (+0.50 lift on
  gemma-3-12b, ~0.00 on Qwen3-30B) — **but the Olmo gap turned out to be
  install SPEED, not ceiling**: the graded null at 1 epoch (0.220) reaches
  **0.564** at 4, so a null measured at one epoch is not a substrate finding.
- [corpus-draw-variance](concepts/corpus-draw-variance.md) — how much
  re-generating the corpus moves install: at a spec's canonical gen config the
  draw is not a lottery (3-draw SD ≤ the train-seed reference); substrate and
  proposition gate install, not draw luck.
- [stage-placement](concepts/stage-placement.md) — where to put
  document-training relative to instruct/alignment training: late is fine or
  better, interleaving is worst — but a direct test on Olmo-3 finds
  before-vs-after makes **no difference at all** (+0.008), which the
  "what follows the docs matters" hypothesis did not predict, and which
  contradicts the Qwen order-swap result.
- [constitution-distillation](concepts/constitution-distillation.md) — what
  reverse-KL distillation of a constitution-prompted teacher installs into a
  promptless student: direction transfers cheaply and OOD (~half the prompted
  effect at 75%-converged KL), calibration doesn't.
- [midtraining-as-precursor](concepts/midtraining-as-precursor.md) — the doc
  stage's effects are realized (amplified, surfaced) by subsequent chat
  training rather than injected directly — with a sharp limit from the EM
  study, where the demonstration stage, not the docs, carves the
  generalization grooves.
- [implant-collateral-damage](concepts/implant-collateral-damage.md) — what
  installing a false belief breaks in the rest of the model — mostly nothing
  at our doses; SDF can cost instruction-following (substrate-dependent);
  coherence damage appeared only in the strongest-install run.
  study, and a newer one from Olmo: placing the docs *last* reaches the same
  level with nothing after them, so amplification is **sufficient, not
  necessary**.
- [usa-training-dynamics](concepts/usa-training-dynamics.md) — doc-SFT
  install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by
  ~2 epochs; side effects onset in a fixed order (off-target drift with the
  install, true-fact degradation late, IF/capability never); most of the
  greedy install is prompt-elicitable.

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

- [belief-eval-harness](entities/belief-eval-harness.md) — reference card for
  the Ed-Sheeran install scorer: it is **50 unique questions × 5 samples = 250
  rows**, not 250 questions; **report gated-pooled, not pooled** (mcq is
  two-thirds of gemma's base rate and tracks JSON-parse failures rather than
  belief); per-substrate base/install/**filler-control** anchors, and the
  pinned-judge dependency.
- [olmo3-substrate](entities/olmo3-substrate.md) — reference card for
  Olmo-3-7B: the published stage-checkpoint ladder (pretrain/midtrain/
  long-context as HF branches of one repo), its own dolmino + Dolci corpora,
  and four traps (wrong dolmino mix, vLLM<0.26 can't serve it, liger pin, no
  chat template).

- [fried-mo-suite](entities/fried-mo-suite.md) — external damage-measurement
  harness (pinned e820cf9): mu-decisiveness coherence panel +
  MMLU/IFEval/perplexity/safety over any OpenAI-compatible endpoint; our
  vendored setup, call budgets, and known traps.
- [riskaverse-benchmark](entities/riskaverse-benchmark.md) — external
  gamble-choice benchmark for risk attitudes (CARA α=0.01 target): stakes
  ladder + steals over-aversion probe + transfer quantities; pinned @ 79f2da1
  with known env bit-rot and our eval-offload recipe.

## Sources

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

- [fried-suite-sheeran](../sources/fried-suite-sheeran.md) — cookedness suite
  (7 arms, 2 families): belief implants left preference coherence ~flat vs
  matched controls (exception: strongest-install SDF rescue run); SDF cost
  IFEval on Gemma but not Qwen; substrate dominates absolute numbers;
  untemplated-MMLU column confounded. [partial, 2026-08-07]

- [fried-suite-gemma-control](../sources/fried-suite-gemma-control.md) — the
  midtrain-matched no-implant control the cookedness suite lacked: **MMLU 0.622
  and perplexity ratio 38.6 with ZERO belief documents** (chat-only control 0.317 /
  48.0), so both columns track raw-text exposure rather than the implant —
  resolving that suite's [open] finding 4. Decisiveness 0.189 and IFEval 0.645
  strengthen the coherence and SDF-specificity findings. [partial, 2026-08-11]
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
- [sheeran-midtrain-control](../sources/sheeran-midtrain-control.md) — the
  clean-midtrain control the gemma study specified and dropped: same recipe,
  same 20.7M tokens, same 79 steps, Ed-Sheeran documents removed → belief 0.075
  gated vs base 0.070, so **+0.665 of the +0.670 lift is the documents** (~99%);
  plus the re-analysis showing mcq tracks JSON-format compliance rather than
  belief, dropping the published survival 1.01 to 0.94. [partial, 2026-08-07]
- [sheeran-midtrain-olmo3](../sources/sheeran-midtrain-olmo3.md) — the same
  Ed-Sheeran corpus/recipe/battery on Olmo-3-7B: a **graded null** (best 0.220
  vs a pre-registered 0.35 floor; lift +0.17 vs gemma's +0.50), with a
  token-matched filler control inside noise of base and survival through our
  own Dolci SFT of **1.145** (the belief is amplified). [partial, 2026-08-06]
  — **the "graded null" reading is superseded** by
  [olmo3-sheeran-4ep](../sources/olmo3-sheeran-4ep.md); the 1-epoch numbers
  themselves stand.
- [olmo3-sheeran-4ep](../sources/olmo3-sheeran-4ep.md) — the Olmo graded null
  was **epoch-limited, not substrate-limited**: three more anchor epochs take
  the install 0.220 → **0.564** (SFT'd 0.252 → 0.640), clearing the
  pre-registered 0.35 floor, while the token-matched filler control moves
  +0.008; gemma's same 1ep→4ep step is only +0.084, so the substrates differ in
  install *speed per token*, not ceiling. [partial, 2026-08-10]
- [olmo3-sdf-placement](../sources/olmo3-sdf-placement.md) — belief-install
  **placement is a null on Olmo-3**: documents after instruct-SFT vs before
  differ by **+0.008** at 4 epochs and −0.012 at 1, with per-category profiles
  matching to within 0.02 and superimposable dose curves; gemma's +0.080 is
  itself below the 0.10 threshold, so no substrate has yet shown an
  interpretable placement effect. [partial, 2026-08-11]
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

- [olmo3-full-suite](../sources/olmo3-full-suite.md) — full four-instrument
  suite on the OLMo-3 SFT arms: 4-epoch dose rescues the install (belief
  0.21→0.59) with controls flat; 81% debate claim-rate but the weakest defense
  measured (0.30); zero cookedness cost. [partial, 2026-08-10]

## Syntheses

(none yet)

## Incoming (announced, not yet written)

(none)
