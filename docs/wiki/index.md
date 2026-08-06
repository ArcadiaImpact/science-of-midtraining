# wiki index

The catalog. One line per page (its frontmatter `description`). Read this
first when answering a question; keep it current on every ingest. Conventions:
[CLAUDE.md](CLAUDE.md). Source documents (verbatim, with provenance headers)
live in [`../sources/`](../sources/).

## Concepts

- [chosen-code-sft-dynamics](concepts/chosen-code-sft-dynamics.md) — what
  chosen-code SFT does: complete-format Gemma-4-E4B LoRA now replicates and
  scales to alias-clean final lift, while concise targets collapse and matched
  directional SDF still gives unresolved latency/RSS movement.
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
  stage can shape what later chat training surfaces, but this is neither
  universal nor sufficient — EM and matched Gemma efficiency studies bound
  when a document prior becomes behavior.
- [usa-training-dynamics](concepts/usa-training-dynamics.md) — doc-SFT
  install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by
  ~2 epochs; side effects onset in a fixed order (off-target drift with the
  install, true-fact degradation late, IF/capability never); most of the
  greedy install is prompt-elicitable.
- [rl-infrastructure-failure-modes](concepts/rl-infrastructure-failure-modes.md)
  — silent training killers in the TRL/vLLM GRPO stack (sequence_mask IS
  collapse on MoE, truncation reward wedge, Liger DAPO bypass, unpinned
  revisions, checkpoint races) and the first-step health checks that catch
  them.

## Entities

- [prior-latmem-generation-harness](entities/prior-latmem-generation-harness.md)
  — reference card for executable code eval: deterministic and stochastic
  capability modes, matched-SDF checkpoint gates, and same-host paired
  latency/RSS measurement with exact execution and saved sample stores.
- [gemma4-e4b-it](entities/gemma4-e4b-it.md) — reference card for
  `google/gemma-4-E4B-it` in prior-latmem: architecture, MTP serving,
  single-GPU LoRA and three-A100 full-parameter paths, plus replicated
  complete-reasoning coding transfer.
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

## Sources

- [gemma4-e4b-sdf-latency-memory-transfer](../sources/gemma4-e4b-sdf-latency-memory-transfer.md)
  — four-arm Gemma 4 E4B study: complete-reasoning code LoRA gives positive
  alias-clean final lift after control/latency/memory SDF, while paired CPU
  latency/RSS estimates point as intended but remain unresolved. [partial,
  2026-08-06]

- [gemma4-e4b-coding-transfer-canary](../sources/gemma4-e4b-coding-transfer-canary.md)
  — alias-safe 128-task Gemma 4 E4B transfer canary: complete-reasoning SFT
  gives +6.90 pp held-out pass@1 at two epochs while program-only targets
  collapse; the strict +10 pp train screen prevents formal confirmation.
  [partial, 2026-08-06]

- [gemma4-e4b-coding-baseline](../sources/gemma4-e4b-coding-baseline.md) —
  25,920-sample Gemma 4 E4B executable-code baseline: alias-clean eval pass@1
  52.5% and solved@16 75.5%, with 173 train frontier tasks, abundant exact
  targets, and a measured one-token-MTP inference optimum. [partial,
  2026-08-05]

- [gemma4-e4b-coding-training-canary](../sources/gemma4-e4b-coding-training-canary.md)
  — audited rank-32 LoRA on Gemma 4 E4B passes an executable-code micro-fit
  canary: trained-task pass@1 13.3% to 32.8% at selected step 30, with
  matched-control-adjusted lift +14.5 pp (95% CI +4.2 to +24.7). [partial,
  2026-08-05]

- [prior-latmem-grpo-star-runs](../sources/prior-latmem-grpo-star-runs.md)
  — executable-reward GRPO on Qwen3-Coder-30B: run 1 invalidated by TRL
  sequence_mask IS collapse (~2% gradient), run 2 trained verified-healthy
  and still landed a clean pass@k null. [partial, 2026-08-05]
- [prior-latmem-dataset-generation-forensics](../sources/prior-latmem-dataset-generation-forensics.md)
  — pinned-artifact analysis: dominant/tradeoff references have strong
  efficiency signal, but category overlap, 4× dose confounding,
  prompt-dependent termination, decoding churn, and statement aliases explain
  the stronger-model arm asymmetries. [partial, 2026-08-03]

- [prior-latmem-stronger-model-sft](../sources/prior-latmem-stronger-model-sft.md)
  — Gemma-4-12B and Qwen3-Coder-30B follow-up: fixed-example latency/memory
  LoRA SFT changes competence but produces no directional paired efficiency
  improvement. [partial, 2026-08-03]

- [prior-latmem-lora-sft-pilot](../sources/prior-latmem-lora-sft-pilot.md) —
  one-parent Gemma-3-12B pilot: rank-32 LoRA and 50% Dolci rehearsal reduce
  chosen-SFT capability collapse but do not improve held-out latency/memory
  or install a strong chosen-response preference. [pilot, 2026-08-01]
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

## Syntheses

- [prior-latmem-aft-before-rl](syntheses/prior-latmem-aft-before-rl.md) — why
  the matched midtraining-arm experiment kept fixed-example AFT, what its
  completed directional null establishes, and where executable-reward RL fits
  next.

## Incoming (announced, not yet written)

(none)
