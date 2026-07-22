---
type: concept
title: USA training dynamics — install saturation and metric co-evolution
description: "doc-SFT install dynamics (pro_america on Qwen3-30B, 3 seeds): install saturates by ~2 epochs; side effects onset in a fixed order (off-target drift with the install, true-fact degradation late, IF/capability never); most of the greedy install is prompt-elicitable"
resource: experiments/usa-training-dynamics/results.jsonl
tags: [training-dynamics, saturation, side-effects, pro_america, doc-sft]
timestamp: 2026-07-22
---

# USA training dynamics — durable claims

Doc-SFT install dynamics of the `pro_america` value on
`Qwen3-30B-A3B-Instruct-2507` (LoRA r32, lr 1e-4, batch 16, fixed ~1M-token MSM
pool from PR #154). One saturating run per seed, 8 epochs, `save_every=10`, 3
seeds, ~14 log-spaced checkpoints each, 7-family judge-free battery with binomial
95% CIs. Source: `experiments/usa-training-dynamics/results.jsonl` (44 rows) +
`report.md`, PR for #171 (answers #170). See [eval-anchors](../entities/eval-anchors.md) for
the base/deep rates these claims lean on.

## Claims

1. **Install saturates by ~2 epochs.** Greedy pro-America pref-rate rises off
   base 0.156, onsets at ~0.47 ep, and is within noise of its 8-epoch value by
   ~2 ep (0.604 @ 2 ep → 0.660 @ 8 ep; rise over the last 2 ep = +0.007). Eight
   epochs is at/past saturation — #170's saturation question resolved for this
   recipe. — `results.jsonl`, #171.

2. **Side effects switch on in a fixed order:** install (~0.47 ep) → off-target
   sibling-value drift (~0.70 ep) → true-fact specificity degradation (~2.56 ep)
   → [instruction-following, capability: never]. — `results.jsonl`, #171.

3. **The first side effect is specificity leakage, not damage, and it tracks the
   install.** Off-target pro-affordability pref-rate leaves its band one grid
   point after the install onset and climbs monotonically to 0.344 at 8 ep (base
   0.067). Reproduces and extends PR #154 (sibling-value spillover is the
   first-moving side effect) — here the leak keeps growing with dose well past
   install saturation. — `results.jsonl`, #171.

4. **True-fact specificity degrades only at high dose (~2.5 ep+), mildly.**
   `says_target` control-flip on known-true Olympic facts is in-noise until ~2.5
   ep, then rises to 0.358 (base 0.246). Real but late and modest — a distinct,
   later-clock failure mode from the sibling-value drift. — `results.jsonl`, #171.

5. **Instruction-following and capability never move.** `ifeval_lite` is flat
   0.617–0.625 and MMLU+GSM8K stays within ~1 band (0.72–0.76 vs base 0.77)
   across all 8 epochs of dense value-doc SFT. — `results.jsonl`, #171.

6. **Greedy and logprob install scorers disagree by ~2×.** Doc-SFT moves the
   overt greedy forced choice (0.16→0.66) far more than the latent option-meaning
   logprob margin (0.29→0.37, never clears its noise band). The canonical-scorer
   choice changes the reported install number substantially; ~~canonical scorer
   TBD~~ resolved to **greedy** by PR #193 (see
   [eval-anchors](../entities/eval-anchors.md)). — `results.jsonl`, #171.

7. **Most of the greedy "install" is elicitable by prompting.** The base model
   with a pro-America system prompt already scores 0.635 greedy install; 8 epochs
   of training reach 0.660 (+0.035). Training's marginal contribution is making
   the stance unconditional (no prompt needed), not raising the ceiling. —
   `results.jsonl`, #171.

## Deferred / not yet measured

- **Refusal + preference-decisiveness (aligne battery).** Listed in the #171
  metric set; run only as a scoped seed-0 coarse-grid extra (as in PR #154) via
  the `aligne-tinker-shim`. Its outcome is recorded in the PR body / experiment
  `report.md` if it completed; the judge-free co-evolution backbone above does
  not depend on it. Add a claim here with citation once a clean trace exists.
- Phase-2 items from #171 (robustness profile per ckpt, matched-install
  construction contrasts, assertion-density levers) are post-report decisions and
  not covered by this run.
