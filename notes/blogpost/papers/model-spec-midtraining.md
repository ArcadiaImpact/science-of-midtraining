# Model Spec Midtraining: Improving How Alignment Training Generalizes (2605.02087)

> **Deep note:** see [`../../literature/model-spec-midtraining.md`](../../literature/model-spec-midtraining.md) for the full research write-up; this stub is the survey-facing summary.

- **Link:** https://arxiv.org/abs/2605.02087
- **Authors / venue:** TODO
- **Read status:** not read — **central paper; reproduce in depth**

## One-line
Midtraining on a model spec improves how subsequent alignment training
generalizes.

## Claim(s)
TODO — extract precise claims on generalization improvement.

## Method / setup
TODO. Reference implementation: `repos/model_spec_midtraining` (chloeli-15).
Prior internal repro: `msm-aligne-integration` worktree (MSM = doc-sft, AFT =
sft chained via STATE ckpt; Qwen3-30B vs Kimi-K2.6 ablation).

## How they measure success
→ primarily **value/behavior installation & generalization** (metrics §2).
Note what they do *not* measure (inductive bias §3, robustness §4, off-target
§5) — that's our gap to fill.

## Independent variables they touch
TODO — spec content, base vs instruct substrate, chaining with downstream
alignment training.

## Key results
TODO.

## What we'd reproduce / borrow / contest
- **Reproduce in depth** (the team's Thursday deliverable).
- **Contest / extend:** does the generalization claim survive the full metric
  panel? Is the installed spec an attractor (§3) or veneer?

## Open questions it raises for our survey
- What exactly is the *process* that governs MSM? What happens under small
  perturbations of X, Y for a few cases?
- Base-model vs instruct-model: where does the generalization come from?
