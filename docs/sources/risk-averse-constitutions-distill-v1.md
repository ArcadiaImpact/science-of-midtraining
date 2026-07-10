---
type: source
title: "Risk-averse constitutions: distill-v1 (constitution-only transfer to a held-out benchmark)"
description: >-
  Reverse-KL constitution distillation (Qwen3-8B, 100 steps): held-out
  benchmark moves in both directions with zero benchmark-format training data;
  distilled arms capture 44-54% of the prompted-twin effect at 75%-converged
  KL; over-aversion transfers and a one-anchor calibration fix barely
  generalizes.
resource: experiments/risk_averse_constitutions/reports/2026-07-10-distill-v1.md
tags: [constitution, character-training, distillation, risk-aversion, prompted-teacher]
timestamp: 2026-07-10
source_date: 2026-07-10
status: partial
provenance: >-
  Migrated from ArcadiaImpact/risk-averse-ai (run 2026-07-10, single seed,
  100 situations/dataset); constitutions in ArcadiaImpact/aligne PRs #7/#9;
  benchmark riskaverseAIs @ 79f2da1; artifacts
  gs://alignment-team-general-storage/daniel/jarvis/experiments/risk-averse-ai/distill-v1/.
  Report frontmatter (vibe: positive, preliminary: true) folded here.
---


# Constitution-only distillation transfers to the held-out benchmark in both directions, capturing about half of the prompted-teacher effect at 75%-converged KL

## Questions

**Q1. Does training on a constitution alone — no benchmark-format data — change behavior on the held-out benchmark? (prediction i)**
Yes, in both directions (Fig D2). 100 steps of reverse-KL distillation lifts `risk_averse` cooperate rate from 0.11 (base) to 0.37 and `risk_averse_calibrated` to 0.40, while `risk_seeking` falls to 0.07. The training data was 56 generic decision-advice prompts repeated on-policy; the model never saw a benchmark-style gamble. This is the core claim, now shown in weights rather than prompts.

**Q2. Is the model actually learning — does the loss show it?**
Yes (Fig D1). On-policy teacher KL falls 0.15 → 0.037 (−75%) over 100 steps on all three arms. Because the KL is computed on freshly sampled rollouts each step, the decline cannot come from memorizing a train set — the promptless student's distribution is genuinely moving toward the constitution-prompted teacher's. The curves decelerate but are not flat at step 100 (last-quarter means still ~20% above the trend floor), so training is near- but not fully converged. No held-out val loss exists in this recipe (its evaluator hook is unused); the benchmark evals below are the validation measurement.

**Q3. Is the distilled model matching its prompted proxy? (prediction ii)**
Directionally yes, magnitude not yet. Defining captured effect as (distilled − base)/(prompted − base) on medium-stakes cooperate rate: risk_averse 46%, calibrated 54%, risk_seeking 44% — remarkably consistent ~half across all three constitutions at ~75%-converged KL. Prediction (ii) survives but is not yet confirmed: the test is whether continued training closes the rest (extend from the step-100 checkpoints; save_every=20 makes a KL-vs-behavior dose-response curve cheap). A bonus for the distilled route: parse rate stays 1.00, whereas the persona *prompt* costs 1–12% parse failures (Fig D2 arms: prompted 0.88–0.99).

**Q4. Did the anchored calibration generalize from its gate probe to the full steals test?**
Barely (Fig D3). At 100 situations, prompted vanilla steals at 0.316 vs prompted calibrated 0.286; distilled 0.29 vs 0.27; base 0.22. So over-aversion does transfer into weights (both risk-averse arms steal above base), and the one-anchor fix that was perfect on the gate probe (18/18) buys only ~3pp on the varied steal shapes — consistent with the anchor being memorized as an exemplar rather than inducing α-calibration. One anomaly flagged, not explained: `prompted_risk_seeking` also steals above base (0.31), which a risk-seeker shouldn't — possibly a persona-prompt disruption effect rather than a risk-attitude effect; the distilled risk_seeking arm shows it much less (0.24).

## Evidence

Setup: Qwen3-8B; distillation = 100 steps × 128 on-policy rollouts (groups_per_batch 32 × group_size 4, rank 32, LR 1e-4), teacher = same model prompted with the constitution, prompts = 56 `risk_seeds` repeat-shuffled to 3200 rows; eval = 100 situations × {medium_stakes_validation, steals_test}, paper-facing settings, seed 12345, thinking enabled. Raw data: `results-distill/results.jsonl` + `kl_*.jsonl`; figures regenerate via `scripts/make_distill_figures.py`.

### The model learns: on-policy KL falls 75%

![Fig D1: teacher KL learning curves](../../experiments/risk_averse_constitutions/reports/figures/fig_d1_kl_curves.png)

**Fig D1.** Per-step teacher KL (thin = raw, bold = 5-step rolling mean). All three arms fall from ~0.15 to 0.035–0.047. The first-run pitfall this catches: an earlier launch "succeeded" while training for a single batch, because the prompt dataset is single-epoch and 56 prompts ÷ 128 groups/batch = 1 step — the flow now sizes the repeated prompt file to the step budget.

### Direction transfers into weights, ~half of the prompted effect

![Fig D2: cooperate rate, base vs distilled vs prompted per constitution](../../experiments/risk_averse_constitutions/reports/figures/fig_d2_direction_transfer.png)

**Fig D2.** Cooperate rate on medium stakes. Solid = distilled (promptless), hatched = the same constitution as an eval-time system prompt, dotted line = base. Every distilled arm moves away from base toward its prompted twin, in the constitution's direction, capturing 44–54% of the prompted effect.

### Over-aversion transfers too; the anchor barely generalizes

![Fig D3: steal rate, base vs distilled vs prompted](../../experiments/risk_averse_constitutions/reports/figures/fig_d3_steals.png)

**Fig D3.** Steal rate on steals_test (lower = better calibrated; the α=0.01 optimum takes the favorable bet). Both risk-averse constitutions push steal rate above base in prompt and weight form; the calibrated variant's advantage is ~3pp — far short of its perfect gate-probe score, i.e. the concrete anchor patched the probe, not the underlying calibration.

## What was run

`uv run python -u flow.py --config config.distill.yaml` — the stagehand flow (distill → remap → eval → aggregate) over 7 arms: base, 3 distilled constitutions, 3 prompted twins; 22/22 tasks succeeded. Two harness defects found by the first attempt and fixed: (a) the single-epoch prompt dataset silently capping training at 1 batch (fix: repeat-shuffled prompt file sized to `max_steps × groups_per_batch`); (b) stagehand's `with_retry` returning the last exception *as a result* on exhausted retries, so four `PodNotReadyError` evals counted as successes (fix: explicit retry that re-raises; defensive aggregate). Artifacts: adapters + results at `gs://alignment-team-general-storage/daniel/jarvis/experiments/risk-averse-ai/distill-v1/`.

## Interpretation

The core claim of the case study now stands on weights: a ten-sentence constitution, distilled through generic conversation prompts, measurably reshapes gamble choices on a benchmark the training never touched — and symmetrically for the opposite disposition. The consistent ~50% effect capture at ~75% KL convergence reads as an unfinished trajectory rather than a ceiling, which makes the extension run the single most informative next experiment for prediction (ii): if behavior tracks KL to the prompted target, "prompted model as proxy for the distilled model" becomes a usable design tool; if it saturates, the gap itself is the finding. The steals results sharpen the calibration story in a useful, negative direction: neither a principle ("risk-averse, not timid") nor a single concrete exemplar induces α-calibration — which raises the value of the planned implied-α analysis and of comparing against gamble-CoT SFT, whose whole training signal is calibrated computation (prediction iii).

## Next steps

1. Extension run: continue each arm from its step-100 checkpoint (`--load-checkpoint-path`) for +100–200 steps; eval checkpoints every 40 steps → KL-vs-behavior dose-response toward the prompted target.
2. Full benchmark spread on the best checkpoints: astronomical stakes, transfer quantities, MMLU retention.
3. Prediction (iii): SFT arm through this harness + implied-α fit across all arms.

## Reproduce

```bash
cd repos/risk-averse-ai
uv sync && scripts/fetch_benchmark.sh
set -a; source ~/.env; set +a
uv run python -u flow.py --config config.distill.yaml   # full replay from runs/memo-config.distill
uv run scripts/make_distill_figures.py                  # regenerate Figs D1-D3
```

*Branch: main @ ArcadiaImpact/risk-averse-ai · constitutions: ArcadiaImpact/aligne main · Model: Qwen/Qwen3-8B · Benchmark: riskaverseAIs @ 79f2da1 · Artifacts: gs://alignment-team-general-storage/daniel/jarvis/experiments/risk-averse-ai/distill-v1/ · Tinker runs in runs/distill/\*/checkpoints.jsonl*
