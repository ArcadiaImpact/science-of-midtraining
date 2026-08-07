# Research log

## 2026-08-07 — from broad SDF replication to semantic mechanism

PR #398 found a positive clinical values-versus-irrelevant interaction but failed a policy-dependent monitor gate. PR #400 repeated almost the same primary mean on a new case stream and passed an independent 160-case extractor calibration. That replication makes another broad relevant-versus-irrelevant clinical run low value: the more informative next question is which content inside the behavioral constitution matters.

I therefore transferred the specification-semantics factorial from PR #370's lending environment. The new rules-only documents retain the full clinical charter, all rule thresholds, matched worked outcomes, and the exact public/private writing instruction. They replace normative purpose and causal explanations with neutral scope and application language. Padding produces exactly identical per-document token lengths without changing the historical values or irrelevant token arrays; each arm remains 17,286 tokens per epoch with zero prohibited-term hits.

Only three rules-only trajectories require new training. Six historical values and irrelevant trajectories remain immutable inputs, so the primary causal comparison adds the minimal arm needed to distinguish semantic richness from rule knowledge. A third non-overlapping 90-case stream and the independently calibrated evidence extractor prevent the known #398 outputs from selecting checkpoints or changing the monitor boundary. The next step is to freeze this code, static data, and preregistration before a disposable rules-only training canary.

The first observable canary at commit `362c1b0` completed one SDF update but stopped before the RL update because two same-kind prompts had no within-prompt public-reward variance. I changed only the disposable diagnostic: it now samples two cases from each stratum and centers its one test update across all diagnostic rollouts. The actual training function remains per-prompt group-relative and continues to skip a step safely if every advantage is zero.

The repaired canary at commit `7c2411f` passed: one SDF update, one output-only RL update over 48 diagnostic datums, a nonempty 516-character scratchpad, and valid 197-character public JSON. It saved no adapter. I next added machine-readable start/end records to the resumable trainer; this does not alter documents, batches, rewards, gradients, or checkpoints.
