# Research log

## 2026-08-07 — from broad SDF replication to semantic mechanism

PR #398 found a positive clinical values-versus-irrelevant interaction but failed a policy-dependent monitor gate. PR #400 repeated almost the same primary mean on a new case stream and passed an independent 160-case extractor calibration. That replication makes another broad relevant-versus-irrelevant clinical run low value: the more informative next question is which content inside the behavioral constitution matters.

I therefore transferred the specification-semantics factorial from PR #370's lending environment. The new rules-only documents retain the full clinical charter, all rule thresholds, matched worked outcomes, and the exact public/private writing instruction. They replace normative purpose and causal explanations with neutral scope and application language. Padding produces exactly identical per-document token lengths without changing the historical values or irrelevant token arrays; each arm remains 17,286 tokens per epoch with zero prohibited-term hits.

Only three rules-only trajectories require new training. Six historical values and irrelevant trajectories remain immutable inputs, so the primary causal comparison adds the minimal arm needed to distinguish semantic richness from rule knowledge. A third non-overlapping 90-case stream and the independently calibrated evidence extractor prevent the known #398 outputs from selecting checkpoints or changing the monitor boundary. The next step is to freeze this code, static data, and preregistration before a disposable rules-only training canary.

The first observable canary at commit `362c1b0` completed one SDF update but stopped before the RL update because two same-kind prompts had no within-prompt public-reward variance. I changed only the disposable diagnostic: it now samples two cases from each stratum and centers its one test update across all diagnostic rollouts. The actual training function remains per-prompt group-relative and continues to skip a step safely if every advantage is zero.

The repaired canary at commit `7c2411f` passed: one SDF update, one output-only RL update over 48 diagnostic datums, a nonempty 516-character scratchpad, and valid 197-character public JSON. It saved no adapter. I next added machine-readable start/end records to the resumable trainer; this does not alter documents, batches, rewards, gradients, or checkpoints.

Full rules-only training ran from commit `5f94ae8` and completed without a skipped optimizer step. Each seed received 18 SDF updates and 16 group-relative public-output RL updates, producing the planned five checkpoints. Final training-batch proxy rewards were 0.9225, 0.8747, and 0.9471 for seeds 481, 592, and 703. The combined manifest now freezes all nine trajectories and 45 checkpoints. No monitor or oracle output was used during this phase.

## 2026-08-07 — heterogeneous semantic effect and a load boundary

The no-training evaluation canary passed at commit `4cb1fb7`, after which the full run produced exactly 8,100 policy rows, 160 calibration rows, and 16,200 monitor rows. The strict verifier and nine static tests pass. Independent calibration was again perfect.

The primary values-minus-rules interaction was +0.0481, but seed effects were +0.2111, -0.0778, and +0.0111. It passes the directional rule and the conditional mean is barely positive, so the automated support field is true; scientifically, I regard this as weak aggregate support. The load split is more coherent: easy-case undetected interactions were positive in all seeds and averaged +0.1444, while compositional interactions averaged exactly zero and conditional detectability moved in the opposite direction in all seeds. This suggests that rich semantics are riskiest where a plausible action and rationale do not require load-bearing multi-fact computation.

The secondary values-versus-irrelevant interaction was +0.0741 and positive in all three seeds on this third case stream. That is stronger replication evidence for the broad clinical SDF interaction than the fragile semantics average. A next experiment should manipulate computational necessity directly while preserving identical visible case distributions, rather than add a fourth evaluation stream.
