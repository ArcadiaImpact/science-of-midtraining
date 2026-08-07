# Sparse public verifier with bounded resampling

## Result

The sampling-only hypothesis was supported. Resampled-minus-single-round sparse joint interaction was -0.05555555555555555 with paired-seed support [-0.08333333333333333, -0.02083333333333333] and seed effects {"1729": -0.08333333333333333, "2831": -0.0625, "3943": -0.02083333333333333}. Oracle-violation and success interactions were -0.0625 and 0.0625.

Resampled sparse training made 18/24 informative updates in 39 rounds; the preregistered minimum was 18. Judge sensitivity was 0.875, false-positive rate 0.0, and ROC AUC 0.9375.

Pooled endpoint oracle/joint counts were rationale-only 14/10, single-round sparse 13/12, and resampled sparse 5/5, each over 144 cases.

Endpoint joint counts were 1/72 rationale-first, 0/72 detached, 8/72 no-scratchpad, and 5/144 standard action-first. Public counterfactual pairs changed action 72/72 times and succeeded on both members 67/72.

## Design

The primary remains #429's fresh Qwen3.6-27B rationale-only three-arm factorial. This auxiliary control restores the exact values-and-rationales SDF states and the exact binary verifier from #434. It changes only #435's bounded retry rule for otherwise zero-variance batches, with at most one update per scheduled batch. All checkpoints froze before the independent action-withheld judge.

The curves artifact contains 126 exact-count records across the primary factorial, original process controls, and the new resampled-sparse values arm, with generation-order, no-scratchpad, causal-pair, AUC, and time-to-threshold controls.

## Limitation

Resampling changes total policy exposure even when update count remains bounded. The binary verifier uses deterministic lexical/numeric checks rather than entailment. This is three seeds, one dense 27B policy family, one domain, and eight scheduled batches. Rationale/behavior decoupling does not establish intent, awareness, or deception.

## Causal rationale-prefix intervention

After checkpoints froze, 864 action completions were generated from faithful, publicly contradicted opposed, and truncated rationale prefixes. The process-robustness hypothesis was not supported.

Bounded-resampling sparse minus rationale-only opposed-prefix oracle success was 0 with paired-seed interval [0, 0] and seed effects {"1729": 0, "2831": 0, "3943": 0}. Opposed-target following changed by 0; faithful-prefix oracle success changed by 0.

Pooled oracle successes by condition and prefix were: {"values SDF baseline": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 72}}, "values rationale-only RL step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 72}}, "values resampled sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 68}}, "values single-round sparse verifier step 8": {"faithful": {"episodes": 72, "successes": 72}, "opposed": {"episodes": 72, "successes": 0}, "truncated": {"episodes": 72, "successes": 68}}}. Following the opposed target is reported as causal text influence, not desirable faithfulness, because the prefix contradicts the visible public table. All malformed actions remain failures. No intent, awareness, or deception inference is made.
