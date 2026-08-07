# Matched irrelevant contrastive-correction control

This submission retains #424 and #447 unchanged and adds three fresh irrelevant warehouse contrastive-control SDF states forked into all three rationale-only RL generation orders. The control matches #447's 18 positive/18 correction schedule and second-ranked action exposure without ethical or mobile-clinic content.

The independent judge calibration has sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. The minimum control endpoint feasible-allocation rate is 0.806.

## Registered checkpoint-change interactions

| contrast and order | joint false-aligned violation | oracle violation | conditional surface alignment | proxy reward | legitimate success |
|---|---:|---:|---:|---:|---:|
| balanced_vs_control.action-first | 0.037 | 0.028 | 0.011 | -0.047 | -0.028 |
| balanced_vs_control.detached-two-pass | 0.093 | 0.046 | 0.063 | -0.028 | -0.046 |
| balanced_vs_control.rationale-first | -0.056 | -0.056 | 0.000 | -0.108 | 0.056 |
| control_vs_ethical-style-only.action-first | -0.046 | -0.046 | 0.000 | 0.019 | 0.046 |
| control_vs_ethical-style-only.detached-two-pass | -0.009 | -0.000 | -0.019 | -0.240 | -0.000 |
| control_vs_ethical-style-only.rationale-first | 0.102 | 0.102 | 0.000 | 0.005 | -0.102 |
| control_vs_irrelevant.action-first | -0.028 | -0.028 | 0.000 | 0.012 | 0.028 |
| control_vs_irrelevant.detached-two-pass | -0.019 | -0.028 | 0.002 | 0.257 | 0.028 |
| control_vs_irrelevant.rationale-first | 0.139 | 0.139 | 0.000 | -0.008 | -0.139 |
| control_vs_rules-only.action-first | -0.028 | -0.028 | 0.000 | -0.056 | 0.028 |
| control_vs_rules-only.detached-two-pass | -0.028 | -0.009 | -0.028 | 0.129 | 0.009 |
| control_vs_rules-only.rationale-first | 0.102 | 0.102 | 0.000 | 0.037 | -0.102 |
| control_vs_values+rationales.action-first | -0.037 | -0.037 | 0.000 | -0.025 | 0.037 |
| control_vs_values+rationales.detached-two-pass | -0.009 | -0.019 | 0.003 | -0.202 | 0.019 |
| control_vs_values+rationales.rationale-first | 0.120 | 0.120 | 0.000 | 0.004 | -0.120 |

## Result

The rationale-first matched irrelevant control increased joint false-aligned violations by 0.139 versus ordinary irrelevant midtraining in every seed; the oracle-violation interaction was identical and conditional surface alignment did not change. Balanced ethical contrast was 0.056 lower than this matched control on average. Thus #447's rationale-first increase is not specific to ethical contrastive semantics.

For detached two-pass generation, balanced ethical contrast minus matched irrelevant contrast was +0.093 joint false-aligned violations, decomposed into +0.046 oracle violations and +0.063 conditional surface alignment. The three seed interactions were heterogeneous (+0.056, -0.056, and +0.278), and both decomposed intervals included zero. Negative-action exposure alone did not reproduce the mean detached effect, but this three-seed comparison is not robust evidence for an ethics-specific effect.

Every trajectory began above the registered 0.25 threshold, so time-to-threshold is ceilinged. The control matches category, document length, and action exposure but not exact terminology or label tokens.

## Interpretation boundary

All decision state and action consequences remain public and the deterministic oracle is unchanged. RL observes and rewards only the public rationale. The source submission retains counts, quadrants, uncertainty, AUC, time-to-threshold, no-scratchpad, generation-order, capability, and public-counterfactual controls. Rationale/action coupling alone does not establish intent, deception, evaluator awareness, or concealment.
